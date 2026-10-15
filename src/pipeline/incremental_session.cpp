// Incremental (updated-reference) DIC over a frame sequence.
//
// Every frame is solved by the ordinary full-field pipeline against the
// current reference, on that reference's own integer grid. The increment is
// then sampled at each frame-0 material point's position in the reference and
// added to the point's total displacement at that reference:
//
//     U(X0, t) = U(X0, k) + u_k( X0 + U(X0, k) )
//
// Output stays on the frame-0 grid, so the packed layout, the VSG strain (now
// of the total displacement, i.e. total Lagrangian) and every consumer of the
// output are unchanged.
//
// While the reference is still frame 0 the solve writes straight into the
// caller's buffer: that is run_full_field itself, so update_interval == 0 is
// a fixed-reference run and the first frame of any interval is bit-identical
// to it. When the reference moves, the frame just solved is adopted as-is —
// the solver already prepared its image — and the frame-0 ROI is warped into
// it so the new reference's grid covers the same material.

#include <semper/incremental.hpp>
#include <semper/strain.hpp>
#include <semper/tuning.hpp>

#include "full_field_internal.hpp"
#include "incremental_internal.hpp"
#include "util/log.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>

#undef LOG_TAG
#define LOG_TAG "SemperIncremental"

namespace Semper {
namespace pipeline {

using namespace Semper::pipeline::internal;

struct IncrementalSession::Impl {
    ReferenceCache cache;
    IncrementalParams params;
    bool initialized = false;
    int width = 0, height = 0;

    cv::Mat user_mask0;       // as given, resized to the image; may be empty
    cv::Mat material_mask0;   // CV_8U 255/0: the material the frame-0 grid samples
    GridGeom grid0;
    int frame0_points = -1;   // nodes passing the frame-0 scan; known after frame 1

    // Total displacement of each frame-0 node at the current reference, and
    // whether it is known. A node unknown at a reference stays unknown.
    std::vector<float> Ubase, Vbase;
    std::vector<unsigned char> alive;
    // Pixels a node has travelled on fitted increments since it was last
    // measured and, if need be, confirmed (0 while it is). MATHEMATICS 9.5.
    std::vector<float> coast_travel;
    // Subset of a node near a gap, cut on a reference where it was clean;
    // kept while it coasts, to confirm it by when it can be measured again.
    std::unordered_map<size_t, CoastAnchor> anchors;

    int ref_index = 0;            // frame index of the current reference
    int frames_since_update = 0;
    int ref_reported = -1;        // points reported on the first frame against this reference
    int frame_count = 0;          // frames solved so far

    FullFieldParams params_k;     // grid on the current reference (k > 0)
    cv::Mat mask_k;
    std::vector<float> scratch;   // packed output of an increment solve (k > 0)
    std::string debug_dir;

    void clear() {
        cache.reset();
        initialized = false;
        width = height = 0;
        user_mask0.release();
        material_mask0.release();
        grid0 = GridGeom();
        frame0_points = -1;
        Ubase.clear(); Vbase.clear(); alive.clear(); coast_travel.clear();
        anchors.clear();
        ref_index = 0;
        frames_since_update = 0;
        ref_reported = -1;
        frame_count = 0;
        params_k = FullFieldParams();
        mask_k.release();
        scratch.clear();
    }

    int alive_count() const {
        int n = 0;
        for (unsigned char a : alive) n += a ? 1 : 0;
        return n;
    }

    // Move the reference to the frame just solved. U/V/valid are that frame's
    // totals on the frame-0 grid.
    void move_reference(FieldCapture &cap, const std::vector<float> &U,
                        const std::vector<float> &V,
                        const std::vector<unsigned char> &valid, int new_ref_index) {
        Ubase = U;
        Vbase = V;
        alive = valid;

        const int half = params.field.subset_size / 2;
        const int margin = (tuning::kIncMaskExtrapCells + 1) * grid0.step + half;
        mask_k = warp_material_mask(material_mask0, grid0, Ubase, Vbase, alive, margin);
        cache.adopt(std::move(cap.deformed), std::move(cap.deformed_gray), mask_k);

        params_k = params.field;
        const cv::Rect bb = cv::boundingRect(mask_k);
        // The new grid spans where the tracked frame-0 nodes now sit, plus one
        // step on the far sides: the grid stops at rect + (n-1)*step, which
        // would otherwise leave the last strip of points without a cell to
        // sample. Nothing wider: a node further out has a subset leaving the
        // mask, so the scan would skip it anyway, and a needlessly large grid
        // makes the anchor lattice sparser and the solve slower. Static
        // material gets back exactly the frame-0 grid.
        float min_x = 1e30f, min_y = 1e30f, max_x = -1e30f, max_y = -1e30f;
        for (int j = 0; j < grid0.gridH; ++j) {
            for (int i = 0; i < grid0.gridW; ++i) {
                const size_t n = (size_t)j * grid0.gridW + i;
                if (!alive[n]) continue;
                const float x = grid0.rect_x + i * grid0.step + Ubase[n];
                const float y = grid0.rect_y + j * grid0.step + Vbase[n];
                min_x = std::min(min_x, x); max_x = std::max(max_x, x);
                min_y = std::min(min_y, y); max_y = std::max(max_y, y);
            }
        }
        cv::Rect grid_rect;
        if (min_x <= max_x) {
            const int x0 = (int)std::floor(min_x), y0 = (int)std::floor(min_y);
            grid_rect = cv::Rect(x0, y0, (int)std::ceil(max_x) - x0 + grid0.step,
                                 (int)std::ceil(max_y) - y0 + grid0.step)
                        & cv::Rect(0, 0, width, height);
        }
        params_k.rect_x = grid_rect.x;
        params_k.rect_y = grid_rect.y;
        params_k.rect_w = grid_rect.width;
        params_k.rect_h = grid_rect.height;
        if (bb.area() == 0) std::fill(alive.begin(), alive.end(), 0);
        ref_index = new_ref_index;
        frames_since_update = 0;
        ref_reported = -1;
    }
};

IncrementalSession::IncrementalSession() : impl_(std::make_unique<Impl>()) {}
IncrementalSession::~IncrementalSession() = default;
IncrementalSession::IncrementalSession(IncrementalSession&&) noexcept = default;
IncrementalSession& IncrementalSession::operator=(IncrementalSession&&) noexcept = default;

bool IncrementalSession::initialized() const { return impl_ && impl_->initialized; }
int IncrementalSession::frames_processed() const { return impl_ ? impl_->frame_count : 0; }
int IncrementalSession::max_points() const {
    return impl_ ? impl_->grid0.gridW * impl_->grid0.gridH : 0;
}
void IncrementalSession::set_debug_dir(std::string dir) {
    if (impl_) impl_->debug_dir = std::move(dir);
}
void IncrementalSession::reset() {
    if (!impl_) impl_ = std::make_unique<Impl>();
    impl_->clear();
}

int IncrementalSession::init(const cv::Mat& ref_gray, const cv::Mat& roi_mask,
                             const IncrementalParams& params) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    Impl& s = *impl_;
    s.clear();

    if (ref_gray.empty() || ref_gray.type() != CV_8UC1) return -3;
    if (params.update_interval < 0) return -3;
    const FullFieldParams& f = params.field;
    if (f.step <= 0 || f.subset_size <= 0) return -2;
    const int gridW = f.rect_w / f.step, gridH = f.rect_h / f.step;
    if (gridW <= 0 || gridH <= 0) return -2;
    if (StrainCalculator::vsg_window_node_count(f.step, f.strain_window) < 3) return kBadStrainWindow;

    s.params = params;
    s.width = ref_gray.cols;
    s.height = ref_gray.rows;
    s.grid0 = GridGeom{gridW, gridH, f.rect_x, f.rect_y, f.step};

    if (!roi_mask.empty()) {
        cv::Mat m = roi_mask;
        if (m.cols != s.width || m.rows != s.height) {
            cv::resize(m, m, cv::Size(s.width, s.height), 0, 0, cv::INTER_NEAREST);
        }
        s.user_mask0 = m.clone();
        cv::compare(s.user_mask0, 128, s.material_mask0, cv::CMP_GE);
    } else {
        // No mask: the material is the ROI rect, plus the reach of its edge
        // nodes' subsets. Subsets stay axis-aligned on every reference while
        // the material turns, so a node's subset needs up to half*sqrt(2) of
        // material around it (at 45 deg), not just half; with only half the
        // edge nodes of a rotated ROI fall out of the scan and the chain
        // erodes inwards.
        const int half = (int)std::ceil(f.subset_size / 2 * std::sqrt(2.0)) + 1;
        s.material_mask0 = cv::Mat::zeros(s.height, s.width, CV_8U);
        const cv::Rect r = cv::Rect(f.rect_x - half, f.rect_y - half,
                                    f.rect_w + 2 * half, f.rect_h + 2 * half)
                           & cv::Rect(0, 0, s.width, s.height);
        s.material_mask0(r).setTo(255);
    }

    s.cache.set_from_gray(ref_gray, s.user_mask0);
    const size_t n = (size_t)gridW * gridH;
    s.Ubase.assign(n, 0.0f);
    s.Vbase.assign(n, 0.0f);
    s.alive.assign(n, 1);
    s.coast_travel.assign(n, 0.0f);
    s.initialized = true;
    return 0;
}

int IncrementalSession::process_frame(const cv::Mat& def_gray, float* output, int output_capacity,
                                      float* metrics, int metrics_len, CancelToken& cancel,
                                      ProgressCallback on_progress, IncrementalFrameInfo* info) {
    detail::ScopedCancelToken bind(cancel);
    return process_frame(def_gray, output, output_capacity, metrics, metrics_len,
                         std::move(on_progress), info);
}

int IncrementalSession::process_frame(const cv::Mat& def_gray, float* output, int output_capacity,
                                      float* metrics, int metrics_len,
                                      ProgressCallback on_progress, IncrementalFrameInfo* info) {
    if (!impl_ || !impl_->initialized || output == nullptr) return -3;
    Impl& s = *impl_;
    if (def_gray.empty() || def_gray.type() != CV_8UC1 ||
        def_gray.cols != s.width || def_gray.rows != s.height) {
        return -3;
    }

    IncrementalFrameInfo fi;
    fi.reference_frame = s.ref_index;
    const int N = s.params.update_interval;
    bool will_update = N > 0 && s.frames_since_update + 1 >= N;
    const GridGeom& g0 = s.grid0;
    const size_t n0 = (size_t)g0.gridW * g0.gridH;

    // Nothing left to follow on a moved reference: no solve, no points.
    if (s.ref_index > 0 && s.alive_count() == 0) {
        s.frame_count++;
        fi.frame0_points = std::max(s.frame0_points, 0);
        if (info) *info = fi;
        return 0;
    }

    FieldCapture cap;
    // With N > 1 any frame may become the reference early (kIncEarlyUpdateDrop).
    cap.keep_deformed = N > 0;
    if (!s.debug_dir.empty()) s.cache.debug_dir = s.debug_dir;

    std::vector<float> U(n0, 0.0f), V(n0, 0.0f);
    std::vector<unsigned char> valid(n0, 0);     // measured on this frame
    std::vector<unsigned char> carried(n0, 0);   // measured or coasting: kept at an update
    std::vector<float> coast_step(n0, 0.0f);     // |fitted increment| of a coasting node
    std::vector<AnchorNeed> need(n0, AnchorNeed::kKeep);   // for measured nodes
    int coasting = 0;
    int written = 0;

    // A node the solve did not reach coasts on the plane through the solved
    // increments around it, unreported, until the solve reaches it again or
    // it has travelled too far. (px, py): where the node sits on the solve's
    // reference; `inc` holds that solve's increments.
    const int subset_size = s.params.field.subset_size;
    const float max_travel = tuning::kIncCoastMaxTravelSubsets * (float)subset_size;
    auto carry = [&](size_t i, float du, float dv) {
        const float travel = std::hypot(du, dv);
        if (s.coast_travel[i] + travel > max_travel) return;
        U[i] = s.Ubase[i] + du;
        V[i] = s.Vbase[i] + dv;
        carried[i] = 1;
        coast_step[i] = travel;
        coasting++;
    };
    // Only a node that has been measured coasts: one that frame 1 cannot
    // measure may sit in a gap, or on texture too weak to correlate at all,
    // and nothing would ever confirm it. Nor does one that has left the image.
    auto coast = [&](size_t i, const NodeField &inc, float px, float py) {
        if (px < 0.0f || py < 0.0f || px > (float)(s.width - 1) || py > (float)(s.height - 1)) return;
        const float radius = (float)(tuning::kIncCoastRadiusCells * inc.step);
        const IncrementSample fit = fit_increment(inc, px, py, radius);
        if (!fit.ok) return;
        carry(i, fit.u, fit.v);
    };
    const float trusted_travel = tuning::kIncCoastTrustedTravelSubsets * (float)subset_size;
    int reanchored = 0;

    if (s.ref_index == 0) {
        // Same call as run_full_field, straight into the caller's buffer.
        const int rc = run_full_field_impl(s.cache, def_gray, s.user_mask0, s.params.field,
                                           output, output_capacity, metrics, metrics_len,
                                           std::move(on_progress), &cap);
        if (rc < 0) return rc;
        written = rc;
        const NodeField nf = node_field_from_capture(cap);
        U = nf.u;
        V = nf.v;
        valid = nf.valid;
        carried = valid;
        if (s.frame0_points < 0) s.frame0_points = cap.attempted;
        const GapDistance gd = gap_distance(nf);
        for (size_t i = 0; i < n0; ++i) {
            if (valid[i]) {
                need[i] = anchor_need(nf, gd, nf.node_x((int)(i % g0.gridW)),
                                      nf.node_y((int)(i / g0.gridW)), subset_size);
            }
        }
    } else {
        const int gw = s.params_k.rect_w / s.params_k.step;
        const int gh = s.params_k.rect_h / s.params_k.step;
        s.scratch.resize((size_t)std::max(gw * gh, 1) * 8);
        const int rc = run_full_field_impl(s.cache, def_gray, s.mask_k, s.params_k,
                                           s.scratch.data(), (int)s.scratch.size(),
                                           metrics, metrics_len, std::move(on_progress), &cap);
        if (rc < 0) return rc;
        const NodeField inc = node_field_from_capture(cap);
        const GapDistance gd = gap_distance(inc);

        ResultGrid grid(g0.gridH, std::vector<GridPoint>(g0.gridW));
        for (int gy = 0; gy < g0.gridH; ++gy) {
            for (int gx = 0; gx < g0.gridW; ++gx) {
                const size_t i = (size_t)gy * g0.gridW + gx;
                GridPoint& gp = grid[gy][gx];
                gp.x = (float)(g0.rect_x + gx * g0.step);
                gp.y = (float)(g0.rect_y + gy * g0.step);
                gp.corr = tuning::kCorrInvalid;
                if (!s.alive[i]) continue;
                const float px = gp.x + s.Ubase[i], py = gp.y + s.Vbase[i];
                const IncrementSample smp = sample_increment(inc, px, py);
                if (!smp.ok) {
                    coast(i, inc, px, py);   // no solved node close enough
                    continue;
                }
                U[i] = s.Ubase[i] + smp.u;
                V[i] = s.Vbase[i] + smp.v;
                float corr = smp.corr;
                need[i] = anchor_need(inc, gd, px, py, subset_size);
                if (s.coast_travel[i] > trusted_travel) {
                    // It coasted across an update, so it sits where a fitted
                    // plane put it, and this increment was measured from
                    // there. Report it again only once its subset, from where
                    // it was last clean, matches it here, and only try where
                    // its subset here is clean too: one still reaching into
                    // the gap can match with a bias.
                    const auto a = s.anchors.find(i);
                    const float max_shift = std::max(tuning::kIncReanchorMinShiftPx,
                                                     tuning::kIncReanchorMaxShiftFraction * s.coast_travel[i]);
                    float x = 0.0f, y = 0.0f;
                    if (need[i] == AnchorNeed::kKeep || a == s.anchors.end() || !cap.deformed ||
                        !match_anchor(a->second, *cap.deformed, subset_size,
                                      s.params.field.use_6x6_interpolator, gp.x + U[i], gp.y + V[i],
                                      max_shift, x, y, corr)) {
                        // Unconfirmed: it moves on the measured increment,
                        // unreported, and adds nothing to its coasted
                        // distance. It is tried again on the next frame.
                        carried[i] = 1;
                        coasting++;
                        continue;
                    }
                    U[i] = x - gp.x;
                    V[i] = y - gp.y;
                    reanchored++;
                }
                valid[i] = 1;
                carried[i] = 1;
                if (smp.taylor) fi.taylor_fallbacks++;
                gp.u = U[i];
                gp.v = V[i];
                gp.corr = corr;
                gp.solved = true;
            }
        }

        // pack_full_field_output runs the displacement median test on the
        // total field and drops what it rejects from this frame's output.
        // That rejection is deliberately not fed back into `valid`: losing a
        // node at a reference update is permanent, so a false positive (an
        // edge node with few neighbours, under noise) would compound over a
        // long sequence. Increments are already median-tested before they are
        // sampled (node_field_from_capture).
        const PackedFieldResult packed = pack_full_field_output(
                grid, g0.gridW, g0.gridH, g0.step, s.params.field.strain_window,
                output, output_capacity);
        written = packed.valid_count;
    }

    for (unsigned char v : carried) fi.tracked_points += v ? 1 : 0;
    fi.frame0_points = std::max(s.frame0_points, 0);

    // Early update on a drop in the points reported (written), not in the
    // nodes sampled: a sparse loss of increment nodes still samples through
    // the Taylor fallback but fails the median test and the strain window.
    if (!will_update && N > 1) {
        if (s.ref_reported < 0) {
            s.ref_reported = written;
        } else if ((float)written < (1.0f - tuning::kIncEarlyUpdateDrop) * (float)s.ref_reported) {
            will_update = true;
        }
    }

    s.frame_count++;
    if (will_update) {
        const int before = (s.ref_index == 0) ? fi.frame0_points : s.alive_count();
        // Anchors are cut on the frame about to become the reference, where
        // a measured node sits at X0 + U. A coasting node keeps the one it has.
        for (size_t i = 0; i < n0; ++i) {
            s.coast_travel[i] = (carried[i] && !valid[i]) ? s.coast_travel[i] + coast_step[i] : 0.0f;
            if (!carried[i] || (valid[i] && need[i] == AnchorNeed::kDrop)) {
                s.anchors.erase(i);
            } else if (valid[i] && need[i] == AnchorNeed::kRefresh) {
                CoastAnchor a;
                if (make_anchor(cap.deformed_gray, g0.rect_x + (int)(i % g0.gridW) * g0.step + U[i],
                                g0.rect_y + (int)(i / g0.gridW) * g0.step + V[i], subset_size, a)) {
                    s.anchors[i] = std::move(a);
                }
            }
        }
        s.move_reference(cap, U, V, carried, s.frame_count);
        fi.reference_updated = true;
        fi.lost_this_frame = std::max(0, before - s.alive_count());
        LOGD("Reference moved to frame %d: %d of %d frame-0 nodes tracked (%d coasting, %d re-anchored)",
             s.ref_index, s.alive_count(), fi.frame0_points, coasting, reanchored);
    } else {
        s.frames_since_update++;
    }

    if (info) *info = fi;
    return written;
}

} // namespace pipeline
} // namespace Semper
