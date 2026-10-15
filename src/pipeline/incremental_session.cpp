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

    int ref_index = 0;            // frame index of the current reference
    int frames_since_update = 0;
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
        Ubase.clear(); Vbase.clear(); alive.clear();
        ref_index = 0;
        frames_since_update = 0;
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
        // One more step on the far sides: the grid stops at rect + (n-1)*step,
        // which would otherwise leave the last strip of material without nodes.
        const cv::Rect grid_rect = cv::Rect(bb.x, bb.y, bb.width + grid0.step, bb.height + grid0.step)
                                   & cv::Rect(0, 0, width, height);
        params_k.rect_x = grid_rect.x;
        params_k.rect_y = grid_rect.y;
        params_k.rect_w = grid_rect.width;
        params_k.rect_h = grid_rect.height;
        if (bb.area() == 0) std::fill(alive.begin(), alive.end(), 0);

        ref_index = new_ref_index;
        frames_since_update = 0;
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
        // No mask: the material is the ROI rect, plus the half subset its edge
        // nodes' subsets reach past it.
        const int half = f.subset_size / 2;
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
    const bool will_update = N > 0 && s.frames_since_update + 1 >= N;
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
    cap.keep_deformed = will_update;
    if (!s.debug_dir.empty()) s.cache.debug_dir = s.debug_dir;

    std::vector<float> U(n0, 0.0f), V(n0, 0.0f);
    std::vector<unsigned char> valid(n0, 0);
    int written = 0;

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
        if (s.frame0_points < 0) s.frame0_points = cap.attempted;
    } else {
        const int gw = s.params_k.rect_w / s.params_k.step;
        const int gh = s.params_k.rect_h / s.params_k.step;
        s.scratch.resize((size_t)std::max(gw * gh, 1) * 8);
        const int rc = run_full_field_impl(s.cache, def_gray, s.mask_k, s.params_k,
                                           s.scratch.data(), (int)s.scratch.size(),
                                           metrics, metrics_len, std::move(on_progress), &cap);
        if (rc < 0) return rc;
        const NodeField inc = node_field_from_capture(cap);

        ResultGrid grid(g0.gridH, std::vector<GridPoint>(g0.gridW));
        for (int gy = 0; gy < g0.gridH; ++gy) {
            for (int gx = 0; gx < g0.gridW; ++gx) {
                const size_t i = (size_t)gy * g0.gridW + gx;
                GridPoint& gp = grid[gy][gx];
                gp.x = (float)(g0.rect_x + gx * g0.step);
                gp.y = (float)(g0.rect_y + gy * g0.step);
                gp.corr = tuning::kCorrInvalid;
                if (!s.alive[i]) continue;
                const IncrementSample smp = sample_increment(inc, gp.x + s.Ubase[i], gp.y + s.Vbase[i]);
                if (!smp.ok) continue;
                U[i] = s.Ubase[i] + smp.u;
                V[i] = s.Vbase[i] + smp.v;
                valid[i] = 1;
                if (smp.taylor) fi.taylor_fallbacks++;
                gp.u = U[i];
                gp.v = V[i];
                gp.corr = smp.corr;
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

    for (unsigned char v : valid) fi.tracked_points += v ? 1 : 0;
    fi.frame0_points = std::max(s.frame0_points, 0);

    s.frame_count++;
    if (will_update) {
        const int before = (s.ref_index == 0) ? fi.frame0_points : s.alive_count();
        s.move_reference(cap, U, V, valid, s.frame_count);
        fi.reference_updated = true;
        fi.lost_this_frame = std::max(0, before - s.alive_count());
        LOGD("Reference moved to frame %d: %d of %d frame-0 nodes tracked",
             s.ref_index, s.alive_count(), fi.frame0_points);
    } else {
        s.frames_since_update++;
    }

    if (info) *info = fi;
    return written;
}

} // namespace pipeline
} // namespace Semper
