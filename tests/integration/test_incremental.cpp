// IncrementalSession: updated-reference DIC over a frame sequence.
//
// Ground truth is exact (framework/sequence.h): frames are rendered from one
// analytic speckle under a known motion, so total displacement and
// Green-Lagrange strain at every frame-0 grid point are known in closed form.
//
// Contracts pinned here:
//  - update_interval = 0 is run_full_field, bit for bit, every frame;
//  - the first frame of any interval is run_full_field, bit for bit;
//  - the reference moves on the documented schedule;
//  - a node lost at a reference update never comes back;
//  - the same sequence gives the same bits twice;
//  - errors and cancellation leave the session unchanged;
//  - it follows large deformation a fixed reference mismatches on, and stays
//    bounded over a 160-frame horizon.
//
// SEMPER_INC_TRACE=1 prints per-frame error, an error map and a bias fit.
#include "framework/test_framework.h"

#if defined(DIC_HAVE_OPENCV)

#include "framework/sequence.h"

#include <semper/cancel.hpp>
#include <semper/incremental.hpp>
#include <semper/pipeline.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using Semper::pipeline::CancelToken;
using Semper::pipeline::FullFieldParams;
using Semper::pipeline::IncrementalFrameInfo;
using Semper::pipeline::IncrementalParams;
using Semper::pipeline::IncrementalSession;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::kBadStrainWindow;
using Semper::pipeline::kCancelled;
using Semper::pipeline::run_full_field;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Sequence {
    int w = 0, h = 0;
    cv::Mat ref;
    std::vector<cv::Mat> frames;
    std::vector<dictest::Motion> motion;   // motion[t] maps frame 0 -> frames[t]
};

// Frames 1..T of stretch e(t) = e_max * t/T with rotation theta(t) and a
// rigid drift (tx, ty) * t/T likewise, about the image centre.
Sequence stretch_rotate_sequence(int w, int h, int T, double e_max, double theta_max_deg,
                                 double nu = 0.3, float noise = 0.0f, unsigned seed = 31,
                                 double tx_max = 0.0, double ty_max = 0.0) {
    Sequence s;
    s.w = w; s.h = h;
    const dictest::SmoothSpeckleField field(seed, w, h);
    const dictest::Vec2 c{w / 2.0, h / 2.0};
    s.ref = dictest::render_frame(field, w, h, dictest::identity_motion(), noise, 1000);
    for (int t = 1; t <= T; ++t) {
        const double f = (double)t / T;
        dictest::Motion m = dictest::stretch_rotate(e_max * f, nu, theta_max_deg * f * kPi / 180.0,
                                                    dictest::Vec2{tx_max * f, ty_max * f}, c);
        s.frames.push_back(dictest::render_frame(field, w, h, m, noise, 1000 + t));
        s.motion.push_back(std::move(m));
    }
    return s;
}

FullFieldParams centred_params(int w, int h, int roi, int step = 10, int subset = 21) {
    FullFieldParams p;
    p.rect_x = (w - roi) / 2;
    p.rect_y = (h - roi) / 2;
    p.rect_w = roi;
    p.rect_h = roi;
    p.step = step;
    p.subset_size = subset;
    p.strain_window = 2 * step + 1;
    p.use_6x6_interpolator = false;
    return p;
}

int capacity_for(const FullFieldParams &p) { return (p.rect_w / p.step) * (p.rect_h / p.step) * 8; }

struct Frame {
    int rc = 0;
    std::vector<float> out;
    IncrementalFrameInfo info;
    float attempted = 0, solved = 0;   // metrics slots 0 and 1: the increment solve
};

bool bit_equal(const Frame &a, const Frame &b) {
    if (a.rc != b.rc) return false;
    if (a.rc <= 0) return true;
    return std::memcmp(a.out.data(), b.out.data(), (size_t)a.rc * 8 * sizeof(float)) == 0;
}

std::vector<Frame> run_session(const Sequence &s, const cv::Mat &mask, const FullFieldParams &p,
                               int interval) {
    IncrementalSession session;
    IncrementalParams ip;
    ip.field = p;
    ip.update_interval = interval;
    std::vector<Frame> frames;
    if (session.init(s.ref, mask, ip) != 0) return frames;
    const int cap = capacity_for(p);
    for (const cv::Mat &f : s.frames) {
        Frame fr;
        fr.out.assign((size_t)cap, 0.0f);
        float metrics[23] = {};
        fr.rc = session.process_frame(f, fr.out.data(), cap, metrics, 23, nullptr, &fr.info);
        fr.attempted = metrics[0];
        fr.solved = metrics[1];
        frames.push_back(std::move(fr));
    }
    return frames;
}

std::vector<Frame> run_fixed(const Sequence &s, const cv::Mat &mask, const FullFieldParams &p) {
    ReferenceCache cache;
    cache.set_from_gray(s.ref, mask);
    const int cap = capacity_for(p);
    std::vector<Frame> frames;
    for (const cv::Mat &f : s.frames) {
        Frame fr;
        fr.out.assign((size_t)cap, 0.0f);
        float metrics[23] = {};
        fr.rc = run_full_field(cache, f, mask, p, fr.out.data(), cap, metrics, 23, nullptr);
        frames.push_back(std::move(fr));
    }
    return frames;
}

struct Accuracy {
    int points = 0;
    int within = 0;           // |error| <= the tolerance passed to evaluate
    double rms_u = 0;         // vector displacement error, px
    double rms_e = 0;         // over exx, eyy, exy
    double max_u = 0;
};

Accuracy evaluate(const Frame &fr, const dictest::Motion &m, double within_px = 0.2) {
    Accuracy a;
    double su = 0, se = 0;
    for (int i = 0; i < std::max(fr.rc, 0); ++i) {
        const float *p = &fr.out[(size_t)i * 8];
        const dictest::Vec2 d = m.displacement(p[0], p[1]);
        const double eu = p[2] - d.x, ev = p[3] - d.y;
        const double err = std::sqrt(eu * eu + ev * ev);
        double exx, eyy, exy;
        m.strain(p[0], p[1], exx, eyy, exy);
        const double e1 = p[4] - exx, e2 = p[5] - eyy, e3 = p[6] - exy;
        su += err * err;
        se += (e1 * e1 + e2 * e2 + e3 * e3) / 3.0;
        a.max_u = std::max(a.max_u, err);
        if (err <= within_px) a.within++;
        a.points++;
    }
    if (a.points > 0) {
        a.rms_u = std::sqrt(su / a.points);
        a.rms_e = std::sqrt(se / a.points);
    }
    return a;
}

cv::Mat disc_mask(int w, int h, double r_out, double r_in) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double r2 = (x - w / 2.0) * (x - w / 2.0) + (y - h / 2.0) * (y - h / 2.0);
            if (r2 < r_out * r_out && r2 >= r_in * r_in) m.at<uchar>(y, x) = 255;
        }
    return m;
}

// Resident set size in bytes (Linux); 0 where unsupported.
long resident_bytes() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    long pages_total = 0, pages_resident = 0;
    if (statm >> pages_total >> pages_resident) return pages_resident * 4096L;
#endif
    return 0;
}

} // namespace

TEST_CASE(Incremental, IntervalZero_BitIdenticalToRunFullField) {
    const Sequence s = stretch_rotate_sequence(320, 320, 5, 0.04, 2.0);
    const FullFieldParams p = centred_params(320, 320, 160);
    for (int with_mask = 0; with_mask <= 1; ++with_mask) {
        const cv::Mat mask = with_mask ? disc_mask(320, 320, 85, 15) : cv::Mat();
        const std::vector<Frame> fixed = run_fixed(s, mask, p);
        const std::vector<Frame> inc = run_session(s, mask, p, 0);
        REQUIRE(inc.size() == fixed.size());
        for (size_t t = 0; t < fixed.size(); ++t) {
            CHECK(fixed[t].rc > 0);
            CHECK(bit_equal(inc[t], fixed[t]));
            CHECK(inc[t].info.reference_frame == 0);
            CHECK(!inc[t].info.reference_updated);
        }
    }
}

TEST_CASE(Incremental, FirstFrame_BitIdenticalToRunFullField) {
    const Sequence s = stretch_rotate_sequence(320, 320, 2, 0.02, 1.0);
    const FullFieldParams p = centred_params(320, 320, 160);
    const std::vector<Frame> fixed = run_fixed(s, cv::Mat(), p);
    for (int interval : {1, 3}) {
        const std::vector<Frame> inc = run_session(s, cv::Mat(), p, interval);
        REQUIRE(!inc.empty());
        CHECK(bit_equal(inc[0], fixed[0]));
    }
}

TEST_CASE(Incremental, UpdateSchedule_FollowsInterval) {
    const Sequence s = stretch_rotate_sequence(320, 320, 8, 0.04, 2.0);
    const FullFieldParams p = centred_params(320, 320, 160);
    const std::vector<Frame> n1 = run_session(s, cv::Mat(), p, 1);
    const std::vector<Frame> n3 = run_session(s, cv::Mat(), p, 3);
    REQUIRE(n1.size() == 8 && n3.size() == 8);
    for (int t = 0; t < 8; ++t) {
        // Frame t+1 (1-based) is solved against frame t under N = 1 ...
        CHECK(n1[t].info.reference_frame == t);
        CHECK(n1[t].info.reference_updated);
        // ... and against the last multiple of 3 under N = 3.
        CHECK(n3[t].info.reference_frame == (t / 3) * 3);
        CHECK(n3[t].info.reference_updated == ((t + 1) % 3 == 0));
        CHECK(n1[t].rc > 0);
        CHECK(n3[t].rc > 0);
    }
}

TEST_CASE(Incremental, StretchRotation_TracksWhereFixedReferenceDegrades) {
    // 40% stretch and 30 degrees over 20 frames: past where a fixed
    // reference's square subsets still match the deformed pattern.
    const int T = 20;
    const Sequence s = stretch_rotate_sequence(448, 448, T, 0.40, 30.0);
    const FullFieldParams p = centred_params(448, 448, 160);
    const std::vector<Frame> inc = run_session(s, cv::Mat(), p, 1);
    const std::vector<Frame> fixed = run_fixed(s, cv::Mat(), p);
    REQUIRE((int)inc.size() == T);

    if (std::getenv("SEMPER_INC_TRACE")) {
        for (int t = 0; t < T; ++t) {
            const Accuracy ai = evaluate(inc[t], s.motion[t]);
            const Accuracy af = evaluate(fixed[t], s.motion[t]);
            std::printf("    t=%2d inc: rc=%3d tracked=%3d taylor=%3d rms=%.4f max=%.3f | fixed: rc=%3d rms=%.4f max=%.3f\n",
                        t + 1, inc[t].rc, inc[t].info.tracked_points, inc[t].info.taylor_fallbacks,
                        ai.rms_u, ai.max_u, fixed[t].rc, af.rms_u, af.max_u);
        }
        // Error map of the last frame, in 0.1 px (frame-0 grid; '.' = absent).
        const int gw = p.rect_w / p.step, gh = p.rect_h / p.step;
        std::vector<int> map((size_t)gw * gh, -1);
        for (int i = 0; i < inc[T - 1].rc; ++i) {
            const float *q = &inc[T - 1].out[(size_t)i * 8];
            const dictest::Vec2 d = s.motion[T - 1].displacement(q[0], q[1]);
            const double e = std::hypot(q[2] - d.x, q[3] - d.y);
            const int gx = (int)((q[0] - p.rect_x) / p.step), gy = (int)((q[1] - p.rect_y) / p.step);
            map[(size_t)gy * gw + gx] = std::min(9, (int)(e * 10));
        }
        for (int gy = 0; gy < gh; ++gy) {
            std::printf("    ");
            for (int gx = 0; gx < gw; ++gx) {
                const int v = map[(size_t)gy * gw + gx];
                std::printf("%c", v < 0 ? '.' : (char)('0' + v));
            }
            std::printf("\n");
        }
    }
    const Accuracy a = evaluate(inc[T - 1], s.motion[T - 1]);
    const Accuracy f = evaluate(fixed[T - 1], s.motion[T - 1]);
    // VSG strain needs a nearly full window, so the outermost ring of nodes
    // never reaches the output in either mode; frame 1's count is the most
    // this grid can show.
    const int full = inc[0].rc;
    std::printf("  [stretch+rot T=%d] incremental: %d/%d pts, rms u %.4f px, max %.3f, rms E %.2e, "
                "within 0.2px %d | fixed: %d pts, rms u %.4f, within 0.2px %d\n",
                T, a.points, full, a.rms_u, a.max_u, a.rms_e, a.within, f.points, f.rms_u, f.within);
    REQUIRE(full > 0);
    CHECK(inc[T - 1].info.tracked_points >= (int)(0.98 * inc[T - 1].info.frame0_points));
    CHECK(a.points >= (int)(0.95 * full));
    CHECK(a.rms_u <= 0.10);
    CHECK(a.rms_e <= 5e-3);
    CHECK(a.within > f.within);
}

TEST_CASE(Incremental, Rotation90_200Frames_TracksTheTurningRoi) {
    // Steady rotation to 90 degrees, 0.45 per frame. Regressions guarded:
    // edge nodes falling out of the ref-k scan as the ROI turns (axis-aligned
    // subsets need half*sqrt(2) of material at 45 degrees), and the mask warp
    // blanking whole bands of rows (a per-pixel Newton inverse following a
    // false branch where the field folds or was held constant). Before those
    // fixes every mode lost the chain between 50 and 85 degrees.
    //
    // N = 1 is held to 45 degrees only: 100+ updates of a sub-pixel rotation
    // accumulate interpolation bias (MATHEMATICS 9.5). That is the cost the
    // benchmark measures, not a defect of the turning ROI.
    const int T = 200;
    const Sequence s = stretch_rotate_sequence(256, 256, T, 0.0, 90.0);
    FullFieldParams p = centred_params(256, 256, 110, 5);
    p.strain_window = 15;
    struct Case { int interval; bool six; int until; double min_within; double max_rms; };
    for (const Case c : {Case{5, true, T, 0.95, 0.08}, Case{10, false, T, 0.95, 0.20},
                         Case{1, true, T / 2, 0.90, 0.40}}) {
        p.use_6x6_interpolator = c.six;
        const std::vector<Frame> inc = run_session(s, cv::Mat(), p, c.interval);
        REQUIRE((int)inc.size() == T);
        if (std::getenv("SEMPER_INC_TRACE")) {
            for (int t = 0; t < T; ++t) {
                const IncrementalFrameInfo &i = inc[t].info;
                if ((t + 1) % 10 && i.lost_this_frame == 0 && inc[t].rc > 0) continue;
                const Accuracy at = evaluate(inc[t], s.motion[t]);
                std::printf("    t=%3d rc=%3d tracked=%3d/%d taylor=%3d lost=%3d rms=%.4f max=%.3f\n",
                            t + 1, inc[t].rc, i.tracked_points, i.frame0_points, i.taylor_fallbacks,
                            i.lost_this_frame, at.rms_u, at.max_u);
                if (inc[t].rc <= 0) break;
            }
        }
        const Frame &last = inc[c.until - 1];
        const Accuracy a = evaluate(last, s.motion[c.until - 1], 0.5);
        std::printf("  [rotation to %.0f deg N=%d %s] %d/%d pts, %d within 0.5 px, tracked %d/%d, "
                    "rms u %.4f px, max %.3f\n",
                    0.45 * c.until, c.interval, c.six ? "6x6" : "4x4", a.points, inc[0].rc, a.within,
                    last.info.tracked_points, last.info.frame0_points, a.rms_u, a.max_u);
        for (int t = 1; t < c.until; ++t) CHECK(inc[t].rc > 0);
        CHECK(last.info.tracked_points >= (int)(0.98 * last.info.frame0_points));
        // The rms carries one weakly textured corner of this speckle, which
        // drifts by a pixel or more over 100+ updates; everywhere else the
        // error stays under 0.1 px, hence the share within 0.5 px.
        CHECK(a.within >= (int)(c.min_within * inc[0].rc));
        CHECK(a.rms_u <= c.max_rms);
    }
}

TEST_CASE(Incremental, MaskedSequence_TracksMaterialInsideMask) {
    const int T = 10;
    const Sequence s = stretch_rotate_sequence(384, 384, T, 0.15, 8.0);
    const FullFieldParams p = centred_params(384, 384, 180);
    const cv::Mat mask = disc_mask(384, 384, 95, 25);   // ring with a hole
    const std::vector<Frame> inc = run_session(s, mask, p, 1);
    REQUIRE((int)inc.size() == T);
    const Accuracy a = evaluate(inc[T - 1], s.motion[T - 1]);
    std::printf("  [ring mask T=%d] %d pts (frame 1: %d), tracked %d/%d, rms u %.4f px, max %.3f, rms E %.2e\n",
                T, a.points, inc[0].rc, inc[T - 1].info.tracked_points, inc[T - 1].info.frame0_points,
                a.rms_u, a.max_u, a.rms_e);
    CHECK(a.points >= (int)(0.85 * inc[0].rc));
    CHECK(a.rms_u <= 0.15);
    // Every reported point is a frame-0 node inside the material.
    for (int i = 0; i < inc[T - 1].rc; ++i) {
        const float *q = &inc[T - 1].out[(size_t)i * 8];
        CHECK(mask.at<uchar>((int)q[1], (int)q[0]) >= 128);
    }
}

TEST_CASE(Incremental, LostNodesNeverReturn) {
    // A 70 px drift carries the right side of the ROI past the solver's
    // 40 px image-edge margin, so nodes are lost on the way.
    const Sequence s = stretch_rotate_sequence(320, 320, 10, 0.05, 3.0, 0.3, 0.0f, 31, 70.0, 0.0);
    const FullFieldParams p = centred_params(320, 320, 200);
    const std::vector<Frame> inc = run_session(s, cv::Mat(), p, 1);
    REQUIRE(inc.size() == 10);
    // Under N = 1 every frame moves the reference, so the tracked set can
    // only shrink.
    for (size_t t = 1; t < inc.size(); ++t) {
        CHECK(inc[t].info.tracked_points <= inc[t - 1].info.tracked_points);
        CHECK(inc[t].info.lost_this_frame >= 0);
    }
    CHECK(inc.back().info.tracked_points < inc.front().info.frame0_points);
}

TEST_CASE(Incremental, RepeatRun_BitIdentical) {
    const Sequence s = stretch_rotate_sequence(320, 320, 6, 0.1, 5.0);
    const FullFieldParams p = centred_params(320, 320, 160);
    const std::vector<Frame> a = run_session(s, cv::Mat(), p, 2);
    const std::vector<Frame> b = run_session(s, cv::Mat(), p, 2);
    REQUIRE(a.size() == b.size());
    for (size_t t = 0; t < a.size(); ++t) CHECK(bit_equal(a[t], b[t]));
}

TEST_CASE(Incremental, Contracts_ErrorCodes) {
    const Sequence s = stretch_rotate_sequence(256, 256, 1, 0.01, 0.0);
    FullFieldParams p = centred_params(256, 256, 120);
    std::vector<float> out((size_t)capacity_for(p), 0.0f);
    float metrics[23] = {};

    IncrementalSession session;
    CHECK(!session.initialized());
    CHECK(session.process_frame(s.frames[0], out.data(), (int)out.size(), metrics, 23) == -3);

    IncrementalParams ip;
    ip.field = p;
    ip.field.step = 0;
    CHECK(session.init(s.ref, cv::Mat(), ip) == -2);
    ip.field = p;
    ip.field.rect_w = 5;
    CHECK(session.init(s.ref, cv::Mat(), ip) == -2);
    ip.field = p;
    ip.field.strain_window = 3;
    CHECK(session.init(s.ref, cv::Mat(), ip) == kBadStrainWindow);
    ip.field = p;
    ip.update_interval = -1;
    CHECK(session.init(s.ref, cv::Mat(), ip) == -3);
    CHECK(session.init(cv::Mat(), cv::Mat(), IncrementalParams{p, 1}) == -3);

    REQUIRE(session.init(s.ref, cv::Mat(), IncrementalParams{p, 1}) == 0);
    CHECK(session.initialized());
    cv::Mat wrong_size(128, 128, CV_8UC1, cv::Scalar(100));
    CHECK(session.process_frame(wrong_size, out.data(), (int)out.size(), metrics, 23) == -3);
    CHECK(session.process_frame(s.frames[0], nullptr, 0, metrics, 23) == -3);
    CHECK(session.frames_processed() == 0);
    CHECK(session.process_frame(s.frames[0], out.data(), (int)out.size(), metrics, 23) > 0);
    CHECK(session.frames_processed() == 1);
    CHECK(session.max_points() == (p.rect_w / p.step) * (p.rect_h / p.step));

    session.reset();
    CHECK(!session.initialized());
}

TEST_CASE(Incremental, Cancel_LeavesSessionUnchanged) {
    // The cancel is raised from the first progress tick, which races the
    // solve; a dense grid makes the solve long enough (~100 ms) for the tick
    // to land inside it.
    const Sequence s = stretch_rotate_sequence(512, 512, 4, 0.08, 4.0);
    const FullFieldParams p = centred_params(512, 512, 320, 5, 21);
    const std::vector<Frame> clean = run_session(s, cv::Mat(), p, 1);
    REQUIRE(clean.size() == 4);

    IncrementalSession session;
    REQUIRE(session.init(s.ref, cv::Mat(), IncrementalParams{p, 1}) == 0);
    const int cap = capacity_for(p);
    std::vector<float> out((size_t)cap, 0.0f);
    float metrics[23] = {};
    CancelToken token;
    for (size_t t = 0; t < s.frames.size(); ++t) {
        if (t == 2) {
            // Cancel the third frame from inside the solve, then run it again.
            const int rc = session.process_frame(s.frames[t], out.data(), cap, metrics, 23, token,
                                                 [&](int) { token.request(); });
            CHECK(rc == kCancelled);
            CHECK(session.frames_processed() == 2);
        }
        Frame fr;
        fr.out.assign((size_t)cap, 0.0f);
        fr.rc = session.process_frame(s.frames[t], fr.out.data(), cap, metrics, 23, token,
                                      nullptr, &fr.info);
        CHECK(bit_equal(fr, clean[t]));
    }
}

TEST_CASE(Incremental, LongHorizon_160Frames) {
    // Slow loading: 0.1% strain per frame for 160 frames (16% total), the
    // horizon where per-update error accumulation would show.
    const int T = 160;
    const Sequence s = stretch_rotate_sequence(256, 256, T, 0.16, 0.0, 0.3, 1.0f, 57);
    const FullFieldParams p = centred_params(256, 256, 110);

    for (int interval : {1, 5}) {
        IncrementalSession session;
        REQUIRE(session.init(s.ref, cv::Mat(), IncrementalParams{p, interval}) == 0);
        const int cap = capacity_for(p);
        Frame fr;
        fr.out.assign((size_t)cap, 0.0f);
        float metrics[23] = {};
        long rss_early = 0, rss_late = 0;
        int first_rc = 0;
        for (int t = 0; t < T; ++t) {
            fr.rc = session.process_frame(s.frames[t], fr.out.data(), cap, metrics, 23, nullptr, &fr.info);
            REQUIRE(fr.rc > 0);
            if (t == 0) first_rc = fr.rc;
            if (t == 19) rss_early = resident_bytes();
            if (std::getenv("SEMPER_INC_TRACE") && (t % 10 == 9 || fr.info.lost_this_frame > 0)) {
                const Accuracy at = evaluate(fr, s.motion[t]);
                std::printf("    t=%3d ref=%3d out=%3d tracked=%3d taylor=%2d lost=%d rms=%.3f\n",
                            t + 1, fr.info.reference_frame, fr.rc, fr.info.tracked_points,
                            fr.info.taylor_fallbacks, fr.info.lost_this_frame, at.rms_u);
            }
        }
        rss_late = resident_bytes();
        const Accuracy a = evaluate(fr, s.motion[T - 1]);
        if (std::getenv("SEMPER_INC_TRACE")) {
            // Error regressed on position: a slope is a strain bias.
            double sxx = 0, sxe = 0, syy = 0, sye = 0, me_u = 0, me_v = 0;
            const double cx = 128.0, cy = 128.0;
            for (int i = 0; i < fr.rc; ++i) {
                const float *q = &fr.out[(size_t)i * 8];
                const dictest::Vec2 d = s.motion[T - 1].displacement(q[0], q[1]);
                const double eu = q[2] - d.x, ev = q[3] - d.y;
                sxx += (q[0] - cx) * (q[0] - cx); sxe += (q[0] - cx) * eu;
                syy += (q[1] - cy) * (q[1] - cy); sye += (q[1] - cy) * ev;
                me_u += eu; me_v += ev;
            }
            std::printf("    bias: mean eu %.4f ev %.4f px, d(eu)/dx %.2e, d(ev)/dy %.2e (truth exx-ish %.3f)\n",
                        me_u / fr.rc, me_v / fr.rc, sxe / sxx, sye / syy, 0.16);
            ReferenceCache cache;
            cache.set_from_gray(s.ref, cv::Mat());
            Frame fx;
            fx.out.assign((size_t)cap, 0.0f);
            fx.rc = run_full_field(cache, s.frames[T - 1], cv::Mat(), p, fx.out.data(), cap, metrics, 23, nullptr);
            const Accuracy af = evaluate(fx, s.motion[T - 1]);
            std::printf("    fixed reference at frame %d: %d pts, rms u %.4f px, max %.3f, rms E %.2e\n",
                        T, af.points, af.rms_u, af.max_u, af.rms_e);
        }
        std::printf("  [long horizon T=%d N=%d] %d/%d pts, rms u %.4f px, max %.3f, rms E %.2e, "
                    "rss %+.1f MB (frames 20..%d)\n",
                    T, interval, a.points, fr.info.frame0_points, a.rms_u, a.max_u, a.rms_e,
                    (rss_late - rss_early) / 1048576.0, T);
        // Nodes are not lost on a slow, in-frame load. Accumulated error makes
        // some fail the output's displacement median test, so fewer reach the
        // output than on frame 1, but they are still tracked.
        CHECK(fr.info.tracked_points >= (int)(0.95 * fr.info.frame0_points));
        CHECK(a.points >= (int)(0.75 * first_rc));
        // Each update adds a small per-node interpolation bias (a few % of a
        // tiny sub-pixel step, the same sign frame after frame), so error grows
        // with the number of updates: measured 0.31 px (N=1) and 0.22 px (N=5)
        // here against 0.05 px for the fixed reference. These bounds hold that
        // with margin; docs/INCREMENTAL_BENCHMARK.md has the full picture.
        CHECK(a.rms_u <= (interval == 1 ? 0.45 : 0.35));
        CHECK(a.rms_e <= 3.5e-2);
        if (rss_early > 0) CHECK(rss_late - rss_early < 16L * 1048576L);
    }
}

#else

TEST_CASE(Incremental, OpenCvRequired_SkippedWithoutOpenCV) {
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
