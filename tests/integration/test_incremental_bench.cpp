// =====================================================================
// INCREMENTAL BENCHMARK — fixed vs updated reference, measured
//
// Which reference strategy is better depends on the sequence: a fixed
// reference pays interpolation bias once but stops correlating under large
// rotation or strain; an incremental one keeps correlating but pays a little
// bias at every reference update. This suite runs both on identical frames and
// prints the tables behind docs/INCREMENTAL_BENCHMARK.md. It asserts nothing
// about which one wins.
//
// Modes: fixed (update_interval 0, which is run_full_field per frame) and
// incremental with N = 1, 2, 5, 10, all fed the same frames in lockstep.
//
// Ground truth is exact (framework/sequence.h): every frame is rendered from
// one continuous speckle under a known motion, so total displacement and
// Green-Lagrange strain at each frame-0 node are known in closed form. Noise
// is added per frame; "seeds" vary the noise, not the speckle, so the spread
// across seeds is the repeatability of a node.
//
// Run:  SEMPER_RUN_INCBENCH=1 ./dic_tests IncBench
//       SEMPER_INCBENCH_CSV=path     per-frame CSV (every mode, every frame)
//       SEMPER_INCBENCH_QUICK=1      reduced matrix: T = 160, sigma 2, one seed
//       SEMPER_INCBENCH_ONLY=S3      one scenario (S1..S10)
//       SEMPER_INCBENCH_SEEDS=n      noise seeds per sigma > 0 (default 3)
// tools/plot_incremental_bench.py turns the CSV into the plots in the doc.
// =====================================================================

#include "framework/test_framework.h"

#if defined(DIC_HAVE_OPENCV)

#include "framework/bench_util.h"
#include "framework/image_io.h"
#include "framework/sequence.h"

#include <semper/incremental.hpp>
#include <semper/pipeline.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::IncrementalFrameInfo;
using Semper::pipeline::IncrementalParams;
using Semper::pipeline::IncrementalSession;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::run_full_field;
using dictest::CpuClock;
using dictest::Mat2;
using dictest::Motion;
using dictest::Vec2;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kGoodPx = 0.5;   // a point within this of the truth counts as covered
const int kModes[] = {0, 1, 2, 5, 10};
constexpr int kNumModes = 5;

bool env_set(const char *name) { return std::getenv(name) != nullptr; }

int env_int(const char *name, int fallback) {
    const char *v = std::getenv(name);
    return v ? std::max(1, std::atoi(v)) : fallback;
}

std::string mode_label(int n) { return n == 0 ? "fixed" : "N=" + std::to_string(n); }

// ---------------------------------------------------------------------------
// Motions. motion(t, T) is the map from frame 0 to frame t of T.
// ---------------------------------------------------------------------------
using MotionAt = std::function<Motion(int t, int T)>;

Motion smooth_motion(std::function<Vec2(double, double)> u, std::function<Mat2(double, double)> F) {
    Motion m;
    m.forward = [u](double X, double Y) {
        const Vec2 d = u(X, Y);
        return Vec2{X + d.x, Y + d.y};
    };
    m.grad = std::move(F);
    return m;   // inverse by Newton
}

// Gaussian bump of amplitude a, width s about c, moving material along (1, 0.5).
Motion bump(double a, double s, Vec2 c) {
    const auto g = [=](double X, double Y) {
        const double dx = X - c.x, dy = Y - c.y;
        return std::exp(-(dx * dx + dy * dy) / (2.0 * s * s));
    };
    return smooth_motion(
        [=](double X, double Y) { const double v = a * g(X, Y); return Vec2{v, 0.5 * v}; },
        [=](double X, double Y) {
            const double v = a * g(X, Y);
            const double gx = -v * (X - c.x) / (s * s), gy = -v * (Y - c.y) / (s * s);
            return Mat2{1.0 + gx, gy, 0.5 * gx, 1.0 + 0.5 * gy};
        });
}

// u = a sin(2 pi (X - cx) / L): the DIC Challenge sinusoid.
Motion sinusoid(double a, double L, Vec2 c) {
    const double k = 2.0 * kPi / L;
    return smooth_motion(
        [=](double X, double) { return Vec2{a * std::sin(k * (X - c.x)), 0.0}; },
        [=](double X, double) { return Mat2{1.0 + a * k * std::cos(k * (X - c.x)), 0.0, 0.0, 1.0}; });
}

// Beam bending of curvature kappa about c: u = -k x y, v = k x^2 / 2.
Motion bending(double kappa, Vec2 c) {
    return smooth_motion(
        [=](double X, double Y) {
            const double x = X - c.x, y = Y - c.y;
            return Vec2{-kappa * x * y, 0.5 * kappa * x * x};
        },
        [=](double X, double Y) {
            const double x = X - c.x, y = Y - c.y;
            return Mat2{1.0 - kappa * y, -kappa * x, kappa * x, 1.0};
        });
}

// ---------------------------------------------------------------------------
// Scenario matrix
// ---------------------------------------------------------------------------
struct Variant {
    std::string id;       // S1..S9
    std::string name;
    int w = 256, h = 256, T = 200;
    cv::Rect roi;
    bool ring_mask = false;
    bool use_6x6 = false;
    bool long_run = true;           // shortened to 160 frames in quick mode
    std::vector<float> sigmas;
    MotionAt motion;
};

// Named fields only: an aggregate initialised with just the first two
// members trips -Wmissing-field-initializers (a CI error).
Variant variant(std::string id, std::string name) {
    Variant v;
    v.id = std::move(id);
    v.name = std::move(name);
    return v;
}

cv::Rect centred(int w, int h, int rw, int rh) { return cv::Rect((w - rw) / 2, (h - rh) / 2, rw, rh); }

std::vector<Variant> scenario_matrix() {
    std::vector<Variant> v;
    const std::vector<float> sweep = {0.0f, 1.0f, 2.0f, 4.0f};
    const std::vector<float> pair = {0.0f, 2.0f};

    {   // S1: nothing moves; error is noise and its accumulation.
        Variant s = variant("S1", "static, noise only");
        s.T = 300; s.roi = centred(256, 256, 110, 110); s.sigmas = sweep;
        s.motion = [](int, int) { return dictest::identity_motion(); };
        v.push_back(s);
    }
    for (bool six : {false, true}) {   // S2: slow loading, 0.1% per frame.
        Variant s = variant("S2", six ? "slow stretch 0.1%/f (6x6)" : "slow stretch 0.1%/f");
        s.roi = centred(256, 256, 110, 110); s.use_6x6 = six;
        s.sigmas = six ? pair : sweep;
        s.motion = [](int t, int) { return dictest::stretch_rotate(0.001 * t, 0.3, 0.0, Vec2{}, Vec2{128, 128}); };
        v.push_back(s);
    }
    for (int rate : {1, 5, 20}) {      // S3: rigid translation, 200 px in total.
        Variant s = variant("S3", "translate " + std::to_string(rate) + " px/f");
        s.w = 448; s.h = 192; s.T = 200 / rate; s.long_run = (rate == 1);
        s.roi = cv::Rect(48, 48, 96, 96); s.sigmas = pair;
        s.motion = [rate](int t, int) { return dictest::affine_motion(Mat2{}, Vec2{(double)rate * t, 0.0}, Vec2{}); };
        v.push_back(s);
    }
    for (double rate : {0.45, 2.0, 5.0}) {   // S4: rigid rotation to 90 degrees.
        char nm[48];
        std::snprintf(nm, sizeof nm, "rotate %.2g deg/f", rate);
        Variant s = variant("S4", nm);
        s.T = (int)std::lround(90.0 / rate); s.long_run = (rate < 1.0);
        s.roi = centred(256, 256, 110, 110); s.sigmas = pair;
        s.motion = [rate](int t, int) {
            return dictest::stretch_rotate(0.0, 0.0, rate * t * kPi / 180.0, Vec2{}, Vec2{128, 128});
        };
        v.push_back(s);
    }
    for (double e : {0.6, 1.0}) {    // S5: large uniaxial stretch with Poisson contraction.
        Variant s = variant("S5", e < 0.9 ? "stretch to 60%" : "stretch to 100%");
        s.w = 384; s.roi = centred(384, 256, 110, 110); s.sigmas = pair;
        s.motion = [e](int t, int T) {
            return dictest::stretch_rotate(e * t / T, 0.3, 0.0, Vec2{}, Vec2{192, 128});
        };
        v.push_back(s);
    }
    {   // S6: stretch + rotation + translation together.
        Variant s = variant("S6", "stretch 40% + rot 45 + 30 px");
        s.w = 384; s.h = 384; s.roi = centred(384, 384, 110, 110); s.sigmas = pair;
        s.motion = [](int t, int T) {
            const double f = (double)t / T;
            return dictest::stretch_rotate(0.4 * f, 0.3, 45.0 * f * kPi / 180.0, Vec2{30.0 * f, 0.0},
                                           Vec2{192, 192});
        };
        v.push_back(s);
    }
    {   // S7: non-affine fields, ramped.
        Variant b = variant("S7", "bump 8 px, s 30");
        b.roi = centred(256, 256, 110, 110); b.sigmas = pair;
        b.motion = [](int t, int T) { return bump(8.0 * t / T, 30.0, Vec2{128, 128}); };
        v.push_back(b);
        Variant s = b;
        s.name = "sinusoid 2 px, L 80";
        s.motion = [](int t, int T) { return sinusoid(2.0 * t / T, 80.0, Vec2{128, 128}); };
        v.push_back(s);
        Variant k = b;
        k.name = "bending, 10% at edge";
        k.motion = [](int t, int T) { return bending(0.0018 * t / T, Vec2{128, 128}); };
        v.push_back(k);
    }
    for (int T : {5, 20, 50, 200}) {   // S8: the same deformation in fewer or more frames.
        Variant s = variant("S8", "stretch 20% + rot 10 in " + std::to_string(T) + " f");
        s.T = T; s.long_run = (T >= 200);
        s.roi = centred(256, 256, 110, 110); s.sigmas = pair;
        s.motion = [](int t, int TT) {
            const double f = (double)t / TT;
            return dictest::stretch_rotate(0.2 * f, 0.3, 10.0 * f * kPi / 180.0, Vec2{}, Vec2{128, 128});
        };
        v.push_back(s);
    }
    {   // S9: a ROI with a hole, as an open-hole tension specimen.
        Variant s = variant("S9", "ring mask, stretch 30%");
        s.w = 320; s.h = 320; s.roi = centred(320, 320, 150, 150); s.ring_mask = true; s.sigmas = pair;
        s.motion = [](int t, int T) {
            return dictest::stretch_rotate(0.3 * t / T, 0.3, 0.0, Vec2{}, Vec2{160, 160});
        };
        v.push_back(s);
    }
    return v;
}

cv::Mat ring_mask(int w, int h, double r_out, double r_in) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double dx = x - w / 2.0, dy = y - h / 2.0, r2 = dx * dx + dy * dy;
            if (r2 < r_out * r_out && r2 >= r_in * r_in) m.at<uchar>(y, x) = 255;
        }
    return m;
}

FullFieldParams grid_params(const Variant &v) {
    FullFieldParams p;   // the app's default grid
    p.rect_x = v.roi.x;
    p.rect_y = v.roi.y;
    p.rect_w = v.roi.width;
    p.rect_h = v.roi.height;
    p.step = 5;
    p.subset_size = 21;
    p.strain_window = 15;
    p.use_6x6_interpolator = v.use_6x6;
    return p;
}

// ---------------------------------------------------------------------------
// Per-frame measurement
// ---------------------------------------------------------------------------
struct FrameStats {
    int rc = 0, tracked = 0, frame0 = 0, ref_frame = 0, good = 0;
    bool updated = false;
    double mean_eu = 0, mean_ev = 0, std_eu = 0, std_ev = 0, rms_u = 0, max_u = 0, rms_e = 0;
    double bias_exx = 0, bias_eyy = 0, bias_exy = 0, mean_corr = 0, iters = 0;
    double wall_ms = 0, cpu_ms = 0;
};

struct NodeErr { int x, y; float eu, ev; };

FrameStats evaluate(const std::vector<float> &out, int rc, const Motion &m, std::vector<NodeErr> *nodes) {
    FrameStats s;
    s.rc = rc;
    if (rc <= 0) return s;
    double su = 0, sv = 0, su2 = 0, sv2 = 0, se = 0, bxx = 0, byy = 0, bxy = 0, sc = 0;
    for (int i = 0; i < rc; ++i) {
        const float *q = &out[(size_t)i * 8];
        const Vec2 d = m.displacement(q[0], q[1]);
        const double eu = q[2] - d.x, ev = q[3] - d.y, err = std::sqrt(eu * eu + ev * ev);
        double exx, eyy, exy;
        m.strain(q[0], q[1], exx, eyy, exy);
        const double e1 = q[4] - exx, e2 = q[5] - eyy, e3 = q[6] - exy;
        su += eu; sv += ev; su2 += eu * eu; sv2 += ev * ev;
        se += (e1 * e1 + e2 * e2 + e3 * e3) / 3.0;
        bxx += e1; byy += e2; bxy += e3; sc += q[7];
        s.max_u = std::max(s.max_u, err);
        if (err <= kGoodPx) s.good++;
        if (nodes) nodes->push_back({(int)std::lround(q[0]), (int)std::lround(q[1]), (float)eu, (float)ev});
    }
    const double n = rc;
    s.mean_eu = su / n; s.mean_ev = sv / n;
    s.std_eu = std::sqrt(std::max(0.0, su2 / n - s.mean_eu * s.mean_eu));
    s.std_ev = std::sqrt(std::max(0.0, sv2 / n - s.mean_ev * s.mean_ev));
    s.rms_u = std::sqrt((su2 + sv2) / n);
    s.rms_e = std::sqrt(se / n);
    s.bias_exx = bxx / n; s.bias_eyy = byy / n; s.bias_exy = bxy / n;
    s.mean_corr = sc / n;
    return s;
}

// One mode's solver, fed frame by frame.
struct Runner {
    int N = 0;
    std::unique_ptr<ReferenceCache> cache;   // fixed
    IncrementalSession session;              // N >= 1
    cv::Mat mask;
    FullFieldParams p;
    bool dead = false;
    int fails_in_a_row = 0;

    bool init(const cv::Mat &ref, const cv::Mat &m, const FullFieldParams &params, int interval) {
        N = interval; mask = m; p = params;
        if (N == 0) {
            cache = std::make_unique<ReferenceCache>();
            cache->set_from_gray(ref, mask);
            return cache->width > 0;
        }
        return session.init(ref, mask, IncrementalParams{p, N}) == 0;
    }

    int solve(const cv::Mat &def, std::vector<float> &out, float *metrics, IncrementalFrameInfo &info) {
        const int cap = (int)out.size();
        if (N == 0) {
            const int rc = run_full_field(*cache, def, mask, p, out.data(), cap, metrics, 23, nullptr);
            info = IncrementalFrameInfo{};
            info.tracked_points = std::max(rc, 0);
            return rc;
        }
        return session.process_frame(def, out.data(), cap, metrics, 23, nullptr, &info);
    }
};

// Summary of one run (one sequence, one mode).
struct RunSummary {
    int fail_frame = -1, cov90 = -1, cov50 = -1;
    bool monotone = true;
    double wall = 0, cpu = 0, wall_upd = 0, wall_hold = 0, wall_first = 0, wall_last = 0;
    int n = 0, n_upd = 0, n_hold = 0, n_first = 0, n_last = 0;
};

struct Cell {   // averaged over seeds
    int seeds = 0;
    FrameStats sum;
    void add(const FrameStats &s) {
        ++seeds;
        sum.rc += s.rc; sum.good += s.good; sum.tracked += s.tracked;
        sum.mean_eu += s.mean_eu; sum.mean_ev += s.mean_ev; sum.std_eu += s.std_eu; sum.std_ev += s.std_ev;
        sum.rms_u += s.rms_u; sum.max_u = std::max(sum.max_u, s.max_u); sum.rms_e += s.rms_e;
        sum.bias_exx += s.bias_exx; sum.bias_eyy += s.bias_eyy; sum.bias_exy += s.bias_exy;
        sum.iters += s.iters; sum.mean_corr += s.mean_corr;
    }
    double avg(double v) const { return seeds ? v / seeds : 0.0; }
};

struct ModeAgg {
    std::map<int, Cell> at;                       // report frame -> averaged stats
    std::vector<std::vector<NodeErr>> final_nodes; // per seed, last frame
    RunSummary worst;                             // earliest failure across seeds
    double wall = 0, cpu = 0, wall_upd = 0, wall_hold = 0, wall_first = 0, wall_last = 0;
    int runs = 0, upd_runs = 0, hold_runs = 0;
    bool monotone = true;
};

int earliest(int a, int b) {
    if (a < 0) return b;
    if (b < 0) return a;
    return std::min(a, b);
}

// Per-node spread of the error across noise seeds: the repeatability.
double repeatability(const std::vector<std::vector<NodeErr>> &seeds) {
    if (seeds.size() < 2) return -1.0;
    std::map<std::pair<int, int>, std::vector<std::pair<float, float>>> by;
    for (const auto &run : seeds)
        for (const NodeErr &e : run) by[{e.x, e.y}].push_back({e.eu, e.ev});
    double acc = 0;
    int n = 0;
    for (const auto &kv : by) {
        const auto &v = kv.second;
        if (v.size() != seeds.size()) continue;
        double mu = 0, mv = 0;
        for (const auto &e : v) { mu += e.first; mv += e.second; }
        mu /= v.size(); mv /= v.size();
        double var = 0;
        for (const auto &e : v) var += (e.first - mu) * (e.first - mu) + (e.second - mv) * (e.second - mv);
        acc += var / (double)(v.size() - 1);
        ++n;
    }
    return n ? std::sqrt(acc / n) : -1.0;
}

std::vector<int> report_frames(int T) {
    std::vector<int> f;
    for (int t : {1, 10, 50, 100, 150, 200, 300})
        if (t <= T) f.push_back(t);
    if (f.back() != T) f.push_back(T);
    return f;
}

std::string frame_or_dash(int f) { return f < 0 ? "-" : std::to_string(f); }

// ---------------------------------------------------------------------------
// One variant at one noise level, all modes in lockstep, over every seed.
// ---------------------------------------------------------------------------
void run_variant(const Variant &v, int T, float sigma, int seeds, std::ofstream *csv) {
    const FullFieldParams p = grid_params(v);
    const cv::Mat mask = v.ring_mask ? ring_mask(v.w, v.h, 70.0, 20.0) : cv::Mat();
    const int cap = (p.rect_w / p.step + 1) * (p.rect_h / p.step + 1) * 8;
    const std::vector<int> frames = report_frames(T);
    const dictest::SmoothSpeckleField field(101, v.w, v.h);
    ModeAgg agg[kNumModes];
    int base = 0;
    double rss_growth = 0, set_ref_ms = 0;

    for (int seed = 0; seed < seeds; ++seed) {
        const unsigned nseed = 7919u * (unsigned)(seed + 1);
        const cv::Mat ref = dictest::render_frame(field, v.w, v.h, dictest::identity_motion(), sigma, nseed);
        Runner runners[kNumModes];
        RunSummary sum[kNumModes];
        for (int k = 0; k < kNumModes; ++k) REQUIRE(runners[k].init(ref, mask, p, kModes[k]));
        std::vector<float> out((size_t)cap, 0.0f);
        int prev_tracked[kNumModes];
        for (int k = 0; k < kNumModes; ++k) prev_tracked[k] = 1 << 30;
        double rss_at_20 = -1;

        for (int t = 1; t <= T; ++t) {
            const Motion m = v.motion(t, T);
            const cv::Mat def = dictest::render_frame(field, v.w, v.h, m, sigma, nseed + (unsigned)t);
            if (t == 1 && seed == 0) {
                // What moving the reference would cost without adopt():
                // decoding aside, set_from_gray prepares the image again.
                ReferenceCache scratch;
                const auto a = std::chrono::steady_clock::now();
                scratch.set_from_gray(def, mask);
                set_ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            }
            const bool last = (t == T);
            const bool report = std::find(frames.begin(), frames.end(), t) != frames.end();
            for (int k = 0; k < kNumModes; ++k) {
                Runner &r = runners[k];
                FrameStats st;
                if (!r.dead) {
                    float metrics[23] = {};
                    IncrementalFrameInfo info;
                    std::fill(out.begin(), out.end(), 0.0f);
                    const auto w0 = std::chrono::steady_clock::now();
                    const CpuClock c0 = CpuClock::now();
                    const int rc = r.solve(def, out, metrics, info);
                    const double cpu = CpuClock::now().total_ms() - c0.total_ms();
                    const double wall = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - w0).count();
                    st = evaluate(out, rc, m, last ? &agg[k].final_nodes.emplace_back() : nullptr);
                    st.tracked = info.tracked_points;
                    st.frame0 = info.frame0_points;
                    st.ref_frame = info.reference_frame;
                    st.updated = info.reference_updated;
                    st.iters = metrics[8];
                    st.wall_ms = wall;
                    st.cpu_ms = cpu;

                    RunSummary &s = sum[k];
                    s.wall += wall; s.cpu += cpu; s.n++;
                    if (r.N >= 2) {
                        if (info.reference_updated) { s.wall_upd += wall; s.n_upd++; }
                        else { s.wall_hold += wall; s.n_hold++; }
                    }
                    if (t <= 20) { s.wall_first += wall; s.n_first++; }
                    if (t > T - 20) { s.wall_last += wall; s.n_last++; }
                    if (r.N == 1) {
                        if (info.tracked_points > prev_tracked[k]) s.monotone = false;
                        prev_tracked[k] = info.tracked_points;
                    }
                    if (rc <= 0) {
                        if (s.fail_frame < 0) s.fail_frame = t;
                        if (++r.fails_in_a_row >= 3) r.dead = true;
                    } else {
                        r.fails_in_a_row = 0;
                    }
                } else if (last) {
                    agg[k].final_nodes.emplace_back();
                }
                if (k == 0 && t == 1) base = std::max(st.rc, 1);
                const double cov = (double)st.good / base;
                if (cov < 0.9 && sum[k].cov90 < 0) sum[k].cov90 = t;
                if (cov < 0.5 && sum[k].cov50 < 0) sum[k].cov50 = t;
                if (report) agg[k].at[t].add(st);
                if (csv) {
                    *csv << v.id << ",\"" << v.name << "\"," << mode_label(r.N) << "," << r.N << ","
                         << sigma << "," << seed << "," << t << "," << st.rc << "," << st.tracked << ","
                         << st.frame0 << "," << st.ref_frame << "," << (st.updated ? 1 : 0) << ","
                         << st.good << "," << cov << "," << st.mean_eu << "," << st.mean_ev << ","
                         << st.std_eu << "," << st.std_ev << "," << st.rms_u << "," << st.max_u << ","
                         << st.rms_e << "," << st.bias_exx << "," << st.bias_eyy << "," << st.bias_exy
                         << "," << st.mean_corr << "," << st.iters << "," << st.wall_ms << ","
                         << st.cpu_ms << "\n";
                }
            }
            if (t == std::min(20, T)) rss_at_20 = dictest::current_rss_mb();
        }
        rss_growth = std::max(rss_growth, dictest::current_rss_mb() - rss_at_20);

        for (int k = 0; k < kNumModes; ++k) {
            const RunSummary &s = sum[k];
            ModeAgg &a = agg[k];
            a.runs++;
            a.worst.fail_frame = earliest(a.worst.fail_frame, s.fail_frame);
            a.worst.cov90 = earliest(a.worst.cov90, s.cov90);
            a.worst.cov50 = earliest(a.worst.cov50, s.cov50);
            a.monotone = a.monotone && s.monotone;
            if (s.n) { a.wall += s.wall / s.n; a.cpu += s.cpu / s.n; }
            if (s.n_upd) { a.wall_upd += s.wall_upd / s.n_upd; a.upd_runs++; }
            if (s.n_hold) { a.wall_hold += s.wall_hold / s.n_hold; a.hold_runs++; }
            if (s.n_first) a.wall_first += s.wall_first / s.n_first;
            if (s.n_last) a.wall_last += s.wall_last / s.n_last;
        }
    }

    // ---- accuracy vs frame -------------------------------------------------
    std::printf("\n  %s  %s  T=%d  sigma=%.0f  seeds=%d  (%d points on frame 1)\n",
                v.id.c_str(), v.name.c_str(), T, sigma, seeds, base);
    std::printf("    rms |U - U_true| (px) / covered %% (within %.1f px) at frame:\n    %-6s", kGoodPx, "mode");
    for (int f : frames) std::printf(" %13d", f);
    std::printf("\n");
    for (int k = 0; k < kNumModes; ++k) {
        std::printf("    %-6s", mode_label(kModes[k]).c_str());
        for (int f : frames) {
            const Cell &c = agg[k].at[f];
            if (c.sum.rc <= 0) std::printf("  %6s/%4.0f%%", "-", 0.0);   // no points: failed
            else std::printf("  %6.3f/%4.0f%%", c.avg(c.sum.rms_u), 100.0 * c.avg(c.sum.good) / base);
        }
        std::printf("\n");
    }

    // ---- final frame and cost ---------------------------------------------
    std::printf("    at frame %d:\n    %-6s %8s %8s %7s %7s %7s %7s %8s %8s %6s %6s %6s %7s %7s %7s %7s %5s\n",
                T, "mode", "bias_u", "bias_v", "std_u", "std_v", "repeat", "max", "rmsE", "biasExx",
                "fail", "cov90", "cov50", "ms/f", "cpu/f", "upd ms", "hold ms", "drift");
    for (int k = 0; k < kNumModes; ++k) {
        const ModeAgg &a = agg[k];
        const Cell &c = a.at.at(T);
        const double rep = sigma > 0 ? repeatability(a.final_nodes) : -1.0;
        char rep_s[16], upd_s[16], hold_s[16];
        if (rep >= 0) std::snprintf(rep_s, sizeof rep_s, "%7.4f", rep);
        else std::snprintf(rep_s, sizeof rep_s, "%7s", "-");
        std::snprintf(upd_s, sizeof upd_s, "%7s", "-");
        std::snprintf(hold_s, sizeof hold_s, "%7s", "-");
        if (a.upd_runs) std::snprintf(upd_s, sizeof upd_s, "%7.1f", a.wall_upd / a.upd_runs);
        if (a.hold_runs) std::snprintf(hold_s, sizeof hold_s, "%7.1f", a.wall_hold / a.hold_runs);
        const double first = a.wall_first / a.runs, lastw = a.wall_last / a.runs;
        std::printf("    %-6s %+8.4f %+8.4f %7.4f %7.4f %s %7.3f %8.2e %+8.1e %6s %6s %6s %7.1f %7.1f %s %s %4.2fx%s\n",
                    mode_label(kModes[k]).c_str(), c.avg(c.sum.mean_eu), c.avg(c.sum.mean_ev),
                    c.avg(c.sum.std_eu), c.avg(c.sum.std_ev), rep_s, c.sum.max_u, c.avg(c.sum.rms_e),
                    c.avg(c.sum.bias_exx), frame_or_dash(a.worst.fail_frame).c_str(),
                    frame_or_dash(a.worst.cov90).c_str(), frame_or_dash(a.worst.cov50).c_str(),
                    a.wall / a.runs, a.cpu / a.runs, upd_s, hold_s, first > 0 ? lastw / first : 0.0,
                    (kModes[k] == 1 && !a.monotone) ? " NOT MONOTONE" : "");
    }
    std::printf("    set_from_gray on this frame size: %.2f ms per update (avoided by adopt)."
                "  RSS growth frames 20..%d: %+.1f MB\n", set_ref_ms, T, rss_growth);
}

// ---------------------------------------------------------------------------
// S10: DICe's OHT CFRP ladder on real images (only five frames exist).
// ---------------------------------------------------------------------------
struct Gold { int x, y; double u, v; };

bool load_gold(const std::string &path, std::vector<Gold> &out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line.rfind("***", 0) == 0 || line.rfind("COORDINATE_X", 0) == 0) continue;
        std::stringstream ss(line);
        std::string cell;
        double col[4];
        int n = 0;
        while (n < 4 && std::getline(ss, cell, ',')) col[n++] = std::atof(cell.c_str());
        if (n == 4) out.push_back({(int)std::lround(col[0]), (int)std::lround(col[1]), col[2], col[3]});
    }
    return !out.empty();
}

cv::Mat load_dice(const char *name) {
    dictest::GrayImage g;
    if (!dictest::load_gray(std::string(DICE_FIXTURES_DIR) + name, g)) return cv::Mat();
    return cv::Mat(g.h, g.w, CV_8UC1, g.px.data()).clone();
}

void run_oht_ladder() {
    const char *images[] = {"/oht_cfrp_01.tiff", "/oht_cfrp_03.tiff", "/oht_cfrp_06.tiff", "/oht_cfrp_11.tiff"};
    const char *golds[] = {"/DICe_solution_01.txt", "/DICe_solution_03.txt", "/DICe_solution_06.txt",
                           "/DICe_solution_11.txt"};
    const cv::Mat ref = load_dice("/oht_cfrp_00.tiff");
    REQUIRE(!ref.empty());
    FullFieldParams p;
    p.step = 7;
    p.rect_x = 61 - 14; p.rect_y = 61 - 14;
    p.rect_w = 280 + 28; p.rect_h = 910 + 28;
    p.subset_size = 27; p.strain_window = 21; p.use_6x6_interpolator = true;
    const int cap = (p.rect_w / p.step + 1) * (p.rect_h / p.step + 1) * 8;
    const int modes[] = {0, 1, 2};

    std::printf("\n  S10  DICe OHT CFRP ladder 00 -> 01 -> 03 -> 06 -> 11 (real images, 400x1040)\n"
                "    agreement with DICe's fixed-reference field: rms du / dv (px), mean du; ms per frame\n"
                "    %-6s %26s %26s %26s %26s\n", "mode", "01", "03", "06", "11");
    for (int n : modes) {
        Runner r;
        REQUIRE(r.init(ref, cv::Mat(), p, n));
        std::printf("    %-6s", mode_label(n).c_str());
        for (int i = 0; i < 4; ++i) {
            const cv::Mat def = load_dice(images[i]);
            std::vector<Gold> gold;
            REQUIRE(load_gold(std::string(DICE_FIXTURES_DIR) + golds[i], gold));
            std::vector<float> out((size_t)cap, 0.0f);
            float metrics[23] = {};
            IncrementalFrameInfo info;
            const auto w0 = std::chrono::steady_clock::now();
            const int rc = r.solve(def, out, metrics, info);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
            std::map<std::pair<int, int>, std::pair<float, float>> at;
            for (int j = 0; j < std::max(rc, 0); ++j) {
                const float *q = &out[(size_t)j * 8];
                at[{(int)std::lround(q[0]), (int)std::lround(q[1])}] = {q[2], q[3]};
            }
            double su = 0, su2 = 0, sv2 = 0;
            int c = 0;
            for (const Gold &g : gold) {
                const auto it = at.find({g.x, g.y});
                if (it == at.end()) continue;
                const double du = it->second.first - g.u, dv = it->second.second - g.v;
                su += du; su2 += du * du; sv2 += dv * dv; ++c;
            }
            if (c == 0) { std::printf(" %26s", "no points"); continue; }
            char cellbuf[64];
            std::snprintf(cellbuf, sizeof cellbuf, "%.4f/%.4f %+.4f %4.0fms", std::sqrt(su2 / c),
                          std::sqrt(sv2 / c), su / c, ms);
            std::printf(" %26s", cellbuf);
        }
        std::printf("\n");
    }
}

} // namespace

TEST_CASE(IncBench, FixedVsIncremental) {
    if (!env_set("SEMPER_RUN_INCBENCH")) {
        std::printf("    skipped (set SEMPER_RUN_INCBENCH=1 to run the benchmark)\n");
        return;
    }
    const bool quick = env_set("SEMPER_INCBENCH_QUICK");
    const int seeds = quick ? 1 : env_int("SEMPER_INCBENCH_SEEDS", 3);
    const char *only = std::getenv("SEMPER_INCBENCH_ONLY");
    std::unique_ptr<std::ofstream> csv;
    if (const char *path = std::getenv("SEMPER_INCBENCH_CSV")) {
        csv = std::make_unique<std::ofstream>(path);
        *csv << "scenario,variant,mode,N,sigma,seed,frame,rc,tracked,frame0,ref_frame,updated,good,"
                "coverage,mean_eu,mean_ev,std_eu,std_ev,rms_u,max_u,rms_e,bias_exx,bias_eyy,bias_exy,"
                "mean_corr,icgn_iters,wall_ms,cpu_ms\n";
    }
    std::printf("  IncBench: subset 21, step 5, strain window 15; modes fixed, N=1,2,5,10%s\n",
                quick ? " (quick)" : "");

    const auto t0 = std::chrono::steady_clock::now();
    std::string last_quick_id;
    for (const Variant &v : scenario_matrix()) {
        if (only && v.id != only) continue;
        int T = v.T;
        std::vector<float> sigmas = v.sigmas;
        if (quick) {
            // One long variant per scenario, at the horizon the regular suite
            // also guards.
            if (!v.long_run || v.id == last_quick_id || v.use_6x6) continue;
            last_quick_id = v.id;
            T = std::min(T, 160);
            sigmas = {2.0f};
        }
        for (float sigma : sigmas) run_variant(v, T, sigma, sigma > 0 ? seeds : 1, csv.get());
    }
    if (!only || std::string(only) == "S10") run_oht_ladder();

    std::printf("\n  IncBench: %.0f s; process peak RSS %.1f MB\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                dictest::process_peak_rss_mb());
}

#else

TEST_CASE(IncBench, OpenCvRequired_SkippedWithoutOpenCV) {
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
