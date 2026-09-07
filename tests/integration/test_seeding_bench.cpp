// =====================================================================
// SEEDING BENCHMARK — measure before replacing
//
// The engine's displacements come from IC-GN; the seeding front-end only
// supplies the initial guess. So "which feature matcher is best" is a question
// about seed accuracy, ROI coverage and cost, not about sub-pixel precision.
// This suite measures the candidates on identical inputs and prints the table;
// it asserts nothing about which one wins.
//
// Two quantities matter, and they are reported separately:
//
//   vertex error   |u - u_true| at the mesh vertices the front-end produced.
//   guess-gradient error
//                  |ux - ux_true| over the grid, taken from the Delaunay mesh
//                  the vertices feed. build_mesh_guess_field derives the
//                  gradients with cv::getAffineTransform over triangle
//                  vertices, so a vertex error e over an edge of length L
//                  becomes a gradient error of order e/L. This is the quantity
//                  the guess actually hands to ICGN.
//
// Ground truth is exact: reference and deformed images are both sampled from
// the same continuous analytic speckle function (framework/synthetic.h), so
// no resampling error enters the comparison.
//
// Run:  SEMPER_RUN_SEEDBENCH=1 ./dic_tests SeedBench
//       SEMPER_SEEDBENCH_CSV=/tmp/seed.csv  (optional machine-readable dump)
// =====================================================================

#include "framework/test_framework.h"

#if DIC_HAVE_OPENCV

#include "framework/synthetic.h"
#include "framework/synthetic_cv.h"
#include "framework/image_io.h"
#include "pipeline/full_field_internal.hpp"

#include <semper/pipeline.hpp>
#include <semper/seeding.hpp>

#include <opencv2/imgproc.hpp>

// Cost accounting below needs process CPU time and resident set size, which
// have no portable spelling. CI and every published measurement in
// docs/SEEDING_BENCHMARK.md are Linux; Windows is a developer host, so it gets
// working equivalents rather than a stub -- an unbuildable test file takes the
// whole dic_tests binary down with it, benchmark or not.
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX          // else the min/max macros shadow std::min/std::max
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#else
#  include <sys/resource.h>
#  include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <functional>
#include <thread>
#include <string>
#include <vector>

using Semper::Image;
using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;
using Semper::SubsetPrecomputer;

namespace {

// ---------------------------------------------------------------------------
// Cost accounting. Wall time alone can hide a CPU regression: the anchor
// lattice is OpenMP-parallel while AKAZE's detectAndCompute is largely serial,
// so a wall-clock win may cost more total CPU — which on a phone is battery.
// ---------------------------------------------------------------------------
struct CpuClock {
    double user_ms = 0.0, sys_ms = 0.0;
    static CpuClock now() {
        CpuClock c;
#ifdef _WIN32
        // GetProcessTimes reports both in 100 ns units and, like getrusage,
        // sums across all threads of the process -- which is the property the
        // OpenMP-vs-serial comparison depends on.
        FILETIME created{}, exited{}, kernel{}, user{};
        if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
            const auto to_ms = [](const FILETIME &ft) {
                return (double)(((unsigned long long)ft.dwHighDateTime << 32) |
                                ft.dwLowDateTime) / 1e4;
            };
            c.user_ms = to_ms(user);
            c.sys_ms  = to_ms(kernel);
        }
#else
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        c.user_ms = ru.ru_utime.tv_sec * 1e3 + ru.ru_utime.tv_usec / 1e3;
        c.sys_ms  = ru.ru_stime.tv_sec * 1e3 + ru.ru_stime.tv_usec / 1e3;
#endif
        return c;
    }
    double total_ms() const { return user_ms + sys_ms; }
};

// ru_maxrss is a process-wide high-water mark: monotonic, so differencing it
// across candidates in one process reports the run order, not the candidate.
// Current RSS from /proc/self/statm can be differenced — it gives the memory a
// candidate *retains* (descriptor matrices, scale-space pyramids), which is the
// comparable quantity. For a true per-candidate peak, run one method per
// process via SEMPER_SEEDBENCH_ONLY and read the peak printed at the end.
double current_rss_mb() {
#ifdef _WIN32
    // WorkingSetSize is the Windows analogue of the statm resident count:
    // pages currently backed by physical memory, and it can be differenced.
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return -1.0;
    return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
#else
    std::ifstream f("/proc/self/statm");
    if (!f) return -1.0;
    long total_pages = 0, rss_pages = 0;
    f >> total_pages >> rss_pages;
    if (!f) return -1.0;
    return (double)rss_pages * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
#endif
}

double process_peak_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return -1.0;
    return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
#else
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#  ifdef __APPLE__
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#  else
    return (double)ru.ru_maxrss / 1024.0;
#  endif
#endif
}

// ---------------------------------------------------------------------------
// Ground truth as a callable, so a scenario need not be a single affine. The
// DIC Challenge sinusoidal sets (Sample 14/15) are non-affine by construction:
// u(x) = A sin(2 pi x / L), whose gradient varies continuously across the field.
// ---------------------------------------------------------------------------
struct Truth {
    float u = 0, v = 0, ux = 0, uy = 0, vx = 0, vy = 0;
};
using TruthFn = std::function<Truth(float x, float y)>;

TruthFn affine_truth(const dictest::AffineDeformation &d) {
    return [d](float x, float y) {
        const float px = x - d.cx, py = y - d.cy;
        Truth t;
        t.u = d.u + d.ux * px + d.uy * py;
        t.v = d.v + d.vx * px + d.vy * py;
        t.ux = d.ux; t.uy = d.uy; t.vx = d.vx; t.vy = d.vy;
        return t;
    };
}

constexpr int W = 640;
constexpr int H = 640;
constexpr int BLOBS = 1600;          // ~11% blob coverage, DIC-representative
constexpr int ROI_SIZE = 480;   // not RECT: windows.h typedefs that
constexpr int RECT_ORIGIN = 80;
constexpr int STEP = 5;
constexpr int SUBSET = 41;
constexpr int STRAIN_WIN = 15;

struct Scenario {
    const char *name;
    dictest::AffineDeformation def;
    float noise_sigma;
};

dictest::AffineDeformation affine(float u, float v, float ux = 0.f, float uy = 0.f,
                                  float vx = 0.f, float vy = 0.f) {
    dictest::AffineDeformation d;
    d.u = u; d.v = v; d.ux = ux; d.uy = uy; d.vx = vx; d.vy = vy;
    d.cx = W / 2.0f; d.cy = H / 2.0f;
    return d;
}

std::vector<Scenario> scenarios() {
    std::vector<Scenario> s;
    s.push_back({"S1a_trans_0.1px",  affine(0.1f, 0.0f),  0.0f});
    s.push_back({"S1b_trans_0.4px",  affine(0.4f, 0.0f),  0.0f});
    s.push_back({"S1c_trans_0.7px",  affine(0.7f, -0.3f), 0.0f});
    s.push_back({"S2_strain_2e-3",   affine(3.0f, 1.0f, 2e-3f, 0.f, 0.f, 2e-3f), 0.0f});
    // A rotation of 0.5 deg expressed in the engine's shape function:
    // A = R - I, so ux = vy = cos-1 and uy = -sin, vx = +sin.
    s.push_back({"S4_rot_0.5deg",    affine(0.0f, 0.0f,
                                            (float)(std::cos(0.5 * CV_PI / 180.0) - 1.0),
                                            (float)(-std::sin(0.5 * CV_PI / 180.0)),
                                            (float)( std::sin(0.5 * CV_PI / 180.0)),
                                            (float)(std::cos(0.5 * CV_PI / 180.0) - 1.0)),
                                     0.0f});
    s.push_back({"S5_trans_12px",    affine(12.0f, -7.0f), 0.0f});
    s.push_back({"S6_trans_noise5",  affine(0.4f, 0.0f),  5.0f});
    return s;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::llround(p * (double)(v.size() - 1));
    return v[std::min(i, v.size() - 1)];
}

// Median over this many runs. 1 keeps the sweep quick while iterating; 5 is
// enough to stop VM scheduling noise from deciding a comparison.
int repeat_count() {
    if (const char *n = std::getenv("SEMPER_SEEDBENCH_REPEATS")) {
        const int v = std::atoi(n);
        if (v >= 1 && v <= 51) return v;
    }
    return 1;
}


// Rotation by theta about the deformation centre, in the engine's shape
// function: A = R - I, so ux = vy = cos(theta) - 1, uy = -sin(theta),
// vx = +sin(theta). Image y runs downward.
dictest::AffineDeformation rotation(double deg) {
    const double t = deg * CV_PI / 180.0;
    return affine(0.0f, 0.0f,
                  (float)(std::cos(t) - 1.0), (float)(-std::sin(t)),
                  (float)( std::sin(t)),      (float)(std::cos(t) - 1.0));
}

// Large-motion sweep uses a smaller, centred ROI: a 100 px translation or a
// 15 deg rotation moves the outer subsets a long way, and they must stay inside
// the image or every candidate fails for reasons that have nothing to do with
// seeding. 320 ROI in a 640 image leaves 160 px of margin.
constexpr int BIG_RECT = 320;
constexpr int BIG_ORIGIN = 160;

FullFieldParams big_motion_params() {
    FullFieldParams p;
    p.rect_x = BIG_ORIGIN;
    p.rect_y = BIG_ORIGIN;
    p.rect_w = BIG_RECT;
    p.rect_h = BIG_RECT;
    p.step = STEP;
    p.subset_size = SUBSET;
    p.strain_window = std::max(STRAIN_WIN, 5 * STEP + 1);
    if (p.strain_window % 2 == 0) p.strain_window += 1;
    p.use_6x6_interpolator = false;
    return p;
}

FullFieldParams bench_params() {
    FullFieldParams p;
    p.rect_x = RECT_ORIGIN;
    p.rect_y = RECT_ORIGIN;
    p.rect_w = ROI_SIZE;
    p.rect_h = ROI_SIZE;
    p.step = STEP;
    p.subset_size = SUBSET;
    p.strain_window = STRAIN_WIN;
    p.use_6x6_interpolator = false;
    return p;
}

// One row of the report.
struct Row {
    std::string scenario;
    std::string method;
    int vertices = 0;
    float coverage = 0.f;
    int quality = -1;              // 2 full / 1 sparse / 0 none
    double seed_ms = 0.0;
    double vtx_med_du = -1.0, vtx_p95_du = -1.0;
    double mesh_frac = 0.0;        // fraction of grid points inside the mesh
    double mesh_med_du = -1.0;
    double mesh_med_dux = -1.0, mesh_p95_dux = -1.0;
    bool phase_locked = false;         // anchor lattice: did phaseCorrelate trust its peak
    int anchors_accepted = 0;
    int anchors_attempted = 0;
    double seed_cpu_ms = 0.0;          // user+sys across the seeding call
    double seed_precompute_ms = 0.0;   // anchor lattice only
    double seed_icgn_ms = 0.0;         // anchor lattice only
    double ff_total_ms = 0.0, ff_seed_ms = 0.0;
    double ff_cpu_ms = 0.0;            // user+sys across run_full_field
    double ff_par = 0.0;               // cpu/wall, i.e. achieved parallelism
    double rss_retained_mb = 0.0;
    double ff_conv_pct = 0.0, ff_mean_iters = 0.0;
    int ff_simplex_calls = 0;
    double ff_rms_u = -1.0;
    int ff_rc = 0;
};

// Seed-level + mesh-level measurement. Calls the seeding front-end and the
// Delaunay mesh builder directly, so the two error columns are separated.
void measure_seed(Row &row, const TruthFn &truth,
                  const cv::Mat &ref_gray, const cv::Mat &def_gray,
                  const FullFieldParams &params) {
    using namespace Semper::pipeline::internal;

    ReferenceCache cache;
    cv::Mat no_mask;
    cache.set_from_gray(ref_gray, no_mask);

    Image def_img(def_gray.cols, def_gray.rows, def_gray.data);
    def_img.prepare_data(false);

    const int gridW = params.rect_w / params.step;
    const int gridH = params.rect_h / params.step;

    // Mirror run_full_field's setup: mark boundary-violating nodes as skipped,
    // then build the Hessian pool the anchor lattice reuses.
    const int reserve = params.subset_size / 2 + 4 + 15;
    ResultGrid grid(gridH, std::vector<GridPoint>(gridW));
    HessianPool pool((size_t)gridW * gridH);
    for (int gy = 0; gy < gridH; ++gy) {
        for (int gx = 0; gx < gridW; ++gx) {
            const int rx = params.rect_x + gx * params.step;
            const int ry = params.rect_y + gy * params.step;
            const bool skip = rx - reserve < 0 || rx + reserve >= cache.width ||
                              ry - reserve < 0 || ry + reserve >= cache.height;
            grid[gy][gx] = {(float)rx, (float)ry, 0, 0, 0, 0, 0, 0,
                            -1.0f, skip, -1, -1, 0, false, 0};
            if (!skip) {
                pool[(size_t)gy * gridW + gx] = SubsetPrecomputer::compute_hessian_only(
                        *cache.ref_img, rx, ry, params.subset_size);
            }
        }
    }

    // Match the pipeline's thread count so seeding cost is comparable to the
    // ff_ columns rather than being an artificially serial number.
    const int cores = std::max(1, (int)std::thread::hardware_concurrency());
    std::atomic<int> solved_count(0), order(1);
    std::vector<ThreadStats> stats((size_t)cores);
    PhaseTimings timings;
    MeshSeedResult seeds;

    // First call warms nothing (there is no descriptor cache any more), but the
    // repeats still give a median that survives VM scheduling noise.
    std::vector<double> seed_walls, seed_cpus;
    for (int rep = 0; rep < repeat_count(); ++rep) {
        ResultGrid g = grid;
        std::atomic<int> sc(0), od(1);
        std::vector<ThreadStats> st((size_t)cores);
        PhaseTimings t;
        auto t0 = std::chrono::high_resolution_clock::now();
        const CpuClock c0 = CpuClock::now();
        seeds = solve_anchor_seeds(cache, def_gray, def_img, params, gridW, gridH,
                                   cores, pool, g, st, t);
        // Publishing is a separate step in the pipeline now, but it is still
        // part of what the seeding phase costs, and `grid` below is expected to
        // carry the published anchors -- so time it here too.
        publish_anchor_results(seeds, sc, od, g, st);
        seed_cpus.push_back(CpuClock::now().total_ms() - c0.total_ms());
        seed_walls.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count());
        if (rep + 1 == repeat_count()) { grid = g; timings = t; }
    }
    row.seed_ms = percentile(seed_walls, 0.50);
    row.seed_cpu_ms = percentile(seed_cpus, 0.50);
    row.seed_precompute_ms = timings.phase_corr;
    row.seed_icgn_ms = timings.anchors;
    row.phase_locked = seeds.phase_locked;
    row.anchors_accepted = seeds.accepted;
    row.anchors_attempted = seeds.attempted;
    (void)solved_count; (void)order; (void)stats;

    row.vertices = (int)seeds.ref_pts.size();
    row.coverage = seeds.coverage;
    row.quality = seeds.quality == MeshQuality::FULL ? 2
                : seeds.quality == MeshQuality::SPARSE ? 1 : 0;

    std::vector<double> dus;
    for (size_t i = 0; i < seeds.ref_pts.size(); ++i) {
        const Truth t = truth(seeds.ref_pts[i].x, seeds.ref_pts[i].y);
        const float mu = seeds.def_pts[i].x - seeds.ref_pts[i].x;
        const float mv = seeds.def_pts[i].y - seeds.ref_pts[i].y;
        dus.push_back(std::sqrt((double)((mu - t.u) * (mu - t.u) + (mv - t.v) * (mv - t.v))));
    }
    row.vtx_med_du = percentile(dus, 0.50);
    row.vtx_p95_du = percentile(dus, 0.95);

    std::vector<AffineTriangle> tris;
    MeshGuessField guess = build_mesh_guess_field(
            cache, params, seeds.ref_pts, seeds.def_pts, seeds.quality,
            seeds.globalU, seeds.globalV, gridW, gridH, "", grid, tris, timings);

    std::vector<double> gdu, gdux;
    int in_mesh = 0;
    for (int gy = 0; gy < gridH; ++gy) {
        for (int gx = 0; gx < gridW; ++gx) {
            const int idx = gy * gridW + gx;
            if (!guess.in_mesh[(size_t)idx]) continue;
            in_mesh++;
            const Truth t = truth((float)(params.rect_x + gx * params.step),
                                  (float)(params.rect_y + gy * params.step));
            const double du = guess.u[(size_t)idx] - t.u;
            const double dv = guess.v[(size_t)idx] - t.v;
            gdu.push_back(std::sqrt(du * du + dv * dv));
            const double e00 = guess.ux[(size_t)idx] - t.ux;
            const double e01 = guess.uy[(size_t)idx] - t.uy;
            const double e10 = guess.vx[(size_t)idx] - t.vx;
            const double e11 = guess.vy[(size_t)idx] - t.vy;
            gdux.push_back(std::sqrt(e00 * e00 + e01 * e01 + e10 * e10 + e11 * e11));
        }
    }
    // Anchors that clear the result gate are already solved, so they are not in
    // the mesh guess field; count them as covered.
    row.mesh_frac = (double)(in_mesh + seeds.solved) / (double)(gridW * gridH);
    row.mesh_med_du = percentile(gdu, 0.50);
    row.mesh_med_dux = percentile(gdux, 0.50);
    row.mesh_p95_dux = percentile(gdux, 0.95);
}

// Downstream measurement: the whole solve, with this candidate seeding it.
void measure_full_field(Row &row, const TruthFn &truth,
                        const cv::Mat &ref_gray, const cv::Mat &def_gray,
                        const FullFieldParams &params) {
    ReferenceCache cache;
    cv::Mat no_mask;
    cache.set_from_gray(ref_gray, no_mask);

    const int gridW = params.rect_w / params.step;
    const int gridH = params.rect_h / params.step;
    std::vector<float> out((size_t)gridW * gridH * 8, 0.0f);
    float metrics[19] = {};

    // Wall and CPU time on a shared VM swing by ~10% run to run, which is
    // larger than some of the differences being judged. Repeat and take the
    // median rather than letting a single sample decide.
    const int repeats = repeat_count();
    std::vector<double> walls, cpus;
    int rc = 0;

    const double rss_before = current_rss_mb();
    for (int rep = 0; rep < repeats; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const CpuClock c0 = CpuClock::now();
        rc = Semper::pipeline::run_full_field(
                cache, def_gray, no_mask, params, out.data(), (int)out.size(),
                metrics, 19, nullptr);
        cpus.push_back(CpuClock::now().total_ms() - c0.total_ms());
        walls.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count());
    }
    const double wall = percentile(walls, 0.50);
    const double cpu = percentile(cpus, 0.50);

    row.ff_rc = rc;
    row.ff_total_ms = wall;
    row.ff_cpu_ms = cpu;
    row.ff_par = wall > 0.0 ? cpu / wall : 0.0;
    row.rss_retained_mb = current_rss_mb() - rss_before;
    row.ff_seed_ms = metrics[10];
    row.ff_conv_pct = metrics[15];
    row.ff_mean_iters = metrics[8];
    row.ff_simplex_calls = (int)metrics[5];

    if (rc > 0) {
        double sse = 0.0;
        int n = 0;
        for (int i = 0; i < rc; ++i) {
            const float x = out[(size_t)i * 8 + 0];
            const float y = out[(size_t)i * 8 + 1];
            const float u = out[(size_t)i * 8 + 2];
            const float v = out[(size_t)i * 8 + 3];
            const Truth t = truth(x, y);
            sse += (double)((u - t.u) * (u - t.u) + (v - t.v) * (v - t.v));
            n++;
        }
        row.ff_rms_u = n > 0 ? std::sqrt(sse / (double)n) : -1.0;
    }
}

void print_table(const std::vector<Row> &rows) {
    std::printf("\n");
    std::printf("%-20s %-15s %6s %6s %3s %5s %8s %8s %11s %11s %7s %11s %10s %10s %6s %8s %8s %7s %11s\n",
                "scenario", "method", "vtx", "cov", "q", "lock", "seed_ms", "seedCPU",
                "vtx_med_du", "mesh_med_dP", "meshfr", "mesh_med_du",
                "ff_wall_ms", "ff_cpu_ms", "par", "rssRet", "conv_%",
                "mean_it", "ff_rms_u");
    std::printf("%s\n", std::string(200, '-').c_str());
    for (const Row &r : rows) {
        char lock[8];
        if (r.anchors_attempted > 0) std::snprintf(lock, sizeof(lock), "%s", r.phase_locked ? "yes" : "NO");
        else std::snprintf(lock, sizeof(lock), "-");
        std::printf("%-20s %-15s %6d %6.3f %3d %5s %8.2f %8.2f %11.4f %11.3e %7.3f %11.4f %10.1f %10.1f %6.2f %8.1f %8.2f %7.2f %11.5f\n",
                    r.scenario.c_str(), r.method.c_str(), r.vertices, (double)r.coverage,
                    r.quality, lock, r.seed_ms, r.seed_cpu_ms, r.vtx_med_du, r.mesh_med_dux,
                    r.mesh_frac, r.mesh_med_du, r.ff_total_ms, r.ff_cpu_ms, r.ff_par,
                    r.rss_retained_mb, r.ff_conv_pct, r.ff_mean_iters, r.ff_rms_u);
    }
    std::printf("    process peak RSS: %.1f MB (comparable only with SEMPER_SEEDBENCH_ONLY)\n\n",
                process_peak_rss_mb());
}

void write_csv(const std::vector<Row> &rows, const char *path) {
    std::FILE *f = std::fopen(path, "w");
    if (!f) {
        std::printf("    (could not open %s for writing)\n", path);
        return;
    }
    std::fprintf(f, "scenario,method,vertices,coverage,quality,phase_locked,anchors_accepted,anchors_attempted,seed_ms,seed_cpu_ms,"
                    "seed_precompute_ms,seed_icgn_ms,"
                    "vtx_med_du,vtx_p95_du,mesh_frac,mesh_med_du,"
                    "mesh_med_dP,mesh_p95_dP,ff_total_ms,ff_cpu_ms,ff_par,"
                    "rss_retained_mb,ff_seed_ms,"
                    "ff_conv_pct,ff_mean_iters,ff_simplex_calls,ff_rms_u,ff_rc\n");
    for (const Row &r : rows) {
        std::fprintf(f, "%s,%s,%d,%.6f,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,"
                        "%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,"
                        "%.3f,%.3f,%.4f,%.2f,%.3f,%.4f,%.4f,%d,%.8f,%d\n",
                     r.scenario.c_str(), r.method.c_str(), r.vertices,
                     (double)r.coverage, r.quality, r.phase_locked ? 1 : 0,
                     r.anchors_accepted, r.anchors_attempted, r.seed_ms, r.seed_cpu_ms,
                     r.seed_precompute_ms, r.seed_icgn_ms, r.vtx_med_du,
                     r.vtx_p95_du, r.mesh_frac, r.mesh_med_du, r.mesh_med_dux,
                     r.mesh_p95_dux, r.ff_total_ms, r.ff_cpu_ms, r.ff_par,
                     r.rss_retained_mb, r.ff_seed_ms, r.ff_conv_pct,
                     r.ff_mean_iters, r.ff_simplex_calls, r.ff_rms_u, r.ff_rc);
    }
    std::fclose(f);
    std::printf("    CSV written to %s\n", path);
}


// ---------------------------------------------------------------------------
// SEM / iDICs DIC Challenge sets, read from an external DICe checkout.
//
// Not vendored: the engine already carries oht_cfrp_* under LICENSE.DICe, but
// these images originate with SEM/iDICs and their redistribution terms have not
// been checked. Point SEMPER_DICE_REPO at a dicengine/dice clone; the tests
// skip cleanly when it is unset.
//
//   Sample 5   512x512, rigid translation, magnitude in the filename.
//   Sample 14  2048x589, u(x) = A sin(2 pi x / L) with A = 0.100 px. The
//              commanded displacement and strain are tabulated per x column in
//              command.csv. Non-affine by construction, so this is the set that
//              actually exercises the guess gradients.
// ---------------------------------------------------------------------------

const char *dice_repo() { return std::getenv("SEMPER_DICE_REPO"); }

bool load_gray_mat(const std::string &path, cv::Mat &out) {
    dictest::GrayImage g;
    if (!dictest::load_gray(path, g)) return false;
    out = cv::Mat(g.h, g.w, CV_8UC1);
    std::memcpy(out.data, g.px.data(), g.px.size());
    return true;
}

// command.csv: x, u_L1, exx_L1, u_L3, exx_L3, u_L5, exx_L5 (strain in microstrain).
// Column index 1 + 2*k selects the displacement for series k.
struct CommandCurve {
    std::vector<double> u;      // px, indexed by integer x
    std::vector<double> exx;    // strain (already converted from microstrain)
};

bool load_command_csv(const std::string &path, int series_index, CommandCurve &out) {
    std::ifstream f(path);
    if (!f) return false;
    const size_t uc = 1 + 2 * (size_t)series_index;
    const size_t ec = uc + 1;
    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> vals;
        while (std::getline(ss, cell, ',')) {
            try { vals.push_back(std::stod(cell)); } catch (...) { vals.clear(); break; }
        }
        if (vals.size() <= ec) continue;
        out.u.push_back(vals[uc]);
        out.exx.push_back(vals[ec] * 1e-6);
    }
    return out.u.size() > 16;
}

// The commanded field varies only along x, so v == 0 and the only nonzero
// gradient component is ux = d(u)/dx, read straight from the commanded strain.
TruthFn sinusoid_truth(const CommandCurve &c) {
    return [&c](float x, float) {
        Truth t;
        int i = (int)std::lround(x);
        if (i < 0) i = 0;
        if (i >= (int)c.u.size()) i = (int)c.u.size() - 1;
        t.u = (float)c.u[(size_t)i];
        t.ux = (float)c.exx[(size_t)i];
        return t;
    };
}

void run_scenario(const char *scenario, const TruthFn &truth,
                  const cv::Mat &ref_gray, const cv::Mat &def_gray,
                  const FullFieldParams &params, std::vector<Row> &rows) {
    Row row;
    row.scenario = scenario;
    row.method = "anchor_lattice";
    std::printf("    [%s] ...\n", row.scenario.c_str());
    std::fflush(stdout);
    measure_seed(row, truth, ref_gray, def_gray, params);
    measure_full_field(row, truth, ref_gray, def_gray, params);
    rows.push_back(row);
}

FullFieldParams params_for(int img_w, int img_h, int subset, int step) {
    // Same reserve run_full_field applies: half a subset, DICe's 4 px
    // interpolation buffer, and a 15 px deformation allowance, plus a margin.
    const int reserve = subset / 2 + 4 + 15 + 5;
    FullFieldParams p;
    p.rect_x = reserve;
    p.rect_y = reserve;
    p.rect_w = img_w - 2 * reserve;
    p.rect_h = img_h - 2 * reserve;
    p.step = step;
    p.subset_size = subset;
    // The VSG window must span several grid nodes or every least-squares fit is
    // rank-deficient, the strain sentinel fires, and the post-filter drops the
    // whole field (run_full_field then returns 0, not an error). DICe's own
    // Sample 14 config uses a 260 px window at step 25 -- about 10x the step.
    p.strain_window = std::max(STRAIN_WIN, 5 * step + 1);
    if (p.strain_window % 2 == 0) p.strain_window += 1;
    p.use_6x6_interpolator = false;
    return p;
}

} // namespace

// ---------------------------------------------------------------------------
// Sign convention of cv::phaseCorrelate against the engine's warp.
//
// The engine's shape function puts a reference point p at p + (u, v) in the
// deformed image. This pins which way round phaseCorrelate reports that, so
// the anchor lattice cannot silently seed with an inverted global guess.
// Cheap and deterministic, so it runs in the default suite.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, PhaseCorrelateSignConvention) {
    const float U = 5.0f, V = -3.0f;
    dictest::SpeckleField field(/*seed=*/7, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const Image def = dictest::make_deformed_image(field, W, H, affine(U, V));

    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 0);
    const cv::Mat def_gray = dictest::gray8(def, 0.0f, 0);

    double u = 0.0, v = 0.0, resp = 0.0;
    const bool ok = Semper::seeding::phase_correlate_roi(
            ref_gray, def_gray, cv::Rect(RECT_ORIGIN, RECT_ORIGIN, ROI_SIZE, ROI_SIZE),
            u, v, resp);

    REQUIRE(ok);
    CHECK(resp > 0.5);
    CHECK_NEAR(u, U, 0.05);
    CHECK_NEAR(v, V, 0.05);
    std::printf("    phaseCorrelate -> (%.4f, %.4f), truth (%.1f, %.1f), response %.4f\n",
                u, v, (double)U, (double)V, resp);
}

// ---------------------------------------------------------------------------
// The always-on half of this file.
//
// Every sweep in this file is gated behind SEMPER_RUN_SEEDBENCH, and three of
// them additionally behind an external DICe checkout, which meant that before
// this case the largest behavioural change in the repo's history was defended
// in CI by exactly one assertion -- the sign convention above.
//
// This case promotes three of the seven CandidateSweep scenarios to always-on
// assertions. They are chosen to cover the three regimes the anchor lattice
// is claimed to handle, not to re-measure the sweep:
//
//   S1b_trans_0.4px  the sub-pixel regime the engine actually ships for
//   S4_rot_0.5deg    small rotation -- docs/SEEDING_BENCHMARK.md 4.5 records
//                    that the lattice degrades above ~5 deg, so the lower end
//                    of that range is the boundary worth pinning
//   S5_trans_12px    motion far outside IC-GN's own capture range, which only
//                    succeeds if the L0 phase-correlation lock is real
//
// The bounds are deliberately loose -- 3x to 5x the measured value, listed
// per row below. This is a regression gate, not a benchmark: it should fire
// when seeding stops working, and stay silent when it merely gets 20% worse
// on one host. The sweep remains the place to read exact numbers.
//
// Reference values are the medians reported by CandidateSweep and the
// criteria in docs/SEEDING_BENCHMARK.md section 9. For scale, the AKAZE
// front-end this replaced measured 0.157 px vertex error and 9.39e-3 gradient
// error at 0.446 mesh coverage on the DICe pair (section 4.1), so every bound
// here is still far tighter than the behaviour that shipped before it.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, AnchorSeedingQualityGate) {
    struct Gate {
        const char *name;
        dictest::AffineDeformation def;
        double max_vtx_med_du;   // px, median vertex displacement error
        double max_mesh_med_dux; // guess displacement-gradient error
        double max_mesh_med_du;  // px, median guess displacement error
    };

    const Gate gates[] = {
        // measured: 0.0103 px / 5.52e-4 / 0.0083 px
        {"S1b_trans_0.4px", affine(0.4f, 0.0f), 0.05, 2.0e-3, 0.04},
        // measured: 0.0106 px / 5.41e-4 / 0.0083 px
        {"S4_rot_0.5deg", rotation(0.5), 0.05, 2.0e-3, 0.04},
        // measured: 0.0000 px / 1.31e-6 / 0.0000 px. Bounded well above the
        // measurement because what is being asserted is that the phase lock
        // carried a 12 px motion at all, not the last digit of an exact hit.
        {"S5_trans_12px", affine(12.0f, -7.0f), 0.02, 5.0e-5, 0.02},
    };

    dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 101);

    for (const Gate &g : gates) {
        const Image def = dictest::make_deformed_image(field, W, H, g.def);
        const cv::Mat def_gray = dictest::gray8(def, 0.0f, 202);

        Row row;
        row.scenario = g.name;
        measure_seed(row, affine_truth(g.def), ref_gray, def_gray,
                     bench_params());

        std::printf("    %-16s lock=%s q=%d cov=%.3f vtx=%.4f px "
                    "dP=%.3e du=%.4f px meshfr=%.3f\n",
                    g.name, row.phase_locked ? "yes" : "NO", row.quality,
                    (double) row.coverage, row.vtx_med_du, row.mesh_med_dux,
                    row.mesh_med_du, row.mesh_frac);

        // A lost phase lock is the single failure that explains most of the
        // others, so check it first and by itself.
        CHECK(row.phase_locked);
        // MeshQuality::FULL. SPARSE means the hull did not cover the ROI and
        // Path B is doing work the mesh was supposed to do.
        CHECK(row.quality == 2);
        CHECK(row.coverage >= 0.90f);
        CHECK(row.mesh_frac >= 0.95);
        // percentile() returns -1 for an empty sample; the bounds below would
        // then pass vacuously.
        REQUIRE(row.vtx_med_du >= 0.0);
        REQUIRE(row.mesh_med_dux >= 0.0);
        CHECK(row.vtx_med_du <= g.max_vtx_med_du);
        CHECK(row.mesh_med_dux <= g.max_mesh_med_dux);
        CHECK(row.mesh_med_du <= g.max_mesh_med_du);
    }
}

// ---------------------------------------------------------------------------
// The publish gate. solve_anchor_seeds records outcomes and publishes nothing;
// publish_anchor_results applies them once the universal median test has run.
// Nothing in the fixtures below produces a blunder that clears kCorrAccept and
// then fails the median test -- that is a periodic-speckle failure and it is
// hard to stage deterministically -- so the gate itself is checked directly.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, PublishAnchorResultsRespectsMedianTest) {
    using Semper::pipeline::internal::AnchorResult;
    using Semper::pipeline::internal::GridPoint;
    using Semper::pipeline::internal::MeshSeedResult;
    using Semper::pipeline::internal::ResultGrid;
    using Semper::pipeline::internal::ThreadStats;
    using Semper::pipeline::internal::publish_anchor_results;

    auto make = [](int gx, int gy, bool output, bool kept, int tid) {
        AnchorResult a;
        a.gx = gx; a.gy = gy;
        a.rx = (float)gx; a.ry = (float)gy;
        a.u = 1.0f + (float)gx; a.v = 2.0f;
        a.corr = 0.05f;
        a.thread_id = tid;
        a.output = output;
        a.kept = kept;
        return a;
    };

    MeshSeedResult seeds;
    seeds.anchors = {
            make(1, 1, true,  true,  0),   // published
            make(2, 1, true,  false, 1),   // the blunder: cleared kCorrAccept,
                                           // rejected by the median test
            make(3, 1, false, true,  1),   // a vertex, but too loose to publish
            make(0, 0, false, false, 0),   // never converged: default slot
            make(9, 9, true,  true,  0),   // outside the grid entirely
    };

    ResultGrid grid(4, std::vector<GridPoint>(4));
    std::atomic<int> solved(0), order(0);
    std::vector<ThreadStats> stats(2);

    const int published = publish_anchor_results(seeds, solved, order, grid, stats);

    CHECK(published == 1);
    CHECK(solved.load() == 1);
    CHECK(order.load() == 1);   // one order number consumed, not five

    CHECK(grid[1][1].solved);
    CHECK_NEAR(grid[1][1].u, 2.0f, 1e-6f);
    CHECK(grid[1][1].mesh_assignment_type == Semper::pipeline::internal::kMeshAnchor);

    // The blunder must not reach the field: Path B would take it as a boundary
    // seed and flood fill from a wrong displacement.
    CHECK(!grid[1][2].solved);
    CHECK(!grid[1][3].solved);
    CHECK(!grid[0][0].solved);

    // Credit lands on the thread that ran the ICGN, not on whoever published.
    CHECK(stats[0].points_solved == 1);
    CHECK(stats[1].points_solved == 0);
}

// ---------------------------------------------------------------------------
// Lattice sizing. No image, no solve -- plan_anchor_lattice is exposed so the
// stride arithmetic can be checked directly, because the failure it guards
// against is a cost blow-up on a geometry no fixture in this repo has.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, AnchorLatticeSizing) {
    using Semper::pipeline::internal::AnchorLattice;
    using Semper::pipeline::internal::plan_anchor_lattice;
    namespace tuning = Semper::tuning;

    // Square ROI: the stride is the isotropic sqrt(nodes / target) and both
    // axes keep it. 64 x 64 = 4096 nodes, 4096 / 256 = 16, sqrt = 4. Each axis
    // carries indices 0, 4, ... 60 plus the appended last index 63 = 17, so
    // 289 anchors -- which is exactly the vertex count CandidateSweep reports.
    // This case pins the per-axis clamp as a no-op on square-ish ROIs.
    {
        const AnchorLattice lat = plan_anchor_lattice(64, 64);
        CHECK(lat.stride_x == 4);
        CHECK(lat.stride_y == 4);
        CHECK(lat.nx == 17);
        CHECK(lat.ny == 17);
        CHECK(lat.count() == 289);
    }

    // The regression this test exists for. A long thin ROI -- a beam, a weld
    // seam -- used to have the stride clamped from the *shorter* axis: 6 / 3 = 2
    // forced on both, giving a lattice of ~2000 nodes against a target of 256.
    // Clamping per axis keeps the short axis at its 3-node floor and pays for
    // it by coarsening the long one.
    {
        const AnchorLattice lat = plan_anchor_lattice(1000, 6);
        CHECK(lat.ny >= 3);   // the floor the old clamp was protecting
        CHECK(lat.nx >= 3);
        // Not 256 exactly: the short axis cannot go below 3, so the product
        // overshoots. It must stay the same order as the target, not 8x it.
        CHECK(lat.count() <= 2 * tuning::kAnchorTarget);
        CHECK(lat.stride_x > lat.stride_y);
    }

    // Same ROI on its side. Nothing in the planner should prefer an axis.
    {
        const AnchorLattice a = plan_anchor_lattice(1000, 6);
        const AnchorLattice b = plan_anchor_lattice(6, 1000);
        CHECK(b.stride_x == a.stride_y);
        CHECK(b.stride_y == a.stride_x);
        CHECK(b.count() == a.count());
    }

    // A grid too small to hold 3 lattice nodes per axis: the floor cannot be
    // met, and asking for it must not loop or return an empty lattice.
    {
        const AnchorLattice lat = plan_anchor_lattice(2, 2);
        CHECK(lat.stride_x == 1);
        CHECK(lat.stride_y == 1);
        CHECK(lat.count() == 4);
    }

    // Degenerate input: no grid, no lattice, no arithmetic on it.
    {
        CHECK(plan_anchor_lattice(0, 10).count() == 0);
        CHECK(plan_anchor_lattice(10, 0).count() == 0);
        CHECK(plan_anchor_lattice(-1, -1).count() == 0);
    }
}

// ---------------------------------------------------------------------------
// The sweep. Measurement only — no assertions on the winner.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, CandidateSweep) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1 to run the sweep)\n");
        return;
    }

    std::vector<Row> rows;
    for (const Scenario &sc : scenarios()) {
        dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
        const Image ref = dictest::make_reference_image(field, W, H);
        const Image def = dictest::make_deformed_image(field, W, H, sc.def);
        const cv::Mat ref_gray = dictest::gray8(ref, sc.noise_sigma, 101);
        const cv::Mat def_gray = dictest::gray8(def, sc.noise_sigma, 202);

        run_scenario(sc.name, affine_truth(sc.def), ref_gray, def_gray,
                       bench_params(), rows);
    }

    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV")) write_csv(rows, csv);
}

// ---------------------------------------------------------------------------
// The same sweep on DICe's real speckle (ref.tif / def.tif, 512x512).
//
// The pair is a rigid translation, so the true displacement-gradient tensor is
// exactly zero regardless of the translation magnitude — mesh_med_dP is
// rigorous ground truth here. The displacement columns use DICe's published
// U = 0.4 px and V = 0.4 px; DICe publishes no V reference, so read the
// displacement columns as indicative and the gradient columns as exact.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, RealSpeckleSweep) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1 to run the sweep)\n");
        return;
    }

    const std::string dir = DICE_FIXTURES_DIR;
    dictest::GrayImage r, d;
    REQUIRE(dictest::load_gray(dir + "/ref.tif", r));
    REQUIRE(dictest::load_gray(dir + "/def.tif", d));
    REQUIRE(r.w == 512 && r.h == 512);
    REQUIRE(d.w == r.w && d.h == r.h);

    cv::Mat ref_gray(r.h, r.w, CV_8UC1, r.px.data());
    cv::Mat def_gray(d.h, d.w, CV_8UC1, d.px.data());

    Scenario sc{};
    sc.name = "REAL_dice_0.4px";
    sc.def.u = 0.4f;
    sc.def.v = 0.4f;
    sc.def.cx = 256.0f;
    sc.def.cy = 256.0f;
    sc.noise_sigma = 0.0f;

    FullFieldParams params;
    params.rect_x = 56;
    params.rect_y = 56;
    params.rect_w = 400;
    params.rect_h = 400;
    params.step = STEP;
    params.subset_size = SUBSET;
    params.strain_window = STRAIN_WIN;
    params.use_6x6_interpolator = false;

    std::vector<Row> rows;
    run_scenario(sc.name, affine_truth(sc.def), ref_gray, def_gray, params, rows);

    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_REAL")) write_csv(rows, csv);
}

// ---------------------------------------------------------------------------
// DIC Challenge Sample 5 — real speckle, exact rigid translation (0.10 px).
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, Challenge5) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1)\n");
        return;
    }
    const char *repo = dice_repo();
    if (repo == nullptr) {
        std::printf("    skipped (set SEMPER_DICE_REPO to a dicengine/dice checkout)\n");
        return;
    }
    const std::string dir = std::string(repo) + "/tests/regression/dic_challenge_5_znssd/images/";

    cv::Mat ref_gray, def_gray;
    REQUIRE(load_gray_mat(dir + "Sample5 Reference.tif", ref_gray));
    REQUIRE(load_gray_mat(dir + "Sample5-001 X0.10 Y0.10 N2 C100 R20.tif", def_gray));
    REQUIRE(ref_gray.cols == 512 && ref_gray.rows == 512);

    dictest::AffineDeformation d;
    d.u = 0.10f; d.v = 0.10f;
    d.cx = 256.0f; d.cy = 256.0f;

    std::vector<Row> rows;
    run_scenario("C5_real_0.10px", affine_truth(d), ref_gray, def_gray,
                 params_for(512, 512, SUBSET, STEP), rows);
    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_C5")) write_csv(rows, csv);
}

// ---------------------------------------------------------------------------
// DIC Challenge Sample 14 — real speckle, u(x) = 0.1 sin(2 pi x / L).
//
// This is the decisive set for the guess gradients: the field is non-affine,
// the amplitude is 0.100 px, and the shortest period (L5, ~205 px) puts a
// continuously varying strain of ~3e-3 across the ROI. A mesh built on
// clustered keypoints has no way to represent that; a uniform lattice does.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, Challenge14Sinusoid) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1)\n");
        return;
    }
    const char *repo = dice_repo();
    if (repo == nullptr) {
        std::printf("    skipped (set SEMPER_DICE_REPO to a dicengine/dice checkout)\n");
        return;
    }
    const std::string base = std::string(repo) + "/tests/regression/dic_challenge_14_vsg/";

    cv::Mat ref_gray;
    REQUIRE(load_gray_mat(base + "images/Sample14 Reference.tif", ref_gray));
    REQUIRE(ref_gray.cols == 2048 && ref_gray.rows == 589);

    // L1 is the longest period and L5 the shortest; series index selects the
    // column pair in command.csv.
    const struct { const char *tag; const char *file; int series; } sets[] = {
            {"C14_L3_sin356px", "images/Sample14 L3 Amp0.1b.tif", 1},
            {"C14_L5_sin205px", "images/Sample14 L5 Amp0.1b.tif", 2},
    };

    std::vector<Row> rows;
    for (const auto &set : sets) {
        cv::Mat def_gray;
        REQUIRE(load_gray_mat(base + set.file, def_gray));

        CommandCurve curve;
        REQUIRE(load_command_csv(base + "command.csv", set.series, curve));

        // 589 rows is short, so the ROI is height-limited; keep the full width
        // where the sinusoid lives.
        run_scenario(set.tag, sinusoid_truth(curve), ref_gray, def_gray,
                       params_for(2048, 589, SUBSET, 10), rows);
    }
    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_C14")) write_csv(rows, csv);
}

// ---------------------------------------------------------------------------
// Rotation and large-motion sweep.
//
// This is the regime the descriptor front-end was actually built for: DICe uses
// its AKAZE Feature_Matching_Initializer for stereo and for re-acquiring lost
// subsets, not for small-displacement 2D DIC. AKAZE is scale- and
// rotation-invariant; phase correlation is not, and the anchor lattice's
// fallback when the correlation peak is untrustworthy is only a +/-15 px
// per-anchor search. So this sweep is expected to be where the lattice loses,
// and it exists to find out where the crossover actually is.
//
// The `lock` column is the diagnostic: `NO` means phaseCorrelate's peak was
// below kPhaseCorrMinResponse and every anchor fell back to its own coarse
// search.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, LargeMotionSweep) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1)\n");
        return;
    }

    std::vector<Scenario> sweep;
    sweep.push_back({"R_rot_2deg",    rotation(2.0),  0.0f});
    sweep.push_back({"R_rot_5deg",    rotation(5.0),  0.0f});
    sweep.push_back({"R_rot_15deg",   rotation(15.0), 0.0f});
    sweep.push_back({"T_trans_25px",  affine(25.0f, -15.0f),  0.0f});
    sweep.push_back({"T_trans_50px",  affine(50.0f, -30.0f),  0.0f});
    sweep.push_back({"T_trans_100px", affine(100.0f, -60.0f), 0.0f});

    const FullFieldParams params = big_motion_params();
    std::vector<Row> rows;
    for (const Scenario &sc : sweep) {
        dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
        const Image ref = dictest::make_reference_image(field, W, H);
        const Image def = dictest::make_deformed_image(field, W, H, sc.def);
        const cv::Mat ref_gray = dictest::gray8(ref, sc.noise_sigma, 101);
        const cv::Mat def_gray = dictest::gray8(def, sc.noise_sigma, 202);
        run_scenario(sc.name, affine_truth(sc.def), ref_gray, def_gray, params, rows);
    }

    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_BIG")) write_csv(rows, csv);
}

#endif // DIC_HAVE_OPENCV
