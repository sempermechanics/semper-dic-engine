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
#include "framework/image_io.h"
#include "pipeline/full_field_internal.hpp"

#include <semper/pipeline.hpp>
#include <semper/seeding.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using Semper::Image;
using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;
using Semper::seeding::SeedMethod;

namespace {

constexpr int W = 640;
constexpr int H = 640;
constexpr int BLOBS = 1600;          // ~11% blob coverage, DIC-representative
constexpr int RECT = 480;
constexpr int RECT_ORIGIN = 80;
constexpr int STEP = 5;
constexpr int SUBSET = 41;
constexpr int STRAIN_WIN = 15;

const SeedMethod kCandidates[] = {
        SeedMethod::AkazePyramid,
        SeedMethod::AkazeFull,
        SeedMethod::Sift,
        SeedMethod::Orb,
        SeedMethod::Brisk,
        SeedMethod::Kaze,
        SeedMethod::AnchorLattice,
};

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

cv::Mat gray8(const Image &img, float noise_sigma, unsigned noise_seed) {
    cv::Mat m(img.height, img.width, CV_8UC1);
    std::mt19937 rng(noise_seed);
    std::normal_distribution<float> gauss(0.0f, noise_sigma);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float v = img.intensities[(size_t)y * img.width + x];
            if (noise_sigma > 0.0f) v += gauss(rng);
            if (v < 0.f) v = 0.f;
            if (v > 255.f) v = 255.f;
            m.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
        }
    }
    return m;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)std::llround(p * (double)(v.size() - 1));
    return v[std::min(i, v.size() - 1)];
}

FullFieldParams bench_params() {
    FullFieldParams p;
    p.rect_x = RECT_ORIGIN;
    p.rect_y = RECT_ORIGIN;
    p.rect_w = RECT;
    p.rect_h = RECT;
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
    double ff_total_ms = 0.0, ff_seed_ms = 0.0;
    double ff_conv_pct = 0.0, ff_mean_iters = 0.0;
    int ff_simplex_calls = 0;
    double ff_rms_u = -1.0;
    int ff_rc = 0;
};

// Seed-level + mesh-level measurement. Calls the seeding front-end and the
// Delaunay mesh builder directly, so the two error columns are separated.
void measure_seed(Row &row, const Scenario &sc, SeedMethod method,
                  const cv::Mat &ref_gray, const cv::Mat &def_gray,
                  const FullFieldParams &params) {
    using namespace Semper::pipeline::internal;

    ReferenceCache cache;
    cv::Mat no_mask;
    cache.set_from_gray(ref_gray, no_mask);
    cache.seed_method = method;

    Image def_img(def_gray.cols, def_gray.rows, def_gray.data);
    def_img.prepare_data(false);

    cv::Mat roi_mask;

    auto t0 = std::chrono::high_resolution_clock::now();
    MeshSeedResult seeds = detect_mesh_seeds(cache, def_gray, def_img, roi_mask, params, "");
    row.seed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

    row.vertices = (int)seeds.ref_pts.size();
    row.coverage = seeds.coverage;
    row.quality = seeds.quality == MeshQuality::FULL ? 2
                : seeds.quality == MeshQuality::SPARSE ? 1 : 0;

    // Vertex displacement error against the analytic warp at each vertex.
    std::vector<double> dus;
    for (size_t i = 0; i < seeds.ref_pts.size(); ++i) {
        const float px = seeds.ref_pts[i].x - sc.def.cx;
        const float py = seeds.ref_pts[i].y - sc.def.cy;
        const float tu = sc.def.u + sc.def.ux * px + sc.def.uy * py;
        const float tv = sc.def.v + sc.def.vx * px + sc.def.vy * py;
        const float mu = seeds.def_pts[i].x - seeds.ref_pts[i].x;
        const float mv = seeds.def_pts[i].y - seeds.ref_pts[i].y;
        dus.push_back(std::sqrt((double)((mu - tu) * (mu - tu) + (mv - tv) * (mv - tv))));
    }
    row.vtx_med_du = percentile(dus, 0.50);
    row.vtx_p95_du = percentile(dus, 0.95);

    // Mesh guess field: the gradients the solve actually receives.
    const int gridW = params.rect_w / params.step;
    const int gridH = params.rect_h / params.step;
    ResultGrid grid(gridH, std::vector<GridPoint>(gridW));
    for (int gy = 0; gy < gridH; ++gy) {
        for (int gx = 0; gx < gridW; ++gx) {
            grid[gy][gx].x = (float)(params.rect_x + gx * params.step);
            grid[gy][gx].y = (float)(params.rect_y + gy * params.step);
        }
    }
    std::vector<AffineTriangle> tris;
    PhaseTimings timings;
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
            const float px = (float)(params.rect_x + gx * params.step) - sc.def.cx;
            const float py = (float)(params.rect_y + gy * params.step) - sc.def.cy;
            const float tu = sc.def.u + sc.def.ux * px + sc.def.uy * py;
            const float tv = sc.def.v + sc.def.vx * px + sc.def.vy * py;
            const double du = guess.u[(size_t)idx] - tu;
            const double dv = guess.v[(size_t)idx] - tv;
            gdu.push_back(std::sqrt(du * du + dv * dv));
            // Frobenius error of the guessed displacement-gradient tensor.
            const double e00 = guess.ux[(size_t)idx] - sc.def.ux;
            const double e01 = guess.uy[(size_t)idx] - sc.def.uy;
            const double e10 = guess.vx[(size_t)idx] - sc.def.vx;
            const double e11 = guess.vy[(size_t)idx] - sc.def.vy;
            gdux.push_back(std::sqrt(e00 * e00 + e01 * e01 + e10 * e10 + e11 * e11));
        }
    }
    row.mesh_frac = (double)in_mesh / (double)(gridW * gridH);
    row.mesh_med_du = percentile(gdu, 0.50);
    row.mesh_med_dux = percentile(gdux, 0.50);
    row.mesh_p95_dux = percentile(gdux, 0.95);
}

// Downstream measurement: the whole solve, with this candidate seeding it.
void measure_full_field(Row &row, const Scenario &sc, SeedMethod method,
                        const cv::Mat &ref_gray, const cv::Mat &def_gray,
                        const FullFieldParams &params) {
    ReferenceCache cache;
    cv::Mat no_mask;
    cache.set_from_gray(ref_gray, no_mask);
    cache.seed_method = method;

    const int gridW = params.rect_w / params.step;
    const int gridH = params.rect_h / params.step;
    std::vector<float> out((size_t)gridW * gridH * 8, 0.0f);
    float metrics[19] = {};

    auto t0 = std::chrono::high_resolution_clock::now();
    const int rc = Semper::pipeline::run_full_field(
            cache, def_gray, no_mask, params, out.data(), (int)out.size(),
            metrics, 19, nullptr);
    const double wall = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

    row.ff_rc = rc;
    row.ff_total_ms = wall;
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
            const float px = x - sc.def.cx, py = y - sc.def.cy;
            const float tu = sc.def.u + sc.def.ux * px + sc.def.uy * py;
            const float tv = sc.def.v + sc.def.vx * px + sc.def.vy * py;
            sse += (double)((u - tu) * (u - tu) + (v - tv) * (v - tv));
            n++;
        }
        row.ff_rms_u = n > 0 ? std::sqrt(sse / (double)n) : -1.0;
    }
}

void print_table(const std::vector<Row> &rows) {
    std::printf("\n");
    std::printf("%-18s %-15s %6s %6s %4s %9s %10s %10s %7s %10s %11s %11s %10s %8s %8s %7s %11s\n",
                "scenario", "method", "vtx", "cov", "q", "seed_ms",
                "vtx_med_du", "vtx_p95_du", "meshfr", "mesh_med_du",
                "mesh_med_dP", "mesh_p95_dP", "ff_total_ms", "conv_%",
                "mean_it", "simplex", "ff_rms_u");
    std::printf("%s\n", std::string(190, '-').c_str());
    for (const Row &r : rows) {
        std::printf("%-18s %-15s %6d %6.3f %4d %9.2f %10.4f %10.4f %7.3f %11.4f %11.3e %11.3e %10.1f %8.2f %8.2f %7d %11.5f\n",
                    r.scenario.c_str(), r.method.c_str(), r.vertices, (double)r.coverage,
                    r.quality, r.seed_ms, r.vtx_med_du, r.vtx_p95_du, r.mesh_frac,
                    r.mesh_med_du, r.mesh_med_dux, r.mesh_p95_dux,
                    r.ff_total_ms, r.ff_conv_pct, r.ff_mean_iters,
                    r.ff_simplex_calls, r.ff_rms_u);
    }
    std::printf("\n");
}

void write_csv(const std::vector<Row> &rows, const char *path) {
    std::FILE *f = std::fopen(path, "w");
    if (!f) {
        std::printf("    (could not open %s for writing)\n", path);
        return;
    }
    std::fprintf(f, "scenario,method,vertices,coverage,quality,seed_ms,"
                    "vtx_med_du,vtx_p95_du,mesh_frac,mesh_med_du,"
                    "mesh_med_dP,mesh_p95_dP,ff_total_ms,ff_seed_ms,"
                    "ff_conv_pct,ff_mean_iters,ff_simplex_calls,ff_rms_u,ff_rc\n");
    for (const Row &r : rows) {
        std::fprintf(f, "%s,%s,%d,%.6f,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,"
                        "%.3f,%.3f,%.4f,%.4f,%d,%.8f,%d\n",
                     r.scenario.c_str(), r.method.c_str(), r.vertices,
                     (double)r.coverage, r.quality, r.seed_ms, r.vtx_med_du,
                     r.vtx_p95_du, r.mesh_frac, r.mesh_med_du, r.mesh_med_dux,
                     r.mesh_p95_dux, r.ff_total_ms, r.ff_seed_ms, r.ff_conv_pct,
                     r.ff_mean_iters, r.ff_simplex_calls, r.ff_rms_u, r.ff_rc);
    }
    std::fclose(f);
    std::printf("    CSV written to %s\n", path);
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

    const cv::Mat ref_gray = gray8(ref, 0.0f, 0);
    const cv::Mat def_gray = gray8(def, 0.0f, 0);

    double u = 0.0, v = 0.0, resp = 0.0;
    const bool ok = Semper::seeding::phase_correlate_roi(
            ref_gray, def_gray, cv::Rect(RECT_ORIGIN, RECT_ORIGIN, RECT, RECT),
            u, v, resp);

    REQUIRE(ok);
    CHECK(resp > 0.5);
    CHECK_NEAR(u, U, 0.05);
    CHECK_NEAR(v, V, 0.05);
    std::printf("    phaseCorrelate -> (%.4f, %.4f), truth (%.1f, %.1f), response %.4f\n",
                u, v, (double)U, (double)V, resp);
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
        const cv::Mat ref_gray = gray8(ref, sc.noise_sigma, 101);
        const cv::Mat def_gray = gray8(def, sc.noise_sigma, 202);

        for (SeedMethod m : kCandidates) {
            Row row;
            row.scenario = sc.name;
            row.method = Semper::seeding::seed_method_name(m);
            std::printf("    [%s / %s] ...\n", row.scenario.c_str(), row.method.c_str());
            std::fflush(stdout);
            measure_seed(row, sc, m, ref_gray, def_gray, bench_params());
            measure_full_field(row, sc, m, ref_gray, def_gray, bench_params());
            rows.push_back(row);
        }
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
    for (SeedMethod m : kCandidates) {
        Row row;
        row.scenario = sc.name;
        row.method = Semper::seeding::seed_method_name(m);
        std::printf("    [%s / %s] ...\n", row.scenario.c_str(), row.method.c_str());
        std::fflush(stdout);
        measure_seed(row, sc, m, ref_gray, def_gray, params);
        measure_full_field(row, sc, m, ref_gray, def_gray, params);
        rows.push_back(row);
    }

    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_REAL")) write_csv(rows, csv);
}

#endif // DIC_HAVE_OPENCV
