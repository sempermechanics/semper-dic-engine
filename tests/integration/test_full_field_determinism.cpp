// Run-to-run determinism of the whole full-field solve (docs/TESTING.md,
// "Numerical reproducibility contract"): the same inputs, solved again in the
// same process, must give byte-identical packed output.
//
// Engine.RepeatSolve_BitIdentical only pins one serial ICGN solve. The
// parallel phases around it (the OpenMP Hessian pre-pass, the Path A mesh
// solve, the Path B RGDIC flood fill) were never repeated, and the flood fill
// was not deterministic: which neighbour reached a cell first, and so which
// initial guess the cell was solved from, depended on thread timing (app
// TECH_DEBT TD-65). These tests repeat the solve with the app's default grid
// (subset 21, step 5, strain window 15) and compare every output byte.
//
// Synthetic speckle rather than the DICe TIFF fixtures: TIFF decoding crashes
// in the MinGW host build, and this guard must run on every host.
#include "framework/test_framework.h"
#include "framework/synthetic.h"

#include <semper/pipeline.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <opencv2/core.hpp>
#include <vector>

#if defined(DIC_HAVE_OPENCV)

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::run_full_field;

namespace {

constexpr int kRepeats = 4;

struct Solve {
    int n = 0;                  // packed points (return code)
    std::vector<float> out;     // n * 8 floats
    float metrics[19] = {};
};

cv::Mat gray8_from_image(const Semper::Image &img) {
    cv::Mat m(img.height, img.width, CV_8UC1);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float v = img.intensities[(size_t)y * img.width + x];
            v = std::min(255.f, std::max(0.f, v));
            m.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
        }
    }
    return m;
}

FullFieldParams app_default_params(int w, int h) {
    FullFieldParams p;
    p.rect_x = 0;
    p.rect_y = 0;
    p.rect_w = w;
    p.rect_h = h;
    p.step = 5;
    p.subset_size = 21;
    p.strain_window = 15;
    p.use_6x6_interpolator = false;
    return p;
}

// One app-style batch: a fresh reference cache (initializeReference), then
// every deformed frame solved in order against it (computeFullFieldDirect).
std::vector<Solve> run_batch(const cv::Mat &ref, const std::vector<cv::Mat> &defs,
                             const cv::Mat &mask, const FullFieldParams &p) {
    ReferenceCache cache;
    cache.set_from_gray(ref, mask);
    const int cap = 8 * ((p.rect_w / p.step) * (p.rect_h / p.step) + 8);
    std::vector<Solve> out;
    for (const auto &def : defs) {
        Solve s;
        s.out.assign(cap, 0.f);
        s.n = run_full_field(cache, def, mask, p, s.out.data(), cap, s.metrics, 19, nullptr);
        s.out.resize(s.n > 0 ? (size_t)s.n * 8 : 0);
        out.push_back(std::move(s));
    }
    return out;
}

// Byte comparison plus a readable summary of any difference, so a failure
// says how far apart the runs are, not only that they differ.
bool same_bytes(const Solve &a, const Solve &b, const char *label, int frame, int run) {
    const bool same = a.n == b.n &&
                      std::memcmp(a.out.data(), b.out.data(), a.out.size() * sizeof(float)) == 0;
    if (same) return true;
    int diff_pts = 0;
    float max_du = 0.f;
    const int n = std::min(a.n, b.n);
    for (int i = 0; i < n; ++i) {
        const float *pa = &a.out[(size_t)i * 8];
        const float *pb = &b.out[(size_t)i * 8];
        if (std::memcmp(pa, pb, 8 * sizeof(float)) == 0) continue;
        ++diff_pts;
        if (pa[0] == pb[0] && pa[1] == pb[1]) max_du = std::max(max_du, std::fabs(pa[2] - pb[2]));
    }
    std::printf("  %s frame %d run %d: points %d vs %d; %d of the first %d differ; "
                "max|du| where aligned %.6f px\n",
                label, frame, run, a.n, b.n, diff_pts, n, max_du);
    return false;
}

void check_repeatable(const char *label, const cv::Mat &ref, const std::vector<cv::Mat> &defs,
                      const cv::Mat &mask, const FullFieldParams &p) {
    const auto first = run_batch(ref, defs, mask, p);
    for (size_t f = 0; f < first.size(); ++f) {
        REQUIRE(first[f].n > 0);
        std::printf("  %s frame %zu: %d points (Path A %.0f, Path B %.0f, mesh %.0f)\n",
                    label, f, first[f].n, first[f].metrics[3], first[f].metrics[4],
                    first[f].metrics[16]);
    }
    for (int r = 1; r < kRepeats; ++r) {
        const auto again = run_batch(ref, defs, mask, p);
        REQUIRE(again.size() == first.size());
        for (size_t f = 0; f < first.size(); ++f) {
            CHECK(same_bytes(first[f], again[f], label, (int)f, r));
            // Path counts are results, not telemetry: a point that moves
            // between paths was solved from a different guess.
            CHECK(first[f].metrics[3] == again[f].metrics[3]);
            CHECK(first[f].metrics[4] == again[f].metrics[4]);
        }
    }
}

dictest::AffineDeformation affine(int w, int h, float u, float v, float ux, float uy,
                                  float vx, float vy) {
    dictest::AffineDeformation d;
    d.u = u; d.v = v;
    d.ux = ux; d.uy = uy; d.vx = vx; d.vy = vy;
    d.cx = w / 2.0f;
    d.cy = h / 2.0f;
    return d;
}

} // namespace

// Two load steps solved as one app batch (one reference cache, frames in
// order), the shape of the TD-65 fixture. The ROI is the whole image, so the
// cells between the AKAZE mesh's hull and the border fall to Path B.
TEST_CASE(FullFieldDeterminism, TwoFrameBatch_BitIdentical) {
    constexpr int W = 360, H = 360;
    dictest::SpeckleField field(/*seed=*/24, W, H, /*blob_count=*/1300);
    const cv::Mat ref = gray8_from_image(dictest::make_reference_image(field, W, H));
    std::vector<cv::Mat> frames;
    for (int k = 1; k <= 2; ++k) {
        frames.push_back(gray8_from_image(dictest::make_deformed_image(
                field, W, H, affine(W, H, 1.2f * k, -0.6f * k, 0.006f * k, 0.f, 0.003f * k, -0.002f * k))));
    }
    check_repeatable("batch", ref, frames, cv::Mat(), app_default_params(W, H));
}

// Affine field under a ROI mask with a hole: Path B has to flood round the
// masked region from the mesh boundary.
TEST_CASE(FullFieldDeterminism, MaskedHole_BitIdentical) {
    constexpr int W = 320, H = 320;
    dictest::SpeckleField field(/*seed=*/29, W, H, /*blob_count=*/900);
    const cv::Mat ref = gray8_from_image(dictest::make_reference_image(field, W, H));
    const cv::Mat deformed = gray8_from_image(dictest::make_deformed_image(
            field, W, H, affine(W, H, 2.5f, -1.5f, 0.004f, 0.002f, 0.f, -0.003f)));

    cv::Mat mask(H, W, CV_8UC1, cv::Scalar(255));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float dx = x - 160.f, dy = y - 170.f;
            if (dx * dx + dy * dy < 45.f * 45.f) mask.at<uchar>(y, x) = 0;
        }
    }
    check_repeatable("masked", ref, {deformed}, mask, app_default_params(W, H));
}

#else

TEST_CASE(FullFieldDeterminism, OpenCvRequired_SkippedWithoutOpenCV) {
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
