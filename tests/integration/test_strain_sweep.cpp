// =====================================================================
// SUITE: StrainSweep — characterization of the large-strain limit
//
// Two strain fields on the exact analytic speckle of framework/synthetic.h:
//
//   Homogeneous   u = ε·x (uniaxial, all other gradients zero).
//   Band          localized uniaxial band, Gaussian in x:
//                   ε(x) = ε₀ · exp(−(x−c)² / 2w²)
//                   u(x) = ε₀ · w·√(π/2) · erf((x−c) / (√2·w))
//                 The first-order shape function cannot represent ε'(x), so
//                 this isolates the shape-function (second-order) residual
//                 ≈ ½·|ε'|·(N/2)² at the subset edge.
//
// Starts compared at subset level (one OptimizationEngine call each):
//   ZG      (u,v) exact, gradients 0, pure ICGN  → Path B seed / Path C
//   ZG+SX   same, Simplex rescue allowed         → production A/B flag
//   TG      (u,v) and local gradient exact, ICGN → ideal Path A start
//
// FullField runs run_full_field end to end (OpenCV builds only).
//
// This is a characterization, not a gate: the tables are printed for
// inspection and only the validated 1% operating point is asserted.
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"

#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using Semper::AnalysisResult;
using Semper::Image;
using Semper::INIT_NO_SEARCH;
using Semper::INIT_NO_SIMPLEX;
using Semper::OptimizationEngine;
using Semper::SubsetData;
using Semper::SubsetPrecomputer;

namespace {

    // A point counts as recovered only if the pipeline's own gate accepts it
    // AND it is correct. Tolerances separate convergence to a wrong basin from
    // ordinary gradient noise (~5e-3 at N = 21 on this pattern).
    constexpr float TOL_U = 0.1f;     // px
    constexpr float TOL_GRAD = 1e-2f; // strain

    constexpr float PI = 3.14159265358979f;

    // Localized uniaxial band about x = c.
    struct Band {
        float eps0, w, c;
        float u(float x) const {
            return eps0 * w * std::sqrt(PI / 2.0f) * std::erf((x - c) / (std::sqrt(2.0f) * w));
        }
        float eps(float x) const {
            const float d = (x - c) / w;
            return eps0 * std::exp(-0.5f * d * d);
        }
        // Reference X with X + u(X) = x (monotone for eps0 > -1): Newton.
        float inverse(float x) const {
            float X = x - u(x);
            for (int k = 0; k < 20; ++k) {
                const float f = X + u(X) - x;
                X -= f / (1.0f + eps(X));
                if (std::fabs(f) < 1e-5f) break;
            }
            return X;
        }
    };

    Image make_band_image(const dictest::SpeckleField &field, int w, int h, const Band &b) {
        std::vector<uint8_t> dummy((size_t)w * h, 0);
        Image img(w, h, dummy.data());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                img.intensities[(size_t)y * w + x] = field.sample(b.inverse((float)x), (float)y);
        img.prepare_data(false);
        return img;
    }

    bool recovered(const AnalysisResult &r, float u_true, float ux_true) {
        return r.status == 0 && r.correlation_score <= Semper::tuning::kCorrAccept &&
               std::fabs(r.u - u_true) <= TOL_U && std::fabs(r.v) <= TOL_U &&
               std::fabs(r.ux - ux_true) <= TOL_GRAD && std::fabs(r.uy) <= TOL_GRAD &&
               std::fabs(r.vx) <= TOL_GRAD && std::fabs(r.vy) <= TOL_GRAD;
    }

    struct SpeckleSpec {
        const char *name;
        int blobs;
        float s0, s1;
    };
    // Blob density scaled so both patterns cover the plane comparably.
    constexpr int SW = 400, SH = 400;
    const SpeckleSpec SPECKLES[] = {
            {"coarse (sigma 1.5-3.5 px)", 2500, 1.5f, 3.5f},
            {"fine (sigma 0.8-1.5 px)", 9000, 0.8f, 1.5f},
    };
    const int DIMS[] = {21, 31, 41};

    // Solve one subset from the three starts; count recoveries.
    void solve_three(const SubsetData &s, const Image &def, float u_true, float ux_true, int ok[3]) {
        OptimizationEngine e;
        e.lm_enabled = true;
        e.lm_alpha = Semper::tuning::kLmAlpha;
        ok[0] += recovered(e.calculate_deformation(s, def, u_true, 0, 0, 0, 0, 0, INIT_NO_SIMPLEX), u_true, ux_true);
        ok[1] += recovered(e.calculate_deformation(s, def, u_true, 0, 0, 0, 0, 0, INIT_NO_SEARCH), u_true, ux_true);
        ok[2] += recovered(e.calculate_deformation(s, def, u_true, 0, ux_true, 0, 0, 0, INIT_NO_SIMPLEX), u_true, ux_true);
    }

    void print_header(const char *col0) {
        std::printf("  %7s | %-18s | %-18s | %-18s\n", col0, "ZG", "ZG+SX", "TG");
        std::printf("  %7s |", "");
        for (int k = 0; k < 3; ++k) std::printf(" N=21  N=31  N=41  |");
        std::printf("\n");
    }

} // namespace

TEST_CASE(StrainSweep, SubsetHomogeneous) {
    const float EPS[] = {0.01f, 0.05f, 0.10f, 0.20f, 0.30f, 0.50f, 0.75f, 1.00f};
    const int OFFS[] = {-40, -20, 0, 20, 40}; // 5×5 subset centres about c
    const float CX = SW / 2.0f, CY = SH / 2.0f;

    for (const auto &sp : SPECKLES) {
        dictest::SpeckleField field(/*seed=*/2024, SW, SH, sp.blobs, sp.s0, sp.s1);
        const Image ref = dictest::make_reference_image(field, SW, SH);

        std::printf("\n  Homogeneous u_x = eps, %s: %% of 25 subsets recovered\n", sp.name);
        print_header("eps");
        for (float eps : EPS) {
            dictest::AffineDeformation truth;
            truth.ux = eps;
            truth.cx = CX;
            truth.cy = CY;
            const Image def = dictest::make_deformed_image(field, SW, SH, truth);

            int pct[3][3] = {};
            for (int di = 0; di < 3; ++di) {
                int ok[3] = {0, 0, 0}, total = 0;
                for (int oy : OFFS)
                    for (int ox : OFFS) {
                        SubsetData s;
                        SubsetPrecomputer::precompute_subset(s, ref, (int)CX + ox, (int)CY + oy, DIMS[di]);
                        if (!s.is_initialized) continue;
                        total++;
                        solve_three(s, def, eps * (float)ox, eps, ok);
                    }
                REQUIRE(total > 0);
                for (int k = 0; k < 3; ++k) pct[k][di] = (100 * ok[k]) / total;
            }
            std::printf("  %6.0f%% |", eps * 100.0f);
            for (int k = 0; k < 3; ++k) std::printf(" %4d%% %4d%% %4d%% |", pct[k][0], pct[k][1], pct[k][2]);
            std::printf("\n");

            if (eps == 0.01f)
                for (int k = 0; k < 3; ++k)
                    for (int di = 0; di < 3; ++di) CHECK(pct[k][di] == 100);
        }
    }
}

TEST_CASE(StrainSweep, SubsetBand) {
    const float EPS0[] = {0.01f, 0.05f, 0.10f, 0.20f, 0.30f, 0.50f};
    const float WIDTHS[] = {40.0f, 20.0f, 10.0f};
    const float CX = SW / 2.0f, CY = SH / 2.0f;

    for (const auto &sp : SPECKLES) {
        dictest::SpeckleField field(/*seed=*/2024, SW, SH, sp.blobs, sp.s0, sp.s1);
        const Image ref = dictest::make_reference_image(field, SW, SH);

        for (float w : WIDTHS) {
            // Subset centres across the band: x ∈ c + {0, ±w/2, ±w, ±1.5w, ±2w},
            // 3 rows in y → 27 subsets. |eps'| peaks at ±w.
            std::vector<int> xs;
            for (float f : {-2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f})
                xs.push_back((int)std::lround(CX + f * w));
            const int ys[] = {(int)CY - 30, (int)CY, (int)CY + 30};

            std::printf("\n  Band w = %.0f px, %s, 27 subsets across the band.\n"
                        "  acc = accepted by the engine (status 0, ZNSSD<=%.2f) AND |du|<=%.1f px; rej = converged but ZNSSD\n"
                        "  above gate; div = ICGN not converged; remainder = accepted by the engine but biased beyond |du|.\n"
                        "  |dux| = median gradient error of acc TG points vs point eps(x).\n"
                        "  r2 = 1/2|eps'|max(N/2)^2 at N=31, px.\n",
                        w, sp.name, Semper::tuning::kCorrAccept, TOL_U);
            std::printf("  %6s | %-17s | %-17s | %-17s | %-17s | %-8s | %s\n", "eps0", "TG acc", "TG rej",
                        "TG div", "ZG acc", "|dux|N31", "r2");
            for (float eps0 : EPS0) {
                const Band band{eps0, w, CX};
                const Image def = make_band_image(field, SW, SH, band);

                int acc[3] = {}, rej[3] = {}, div[3] = {}, zacc[3] = {}, tot[3] = {};
                std::vector<float> dux31;
                for (int di = 0; di < 3; ++di) {
                    for (int y : ys)
                        for (int x : xs) {
                            SubsetData s;
                            SubsetPrecomputer::precompute_subset(s, ref, x, y, DIMS[di]);
                            if (!s.is_initialized) continue;
                            tot[di]++;
                            const float ut = band.u((float)x), et = band.eps((float)x);
                            OptimizationEngine e;
                            e.lm_enabled = true;
                            e.lm_alpha = Semper::tuning::kLmAlpha;
                            auto on_target = [&](const AnalysisResult &r) {
                                return std::fabs(r.u - ut) <= TOL_U && std::fabs(r.v) <= TOL_U;
                            };
                            const auto rt = e.calculate_deformation(s, def, ut, 0, et, 0, 0, 0, INIT_NO_SIMPLEX);
                            if (rt.status != 0) div[di]++;
                            else if (rt.correlation_score > Semper::tuning::kCorrAccept) rej[di]++;
                            else if (on_target(rt)) {
                                acc[di]++;
                                if (DIMS[di] == 31) dux31.push_back(std::fabs(rt.ux - et));
                            }
                            const auto rz = e.calculate_deformation(s, def, ut, 0, 0, 0, 0, 0, INIT_NO_SEARCH);
                            if (rz.status == 0 && rz.correlation_score <= Semper::tuning::kCorrAccept && on_target(rz)) zacc[di]++;
                        }
                    REQUIRE(tot[di] > 0);
                }
                float med = -1.f;
                if (!dux31.empty()) {
                    std::nth_element(dux31.begin(), dux31.begin() + dux31.size() / 2, dux31.end());
                    med = dux31[dux31.size() / 2];
                }
                auto pc = [&](const int *v, int di) { return (100 * v[di]) / tot[di]; };
                const float deps_max = eps0 / w * std::exp(-0.5f); // |eps'| at x = c ± w
                std::printf("  %5.0f%% |", eps0 * 100.0f);
                for (const int *v : {acc, rej, div, zacc})
                    std::printf(" %4d%% %4d%% %4d%% |", pc(v, 0), pc(v, 1), pc(v, 2));
                std::printf(" %8.4f | %.2f\n", med, 0.5f * deps_max * 15.5f * 15.5f);

                if (eps0 == 0.01f)
                    for (int di = 0; di < 3; ++di) CHECK(acc[di] == tot[di]);
            }
        }
    }
}

#if defined(DIC_HAVE_OPENCV)

#include <semper/pipeline.hpp>
#include <opencv2/core.hpp>

namespace {

    cv::Mat to_gray(const Image &img) {
        cv::Mat m(img.height, img.width, CV_8UC1);
        for (int y = 0; y < img.height; ++y)
            for (int x = 0; x < img.width; ++x) {
                float v = std::min(255.f, std::max(0.f, img.intensities[(size_t)y * img.width + x]));
                m.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
            }
        return m;
    }

    struct FieldRun {
        int n = 0;
        float m[19] = {};
        std::vector<float> out;
    };

    FieldRun run(Semper::pipeline::ReferenceCache &cache, const cv::Mat &def,
                 const Semper::pipeline::FullFieldParams &p) {
        FieldRun r;
        const int grid_pts = (p.rect_w / p.step) * (p.rect_h / p.step);
        r.out.assign((size_t)8 * (grid_pts + 16), 0.f);
        r.n = Semper::pipeline::run_full_field(cache, def, cv::Mat(), p, r.out.data(),
                                               (int)r.out.size(), r.m, 19);
        return r;
    }

    const char *SEED_NAME[] = {"PathC", "sparse", "full"};

} // namespace

TEST_CASE(StrainSweep, FullField) {
    using Semper::pipeline::FullFieldParams;
    using Semper::pipeline::ReferenceCache;

    constexpr int W = 512, H = 512;
    constexpr int STEP = 10, SUBSET = 31, STRAIN_WIN = 41;

    dictest::SpeckleField field(/*seed=*/77, W, H, /*blob_count=*/4000);
    const Image ref_img = dictest::make_reference_image(field, W, H);
    ReferenceCache cache;
    cache.set_from_gray(to_gray(ref_img), cv::Mat());

    FullFieldParams p;
    p.step = STEP;
    p.subset_size = SUBSET;
    p.strain_window = STRAIN_WIN;
    p.use_6x6_interpolator = false;

    // --- Homogeneous: ROI inset so the stretched field stays inside the image.
    {
        constexpr int ROI = 160;
        p.rect_x = p.rect_y = ROI;
        p.rect_w = p.rect_h = W - 2 * ROI;
        std::printf("\n  Full field, homogeneous u_x = eps: %dx%d, ROI %d..%d, step %d, subset %d, VSG %d\n"
                    "  (out %% < disp %% at every eps is the VSG 90%% fill rule on the ROI rim)\n",
                    W, H, ROI, W - ROI, STEP, SUBSET, STRAIN_WIN);
        std::printf("  %6s | %-6s | %7s %7s | %6s %6s | %9s | %10s\n", "eps", "seed", "disp%",
                    "out%", "meshA", "floodB", "rescue", "med|dExx|");
        for (float eps : {0.01f, 0.05f, 0.10f, 0.20f, 0.30f, 0.50f, 0.75f, 1.00f}) {
            dictest::AffineDeformation truth;
            truth.ux = eps;
            truth.cx = W / 2.0f;
            truth.cy = H / 2.0f;
            FieldRun r = run(cache, to_gray(dictest::make_deformed_image(field, W, H, truth)), p);
            REQUIRE(r.n >= 0);

            const float exx_true = eps + 0.5f * eps * eps; // Green-Lagrange
            std::vector<float> err;
            for (int i = 0; i < r.n; ++i) err.push_back(std::fabs(r.out[(size_t)8 * i + 4] - exx_true));
            float med = -1.f;
            if (!err.empty()) {
                std::nth_element(err.begin(), err.begin() + err.size() / 2, err.end());
                med = err[err.size() / 2];
            }
            const float att = std::max(1.f, r.m[0]);
            const float disp_pct = 100.f * (r.m[3] + r.m[4]) / att, out_pct = 100.f * (float)r.n / att;
            std::printf("  %5.0f%% | %-6s | %6.1f%% %6.1f%% | %6.0f %6.0f | %4.0f/%-4.0f | %10.2e\n",
                        eps * 100.f, SEED_NAME[(int)r.m[16]], disp_pct, out_pct, r.m[3], r.m[4],
                        r.m[6], r.m[5], med);

            if (eps == 0.01f) {
                CHECK(disp_pct >= 99.f);
                CHECK(med >= 0.f && med <= 2e-3f);
            }
        }
    }

    // --- Band: ROI spans the band; coverage and exx error reported inside the
    //     central core |x − c| ≤ w, where eps is largest and varies fastest.
    {
        // Fresh cache: cached AKAZE reference keypoints are stored in padded-ROI
        // coordinates and are invalidated only on a scale change, not on an ROI
        // change (full_field_akaze.cpp), so reusing the homogeneous-block cache
        // with a different ROI would offset every seed.
        ReferenceCache band_cache;
        band_cache.set_from_gray(to_gray(ref_img), cv::Mat());
        constexpr int ROI = 96;
        p.rect_x = p.rect_y = ROI;
        p.rect_w = p.rect_h = W - 2 * ROI;
        const float c = W / 2.0f;
        for (float w : {40.0f, 20.0f, 10.0f}) {
            std::printf("\n  Full field, band w = %.0f px: core |x-c| <= w; peak Exx error at the band centre\n", w);
            std::printf("  %6s | %-6s | %7s | %9s | %10s %10s\n", "eps0", "seed", "disp%",
                        "core out%", "Exx peak", "Exx meas");
            for (float eps0 : {0.01f, 0.05f, 0.10f, 0.20f, 0.30f, 0.50f}) {
                const Band band{eps0, w, c};
                FieldRun r = run(band_cache, to_gray(make_band_image(field, W, H, band)), p);
                REQUIRE(r.n >= 0);

                // Core grid columns: packed x within [c − w, c + w].
                int core_cols = 0;
                for (int gx = 0; gx < p.rect_w / STEP; ++gx) {
                    const float x = (float)(p.rect_x + gx * STEP);
                    if (std::fabs(x - c) <= w) core_cols++;
                }
                const int core_total = core_cols * (p.rect_h / STEP);
                int core_out = 0;
                float peak_meas = 0.f;
                int peak_n = 0;
                for (int i = 0; i < r.n; ++i) {
                    const float x = r.out[(size_t)8 * i];
                    if (std::fabs(x - c) <= w) core_out++;
                    if (std::fabs(x - c) <= STEP / 2.0f) {
                        peak_meas += r.out[(size_t)8 * i + 4];
                        peak_n++;
                    }
                }
                const float att = std::max(1.f, r.m[0]);
                const float exx_peak = eps0 + 0.5f * eps0 * eps0;
                // disp% is field-wide (per-point displacement status is not packed);
                // core out% counts packed points, i.e. displacement AND strain valid.
                // Exx meas = mean packed exx in the centre column, -1 if none.
                std::printf("  %5.0f%% | %-6s | %6.1f%% | %8.1f%% | %10.4f %10.4f\n",
                            eps0 * 100.f, SEED_NAME[(int)r.m[16]], 100.f * (r.m[3] + r.m[4]) / att,
                            core_total ? 100.f * core_out / core_total : 0.f, exx_peak,
                            peak_n ? peak_meas / peak_n : -1.f);
            }
        }
    }
}

#endif // DIC_HAVE_OPENCV
