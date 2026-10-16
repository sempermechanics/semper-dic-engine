// Path C — central-tendency fallback seeding, used when the anchor
// lattice could not produce even a sparse mesh. Lifted verbatim out of
// full_field_solver.cpp.

#include "full_field_internal.hpp"

#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

// =========================================================
// 🚀 PATH C: CENTRAL SEEDING (Industry Standard Fallback)
// =========================================================
int run_path_c(
        const SolveContext& ctx,
        const std::vector<cv::Point2f>& seed_ref_pts,
        ResultGrid& resultGrid,
        int& path_c_seed_x,
        int& path_c_seed_y,
        float& globalU,
        float& globalV) {

    const FullFieldParams& params = ctx.params;
    const int gridW = ctx.gridW;
    const int gridH = ctx.gridH;

    LOGD("ROUTING: Mesh Insufficient. Falling back to Central-Tendency Seeding.");

    int best_grid_x = -1;
    int best_grid_y = -1;

    // 1. Seed from whichever mesh vertex sits nearest the grid centre.
    if (!seed_ref_pts.empty()) {
        float grid_cx = params.rect_x + (gridW / 2.f) * params.step;
        float grid_cy = params.rect_y + (gridH / 2.f) * params.step;
        float min_dist = 1e9f;
        int best_seed_idx = -1;

        for (size_t i = 0; i < seed_ref_pts.size(); ++i) {
            float dx = seed_ref_pts[i].x - grid_cx;
            float dy = seed_ref_pts[i].y - grid_cy;
            float dist = dx*dx + dy*dy;
            if (dist < min_dist) {
                min_dist = dist;
                best_seed_idx = (int)i;
            }
        }

        if (best_seed_idx >= 0) {
            best_grid_x = std::max(0, std::min(gridW - 1, (int)std::round((seed_ref_pts[best_seed_idx].x - params.rect_x) / params.step)));
            best_grid_y = std::max(0, std::min(gridH - 1, (int)std::round((seed_ref_pts[best_seed_idx].y - params.rect_y) / params.step)));
        }
    }

    // 2. THE ZERO-SEED / HOLE BYPASS (Mask-Aware Center Search)
    // If seeding produced nothing, OR the chosen vertex is inside a masked hole/dead zone
    if (best_grid_x == -1 || resultGrid[best_grid_y][best_grid_x].solved) {
        LOGD("PATH C: Blind searching for the nearest valid unmasked seed...");
        float min_dist = 1e9f;
        float center_x = gridW / 2.0f;
        float center_y = gridH / 2.0f;

        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                // 🚀 INDUSTRY STANDARD: Must be strictly inside the User's ROI and have good speckles!
                if (!resultGrid[y][x].solved) {
                    float dx = x - center_x;
                    float dy = y - center_y;
                    float dist = dx * dx + dy * dy;
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_grid_x = x;
                        best_grid_y = y;
                    }
                }
            }
        }
    }

    // 3. Fatal Error Check: If it's still -1, the entire ROI is masked or textureless
    if (best_grid_x == -1) {
        LOGE("PATH C FAILED: Entire ROI is masked or lacks valid speckle contrast.");
        return -2;
    }

    LOGD("PATH C: Executing NCC Brute-Force Anchor Search at grid [%d, %d]", best_grid_x, best_grid_y);
    path_c_seed_x = best_grid_x;
    path_c_seed_y = best_grid_y;

    int realX = params.rect_x + path_c_seed_x * params.step;
    int realY = params.rect_y + path_c_seed_y * params.step;
    int flat_idx = path_c_seed_y * gridW + path_c_seed_x;

    Semper::SubsetData seed_subset;
    SubsetPrecomputer::precompute_subset_fast(seed_subset, *ctx.cache.ref_img, realX, realY, params.subset_size, ctx.hessian_pool[flat_idx]);

    if (seed_subset.is_initialized) {
        OptimizationEngine seed_engine;
        seed_engine.use_6x6_interpolator = params.use_6x6_interpolator;
        // Seed the search from the phase-correlation shift when there is one.
        // INIT_AUTO_SEARCH honours a non-zero guess and otherwise falls back to
        // its own +/-15 px scan, so passing the global shift both starts Path C
        // closer and extends its reach past that scan window. globalU/globalV
        // are overwritten below only on success, so a failure leaves the
        // phase-correlation values intact for Path B.
        auto res = seed_engine.calculate_deformation(seed_subset, ctx.def_img, globalU, globalV, 0.0f, 0.0f, 0.0f, 0.0f, INIT_AUTO_SEARCH);

        if (res.status == 0 && res.correlation_score <= tuning::kCorrAccept) {
            globalU = res.u; globalV = res.v;
            LOGD("PATH C SUCCESS: Seed locked at u=%.2f, v=%.2f", globalU, globalV);
        } else {
            LOGE("PATH C FAILED: Central anchor search diverged.");
            return -1;
        }
    } else { return -1; }

    return 0;
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
