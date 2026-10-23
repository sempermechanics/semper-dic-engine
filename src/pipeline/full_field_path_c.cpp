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
#include <utility>
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

    // Candidates in the order they are tried. The first is the node the old
    // single-shot search picked, so a field that seeded before seeds the same:
    // the mesh vertex nearest the grid centre if its node is free, else the
    // free node nearest the centre (first in row order on a tie).
    std::vector<int> candidates;   // flat grid indices

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
            const int gx = std::max(0, std::min(gridW - 1, (int)std::round((seed_ref_pts[best_seed_idx].x - params.rect_x) / params.step)));
            const int gy = std::max(0, std::min(gridH - 1, (int)std::round((seed_ref_pts[best_seed_idx].y - params.rect_y) / params.step)));
            if (!resultGrid[gy][gx].solved) candidates.push_back(gy * gridW + gx);
        }
    }

    // 2. THE ZERO-SEED / HOLE BYPASS (Mask-Aware Center Search): the free
    // nodes, nearest the grid centre first. Skipped nodes (masked, off the
    // image, no contrast) are marked solved and never tried.
    {
        const float center_x = gridW / 2.0f;
        const float center_y = gridH / 2.0f;
        std::vector<std::pair<float, int>> free_nodes;
        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                if (resultGrid[y][x].solved) continue;
                const float dx = x - center_x;
                const float dy = y - center_y;
                free_nodes.emplace_back(dx * dx + dy * dy, y * gridW + x);
            }
        }
        std::stable_sort(free_nodes.begin(), free_nodes.end(),
                         [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                             return a.first < b.first;
                         });
        for (const auto& n : free_nodes) {
            if ((int)candidates.size() >= tuning::kPathCMaxCandidates) break;
            if (!candidates.empty() && candidates.front() == n.second) continue;
            candidates.push_back(n.second);
        }
    }

    // 3. Fatal Error Check: nothing free -- the entire ROI is masked or textureless
    if (candidates.empty()) {
        LOGE("PATH C FAILED: Entire ROI is masked or lacks valid speckle contrast.");
        return -2;
    }

    // The same engine configuration as every other solve (it used to leave
    // LM damping off). Seed the search from the global model's shift when
    // there is one: INIT_AUTO_SEARCH honours a non-zero guess and otherwise
    // falls back to its own +/-15 px scan, so passing the global shift both
    // starts Path C closer and extends its reach past that scan window.
    // globalU/globalV are overwritten below only on success, so a failure
    // leaves the global model's values intact for Path B.
    //
    // One node used to decide: if it sat on a scratch, a glare spot or a
    // patch the speckle missed, the whole solve returned kSeedFailed though
    // the rest of the ROI was fine. The next nearest are tried in turn.
    Semper::SubsetData seed_subset;
    OptimizationEngine seed_engine = make_engine(params.use_6x6_interpolator);
    const float guess[6] = {globalU, globalV, 0.0f, 0.0f, 0.0f, 0.0f};
    for (const int flat_idx : candidates) {
        if (ctx.cancel.requested()) return kCancelled;
        const int gx = flat_idx % gridW;
        const int gy = flat_idx / gridW;
        LOGD("PATH C: Executing NCC Brute-Force Anchor Search at grid [%d, %d]", gx, gy);
        const int realX = params.rect_x + gx * params.step;
        const int realY = params.rect_y + gy * params.step;
        const NodeSolve node = solve_node(seed_engine, seed_subset, *ctx.cache.ref_img, ctx.def_img,
                                          realX, realY, params.subset_size,
                                          ctx.hessian_pool[flat_idx], guess, INIT_AUTO_SEARCH);
        if (!node.initialized) continue;
        const AnalysisResult& res = node.res;
        if (res.status == 0 && res.correlation_score <= tuning::kCorrAccept) {
            path_c_seed_x = gx;
            path_c_seed_y = gy;
            globalU = res.u; globalV = res.v;
            LOGD("PATH C SUCCESS: Seed locked at u=%.2f, v=%.2f", globalU, globalV);
            return 0;
        }
    }

    // With an AKAZE lock the solve goes on to Path B, which falls back to this
    // node as its seed: the first candidate, as when only one was tried.
    path_c_seed_x = candidates.front() % gridW;
    path_c_seed_y = candidates.front() / gridW;
    LOGE("PATH C FAILED: Central anchor search diverged at all %d candidate nodes.",
         (int)candidates.size());
    return kSeedFailed;
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
