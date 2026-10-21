// Path A — AKAZE matches refined into IC-GN mesh vertices.
//
// A keypoint pair gives the displacement at the keypoint to within its
// detection error: about a pixel at full scale, several at the 0.25x scale the
// pyramid usually settles on. build_mesh_guess_field turns each triangle's
// three vertices into an affine map with cv::getAffineTransform, so a vertex
// error e on an edge of length L becomes a gradient error of order e / L. On
// the short edges of a dense keypoint mesh that is enough to send IC-GN into
// timeouts and Simplex rescues, worst under rotation, where the guess gradients
// are large.
//
// Here each match is moved to its nearest grid node and solved there with the
// engine's own IC-GN, seeded from the match and the RANSAC homography's
// Jacobian. A converged node is accurate to the solver's ~0.01 px, so it makes
// a much better vertex; and since it passed the same gate Path A applies, it is
// also a final result for that node, so the refinement costs almost nothing
// beyond the field itself. The node reuses the pooled Hessian. Matches that do
// not converge, or that fail the normalized median test against the other
// vertices, are dropped; the rest of the pipeline is unchanged.

#include "full_field_internal.hpp"

#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

namespace {

struct Candidate {
    int gx = 0, gy = 0;
    float dist2 = 0.0f;          // keypoint to node, squared (tie-break)
    int match = 0;               // index into the match lists
    float gu = 0, gv = 0, gux = 0, guy = 0, gvx = 0, gvy = 0;
    AnalysisResult res;
    int tid = 0;
    bool ok = false;
};

float median_of(std::vector<float> &v) {
    const size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    float hi = v[m];
    if (v.size() % 2 == 1) return hi;
    float lo = *std::max_element(v.begin(), v.begin() + m);
    return 0.5f * (lo + hi);
}

} // namespace

bool homography_at(const cv::Mat &H, float px, float py, double &X, double &Y, double J[4]) {
    const double *h = H.ptr<double>(0);
    const double w = h[6] * px + h[7] * py + h[8];
    if (!std::isfinite(w) || std::abs(w) < 1e-12) return false;
    X = (h[0] * px + h[1] * py + h[2]) / w;
    Y = (h[3] * px + h[4] * py + h[5]) / w;
    J[0] = (h[0] - X * h[6]) / w;
    J[1] = (h[1] - X * h[7]) / w;
    J[2] = (h[3] - Y * h[6]) / w;
    J[3] = (h[4] - Y * h[7]) / w;
    return std::isfinite(X) && std::isfinite(Y) && std::isfinite(J[0]) && std::isfinite(J[1]) &&
           std::isfinite(J[2]) && std::isfinite(J[3]);
}

VertexRefineResult refine_mesh_vertices(
        const SolveContext &ctx,
        const MeshSeedResult &seeds,
        ResultGrid &resultGrid,
        std::vector<ThreadStats> &stats) {

    VertexRefineResult out;
    auto t0 = std::chrono::high_resolution_clock::now();
    const FullFieldParams &params = ctx.params;
    const int gridW = ctx.gridW, gridH = ctx.gridH;
    const bool have_H = seeds.homography.rows == 3 && seeds.homography.cols == 3;

    // One candidate per grid node: the match nearest to it.
    std::vector<int> owner((size_t)gridW * gridH, -1);
    std::vector<Candidate> cand;
    for (size_t i = 0; i < seeds.ref_pts.size(); ++i) {
        const cv::Point2f r = seeds.ref_pts[i];
        const int gx = (int)std::lround((r.x - params.rect_x) / (float)params.step);
        const int gy = (int)std::lround((r.y - params.rect_y) / (float)params.step);
        if (gx < 0 || gx >= gridW || gy < 0 || gy >= gridH) continue;
        if (resultGrid[gy][gx].solved) continue;  // boundary- or mask-rejected
        const float nx = (float)(params.rect_x + gx * params.step);
        const float ny = (float)(params.rect_y + gy * params.step);
        Candidate c;
        c.gx = gx; c.gy = gy; c.match = (int)i;
        c.dist2 = (r.x - nx) * (r.x - nx) + (r.y - ny) * (r.y - ny);
        const size_t node = (size_t)gy * gridW + gx;
        if (owner[node] < 0) {
            owner[node] = (int)cand.size();
            cand.push_back(c);
        } else if (c.dist2 < cand[owner[node]].dist2) {
            cand[owner[node]] = c;
        }
    }

    // Guess at the node: the match's displacement carried from the keypoint to
    // the node by the homography's local linear map, with that map's gradients.
    for (auto &c : cand) {
        const cv::Point2f r = seeds.ref_pts[c.match];
        const cv::Point2f d = seeds.def_pts[c.match];
        const float nx = (float)(params.rect_x + c.gx * params.step);
        const float ny = (float)(params.rect_y + c.gy * params.step);
        double X, Y, J[4] = {1, 0, 0, 1};
        if (!have_H || !homography_at(seeds.homography, r.x, r.y, X, Y, J)) {
            J[0] = 1; J[1] = 0; J[2] = 0; J[3] = 1;
        }
        const double ex = nx - r.x, ey = ny - r.y;
        c.gu = (float)(d.x + J[0] * ex + J[1] * ey - nx);
        c.gv = (float)(d.y + J[2] * ex + J[3] * ey - ny);
        c.gux = (float)(J[0] - 1.0); c.guy = (float)J[1];
        c.gvx = (float)J[2];         c.gvy = (float)(J[3] - 1.0);
    }
    out.tried = (int)cand.size();

    // Solve every candidate node: pure IC-GN from the guess, no Simplex rescue
    // (a node that needs one is not a trustworthy vertex; Path A retries it).
#pragma omp parallel num_threads(ctx.safe_cores)
    {
        const int tid = omp_get_thread_num();
        OptimizationEngine engine;
        Semper::SubsetData subset;
        engine.lm_enabled = true;
        engine.lm_alpha = tuning::kLmAlpha;
        engine.use_6x6_interpolator = params.use_6x6_interpolator;
        double local_hessian = 0.0, local_wait = 0.0;
        int local_pts = 0;
        EngineStatFlusher flusher(engine, stats[tid], local_pts, local_hessian, local_wait);
#pragma omp for schedule(dynamic, 4)
        for (int k = 0; k < (int)cand.size(); ++k) {
            if (cancel_requested()) continue;
            Candidate &c = cand[k];
            const int idx = c.gy * gridW + c.gx;
            const int nx = params.rect_x + c.gx * params.step;
            const int ny = params.rect_y + c.gy * params.step;
            auto th = std::chrono::high_resolution_clock::now();
            SubsetPrecomputer::precompute_subset_fast(subset, *ctx.cache.ref_img, nx, ny, params.subset_size, ctx.hessian_pool[idx]);
            local_hessian += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - th).count();
            if (!subset.is_initialized) continue;
            c.res = engine.calculate_deformation(subset, ctx.def_img, c.gu, c.gv, c.gux, c.guy, c.gvx, c.gvy, INIT_NO_SIMPLEX);
            reject_if_ghosted(c.res, params.subset_size);
            stats[tid].icgn_iters += c.res.iters;
            c.tid = tid;
            c.ok = c.res.status == 0 && c.res.correlation_score <= tuning::kCorrAccept;
        }
    }
    if (cancel_requested()) return out;

    // Normalized median test (Westerweel & Scarano) of each converged vertex
    // against its nearest converged neighbours. RANSAC already dropped gross
    // mismatches; this catches an IC-GN lock on the wrong speckle.
    std::vector<int> conv;
    for (int k = 0; k < (int)cand.size(); ++k) if (cand[k].ok) conv.push_back(k);
    const int kNeighbours = 8;
    std::vector<char> keep(cand.size(), 0);
    for (int a : conv) {
        std::vector<std::pair<int, int>> nb;  // (squared grid distance, candidate)
        for (int b : conv) {
            if (b == a) continue;
            const int dx = cand[b].gx - cand[a].gx, dy = cand[b].gy - cand[a].gy;
            nb.push_back({dx * dx + dy * dy, b});
        }
        if ((int)nb.size() < tuning::kOutlierMinNeighbours) { keep[a] = 1; continue; }
        const size_t n = std::min<size_t>(kNeighbours, nb.size());
        std::partial_sort(nb.begin(), nb.begin() + n, nb.end());
        bool pass = true;
        for (int comp = 0; comp < 2 && pass; ++comp) {
            std::vector<float> vals, resid;
            for (size_t j = 0; j < n; ++j) vals.push_back(comp == 0 ? cand[nb[j].second].res.u : cand[nb[j].second].res.v);
            const float med = median_of(vals);
            for (size_t j = 0; j < n; ++j) {
                const float x = comp == 0 ? cand[nb[j].second].res.u : cand[nb[j].second].res.v;
                resid.push_back(std::abs(x - med));
            }
            const float rm = median_of(resid);
            const float self = comp == 0 ? cand[a].res.u : cand[a].res.v;
            if (std::abs(self - med) / (rm + tuning::kOutlierEpsPx) > tuning::kOutlierThreshold) pass = false;
        }
        keep[a] = pass ? 1 : 0;
    }

    // Publish the survivors as final results, in grid order so compute_order
    // and the vertex list do not depend on thread timing.
    std::vector<int> order;
    for (int a : conv) if (keep[a]) order.push_back(a);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return cand[a].gy != cand[b].gy ? cand[a].gy < cand[b].gy : cand[a].gx < cand[b].gx;
    });
    for (int a : order) {
        const Candidate &c = cand[a];
        const float nx = (float)(params.rect_x + c.gx * params.step);
        const float ny = (float)(params.rect_y + c.gy * params.step);
        const int ord = ctx.compute_order_counter.fetch_add(1, std::memory_order_relaxed);
        GridPoint &gp = resultGrid[c.gy][c.gx];
        gp = {nx, ny, c.res.u, c.res.v, c.res.ux, c.res.uy, c.res.vx, c.res.vy,
              c.res.correlation_score, true, c.tid, ord, kMeshVertexTag, false, c.res.iters};
        gp.guess_u = c.gu; gp.guess_v = c.gv; gp.guess_ux = c.gux;
        gp.guess_uy = c.guy; gp.guess_vx = c.gvx; gp.guess_vy = c.gvy;
        ctx.global_points_solved.fetch_add(1, std::memory_order_relaxed);
        stats[c.tid].points_solved++;
        out.ref_pts.push_back(cv::Point2f(nx, ny));
        out.def_pts.push_back(cv::Point2f(nx + c.res.u, ny + c.res.v));
    }
    out.converged = (int)conv.size();
    out.accepted = (int)order.size();
    out.time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
