// Sampling and mask warping for incremental DIC. See incremental_internal.hpp.

#include "incremental_internal.hpp"

#include <semper/strain.hpp>
#include <semper/tuning.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Semper {
namespace pipeline {
namespace internal {

NodeField node_field_from_capture(const FieldCapture &cap, int *rejected) {
    NodeField f;
    f.gridW = cap.gridW;
    f.gridH = cap.gridH;
    f.rect_x = cap.rect_x;
    f.rect_y = cap.rect_y;
    f.step = cap.step;
    const size_t n = f.size();
    f.u.assign(n, 0.0f); f.v.assign(n, 0.0f);
    f.ux.assign(n, 0.0f); f.uy.assign(n, 0.0f);
    f.vx.assign(n, 0.0f); f.vy.assign(n, 0.0f);
    f.corr.assign(n, tuning::kCorrInvalid);
    f.valid.assign(n, 0);

    DisplacementField disp;
    disp.width = f.gridW;
    disp.height = f.gridH;
    disp.step = f.step;
    disp.u.assign(n, 0.0f);
    disp.v.assign(n, 0.0f);
    disp.valid.assign(n, false);

    for (int y = 0; y < f.gridH; ++y) {
        for (int x = 0; x < f.gridW; ++x) {
            const GridPoint &g = cap.grid[y][x];
            const size_t i = (size_t)y * f.gridW + x;
            if (!(g.solved && g.corr >= 0.0f)) continue;
            f.u[i] = g.u; f.v[i] = g.v;
            f.ux[i] = g.ux; f.uy[i] = g.uy;
            f.vx[i] = g.vx; f.vy[i] = g.vy;
            f.corr[i] = g.corr;
            disp.u[i] = g.u; disp.v[i] = g.v;
            disp.valid[i] = true;
        }
    }

    const int dropped = StrainCalculator::reject_displacement_outliers(disp);
    if (rejected) *rejected = dropped;
    for (size_t i = 0; i < n; ++i) f.valid[i] = disp.valid[i] ? 1 : 0;
    return f;
}

IncrementSample sample_increment(const NodeField &f, float x, float y) {
    IncrementSample s;
    if (f.gridW <= 0 || f.gridH <= 0 || f.step <= 0) return s;

    const float gx = (x - (float)f.rect_x) / (float)f.step;
    const float gy = (y - (float)f.rect_y) / (float)f.step;
    const int i0 = (int)std::floor(gx);
    const int j0 = (int)std::floor(gy);
    const float fx = gx - (float)i0;
    const float fy = gy - (float)j0;

    auto idx = [&](int i, int j) { return (size_t)j * f.gridW + i; };
    auto in = [&](int i, int j) { return i >= 0 && i < f.gridW && j >= 0 && j < f.gridH; };

    // Exactly on a node: no interpolation, so the node's own value comes back
    // unchanged even when a neighbour is missing.
    if (fx == 0.0f && fy == 0.0f && in(i0, j0) && f.valid[idx(i0, j0)]) {
        const size_t k = idx(i0, j0);
        s.ok = true;
        s.u = f.u[k];
        s.v = f.v[k];
        s.corr = f.corr[k];
        return s;
    }

    if (in(i0, j0) && in(i0 + 1, j0 + 1)) {
        const size_t a = idx(i0, j0), b = idx(i0 + 1, j0);
        const size_t c = idx(i0, j0 + 1), d = idx(i0 + 1, j0 + 1);
        if (f.valid[a] && f.valid[b] && f.valid[c] && f.valid[d]) {
            const float wa = (1 - fx) * (1 - fy), wb = fx * (1 - fy);
            const float wc = (1 - fx) * fy, wd = fx * fy;
            s.ok = true;
            s.u = wa * f.u[a] + wb * f.u[b] + wc * f.u[c] + wd * f.u[d];
            s.v = wa * f.v[a] + wb * f.v[b] + wc * f.v[c] + wd * f.v[d];
            s.corr = std::max(std::max(f.corr[a], f.corr[b]), std::max(f.corr[c], f.corr[d]));
            return s;
        }
    }

    // First-order expansion from the nearest valid node in the 4x4 block
    // around the cell.
    const float max_d = tuning::kIncTaylorMaxCells * (float)f.step;
    float best_d2 = std::numeric_limits<float>::infinity();
    int best_i = -1, best_j = -1;
    for (int j = j0 - 1; j <= j0 + 2; ++j) {
        for (int i = i0 - 1; i <= i0 + 2; ++i) {
            if (!in(i, j) || !f.valid[idx(i, j)]) continue;
            const float dx = x - f.node_x(i), dy = y - f.node_y(j);
            const float d2 = dx * dx + dy * dy;
            if (d2 < best_d2) { best_d2 = d2; best_i = i; best_j = j; }
        }
    }
    if (best_i < 0 || best_d2 > max_d * max_d) return s;

    const size_t k = idx(best_i, best_j);
    const float dx = x - f.node_x(best_i), dy = y - f.node_y(best_j);
    s.ok = true;
    s.taylor = true;
    s.u = f.u[k] + f.ux[k] * dx + f.uy[k] * dy;
    s.v = f.v[k] + f.vx[k] * dx + f.vy[k] * dy;
    s.corr = f.corr[k];
    return s;
}

namespace {

// Fill invalid nodes from valid ones: each pass gives every still-empty node
// with a filled 4-neighbour the mean of those neighbours, reading only the
// previous pass's state, until nothing changes. Returns false if no node is
// valid to start from.
bool fill_field(int w, int h, std::vector<float> &U, std::vector<float> &V,
                const std::vector<unsigned char> &valid) {
    const size_t n = (size_t)w * h;
    std::vector<unsigned char> filled(valid.begin(), valid.end());
    bool any = false;
    for (size_t i = 0; i < n; ++i) {
        if (filled[i]) any = true;
        else { U[i] = 0.0f; V[i] = 0.0f; }
    }
    if (!any) return false;

    std::vector<unsigned char> next;
    std::vector<float> nu(U), nv(V);
    for (int pass = 0; pass < w + h + 2; ++pass) {
        next = filled;
        bool changed = false;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = (size_t)y * w + x;
                if (filled[i]) continue;
                float su = 0, sv = 0;
                int cnt = 0;
                const int nx[4] = {x - 1, x + 1, x, x};
                const int ny[4] = {y, y, y - 1, y + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || nx[k] >= w || ny[k] < 0 || ny[k] >= h) continue;
                    const size_t j = (size_t)ny[k] * w + nx[k];
                    if (!filled[j]) continue;
                    su += U[j]; sv += V[j]; ++cnt;
                }
                if (cnt == 0) continue;
                nu[i] = su / (float)cnt;
                nv[i] = sv / (float)cnt;
                next[i] = 1;
                changed = true;
            }
        }
        U = nu; V = nv;
        filled.swap(next);
        if (!changed) break;
    }
    return true;
}

// Continuous field over the filled grid, in image pixels. Bilinear inside.
// Outside, the plane through the nearest grid point with that point's
// gradient: exact for any affine field, so a turning body stays a rigid turn
// out to the warp window. Two extensions that fail under large rotation:
// holding the value constant (a pure translation, which gives pixels outside
// the material a second, false preimage that a row's warm start can follow
// across the material), and extending the edge cell's bilinear (its cross term
// grows quadratically off the corners and folds the map). The caller bounds
// how far out it evaluates (the warp window).
struct ContinuousField {
    const GridGeom &g;
    const std::vector<float> &U, &V;

    // Value and derivatives (per pixel) at image point (x, y).
    void eval(double x, double y, double &u, double &v,
              double &ux, double &uy, double &vx, double &vy) const {
        const double gx_raw = (x - g.rect_x) / g.step;
        const double gy_raw = (y - g.rect_y) / g.step;
        const double gx = std::max(0.0, std::min(gx_raw, (double)(g.gridW - 1)));
        const double gy = std::max(0.0, std::min(gy_raw, (double)(g.gridH - 1)));
        const double ox = (gx_raw - gx) * g.step, oy = (gy_raw - gy) * g.step;   // px outside

        int i0 = (int)std::floor(gx), j0 = (int)std::floor(gy);
        i0 = std::max(0, std::min(i0, std::max(g.gridW - 2, 0)));
        j0 = std::max(0, std::min(j0, std::max(g.gridH - 2, 0)));
        const int i1 = std::min(i0 + 1, g.gridW - 1);
        const int j1 = std::min(j0 + 1, g.gridH - 1);
        const double tx = (i1 == i0) ? 0.0 : gx - i0;
        const double ty = (j1 == j0) ? 0.0 : gy - j0;

        auto at = [&](const std::vector<float> &F, int i, int j) {
            return (double)F[(size_t)j * g.gridW + i];
        };
        auto bil = [&](const std::vector<float> &F, double &val, double &dx, double &dy) {
            const double a = at(F, i0, j0), b = at(F, i1, j0);
            const double c = at(F, i0, j1), d = at(F, i1, j1);
            val = a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty;
            dx = (i1 == i0) ? 0.0 : ((b - a) * (1 - ty) + (d - c) * ty) / g.step;
            dy = (j1 == j0) ? 0.0 : ((c - a) * (1 - tx) + (d - b) * tx) / g.step;
        };
        bil(U, u, ux, uy);
        bil(V, v, vx, vy);
        u += ux * ox + uy * oy;
        v += vx * ox + vy * oy;
    }
};

} // namespace

cv::Mat warp_material_mask(const cv::Mat &material_mask0,
                           const GridGeom &geom,
                           const std::vector<float> &U_in,
                           const std::vector<float> &V_in,
                           const std::vector<unsigned char> &valid,
                           int margin_px) {
    const int W = material_mask0.cols, H = material_mask0.rows;
    cv::Mat out = cv::Mat::zeros(H, W, CV_8U);
    if (geom.gridW <= 0 || geom.gridH <= 0 || geom.step <= 0) return out;

    std::vector<float> U(U_in), V(V_in);
    if (!fill_field(geom.gridW, geom.gridH, U, V, valid)) return out;

    const ContinuousField field{geom, U, V};

    // Forward warp: every frame-0 material sample X lands at X + U(X). There
    // is no inverse to solve, so no branch to pick: where a few drifted nodes
    // fold the field, samples overlap or spread locally, and the rest of the
    // mask is untouched. (A per-pixel Newton inverse can follow a fold, or a
    // false preimage in an extrapolated zone, across the material and blank a
    // whole band of rows.)
    //
    // Sample spacing: fine enough that neighbouring samples land within 0.7 px
    // of each other at the largest local stretch, so a 3x3 close fills every
    // pinhole. Rounded to a whole number of samples per pixel.
    double stretch = 1.0;
    for (int gy = 0; gy < geom.gridH; ++gy) {
        for (int gx = 0; gx < geom.gridW; ++gx) {
            const size_t i = (size_t)gy * geom.gridW + gx;
            const size_t ix = gx + 1 < geom.gridW ? i + 1 : i;
            const size_t iy = gy + 1 < geom.gridH ? i + geom.gridW : i;
            const double ux = (U[ix] - U[i]) / geom.step, uy = (U[iy] - U[i]) / geom.step;
            const double vx = (V[ix] - V[i]) / geom.step, vy = (V[iy] - V[i]) / geom.step;
            stretch = std::max(stretch, std::max(std::abs(1 + ux) + std::abs(uy),
                                                 std::abs(vx) + std::abs(1 + vy)));
        }
    }
    const int per_px = std::min(tuning::kIncMaskMaxSamplesPerPx,
                                std::max(2, (int)std::ceil(stretch / 0.7)));
    const double hs = 1.0 / per_px;

    // Frame-0 window: the grid widened by the margin (the material the nodes'
    // subsets can reach).
    const int X0 = std::max(0, geom.rect_x - margin_px);
    const int X1 = std::min(W - 1, geom.rect_x + (geom.gridW - 1) * geom.step + margin_px);
    const int Y0 = std::max(0, geom.rect_y - margin_px);
    const int Y1 = std::min(H - 1, geom.rect_y + (geom.gridH - 1) * geom.step + margin_px);
    if (X0 > X1 || Y0 > Y1) return out;

    unsigned char *dst = out.data;
    const size_t dstep = out.step;
#pragma omp parallel for schedule(static)
    for (int sy = Y0; sy <= Y1; ++sy) {
        const unsigned char *mrow = material_mask0.ptr<unsigned char>(sy);
        for (int sx = X0; sx <= X1; ++sx) {
            if (mrow[sx] < 128) continue;
            for (int b = 0; b < per_px; ++b) {
                for (int a = 0; a < per_px; ++a) {
                    const double X = sx - 0.5 + (a + 0.5) * hs;
                    const double Y = sy - 0.5 + (b + 0.5) * hs;
                    double u, v, ux, uy, vx, vy;
                    field.eval(X, Y, u, v, ux, uy, vx, vy);
                    const long px = std::lround(X + u), py = std::lround(Y + v);
                    if (px < 0 || px >= W || py < 0 || py >= H) continue;
                    // Every writer stores the same value; atomic keeps it
                    // race-free, and the result does not depend on order.
                    unsigned char *cell = dst + (size_t)py * dstep + (size_t)px;
#pragma omp atomic write
                    *cell = 255;
                }
            }
        }
    }
    cv::morphologyEx(out, out, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U));
    return out;
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
