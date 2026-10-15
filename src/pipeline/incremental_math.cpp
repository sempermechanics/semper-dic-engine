// Sampling and mask warping for incremental DIC. See incremental_internal.hpp.

#include "incremental_internal.hpp"

#include <semper/strain.hpp>
#include <semper/tuning.hpp>

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

// Continuous field over the filled grid, in image pixels. Bilinear inside,
// linear extrapolation of the edge cell up to `extrap` cells out, then held.
struct ContinuousField {
    const GridGeom &g;
    const std::vector<float> &U, &V;
    float extrap;

    // Value and derivatives (per pixel) at image point (x, y).
    void eval(double x, double y, double &u, double &v,
              double &ux, double &uy, double &vx, double &vy) const {
        double gx = (x - g.rect_x) / g.step;
        double gy = (y - g.rect_y) / g.step;
        bool clamp_x = false, clamp_y = false;
        const double lo = -extrap, hix = (g.gridW - 1) + extrap, hiy = (g.gridH - 1) + extrap;
        if (gx < lo) { gx = lo; clamp_x = true; }
        if (gx > hix) { gx = hix; clamp_x = true; }
        if (gy < lo) { gy = lo; clamp_y = true; }
        if (gy > hiy) { gy = hiy; clamp_y = true; }

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
        if (clamp_x) { ux = 0.0; vx = 0.0; }
        if (clamp_y) { uy = 0.0; vy = 0.0; }
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

    const ContinuousField field{geom, U, V, (float)tuning::kIncMaskExtrapCells};

    // Window: the forward-mapped grid, widened by the margin.
    double minx = 1e300, miny = 1e300, maxx = -1e300, maxy = -1e300;
    for (int gy = 0; gy < geom.gridH; ++gy) {
        for (int gx = 0; gx < geom.gridW; ++gx) {
            const size_t i = (size_t)gy * geom.gridW + gx;
            const double px = geom.rect_x + gx * geom.step + U[i];
            const double py = geom.rect_y + gy * geom.step + V[i];
            minx = std::min(minx, px); maxx = std::max(maxx, px);
            miny = std::min(miny, py); maxy = std::max(maxy, py);
        }
    }
    const int x0 = std::max(0, (int)std::floor(minx) - margin_px);
    const int x1 = std::min(W - 1, (int)std::ceil(maxx) + margin_px);
    const int y0 = std::max(0, (int)std::floor(miny) - margin_px);
    const int y1 = std::min(H - 1, (int)std::ceil(maxy) + margin_px);
    if (x0 > x1 || y0 > y1) return out;

    const double tol = tuning::kIncMaskNewtonTolPx;
    const int iters = tuning::kIncMaskNewtonIters;

#pragma omp parallel for schedule(static)
    for (int py = y0; py <= y1; ++py) {
        unsigned char *row = out.ptr<unsigned char>(py);
        double X = 0, Y = 0;
        bool warm = false;
        for (int px = x0; px <= x1; ++px) {
            double u, v, ux, uy, vx, vy;
            if (warm) {
                X += 1.0;   // the neighbour's preimage, one pixel on
            } else {
                field.eval(px, py, u, v, ux, uy, vx, vy);
                X = px - u;
                Y = py - v;
            }
            double rx = 0, ry = 0;
            bool converged = false;
            for (int it = 0; it < iters; ++it) {
                field.eval(X, Y, u, v, ux, uy, vx, vy);
                rx = X + u - px;
                ry = Y + v - py;
                if (std::abs(rx) < tol && std::abs(ry) < tol) { converged = true; break; }
                const double a = 1 + ux, b = uy, c = vx, d = 1 + vy;
                const double det = a * d - b * c;
                if (std::abs(det) < 1e-6) break;
                X -= ( d * rx - b * ry) / det;
                Y -= (-c * rx + a * ry) / det;
            }
            if (!converged) {
                field.eval(X, Y, u, v, ux, uy, vx, vy);
                rx = X + u - px;
                ry = Y + v - py;
                converged = std::abs(rx) < 0.5 && std::abs(ry) < 0.5;
            }
            warm = converged;
            if (!converged) continue;
            const int sx = (int)std::lround(X), sy = (int)std::lround(Y);
            if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
            if (material_mask0.at<unsigned char>(sy, sx) >= 128) row[px] = 255;
        }
    }
    return out;
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
