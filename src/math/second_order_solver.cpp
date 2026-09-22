#include "math/second_order_solver.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace Semper {
namespace experimental {

namespace {

// Displacement basis at offset (x, y): du/dp_k for the first 6 or 12 params,
// in the fixed order u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy.
inline void basis(float x, float y, double bx[12], double by[12]) {
    for (int k = 0; k < 12; ++k) bx[k] = by[k] = 0.0;
    bx[0] = 1.0;
    by[1] = 1.0;
    bx[2] = x;  bx[3] = y;
    by[4] = x;  by[5] = y;
    bx[6] = 0.5 * x * x; bx[7] = (double)x * y; bx[8] = 0.5 * y * y;
    by[9] = 0.5 * x * x; by[10] = (double)x * y; by[11] = 0.5 * y * y;
}

} // namespace

ShapeFitResult solve_shape_fagn(const SubsetData &subset, const Image &def_img,
                                int order, const float p0[12], int max_iter) {
    ShapeFitResult res;
    const int m = (order == 2) ? 12 : 6;
    const size_t n = subset.x_offsets_f.size();
    for (int k = 0; k < 12; ++k) res.p[k] = (k < m) ? p0[k] : 0.0f;
    if (n == 0) return res;

    // Parameter scaling: step convergence is judged by the displacement each
    // update produces at the subset edge (h = half-size), in px.
    const double h = 0.5 * (subset.dim - 1);
    double scale[12];
    for (int k = 0; k < 12; ++k) scale[k] = (k < 2) ? 1.0 : (k < 6 ? h : 0.5 * h * h);

    const std::vector<float> &f = subset.ref_intensities;
    double f_mean = 0.0;
    for (size_t i = 0; i < n; ++i) f_mean += f[i];
    f_mean /= (double)n;

    std::vector<float> g(n), gx(n), gy(n);
    constexpr float H_FD = 0.5f; // central-difference step for ∇g (Jacobian only)
    const float lo = 4.0f, hi_x = (float)def_img.width - 5.0f, hi_y = (float)def_img.height - 5.0f;

    for (int iter = 0; iter < max_iter; ++iter) {
        res.iters = iter + 1;

        // 1. Warp, sample g and ∇g.
        for (size_t i = 0; i < n; ++i) {
            const float x = subset.x_offsets_f[i], y = subset.y_offsets_f[i];
            const float *p = res.p;
            const float du = p[0] + p[2] * x + p[3] * y + 0.5f * p[6] * x * x + p[7] * x * y + 0.5f * p[8] * y * y;
            const float dv = p[1] + p[4] * x + p[5] * y + 0.5f * p[9] * x * x + p[10] * x * y + 0.5f * p[11] * y * y;
            const float X = subset.cx + x + du, Y = subset.cy + y + dv;
            if (!(X >= lo && X <= hi_x && Y >= lo && Y <= hi_y)) return res; // left the image
            g[i] = def_img.interpolate_bicubic(X, Y);
            gx[i] = 0.5f / H_FD * (def_img.interpolate_bicubic(X + H_FD, Y) - def_img.interpolate_bicubic(X - H_FD, Y));
            gy[i] = 0.5f / H_FD * (def_img.interpolate_bicubic(X, Y + H_FD) - def_img.interpolate_bicubic(X, Y - H_FD));
        }

        // 2. Closed-form gain/offset and the ZNSSD of the current warp.
        double g_mean = 0.0;
        for (size_t i = 0; i < n; ++i) g_mean += g[i];
        g_mean /= (double)n;
        double sfg = 0.0, sgg = 0.0, sff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double df = f[i] - f_mean, dg = g[i] - g_mean;
            sfg += df * dg; sgg += dg * dg; sff += df * df;
        }
        if (sgg < 1e-9 || sff < 1e-9) return res;
        const double a = sfg / sgg, b = f_mean - a * g_mean;
        const double zncc = sfg / std::sqrt(sff * sgg);
        res.correlation_score = (float)(2.0 * (1.0 - zncc));

        // 3. Gauss-Newton normal equations in scaled parameters.
        //    r = f − a·g − b ;  ∂r/∂p = −a ∇g·∂W/∂p ;  offset b absorbs the mean.
        Eigen::MatrixXd JtJ = Eigen::MatrixXd::Zero(m, m);
        Eigen::VectorXd Jtr = Eigen::VectorXd::Zero(m);
        Eigen::VectorXd Jsum = Eigen::VectorXd::Zero(m);
        std::vector<Eigen::VectorXd> J(n, Eigen::VectorXd(m));
        double bx[12], by[12];
        for (size_t i = 0; i < n; ++i) {
            basis(subset.x_offsets_f[i], subset.y_offsets_f[i], bx, by);
            for (int k = 0; k < m; ++k) J[i](k) = a * (gx[i] * bx[k] + gy[i] * by[k]) / scale[k];
            Jsum += J[i];
        }
        const Eigen::VectorXd Jmean = Jsum / (double)n;
        for (size_t i = 0; i < n; ++i) {
            const Eigen::VectorXd Jc = J[i] - Jmean;
            const double r = f[i] - a * g[i] - b;
            JtJ.noalias() += Jc * Jc.transpose();
            Jtr.noalias() += Jc * r;
        }

        const Eigen::VectorXd dq = JtJ.ldlt().solve(Jtr); // scaled step
        if (!dq.allFinite()) return res;
        for (int k = 0; k < m; ++k) res.p[k] += (float)(dq(k) / scale[k]);

        if (dq.norm() < 1e-3) { // < 1e-3 px of motion anywhere on the subset
            res.status = 0;
            return res;
        }
    }
    return res; // iteration budget exhausted: status stays 1
}

} // namespace experimental
} // namespace Semper
