#ifndef DICTEST_SEQUENCE_H
#define DICTEST_SEQUENCE_H

// =====================================================================
// FRAME SEQUENCES WITH EXACT GROUND TRUTH
//
// Every frame is rendered from the same continuous speckle (synthetic.h) by
// inverse-mapping each deformed pixel to the reference. A Motion gives, for a
// reference (frame-0) point X, its deformed position x(X) and the deformation
// gradient F(X) = dx/dX; ground truth follows exactly:
//
//     U(X) = x(X) - X,      E = 1/2 (F^T F - I)   (Green-Lagrange)
//
// matching the engine's exy convention 1/2(uy + vx + ux*uy + vx*vy).
// Affine motions invert in closed form; smooth non-affine ones by Newton.
// =====================================================================

#include "framework/synthetic.h"

#include <opencv2/core.hpp>

#include <cmath>
#include <functional>
#include <algorithm>
#include <random>
#include <vector>

namespace dictest {

struct Vec2 { double x = 0, y = 0; };
struct Mat2 { double a = 1, b = 0, c = 0, d = 1; };   // [[a, b], [c, d]]

// A motion: forward map X -> x and its gradient. Inverse by Newton unless the
// motion provides a closed form.
struct Motion {
    std::function<Vec2(double, double)> forward;
    std::function<Mat2(double, double)> grad;
    std::function<Vec2(double, double)> inverse;   // optional closed form

    Vec2 inv(double x, double y) const {
        if (inverse) return inverse(x, y);
        double X = x, Y = y;
        for (int it = 0; it < 30; ++it) {
            const Vec2 f = forward(X, Y);
            const double rx = f.x - x, ry = f.y - y;
            if (std::abs(rx) < 1e-9 && std::abs(ry) < 1e-9) break;
            const Mat2 J = grad(X, Y);
            const double det = J.a * J.d - J.b * J.c;
            X -= ( J.d * rx - J.b * ry) / det;
            Y -= (-J.c * rx + J.a * ry) / det;
        }
        return {X, Y};
    }

    Vec2 displacement(double X, double Y) const {
        const Vec2 f = forward(X, Y);
        return {f.x - X, f.y - Y};
    }

    // Green-Lagrange (exx, eyy, exy).
    void strain(double X, double Y, double &exx, double &eyy, double &exy) const {
        const Mat2 F = grad(X, Y);
        exx = 0.5 * (F.a * F.a + F.c * F.c - 1.0);
        eyy = 0.5 * (F.b * F.b + F.d * F.d - 1.0);
        exy = 0.5 * (F.a * F.b + F.c * F.d);
    }
};

// x = c + F (X - c) + t
inline Motion affine_motion(Mat2 F, Vec2 t, Vec2 c) {
    Motion m;
    m.forward = [=](double X, double Y) {
        const double px = X - c.x, py = Y - c.y;
        return Vec2{c.x + F.a * px + F.b * py + t.x, c.y + F.c * px + F.d * py + t.y};
    };
    m.grad = [=](double, double) { return F; };
    m.inverse = [=](double x, double y) {
        const double det = F.a * F.d - F.b * F.c;
        const double bx = x - c.x - t.x, by = y - c.y - t.y;
        return Vec2{c.x + ( F.d * bx - F.b * by) / det, c.y + (-F.c * bx + F.a * by) / det};
    };
    return m;
}

// Rotation by theta (rad) of a stretch diag(1 + e, 1 - nu*e), about c, then t.
inline Motion stretch_rotate(double e, double nu, double theta, Vec2 t, Vec2 c) {
    const double cs = std::cos(theta), sn = std::sin(theta);
    const double sx = 1.0 + e, sy = 1.0 - nu * e;
    return affine_motion(Mat2{cs * sx, -sn * sy, sn * sx, cs * sy}, t, c);
}

// Gaussian-blob speckle that is smooth everywhere. SpeckleField::sample cuts
// each blob off at r^2 < 40, which leaves steps of up to ~17 grey levels where
// a blob of sigma 3.5 px ends; a pair of frames sees those steps move, and an
// incremental chain keeps whatever error that causes. Here every blob runs to
// 4 sigma, where its remaining contribution is below 0.03 grey levels.
// Blobs cover a margin around the image so content moving in is speckled too.
class SmoothSpeckleField {
public:
    SmoothSpeckleField(unsigned seed, int width, int height, double blobs_per_px = 600.0 / 65536.0,
                       int margin = 64) {
        std::mt19937 rng(seed);
        x0_ = -margin; y0_ = -margin;
        const int W = width + 2 * margin, H = height + 2 * margin;
        const int count = (int)(blobs_per_px * W * H);
        std::uniform_real_distribution<float> ux(0.0f, (float)W), uy(0.0f, (float)H);
        std::uniform_real_distribution<float> usigma(1.5f, 3.5f), uamp(-90.0f, 90.0f);
        for (int i = 0; i < count; ++i) {
            const float s = usigma(rng);
            cx_.push_back(x0_ + ux(rng));
            cy_.push_back(y0_ + uy(rng));
            inv2s2_.push_back(1.0f / (2.0f * s * s));
            r2max_.push_back(16.0f * s * s);
            amp_.push_back(uamp(rng));
        }
        bw_ = (int)std::ceil(W / kCell);
        bh_ = (int)std::ceil(H / kCell);
        std::vector<int> cell(cx_.size()), count_per((size_t)bw_ * bh_, 0);
        for (size_t i = 0; i < cx_.size(); ++i) {
            const int bx = std::min(bw_ - 1, std::max(0, (int)((cx_[i] - x0_) / kCell)));
            const int by = std::min(bh_ - 1, std::max(0, (int)((cy_[i] - y0_) / kCell)));
            cell[i] = by * bw_ + bx;
            count_per[cell[i]]++;
        }
        start_.assign(count_per.size() + 1, 0);
        for (size_t c = 0; c < count_per.size(); ++c) start_[c + 1] = start_[c] + count_per[c];
        std::vector<int> fill(start_.begin(), start_.end() - 1);
        idx_.assign(cx_.size(), 0);
        for (size_t i = 0; i < cx_.size(); ++i) idx_[fill[cell[i]]++] = (int)i;
    }

    float sample_fast(float x, float y) const {
        const int qx = (int)std::floor((x - x0_) / kCell), qy = (int)std::floor((y - y0_) / kCell);
        float v = 128.0f;
        for (int by = std::max(qy - 1, 0); by <= std::min(qy + 1, bh_ - 1); ++by) {
            for (int bx = std::max(qx - 1, 0); bx <= std::min(qx + 1, bw_ - 1); ++bx) {
                const int c = by * bw_ + bx;
                for (int k = start_[c]; k < start_[c + 1]; ++k) {
                    const int i = idx_[k];
                    const float dx = x - cx_[i], dy = y - cy_[i];
                    const float r2 = dx * dx + dy * dy;
                    if (r2 < r2max_[i]) v += amp_[i] * std::exp(-r2 * inv2s2_[i]);
                }
            }
        }
        if (v < 5.0f) v = 5.0f;
        if (v > 250.0f) v = 250.0f;
        return v;
    }

private:
    static constexpr float kCell = 14.0f;   // 4 * the largest sigma
    float x0_ = 0, y0_ = 0;
    int bw_ = 1, bh_ = 1;
    std::vector<float> cx_, cy_, inv2s2_, r2max_, amp_;
    std::vector<int> start_, idx_;
};

// Render a frame: pixel q takes the speckle at the reference point mapping to q.
// Same arithmetic as gray8 (noise in float, clamp, truncate v + 0.5).
template <class Field>
inline cv::Mat render_frame(const Field &field, int w, int h, const Motion &m,
                            float noise_sigma = 0.0f, unsigned noise_seed = 0) {
    cv::Mat out(h, w, CV_8UC1);
    std::mt19937 rng(noise_seed);
    std::normal_distribution<float> gauss(0.0f, noise_sigma);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const Vec2 r = m.inv(x, y);
            float v = field.sample_fast((float)r.x, (float)r.y);
            if (noise_sigma > 0.0f) v += gauss(rng);
            if (v < 0.f) v = 0.f;
            if (v > 255.f) v = 255.f;
            out.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
        }
    }
    return out;
}

inline Motion identity_motion() {
    return affine_motion(Mat2{}, Vec2{}, Vec2{});
}

} // namespace dictest

#endif // DICTEST_SEQUENCE_H
