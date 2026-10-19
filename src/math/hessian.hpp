#ifndef SEMPER_MATH_HESSIAN_HPP
#define SEMPER_MATH_HESSIAN_HPP

// Shared checks for the 6x6 Gauss-Newton Hessian of a subset and for the
// numbers IC-GN produces from it. Internal to src/math.

#include <semper/tuning.hpp>
#include <semper/types.hpp>

#include <cstdint>
#include <cstring>

namespace Semper {
namespace hessian {

using Matrix6f = Eigen::Matrix<float, 6, 6>;

// True unless f is Inf or NaN, read from the bits. The engine is built with
// -ffast-math, under which std::isfinite may be folded to true.
inline bool finite_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return (u & 0x7f800000u) != 0x7f800000u;
}

template <typename Derived>
bool all_finite_bits(const Eigen::MatrixBase<Derived> &m) {
    for (Eigen::Index r = 0; r < m.rows(); ++r)
        for (Eigen::Index c = 0; c < m.cols(); ++c)
            if (!finite_bits(m(r, c))) return false;
    return true;
}

// Whether H is safe to invert in float. The steepest-descent columns for the
// four displacement gradients are the intensity gradient times a pixel offset
// of up to `half`, so their rows and columns of H grow as half^2; dividing
// them by `half` first makes the conditioning a property of the texture, not
// of the subset size. The determinant used before was scale-dependent: about
// 1e20 at 41 px, Inf past roughly 170 px, and a singular Hessian's rounding
// noise cleared its 1e-6 floor by orders of magnitude.
//
// The reciprocal condition number is computed, not estimated:
// 1 / (||Hn||_1 ||Hn^-1||_1) from the explicit inverse, which a 6x6 affords.
// The LDLT rcond() estimator was tried first and is not safe here: on a
// rotation-symmetric texture (SubsetPrecomputer.IllConditioned_Rejected) it
// reported 0.27 for a matrix whose eigenvalues span 1e-6 .. 30 under clang.
inline bool well_conditioned(const Matrix6f &H, int half) {
    const float s = half > 1 ? 1.0f / (float)half : 1.0f;
    Eigen::Matrix<float, 6, 1> d;
    d << 1.0f, 1.0f, s, s, s, s;
    const Matrix6f Hn = d.asDiagonal() * H * d.asDiagonal();
    if (!all_finite_bits(Hn)) return false;
    const Matrix6f Hn_inv = Hn.inverse();
    if (!all_finite_bits(Hn_inv)) return false;
    const float norm = Hn.cwiseAbs().colwise().sum().maxCoeff();
    const float norm_inv = Hn_inv.cwiseAbs().colwise().sum().maxCoeff();
    // Written so that a NaN fails.
    return 1.0f / (norm * norm_inv) > tuning::kHessianMinRcond;
}

} // namespace hessian
} // namespace Semper

#endif // SEMPER_MATH_HESSIAN_HPP
