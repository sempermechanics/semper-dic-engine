#ifndef SEMPER_SECOND_ORDER_SOLVER_HPP
#define SEMPER_SECOND_ORDER_SOLVER_HPP

// EXPERIMENTAL — not linked into semper_math or the pipeline; built only into
// the host test binary (tests/CMakeLists.txt) for the StrainSweep comparison.
//
// Forward-additive Gauss-Newton on a first- or second-order subset shape
// function. For a reference offset (x, y) from the subset centre:
//
//   u(x,y) = u + u_x x + u_y y + ½u_xx x² + u_xy xy + ½u_yy y²
//   v(x,y) = v + v_x x + v_y y + ½v_xx x² + v_xy xy + ½v_yy y²
//
// Parameter order: u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy.
// Order 1 fixes the last six at zero (same model as OptimizationEngine).
//
// Intensity model: each iteration fits gain a and offset b in closed form,
// minimising Σ (f − a·g − b)². At the optimum that sum equals
// n·σ_f²·(1 − ZNCC²), so the minimiser is the ZNSSD minimiser (ZNCC > 0);
// the reported score is ZNSSD / n = 2(1 − ZNCC), the engine's convention.

#include <semper/image.hpp>
#include <semper/types.hpp>

namespace Semper {
namespace experimental {

struct ShapeFitResult {
    float p[12] = {};
    int status = 1;              ///< 0 converged, 1 failed
    float correlation_score = 2.0f; ///< ZNSSD / n
    int iters = 0;
};

/**
 * Solve one subset of the reference image centred on (subset.cx, subset.cy).
 * @param order  1 (6 DOF) or 2 (12 DOF)
 * @param p0     initial 12-vector (entries beyond the order are ignored)
 */
ShapeFitResult solve_shape_fagn(const SubsetData &subset, const Image &def_img,
                                int order, const float p0[12], int max_iter = 50);

} // namespace experimental
} // namespace Semper

#endif // SEMPER_SECOND_ORDER_SOLVER_HPP
