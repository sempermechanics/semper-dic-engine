# Mathematical methodology

This document states the mathematics the engine implements: the ZNSSD
criterion, sub-pixel interpolation, the IC-GN solver and its Nelder–Mead
rescue, the seeding of initial guesses, outlier rejection and strain. It is
for readers who verify the algorithms, cite them, or change the numerics.
Named constants (`tuning::k…`) are defined in `include/semper/tuning.hpp`.

For where each equation lives in the code, see
[ARCHITECTURE.md](ARCHITECTURE.md). For how each is verified, see
[TESTING.md](TESTING.md). For seeding measurements, see
[SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md).

---

## Table of Contents

1.  [Introduction & Notation](#1-introduction--notation)
2.  [The Digital Image Correlation Problem](#2-the-digital-image-correlation-problem)
3.  [Correlation Criterion: ZNSSD](#3-correlation-criterion-znssd)
4.  [Image Representation & Interpolation](#4-image-representation--interpolation)
5.  [Optimization Framework](#5-optimization-framework)
6.  [Inverse Compositional Gauss-Newton](#6-inverse-compositional-gauss-newton)
7.  [Simplex Rescue Method](#7-simplex-rescue-method)
8.  [Anchor-Lattice Initialization](#8-anchor-lattice-initialization)
9.  [Strain Tensor Calculation](#9-strain-tensor-calculation)
10. [Error Analysis & Uncertainty](#10-error-analysis--uncertainty)
11. [Implementation Validation: DIC Challenge](#11-implementation-validation-dic-challenge)
12. [References](#12-references)

---

## 1. Introduction & Notation

### 1.1 Mathematical Symbols

| Symbol | Description | Units |
| :--- | :--- | :--- |
| $f(\mathbf{x})$ | Reference image intensity function | [0, 255] |
| $g(\mathbf{x})$ | Deformed image intensity function | [0, 255] |
| $\mathbf{x} = (x, y)^T$ | Spatial coordinates in image plane | pixels |
| $\mathbf{p} = (u, v, u_x, u_y, v_x, v_y)^T$ | Warp parameters | pixels, dimensionless |
| $W(\mathbf{x}; \mathbf{p})$ | Warp function mapping reference to deformed | pixels |
| $S$ | Subset (region of interest) | pixels² |
| $\nabla f$ | Image gradient vector | intensity/pixel |
| $H$ | Hessian matrix | intensity²/pixel² |

### 1.2 Coordinate Systems

* **Reference Frame:** $(x_r, y_r)$ - Undeformed image coordinates
* **Deformed Frame:** $(x_d, y_d)$ - Deformed image coordinates
* **Subset-Local Frame:** $(x_s, y_s)$ - Coordinates relative to subset center

**Transformation:**
$$
\mathbf{x}_d = W(\mathbf{x}_r; \mathbf{p})
$$

---

## 2. The Digital Image Correlation Problem

### 2.1 Problem Statement

**Given:**
* Reference image $f: \mathbb{R}^2 \rightarrow \mathbb{R}$ (captured at time $t_0$)
* Deformed image $g: \mathbb{R}^2 \rightarrow \mathbb{R}$ (captured at time $t_1$)
* Subset $S$ centered at $\mathbf{x}_c = (x_c, y_c)^T$ with dimensions $(2N+1) \times (2N+1)$ pixels

**Find:**
* Warp parameters $\mathbf{p}^*$ that minimize the dissimilarity between the reference subset and its deformed counterpart

**Mathematical formulation:**
$$
\mathbf{p}^* = \arg\min_{\mathbf{p}} C(f, g, \mathbf{p})
$$
where $C$ is a correlation criterion measuring image similarity.

### 2.2 Fundamental Assumptions

1.  **Material Point Uniqueness:** Each material point appears exactly once in both images.
2.  **Intensity Conservation:** $f(\mathbf{x}) \approx g(W(\mathbf{x}; \mathbf{p}))$, up to the affine change of brightness and contrast that ZNSSD removes (§3.2).
3.  **Smoothness:** The warp function $W$ is differentiable.
4.  **Local Homogeneity:** An affine transformation approximates the deformation within a subset.

---

## 3. Correlation Criterion: ZNSSD

### 3.1 Formulation

The **Zero-Normalized Sum of Squared Differences (ZNSSD)** is defined as:
$$
C_{\text{ZNSSD}}(\mathbf{p}) = \sum_{\mathbf{x}_i \in S} \left[ \frac{f(\mathbf{x}_i) - \bar{f}}{\Delta f} - \frac{g(W(\mathbf{x}_i; \mathbf{p})) - \bar{g}}{\Delta g} \right]^2
$$

**Where:**
* $\bar{f} = \frac{1}{|S|} \sum_{\mathbf{x}_i \in S} f(\mathbf{x}_i)$ : Mean intensity of reference subset
* $\bar{g} = \frac{1}{|S|} \sum_{\mathbf{x}_i \in S} g(W(\mathbf{x}_i; \mathbf{p}))$ : Mean intensity of warped deformed subset
* $\Delta f = \sqrt{\frac{1}{|S|} \sum_{\mathbf{x}_i \in S} [f(\mathbf{x}_i) - \bar{f}]^2}$ : Standard deviation of reference
* $\Delta g = \sqrt{\frac{1}{|S|} \sum_{\mathbf{x}_i \in S} [g(W(\mathbf{x}_i; \mathbf{p})) - \bar{g}]^2}$ : Standard deviation of deformed
* $|S| = (2N+1)^2$ : Number of pixels in subset

**What the engine reports.** Every solve returns this sum divided by the
number of pixels it ran over: the per-pixel mean $C_{\text{ZNSSD}} / |S_v|$,
where $S_v$ are the pixels §6.6 calls valid (`OptimizationEngine`,
`error_sum_sq / valid_pixels`). Every gate compares this mean: the result gate
`tuning::kCorrAccept` = 0.15, the anchor gate `kAnchorAcceptScore` = 0.40
(§8.2), and the Simplex trigger and fail scores (§7). The mean lies in
$[0, 4]$ and does not grow with the subset size.

### 3.2 Properties

**Invariance to Linear Intensity Changes:**
If $g'(\mathbf{x}) = \alpha g(\mathbf{x}) + \beta$ with $\alpha > 0$ (contrast and brightness change), then:
$$
C_{\text{ZNSSD}}(\mathbf{p}) = C_{\text{ZNSSD}}(\mathbf{p}; g')
$$

**Proof:**
$$
\begin{aligned}
\bar{g}' &= \alpha \bar{g} + \beta \\
\Delta g' &= \alpha \Delta g \\
\frac{g'(W(\mathbf{x}; \mathbf{p})) - \bar{g}'}{\Delta g'} &= \frac{\alpha g(W(\mathbf{x}; \mathbf{p})) + \beta - \alpha \bar{g} - \beta}{\alpha \Delta g} \\
&= \frac{g(W(\mathbf{x}; \mathbf{p})) - \bar{g}}{\Delta g}
\end{aligned}
$$
Thus, ZNSSD is invariant to affine intensity transformations.

### 3.3 Relationship to Cross-Correlation

ZNSSD is a direct function of the Zero-Normalized Cross-Correlation (ZNCC):
$$
\text{ZNCC} = \frac{\sum [f - \bar{f}][g - \bar{g}]}{\sqrt{\sum [f - \bar{f}]^2} \sqrt{\sum [g - \bar{g}]^2}}
$$
With $\Delta f$, $\Delta g$ the standard deviations over $|S|$ pixels as
above, each normalized intensity has a sum of squares of exactly $|S|$.
Expanding the square in §3.1 therefore gives
$C_{\text{ZNSSD}} = 2|S|(1 - \text{ZNCC})$, and the per-pixel mean the engine
reports is
$$
C_{\text{ZNSSD}} / |S| = 2(1 - \text{ZNCC})
$$
The 0.15 result gate is therefore $\text{ZNCC} \ge 0.925$.

**Implementation Note:** Semper minimizes ZNSSD. Values closer to 0 indicate a better match.

---

## 4. Image Representation & Interpolation

### 4.1 Discrete-to-Continuous Mapping

Digital images are discrete: $f_{ij} = f(i, j)$ for $i, j \in \mathbb{Z}$.  
DIC requires evaluation at non-integer coordinates: $g(x_d, y_d)$ where $(x_d, y_d) \in \mathbb{R}^2$.

### 4.2 Keys Bicubic Interpolation

The default interpolator is Keys' cubic convolution kernel on a 4×4
neighbourhood, with $a = -0.5$. With that value the kernel is third-order
accurate (Keys, 1981). An optional 6×6 kernel, fourth-order accurate, is
described at the end of this section.

**Kernel Definition:**
For a parameter $a$ (the engine uses $a = -0.5$):
$$
k_a(s) =
\begin{cases}
(a+2)|s|^3 - (a+3)|s|^2 + 1 & \text{for } |s| \leq 1 \\
a|s|^3 - 5a|s|^2 + 8a|s| - 4a & \text{for } 1 < |s| \leq 2 \\
0 & \text{otherwise}
\end{cases}
$$

**Interpolation Formula:**
For point $(x, y)$, let $x_i = \lfloor x \rfloor$ and $y_j = \lfloor y \rfloor$:
$$
g(x, y) = \sum_{m=-1}^{2} \sum_{n=-1}^{2} g_{i+m, j+n} \cdot k_a(x - (i+m)) \cdot k_a(y - (j+n))
$$

**Weights for $a = -0.5$:**
With $s_x = x - x_i$ and $s_y = y - y_j$, the four weights along each axis are
(`get_keys_weights`, evaluated per sample):
$$
\begin{aligned}
w_{-1}(s) &= -0.5s^3 + s^2 - 0.5s \\
w_0(s) &= 1.5s^3 - 2.5s^2 + 1 \\
w_1(s) &= -1.5s^3 + 2s^2 + 0.5s \\
w_2(s) &= 0.5s^3 - 0.5s^2
\end{aligned}
$$

**Implementation:**
```cpp
// src/math/image_processor.cpp, Image::interpolate_bicubic
scalar_t Image::interpolate_bicubic(scalar_t x, scalar_t y) const {
    // Where the 4x4 support would leave the image: bilinear instead.
    if (!(x >= 1.0f && x < width - 2.0f && y >= 1.0f && y < height - 2.0f)) {
        return interpolate_bilinear(x, y);
    }

    int xi = static_cast<int>(x);
    int yi = static_cast<int>(y);

    float dx = x - xi;
    float dy = y - yi;

    float wx[4], wy[4];
    get_keys_weights(dx, wx[0], wx[1], wx[2], wx[3]);
    get_keys_weights(dy, wy[0], wy[1], wy[2], wy[3]);

    float val = 0.0f;
    for (int j = -1; j <= 2; ++j) {
        const scalar_t* row_ptr = &intensities[(yi + j) * width + xi];
        float row_val = 0.0f;
        row_val += row_ptr[-1] * wx[0];
        row_val += row_ptr[0]  * wx[1];
        row_val += row_ptr[1]  * wx[2];
        row_val += row_ptr[2]  * wx[3];
        val += row_val * wy[j + 1];
    }
    return val;
}
```

**Optional 6×6 kernel.** With `use_6x6_interpolator` the engine uses Keys'
six-point kernel (`Image::interpolate_keys_fourth`), summed over
$m, n = -2, \dots, 3$:
$$
k_6(s) =
\begin{cases}
\tfrac{4}{3}|s|^3 - \tfrac{7}{3}|s|^2 + 1 & \text{for } |s| \leq 1 \\
-\tfrac{7}{12}|s|^3 + 3|s|^2 - \tfrac{59}{12}|s| + \tfrac{5}{2} & \text{for } 1 < |s| \leq 2 \\
\tfrac{1}{12}|s|^3 - \tfrac{2}{3}|s|^2 + \tfrac{7}{4}|s| - \tfrac{3}{2} & \text{for } 2 < |s| \leq 3 \\
0 & \text{otherwise}
\end{cases}
$$

**Image border.** Both kernels fall back to bilinear interpolation where
their support would leave the image, and bilinear returns 0 outside it. The
4 px guard of §6.6 is stricter than either fallback boundary, so IC-GN never
samples through the fallback.

### 4.3 Image Gradient Computation

Gradients are computed using **5-point central finite difference**:
$$
\frac{\partial f}{\partial x}(i, j) \approx \frac{-f_{i+2,j} + 8 f_{i+1,j} - 8 f_{i-1,j} + f_{i-2,j}}{12}
$$
and likewise along $y$.

**Accuracy:** This scheme is 4th-order accurate: the truncation error is
$O(h^4)$ for grid spacing $h$ (here $h = 1$ px). Within 2 px of the image
border the gradient is set to 0.

**Implementation:**

```cpp
// src/math/image_processor.cpp, Image::prepare_data (I = intensities, w = width)
for (int y = 2; y < height - 2; ++y) {
    for (int x = 2; x < width - 2; ++x) {
        int idx = y * width + x;
        grad_x[idx] = (-I[idx+2] + 8*I[idx+1] - 8*I[idx-1] + I[idx-2]) / 12.0f;
        grad_y[idx] = (-I[idx+2*w] + 8*I[idx+w] - 8*I[idx-w] + I[idx-2*w]) / 12.0f;
    }
}
```

**Gradient Interpolation:**
None is needed. IC-GN uses gradients of the reference image only, at the
integer pixels of each subset (§6.2), so gradients are never evaluated at
non-integer coordinates.

---

## 5. Optimization Framework

### 5.1 Warp Function Parameterization

Semper uses a **6-parameter affine warp**. A subset pixel at offset
$(x_s, y_s)$ from the subset centre $\mathbf{x}_c$ maps to
$$
W(\mathbf{x}; \mathbf{p}) = \mathbf{x}_c + \begin{pmatrix} 1 + u_x & u_y \\ v_x & 1 + v_y \end{pmatrix} \begin{pmatrix} x_s \\ y_s \end{pmatrix} + \begin{pmatrix} u \\ v \end{pmatrix}
$$

**Expanded:**
$$
\begin{aligned}
x_d &= x_c + x_s + u + u_x x_s + u_y y_s \\
y_d &= y_c + y_s + v + v_x x_s + v_y y_s
\end{aligned}
$$

**Parameter Interpretation:**
| Parameter | Physical Meaning |
| :--- | :--- |
| $u, v$ | Rigid-body translation |
| $u_x, v_y$ | Normal strain (stretching/compression) |
| $u_y, v_x$ | Shear strain and rotation: $\tfrac{1}{2}(u_y + v_x)$ and $\tfrac{1}{2}(v_x - u_y)$ to first order |

**Homogeneous Representation:**
$$
\mathbf{W}(\mathbf{p}) = \begin{pmatrix} 1 + u_x & u_y & u \\ v_x & 1 + v_y & v \\ 0 & 0 & 1 \end{pmatrix}, \qquad
\begin{pmatrix} x_d - x_c \\ y_d - y_c \\ 1 \end{pmatrix} = \mathbf{W}(\mathbf{p}) \begin{pmatrix} x_s \\ y_s \\ 1 \end{pmatrix}
$$
In this form, composing and inverting warps (§6.4) are matrix operations.

**Precision.** The engine sums $x_c + [(1 + u_x) x_s + u_y y_s + u]$ (and
likewise $y_d$) in double and rounds to float once. In float, the three
roundings fall at the magnitude of $x_c$: about $10^{-3}$ px on an 8000 px
frame, comparable to the field's own noise.

### 5.2 Warp Jacobian

The Jacobian of the warp with respect to the parameters, in the order
$\mathbf{p} = (u, v, u_x, u_y, v_x, v_y)$:
$$
\frac{\partial W}{\partial \mathbf{p}} = \begin{pmatrix} 1 & 0 & x_s & y_s & 0 & 0 \\ 0 & 1 & 0 & 0 & x_s & y_s \end{pmatrix}
$$

**Evaluating:** The warp is linear in $\mathbf{p}$, so the Jacobian does not
depend on $\mathbf{p}$. IC-GN evaluates it once per subset, at the identity
warp.

---

## 6. Inverse Compositional Gauss-Newton

### 6.1 Algorithm Derivation

**Objective:** Minimize ZNSSD with respect to an increment $\Delta\mathbf{p}$
applied to the reference subset, with the current warp $\mathbf{p}$ fixed:
$$
\Delta\mathbf{p}^* = \arg\min_{\Delta\mathbf{p}} \sum_{\mathbf{x}_i \in S} \left[ \frac{f(W(\mathbf{x}_i; \Delta\mathbf{p})) - \bar{f}}{\Delta f} - \frac{g(W(\mathbf{x}_i; \mathbf{p})) - \bar{g}}{\Delta g} \right]^2
$$

**Key Insight:** In the inverse compositional formulation, the increment
warps the reference image instead of the deformed image. Every derivative is
then taken on the reference, which does not change between iterations.

**Linearization:**
A first-order Taylor expansion around $\Delta\mathbf{p} = \mathbf{0}$ gives
$$
f(W(\mathbf{x}_i; \Delta\mathbf{p})) \approx f(\mathbf{x}_i) + \nabla f(\mathbf{x}_i)\, \frac{\partial W}{\partial \mathbf{p}}\, \Delta\mathbf{p}
$$
which turns the objective into a linear least-squares problem in $\Delta\mathbf{p}$.

### 6.2 Steepest Descent Images

Define the **steepest descent image** at pixel $\mathbf{x}_i$:
$$
\mathbf{SD}_i = \frac{1}{\Delta f}\, \nabla f(\mathbf{x}_i)\, \frac{\partial W}{\partial \mathbf{p}}
$$

In component form, with $(f_x, f_y)$ the gradient of §4.3 at $\mathbf{x}_i$
and $(x_s, y_s)$ its offset from the subset centre:
$$
\mathbf{SD}_i = \frac{1}{\Delta f} \begin{pmatrix} f_x & f_y & f_x x_s & f_x y_s & f_y x_s & f_y y_s \end{pmatrix}
$$
The factor $1/\Delta f$ comes from the normalization of $f$ in the criterion.

**Implementation:**

```cpp
// src/math/subset_precomputer.cpp, SubsetPrecomputer::precompute_subset_fast
// (simplified; inv_std = 1 / Δf)
for (int oy = -half; oy <= half; ++oy) {
    for (int ox = -half; ox <= half; ++ox) {
        const float gx = data.gx_vec[idx] * inv_std;
        const float gy = data.gy_vec[idx] * inv_std;

        auto& sd = data.steepest_descent_images[idx];
        sd(0) = gx;      sd(1) = gy;
        sd(2) = gx * ox; sd(3) = gx * oy;
        sd(4) = gy * ox; sd(5) = gy * oy;
        idx++;
    }
}
```

### 6.3 Hessian Matrix

The approximate Hessian (Gauss-Newton approximation):
$$
\mathbf{H} = \sum_{\mathbf{x}_i \in S} \mathbf{SD}_i^{\top}\, \mathbf{SD}_i
$$

This is a $6 \times 6$ symmetric positive semi-definite matrix. It is
positive definite, and so invertible, only when the texture constrains all
six parameters. §6.6 tests this before inverting.

**Expanded Form:** $H_{jk} = \sum_i SD_{i,j}\, SD_{i,k}$ for
$j, k = 1, \dots, 6$. For example, $H_{11} = \Delta f^{-2} \sum_i f_x^2$ and
$H_{33} = \Delta f^{-2} \sum_i f_x^2 x_s^2$.

**Why inverse compositional:** In IC-GN, $\mathbf{H}$ depends only on the
reference image, so it is computed once per subset. The pipeline computes it
once per grid node and pools it.

**Implementation:**

```cpp
// src/math/subset_precomputer.cpp, SubsetPrecomputer::compute_hessian_only
// (simplified: the code accumulates the upper triangle and mirrors it)
Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
for (/* each subset pixel outside the Ghost Wall */) {
    H += sd * sd.transpose();
}
if (cond_2x2 > 1.0e12f || !hessian::well_conditioned(H, half)) {
    result.valid = false;          // the subset is not solved (§6.6)
} else {
    result.H_inv = H.inverse();
}
```

### 6.4 Parameter Update

The parameter increment is computed as:
$$
\Delta\mathbf{p} = -\mathbf{H}^{-1} \sum_{\mathbf{x}_i \in S} \mathbf{SD}_i^{\top}\, e_i, \qquad
e_i = \frac{f(\mathbf{x}_i) - \bar{f}}{\Delta f} - \frac{g(W(\mathbf{x}_i; \mathbf{p})) - \bar{g}}{\Delta g}
$$
This is the solution of the normal equations of the linearized objective of
§6.1.

**Warp Composition:**
The current warp is updated via inverse composition:
$$
W(\mathbf{x}; \mathbf{p}) \leftarrow W(\mathbf{x}; \mathbf{p}) \circ W(\mathbf{x}; \Delta\mathbf{p})^{-1}
$$

In matrix form:
$$
\mathbf{W}(\mathbf{p}) \leftarrow \mathbf{W}(\mathbf{p})\, \mathbf{W}(\Delta\mathbf{p})^{-1}
$$

**Levenberg–Marquardt damping.** Every pipeline solve (`make_engine`,
`src/pipeline/full_field_internal.hpp`) damps the two translation entries of
the Hessian, as DICe does:
$$
\mathbf{H}_{\text{LM}} = \mathbf{H} + \operatorname{diag}(\alpha, \alpha, 0, 0, 0, 0), \qquad \alpha = \texttt{tuning::kLmAlpha} = 0.05
$$
The four gradient entries are not damped. Because $\alpha$ is fixed,
$\mathbf{H}_{\text{LM}}$ is inverted once per call.

**Implementation:**

```cpp
// src/math/optimization_engine.cpp, OptimizationEngine::solve_icgn
Eigen::Matrix<float, 6, 1> delta_p = -(*H_step) * dp_sum;   // H_step: see §6.6

Eigen::Matrix3f dW = Eigen::Matrix3f::Identity();
dW(0, 0) += delta_p(2); dW(0, 1) += delta_p(3); dW(0, 2) += delta_p(0);
dW(1, 0) += delta_p(4); dW(1, 1) += delta_p(5); dW(1, 2) += delta_p(1);

W = W * dW.inverse();   // after the checks of §6.6, "When to stop"
```

### 6.5 Convergence Criterion

The algorithm terminates when:
$$
\|\Delta\mathbf{p}\|_2 < \varepsilon, \qquad \varepsilon = 10^{-3}
$$

**Norm Definition:**
$$
\|\Delta\mathbf{p}\|_2 = \sqrt{\Delta u^2 + \Delta v^2 + \Delta u_x^2 + \Delta u_y^2 + \Delta v_x^2 + \Delta v_y^2}
$$
The norm mixes translations (pixels) and gradients (dimensionless) without
weights.

A converged warp is accepted (status 0) only if the valid pixels carry
gradient in both directions, $\min(\sum f_x^2, \sum f_y^2) > 0$, and the score
is finite; otherwise the status is 1. After `tuning::kIcgnMaxIter` = 50
iterations without convergence, the solve ends with status 1 on the last warp.

**Typical Convergence:** 3–5 iterations for well-textured subsets with a good initial guess.

### 6.6 Validity, Conditioning and Failure

**Which pixels count.** Pixel $i$ of the subset contributes to an iteration
when two conditions hold. First, its warped position $\bm{q}_i = W(\bm{x}_i)$
rounds to a pixel at least 4 px inside the image (DICe's guard, stricter than
every interpolator's bilinear fallback boundary, §4.2). Second, its reference
pixel is not in the Ghost Wall ($I_{\text{ref}} \ge -5$; reference pixels
outside the ROI mask hold `tuning::kGhostWallValue` = −10).

Validity depends on where a sample was taken, never on the value read:
$g(\bm{q}_i) = 0$ is black speckle, and a negative value is cubic undershoot
beside it. One sampler (`sample_warped_subset`) serves IC-GN and the Simplex
cost (§7.3), so both see the same pixels. If fewer than 90% of the pixels are
valid (`tuning::kMinValidPixelFraction`), the solve ends (status 1).

**Which Hessian.** With every pixel valid, the step uses the reference
Hessian of §6.3 (LM-damped as in §6.4), fixed for the call. An iteration that
lost pixels uses the Hessian of the pixels it has (here $\nabla I_i$ is the
steepest-descent row $\mathbf{SD}_i$ of §6.2),

$$ \mathbf{H}_{\text{masked}} = \sum_{i \text{ valid}} \nabla I_i^{\top}\nabla I_i , $$

for that iteration only. The next full iteration returns to the reference
Hessian.

**When to invert.** The four gradient columns of the steepest-descent images
are $\nabla I \cdot x$ or $\nabla I \cdot y$, where $x, y$ are subset-local
offsets with $|x|, |y| \le h = \lfloor d/2 \rfloor$ for a subset of size $d$.
Their rows and columns of $\mathbf{H}$ therefore scale as $h^2$, and
$\det \mathbf{H}$ as $h^8$. A test on the determinant thus depends on the
subset size: the determinant overflows float past about 170 px, and a
singular Hessian's rounding noise clears a $10^{-6}$ floor by twenty orders
of magnitude. The engine scales first,

$$ \tilde{\mathbf{H}} = \mathbf{D}\,\mathbf{H}\,\mathbf{D}, \qquad
\mathbf{D} = \operatorname{diag}(1, 1, h^{-1}, h^{-1}, h^{-1}, h^{-1}), $$

and inverts $\mathbf{H}$ only when the reciprocal 1-norm condition number
$1 / (\|\tilde{\mathbf{H}}\|_1 \|\tilde{\mathbf{H}}^{-1}\|_1)$ exceeds
`tuning::kHessianMinRcond` $= 10^{-6}$. The number is computed from the
explicit inverse, not estimated: LDLT's `rcond()` misjudges the
symmetric-texture case below by five orders of magnitude. DICe's $2\times2$
translation-block test applies as well.

The rcond test applies to three Hessians, and the $2\times2$ test to the
first and last of them. A reference Hessian that fails at precompute marks
the subset unsolvable. An LM-damped one that fails falls back to the undamped
inverse. A masked one that fails ends the solve (status 1).

A texture symmetric about the subset centre is the textbook failure. It
cannot see a rotation about that centre, so the $\partial u/\partial y$ and
$\partial v/\partial x$ columns coincide while the translation block is
perfectly conditioned.

**When to stop.** An update $\Delta\bm{p}$ is applied only if it is finite,
$|\det \Delta W| > 10^{-6}$ (the inverse-compositional update needs
$\Delta W^{-1}$), and the composed warp is finite. Otherwise the solve ends
with status 1 and score 2.0 on the last good warp. Finiteness is read from the
float's exponent bits (`util::finite_bits`), a test that holds under any
floating-point flags. The build adds `-fno-finite-math-only` to `-ffast-math`;
without it the compiler may fold `std::isfinite` to true.

---

## 7. Simplex Rescue Method

### 7.1 When ICGN Fails

ICGN may fail to converge if:

1. Initial guess is far from true solution (>5 pixels error).
2. Subset contains occlusion or crack.
3. Intensity gradients are too weak (flat texture).

**Failure Indicators:** A solve calls the rescue when either holds:

* IC-GN ends with status ≠ 0: it was rejected (§6.6), or it reached the
  iteration budget (`tuning::kIcgnMaxIter` = 50) without convergence.
* The score exceeds `tuning::kCorrSimplexTrigger` = 0.4.

**Rescue sequence.** The 6-DOF Simplex (§7.2) starts from the solve's initial
guess, not from where IC-GN stopped. IC-GN then runs again from the Simplex
result, and the engine keeps the better of the two IC-GN runs
(`OptimizationEngine::calculate_deformation`). The first run is kept only if
it converged (status 0) and the second did not converge or scored higher;
otherwise the second is kept. A first run that converged with a passing score
therefore survives a rescue that ends worse.

The solve mode decides whether the rescue may run at all: `INIT_NO_SIMPLEX`
never calls it, and `INIT_AUTO_SEARCH` runs the search of §7.4 instead. The
modes and the paths that use them are listed in
[ARCHITECTURE.md](ARCHITECTURE.md) (*Initialization modes*).

### 7.2 Nelder-Mead Simplex Algorithm

The simplex method (Nelder & Mead, 1965) is a derivative-free optimization
technique suitable for non-smooth objective functions.

**Initialization:**
Create a simplex with $n + 1$ vertices in $n$-dimensional parameter space.
For translation-only mode ($n = 2$):
$$
\mathbf{p}_0 = (u_0, v_0), \qquad \mathbf{p}_1 = \mathbf{p}_0 + (2, 0), \qquad \mathbf{p}_2 = \mathbf{p}_0 + (0, 2)
$$

For full 6-DOF mode ($n = 6$), create 7 vertices: the start $\mathbf{p}_0$
and $\mathbf{p}_j = \mathbf{p}_0 + s_j \mathbf{e}_j$, with perturbations of
scale $s = (2, 2, 0.01, 0.01, 0.01, 0.01)$: 2 px on each translation and 0.01
on each displacement gradient.

**Operations:**
At each iteration, sort vertices by cost: $C(\mathbf{p}_0) \le C(\mathbf{p}_1) \le \dots \le C(\mathbf{p}_n)$.
Compute the centroid of the best $n$ points:
$$
\bar{\mathbf{p}} = \frac{1}{n} \sum_{i=0}^{n-1} \mathbf{p}_i
$$

**Four Possible Moves:**
The first three replace the worst vertex $\mathbf{p}_n$; the shrink moves
all but the best.

1. **Reflection:** $\mathbf{p}_r = \bar{\mathbf{p}} + \alpha(\bar{\mathbf{p}} - \mathbf{p}_n)$ with $\alpha = 1$. Accepted when $C(\mathbf{p}_0) \le C(\mathbf{p}_r) < C(\mathbf{p}_{n-1})$.
2. **Expansion:** $\mathbf{p}_e = \bar{\mathbf{p}} + \gamma(\mathbf{p}_r - \bar{\mathbf{p}})$ with $\gamma = 2$, tried when $C(\mathbf{p}_r) < C(\mathbf{p}_0)$. The better of $\mathbf{p}_e$ and $\mathbf{p}_r$ replaces $\mathbf{p}_n$.
3. **Contraction:** $\mathbf{p}_c = \bar{\mathbf{p}} + \rho(\mathbf{p}_\ast - \bar{\mathbf{p}})$ with $\rho = 0.5$, where $\mathbf{p}_\ast = \mathbf{p}_r$ if $C(\mathbf{p}_r) < C(\mathbf{p}_n)$ and $\mathbf{p}_\ast = \mathbf{p}_n$ otherwise. Accepted when $C(\mathbf{p}_c) < \min(C(\mathbf{p}_r), C(\mathbf{p}_n))$.
4. **Shrink:** $\mathbf{p}_i \leftarrow \mathbf{p}_0 + \sigma(\mathbf{p}_i - \mathbf{p}_0)$ for $i = 1, \dots, n$, with $\sigma = 0.5$, when the contraction is not accepted.

**Termination:** After at most 100 iterations, or earlier when
$|C(\mathbf{p}_n) - C(\mathbf{p}_0)| < 10^{-5}$. The result is the best
vertex. Its status is −2 when its score exceeds `tuning::kSimplexFailScore`
= 0.1. IC-GN runs from it either way (§7.1, §7.4).

**Implementation:**

```cpp
// OptimizationEngine::solve_simplex
for (int iter = 0; iter < 100; ++iter) {
    // Sort vertices by cost (an insertion sort over the 3 or 7 vertices --
    // the order std::sort gives on a range this short)
    for (int a = 1; a < n_pts; ++a) { /* insert idx[a] by y */ }
    
    // Compute centroid
    for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j) 
            p_bar[j] += p[idx[i]][j];
    for (int j = 0; j < DIM; ++j) 
        p_bar[j] /= DIM;
    
    // Reflection
    for (int j = 0; j < DIM; ++j) 
        p_r[j] = p_bar[j] + alpha * (p_bar[j] - p[idx[n_pts-1]][j]);
    
    // ... (expansion, contraction, shrink logic)
}

```

### 7.3 Cost Function Evaluation

At each simplex vertex the cost is the per-pixel mean ZNSSD of §3.1 over the
pixels §6.6 calls valid, sampled by the same routine as IC-GN:
$$
C(\mathbf{p}) =
\begin{cases}
\dfrac{1}{|S_v|} \displaystyle\sum_{\mathbf{x}_i \in S_v} \left[ \frac{f(\mathbf{x}_i) - \bar{f}}{\Delta f} - \frac{g(W(\mathbf{x}_i; \mathbf{p})) - \bar{g}}{\Delta g} \right]^2 & \text{if } |S_v| \ge 0.9\,|S| \\
2.0 & \text{otherwise}
\end{cases}
$$

The 90% floor is `tuning::kMinValidPixelFraction`, the floor IC-GN applies
(§6.6). Without it, the Simplex could move towards the image edge, where a few
surviving pixels can match closely, pass its gate, and hand IC-GN a seed that
IC-GN then rejects. A non-finite score also counts as the worst, 2.0.

**Typical Performance:**

* Evaluation cost: ~0.5 ms per vertex
* Total iterations: up to 100 (stops early when the vertex costs agree within 1e-5)
* Total time: ~2–5 ms, about 10× an IC-GN solve

### 7.4 Coarse Search Without a Guess

A solve in mode `INIT_AUTO_SEARCH` has no reliable guess. Path C and anchors
without a global model (§8.1.2) use it. When the guess is exactly $(0, 0)$,
the solve first runs a coarse integer search
(`OptimizationEngine::estimate_initial_guess`):

1. The candidates are the offsets $(u, v)$ with
   $u, v \in \lbrace -15, -13, \dots, 13, 15 \rbrace$ px: a 2 px step, 256 offsets.
2. The sample is every 9th subset pixel, in row-major order. A sample is left
   out when its reference pixel is in the Ghost Wall, or when its shifted
   position fails the 4 px guard of §6.6.
3. An offset is skipped when fewer than half the sample (or fewer than 3
   pixels) remain, or when either side is flat (standard deviation
   $\le 10^{-5}$).
4. Each remaining offset is scored by the per-pixel mean ZNSSD over its
   sample. The lowest score wins. If no offset qualifies, the guess stays
   $(0, 0)$.

The translation-only Simplex (§7.2, $n = 2$) then starts from the winner, or
from the given guess when it is not zero. IC-GN runs from the Simplex result
with zero gradients, and no further rescue follows.

ZNSSD makes the ranking follow the texture. A raw SSD also ranks by
brightness, so an exposure change between the frames pulls it off.

---

## 8. Anchor-Lattice Initialization

The seeding stage supplies the initial guess for §6's IC-GN iteration. Its job
is to put every subset inside the basin of attraction and, through the
Delaunay mesh of §8.4, to supply a first-order guess of the deformation
gradient. Anchors that pass the result gate and the median test are also kept
as measurements (§8.2).

The solve paths named below are Path A (mesh-guided solve of the nodes the
mesh covers), Path B (reliability-guided flood fill) and Path C (fallback
seeding when no mesh forms). [ARCHITECTURE.md](ARCHITECTURE.md) describes
them.

### 8.1 Global Rigid Shift by Phase Correlation

*The solve takes its global model from the AKAZE homography of §8.1.2. Phase
correlation (§8.1) and its Fourier–Mellin extension (§8.1.1) remain in the
Internal API (`seeding::phase_correlate_roi`,
`seeding::log_polar_rotation_scale`, `seeding::phase_correlate_similarity`),
and no solve path calls them. They are described here because their limits
set the requirements for §8.1.2.*

Let $f$ and $g$ be the reference and deformed intensity fields over the ROI,
windowed by a Hann taper to suppress edge discontinuity. Their cross-power
spectrum is normalised to unit magnitude:

$$ R(\bm{k}) = \frac{F(\bm{k})\,\overline{G(\bm{k})}}{\left|F(\bm{k})\,\overline{G(\bm{k})}\right|} $$

For a pure translation $g(\bm{x}) = f(\bm{x} - \bm{d})$ the normalisation leaves
a pure phase ramp, $R(\bm{k}) = e^{-i\bm{k}\cdot\bm{d}}$, whose inverse transform
is a Dirac impulse at $\bm{d}$. The estimator takes the centroid of that peak,
giving sub-pixel resolution, and its height as a confidence measure. A response
below $k_{\text{PhaseCorrMinResponse}}$ (0.05) is treated as no lock.

Keeping only the phase makes the estimate invariant to the affine intensity
change $a\,f + b$, as the ZNSSD criterion is. It uses every pixel in the ROI
rather than a sparse keypoint set. Its accuracy on a rigid shift (~0.005 px
measured) is therefore two orders of magnitude better than that of the
keypoint detectors in [SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md).

**Limitation.** The estimate is a *translation*. Under a rotation $\theta$ about
the ROI centre the true displacement at radius $r$ is $2r\sin(\theta/2)$, so a
single global shift is increasingly wrong away from the centre. Measured
behaviour of the translation alone: usable to about $2°$, degraded by $5°$, no
lock by $15°$ (see [SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md) §4.5). §8.1.1
removes this limitation.

#### 8.1.1 Rotation and Scale by Fourier–Mellin Correlation

Model the global motion as a similarity about the ROI centre $\bm{c}$: a
reference point $\bm{p}$ appears at

$$ \bm{q} = \bm{c} + sR(\theta)(\bm{p} - \bm{c}) + \bm{t}, \qquad R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix} $$

in image axes ($y$ down), so $g(\bm{x}) = f\big(\bm{c} + (sR)^{-1}(\bm{x} - \bm{c} - \bm{t})\big)$.
In the Fourier transform the translation only multiplies by a phase, while
rotation and scale act on the frequency axis:

$$ |G(\bm{k})| = s^2\,\big|F\big(s\,R(\theta)^{-1}\bm{k}\big)\big| $$

The magnitude spectrum therefore sees $\theta$ and $s$ but not $\bm{t}$. In
log-polar coordinates $(\phi, \rho) = (\arg\bm{k}, \ln|\bm{k}|)$ the rotation
becomes a shift by $+\theta$ along $\phi$ and the scale a shift by $-\ln s$
along $\rho$:

$$ |G|(\phi, \rho) = s^2\,|F|(\phi - \theta,\ \rho + \ln s) $$

The phase correlation of §8.1, applied to the two log-polar images, therefore
returns $(\theta, \ln s)$ as an ordinary translation (Reddy & Chatterji, 1996).

Implementation details, each for a stated reason:

* **Square crop.** The DFT of a non-square crop has different frequency
  spacing on the two axes, so an image rotation is not a rotation of the
  discrete spectrum. The estimate uses the largest centred square in the ROI,
  of even side $N$, capped at $k_{\text{LogPolarMaxSide}} = 512$ to bound the
  cost; below $k_{\text{LogPolarMinSide}} = 64$ it is skipped.
* **Half the angle range.** The spectrum of a real image is point-symmetric,
  $|F(\bm{k})| = |F(-\bm{k})|$, so $\phi$ is sampled over $[0, \pi)$ in 256
  rows. That makes the $\phi$ axis exactly periodic, which is what the FFT
  inside the correlation assumes. It also means $\theta$ is recovered modulo
  $180°$, i.e. in $(-90°, 90°)$.
* **Radial band.** $\rho$ is sampled in 128 columns, log-spaced between
  $N/16$ and $0.45N$. The low end excludes the DC term and the Hann window's
  leakage, where the angle is also barely resolved; the high end stays inside
  the Nyquist circle. The correlation window is a Hann taper along $\rho$ only,
  since tapering the periodic $\phi$ axis would only discard signal.
* **$\log(1 + |F|)$.** This compresses the dynamic range so the strongest low
  frequencies do not dominate the correlation.

With $\theta$ and $s$ known, the deformed image is resampled at
$\bm{c} + sR(\bm{x} - \bm{c})$. By the model above this leaves a pure
translation $\bm{t}' = (sR)^{-1}\bm{t}$ against the reference. The §8.1
correlation then measures $\bm{t}'$, and $\bm{t} = sR\,\bm{t}'$. The resampling
reads from the whole image, so pixels that rotated out of the ROI rectangle are
still found.

The seed is then the full first-order field

$$ \bm{u}_0(\bm{p}) = \bm{t} + (sR - I)(\bm{p} - \bm{c}), \qquad \begin{pmatrix}u_x & u_y\\ v_x & v_y\end{pmatrix}_0 = sR - I $$

which gives IC-GN both the displacement the rotation puts at each anchor and
the matching gradients.

**When the similarity applies.** $sR - I$ is itself a scaled rotation, so its
operator norm is $\sqrt{(s\cos\theta - 1)^2 + (s\sin\theta)^2}$. Multiplied by
the ROI half-diagonal, that is how far the bare translation is wrong at the ROI
corners. `phase_correlate_similarity` returns the similarity only when

1. that corner error exceeds $k_{\text{SimilarityMinCornerPx}} = 4$ px, and
2. the translation correlation on the rotation-corrected image has a higher
   peak than the uncorrected one.

Otherwise it returns the §8.1 translation, bit for bit, so small motion gives
exactly the translation-only seed. The 4 px threshold sits above the 2.8 px
that FullFieldGolden's affine scenario implies, where IC-GN pulls in from the
translation alone. It sits below the ~8 px at which 2° on a 320 px ROI already
costs CPU. The second condition guards against a spurious estimate on content
the log-polar correlation reads poorly.

**Why a full model must fit the lattice.** A third condition applies to any
full global model, and the solve applies it to the homography of §8.1.2: the
anchor lattice the model seeds (§8.2–8.3) must route FULL and keep at least
$k_{\text{SimilarityKeepFraction}} = 90\%$ of the anchors it attempts.
Otherwise the lattice is solved again from the translation alone, and that
lattice is used.

The similarity shows why. It cannot describe an anisotropic stretch, yet
under a strong one the log-polar correlation still returns a peak (−5.8° ×
0.92 for a 45% uniaxial stretch; about 90° once noise is added). The stretch
also weakens the translation peak, so condition 2 lets the estimate through.
Anchors seeded from that wrong model can converge, as a cluster, onto the
wrong speckle. Such a cluster agrees with itself, so neither the ZNSSD gate
nor a median test of residuals from that same model rejects it. Counting kept
anchors cannot choose between the two lattices either: in
`SeedBench.StretchSeedsWithoutBlunders`, measured with the similarity, the
wrong lattice kept 123 anchors and the translation-only one 86. A lattice that
keeps almost every anchor is the evidence that the model fits; a rotation it
fits keeps ~100%.

On pure translation the estimate costs one $N^2$ DFT and a 256×128
correlation, about 1.3 ms serial on a 320 px ROI.

**Where the estimate does not reach.** The correlation needs spectral signal
above the noise in both frames. On a 110 px ROI at $\sigma = 2$ grey-level
noise it finds no peak at all (response ≈ 0 from 5° to 45°). On a 200 px ROI
at the same noise it is within 0.05° at every angle tested. Below that size,
noisy frames therefore get the translation only.

#### 8.1.2 Global Homography from AKAZE Matches

The §8.1.1 model has two blind spots that follow from its construction. The
point-symmetric spectrum confines $\theta$ to $(-90°, 90°)$, so a half turn
reads as no turn. A similarity cannot describe stretch, so it misreads strong
stretch as rotation and scale. The solve therefore fits a planar homography to
feature matches:

$$ \bm{q} = \pi\!\left(H \begin{pmatrix}\bm{p}\\1\end{pmatrix}\right), \qquad \pi(x, y, w) = (x/w,\ y/w) $$

The homography contains translation, rotation at any angle, scale,
anisotropic stretch and shear. It has no angle ambiguity, because it is fitted
to correspondences, not to a spectrum.

**Matches.** AKAZE keypoints and binary (MLDB) descriptors are detected in
both frames, on an area-downsampled copy at scale
$\sigma \in \{0.25, 0.5, 1\}$, over the ROI padded by
$k_{\text{AkazePadPx}} = 40$ px of the working scale. Descriptors are matched
by Hamming distance with Lowe's ratio test ($d_1 < 0.75\,d_2$). Matches whose
reference keypoint falls outside the ROI mask are dropped, and $H$ is fitted
by RANSAC with a 3 px threshold at the working scale. Fitted at scale
$\sigma$ on crop-local coordinates $\bm{p}' = \sigma(\bm{p} - \bm{o})$, the
image-space model is

$$ H = T^{-1} H' T, \qquad T = \begin{pmatrix}\sigma & 0 & -\sigma o_x\\ 0 & \sigma & -\sigma o_y\\ 0 & 0 & 1\end{pmatrix} $$

**Scale.** The engine uses the coarsest scale with at least
$k_{\text{AkazeGoodInliers}} = 25$ inliers. Failing that, it uses the scale
with the most inliers if that scale has $k_{\text{AkazeMinInliers}} = 10$;
otherwise there is no model. The coarse scales are cheap and see the large
speckle of a big ROI. A small ROI has too few pixels at $0.25\times$ and
settles at $0.5\times$ or $1\times$. The scale a solve settles on is the first
one tried on the next solve; coarser ones are tried only if nothing from there
up fits.

**Seed.** At each anchor $\bm{p}$, with $(X, Y, w)^\top = H(\bm{p}, 1)^\top$,

$$ \bm{u}_0(\bm{p}) = \pi\big(H(\bm{p}, 1)^\top\big) - \bm{p}, \qquad
\nabla\bm{u}_0 = \frac{1}{w}\begin{pmatrix} h_{11} - X h_{31}/w & h_{12} - X h_{32}/w\\ h_{21} - Y h_{31}/w & h_{22} - Y h_{32}/w \end{pmatrix} - I $$

i.e. the displacement and the Jacobian of the projective map minus the
identity. The full model seeds the lattice only when it moves a ROI corner
more than $k_{\text{SimilarityMinCornerPx}} = 4$ px away from where its
translation at the ROI centre puts it. Below that, every anchor starts from
that translation. The median test of §8.3 runs on residuals from the model.
A model lattice that is not FULL, or keeps less than
$k_{\text{SimilarityKeepFraction}} = 90\%$, is replaced by the
translation-only lattice, for the reason given under §8.1.1; a rejected model
therefore costs a second lattice solve. Without a model the lattice searches
for its own offsets (`INIT_AUTO_SEARCH`, §7.4).

**Refit.** The anchors that survive the median test are good to about 0.01 px,
the keypoints only to the RANSAC tolerance at the working scale. $H$ is
therefore refitted to the anchor correspondences (RANSAC, 1 px,
$\ge 8$ anchors). The refitted model is the guess for the grid nodes the
Delaunay mesh of §8.4 does not reach, which IC-GN then solves directly.

**Why the keypoints are not mesh vertices.** §8.4 derives gradients from
triangle vertices, so keypoint errors of $e \sim 1$ px on short, clustered
edges would become large gradient errors. Here the keypoints only fit an
eight-parameter model, which averages that error over all inliers, and the
mesh vertices remain the IC-GN anchors.

**Incremental reuse.** The reference's keypoints are kept per scale with the
reference. When `IncrementalSession` promotes the deformed frame to reference,
the keypoints just detected on it are moved into the new padded ROI's
coordinates, $\bm{p}'' = \bm{p}' + \sigma(\bm{o} - \bm{o}'')$, and kept if
inside, provided the old crop covers at least 90% of the new one. A reference
is thus detected once, as a deformed frame.

**Where it does not reach.** AKAZE needs texture that survives the deformation
at some scale. Under a 45% stretch the descriptors no longer match and there
is no model, so the lattice searches. On the tensile video's 80 px ROI,
$0.25\times$ and $0.5\times$ find too few inliers and the model comes from
$1\times$.

### 8.2 Anchor Lattice

A regular lattice of ROI grid nodes with stride

$$ s = \max\left(1,\ \operatorname{round}\sqrt{\frac{N_x N_y}{N_{\text{target}}}}\right) $$

($N_{\text{target}}$ = `tuning::kAnchorTarget` = 256) is solved by the
engine's own IC-GN. It starts from the global model of §8.1.2: under the full
model, at $\bm{u}_0(\bm{p})$ with its gradients. With a model, anchors run
IC-GN only (`INIT_NO_SIMPLEX`); without one, each anchor runs the search of
§7.4. The last row and column are always included, so the convex hull reaches
the ROI boundary. Because the anchors coincide with grid nodes they reuse the
pooled Hessians of §6, and a node whose ZNSSD clears the result gate is
retained as a **final** measurement rather than being re-solved.

Each axis is then adjusted on its own. An axis carrying fewer than three
lattice indices leaves the mesh with no interior edge along it, so each axis
reduces its own stride until it holds at least three. The pair is then
coarsened again until the node count is back within the budget $s$ implied.
A long thin ROI (a beam, a weld seam) thus keeps its short-axis floor without
a lattice several times $N_{\text{target}}$. On a square ROI both axes keep
$s$ and nothing changes.

Two thresholds apply, because a seed and a result are held to different
standards:

| role | gate |
|---|---|
| mesh vertex | $C_{\text{ZNSSD}} \le k_{\text{AnchorAcceptScore}}$ (loose — the median test rejects blunders) |
| output point | $C_{\text{ZNSSD}} \le k_{\text{CorrAccept}}$ (the same bar Path A applies) |

Here $k_{\text{AnchorAcceptScore}} = 0.40$ and $k_{\text{CorrAccept}} = 0.15$,
both on the per-pixel mean of §3.1.

Neither gate publishes on its own. Both are recorded during the lattice solve
and applied only after the median test of §8.3 has run over the whole
lattice. A node that clears the result gate but fails the median test is
therefore dropped from the field as well as from the mesh. The classic
periodic-speckle blunder is exactly that combination: a confident ZNSSD on the
wrong speckle blob, whose displacement disagrees with its neighbours.
Publishing it would also hand Path B a boundary seed to flood fill from.

### 8.3 Outlier Rejection — Universal Median Test

For anchor $i$ with lattice neighbours $\mathcal{N}(i)$, reject when

$$ \left|u_i - \operatorname{median}_{j \in \mathcal{N}(i)} u_j\right| > k_{\text{AnchorMedianTol}} $$

and likewise for $v$ ($k_{\text{AnchorMedianTol}} = 1$ px). $\mathcal{N}(i)$
holds the up to eight lattice neighbours that passed the vertex gate. A
rejected anchor still counts in its neighbours' medians, so one blunder cannot
cascade along a row.

This is the Westerweel–Scarano test from PIV. It is used in preference to a global homography or affine fit because a
deforming specimen does not have an affine displacement field. A global model
rejects signal, not just blunders, whereas the median test assumes no field
shape at all. The lattice makes the neighbourhood lookup $O(1)$.

Under the full global model (§8.1.2) the test runs on the residual
$u_i - u_0(\bm{p}_i)$, not on $u_i$. A rotation is a smooth field, but
neighbours a lattice spacing $\ell$ apart differ by about $\ell\theta$: 5 px
at $15°$ and $\ell = 20$ px. On the lattice edge every neighbour lies on one
side, so the raw median is offset by that amount and an edge anchor fails a
1 px tolerance however accurate it is. Subtracting the global model removes
exactly that offset and makes no assumption about the deformation on top of
it. Without the full model the residual is the displacement itself.

A rejected anchor is neither a mesh vertex nor an output point.

The test needs $|\mathcal{N}(i)| \ge 3$ converged neighbours. An anchor with
fewer cannot be tested, and it is rejected. Such anchors appear where the
lattice is failing locally, and there the ZNSSD does not separate a blunder
from a measurement. On a real PMMA 3-point bend, the translation-only lattice
(§8.1.2's fallback) lost most anchors under the load nose. The few that
converged there matched speckle about 20 px off at ZNSSD 0.17–0.30, and the
mesh built on them guided Path A to the same wrong speckle. At the edge of a
45% stretch with no global model, untested blunders sit at ZNSSD 0.095–0.36,
inside the output gate as well as outside it. Path A or B solves such a node
instead.

The median over an even number of neighbours is the mean of the middle pair
(`util::median_inplace`, shared with §9.4). Taking the upper of the two would
bias the test towards the larger displacements.

### 8.4 Mesh Guess Field

Delaunay triangulation over the surviving anchors gives, per triangle, an affine
map from three reference vertices to their deformed positions, and hence a
6-DOF guess $(u, v, u_x, u_y, v_x, v_y)$ at every enclosed grid point.

The accuracy of the gradient components is what motivates the lattice. A
vertex position error $\varepsilon$ over a triangle edge of length $L$
propagates to a gradient error of order $\varepsilon / L$. Keypoint detectors
give $\varepsilon \sim 1$ px on short, clustered edges; IC-GN anchors give
$\varepsilon \sim 0.01$ px on edges fixed at $s \cdot \text{step}$. On real
speckle this measured a 5–17× improvement in
$\|\nabla\bm{u} - \nabla\bm{u}_{\text{true}}\|_F$
([SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md)).

---

## 9. Strain Tensor Calculation

### 9.1 Displacement Gradient Tensor

From the displacement field $\mathbf{u}(x, y) = (u, v)^T$, define the
displacement gradient tensor:
$$
\nabla\mathbf{u} = \begin{pmatrix} \partial u/\partial x & \partial u/\partial y \\ \partial v/\partial x & \partial v/\partial y \end{pmatrix}, \qquad \mathbf{F} = \mathbf{I} + \nabla\mathbf{u}
$$
where $\mathbf{F}$ is the deformation gradient.

### 9.2 Green-Lagrange Strain Tensor

The Green-Lagrange strain tensor accounts for finite deformations:
$$
\mathbf{E} = \tfrac{1}{2}\left(\mathbf{F}^{\top}\mathbf{F} - \mathbf{I}\right) = \tfrac{1}{2}\left(\nabla\mathbf{u} + \nabla\mathbf{u}^{\top} + \nabla\mathbf{u}^{\top}\nabla\mathbf{u}\right)
$$

**In component form:**
$$
\begin{aligned}
E_{xx} &= \frac{\partial u}{\partial x} + \frac{1}{2}\left[\left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial v}{\partial x}\right)^2\right] \\
E_{yy} &= \frac{\partial v}{\partial y} + \frac{1}{2}\left[\left(\frac{\partial u}{\partial y}\right)^2 + \left(\frac{\partial v}{\partial y}\right)^2\right] \\
E_{xy} &= \frac{1}{2}\left[\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} + \frac{\partial u}{\partial x}\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\frac{\partial v}{\partial y}\right]
\end{aligned}
$$
These are the `exx`, `eyy` and `exy` the engine outputs. $E_{xy}$ is the
tensor shear component, half the engineering shear strain.

**For Small Strain ($|\partial u_i / \partial x_j| \ll 1$):**
Linearized form (Cauchy strain):
$$
\varepsilon_{xx} = \frac{\partial u}{\partial x}, \qquad \varepsilon_{yy} = \frac{\partial v}{\partial y}, \qquad \varepsilon_{xy} = \frac{1}{2}\left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right)
$$

### 9.3 Virtual Strain Gauge (VSG)

**Motivation:** Displacement data from DIC is noisy. Direct differentiation amplifies noise.

**Method:** Fit a plane to the displacement data over a circular window, then differentiate it.

**Model (linear, a plane per component):** $u(x, y) = c_0 + c_1 x + c_2 y$,
likewise $v$; $\partial u / \partial x = c_1$, $\partial u / \partial y = c_2$.
Here $(x, y)$ are offsets from the window centre in pixels (grid offsets
times the step).

**Least-Squares Fitting:**
Given $n$ displacement measurements $(x_k, y_k, u_k)$ at the valid grid nodes
with $x_k^2 + y_k^2 \le R^2$, where $R$ is half the VSG window size:

**Design matrix:**
$$
\mathbf{A} = \begin{pmatrix} 1 & x_1 & y_1 \\ \vdots & \vdots & \vdots \\ 1 & x_n & y_n \end{pmatrix}, \qquad \mathbf{c} = (c_0, c_1, c_2)^T
$$

**Solve normal equations:**
$$
\mathbf{c} = (\mathbf{A}^{\top}\mathbf{A})^{-1}\,\mathbf{A}^{\top}\mathbf{u}
$$
where $\mathbf{u} = (u_1, \dots, u_n)^T$, and likewise for $v$.

**Gradient Recovery:**
From the fitted plane, $\partial u/\partial x = c_1$ and
$\partial u/\partial y = c_2$. A plane has a constant gradient, so this is the
gradient at the window centre, which feeds §9.2.

**Window rules.** A node gets a strain value only if its window holds at least
90% of the nodes of a full circular window, and at least 3. The fit is skipped
when the reciprocal 1-norm condition number of $\mathbf{A}^{\top}\mathbf{A}$
is below $10^{-12}$ (for example, nodes on a line). A skipped node keeps the
sentinel `tuning::kStrainUninitSentinel`.

**Implementation:**

```cpp
// src/strain/strain_calculator.cpp, StrainCalculator::compute_vsg_strain
// (simplified)
for (int dy = -grid_rad; dy <= grid_rad; ++dy) {
    for (int dx = -grid_rad; dx <= grid_rad; ++dx) {
        double phys_dx = dx * disp.step;
        double phys_dy = dy * disp.step;
        
        if ((phys_dx*phys_dx + phys_dy*phys_dy) <= radius_sq) {
            Eigen::Vector3d a(1.0, phys_dx, phys_dy);
            AtA += a * a.transpose();
            AtU += a * disp.u[neighbor];
        }
    }
}

// At least 90% of a full window must be present.
if (fill_ratio >= 0.90 && valid_pts >= 3) {
    // Explicit 3x3 inverse, with its reciprocal condition number computed
    // LAPACK-style (1 / (||AtA||_1 ||AtA^-1||_1)); a window with rcond < 1e-12
    // (nodes on a line) is skipped and keeps the sentinel.
    Eigen::Matrix3d AtA_inv = AtA.inverse();
    Eigen::Vector3d Cu = AtA_inv * AtU;
    double dudx = Cu(1);
    double dudy = Cu(2);
}

```

### 9.4 Displacement Outlier Rejection

**Motivation:** A wrong match can pass the ZNSSD check, and on real images
wrong matches come in clusters: a dozen neighbouring points all off by about
the same amount (≈ 20 px near a bending beam's edge). One such point inside a
VSG window tilts the whole plane fit, so strain there can be off by
10⁵–10⁶ µε.

**Method:** The normalized median test (Westerweel & Scarano, 2005), per
component, before the VSG fit. For a point with value `x` (u or v) and accepted
neighbours `x_i` in its 5x5 block:

```
m  = median(x_i)
rm = median(|x_i − m|)
r  = |x − m| / (rm + ε)          ε = 0.1 px
```

The point is rejected when `r > 3` for u or v. A point with fewer than 3
accepted neighbours is not tested. The test repeats, with rejected points
removed from every neighbourhood, until a pass rejects nothing (at most 5
passes). Inside a cluster the wrong values are the local majority, so the
passes peel the cluster from its edge inward. A straight jump (a crack)
survives, because each side is its own majority.

**Where:** `StrainCalculator::reject_displacement_outliers`, called from
`pack_full_field_output`; constants `tuning::kOutlier*`. The points it rejects
are counted in metrics slot 23.

**Validation (real images):** On the 3-point-bending PMMA series it rejects
211 points over 33 frames, and nothing in 26 of them; each rejected point
checked sat 4–23 px from its neighbours' median. Strain RMSE against the
authors' field over all frames, at a 45 px VSG, drops from 915 / 3874 / 2117 µε
(exx / eyy / exy) to 412 / 394 / 392; at 15 px, from about 20,000 to about 900.
On a steel tensile series (40 frames) it rejects nothing. Threshold 2 rejects
the same points on the PMMA series (and 8 on the steel's last frame, by the
fracture); 3 leaves more room for noise.

### 9.5 Incremental (Updated-Reference) DIC

`include/semper/incremental.hpp`, `src/pipeline/incremental_session.cpp`,
`src/pipeline/incremental_math.cpp`.

By default every frame t is correlated against the first image. Under large
rotation or strain the subsets of frame 0 stop matching frame t: the affine
shape function covers less of the real deformation, and the speckle may
change appearance. Incremental DIC correlates frame t against a recent frame k
instead and composes the result.

**Composition.** Let `X0` be a frame-0 grid node and `U(X0, k)` its total
displacement at the reference frame k, so the material point sits at
`x_k = X0 + U(X0, k)` in image k. The increment field `u_k` is solved on a
regular grid in image k. Then

```
U(X0, t) = U(X0, k) + u_k( X0 + U(X0, k) )
```

The output stays on the frame-0 grid, as total displacement, so the VSG and
Green-Lagrange strain of §9.3 apply to `U(·, t)` unchanged.

**Sampling `u_k` between nodes.** `x_k` is generally not a node of the frame-k
grid:

* If all four corners of the grid cell containing `x_k` are valid, `u_k` is
  interpolated bilinearly. The ZNSSD reported for the point is the worst of
  the four.
* Otherwise, the nearest valid node `n` within `1.5` grid cells
  (`tuning::kIncTaylorMaxCells`) gives a first-order expansion from its solved
  displacement gradient, the same expansion Path B uses for its guesses:
  `u_k(x) ≈ u_n + ∇u_n · (x − x_n)`. Ties go to the lowest row-major index, so
  the result does not depend on thread timing.
* Further away, the node **coasts** (below). A node that cannot coast either is
  lost. A lost node never returns, since there is no `U(X0, k)` to compose
  onto.

Increment nodes go through the median test of §9.4 before they are sampled. The
same test on the total field only drops points from that frame's output and
is not fed back into tracking, because a false positive there would be
permanent.

**Coasting across a gap.** Some bands of an image cannot be correlated for a
while even though the material in them is intact: a stationary reflection or
glare, a translucent specimen passing from a light background onto a dark
rig, a necking edge where the subset starts to take in background. A node in
such a band gets no increment, while the nodes on either side do. Rather than
drop it, the engine estimates its increment by a weighted least-squares plane
`u = a + b·dx + c·dy` (and `v` likewise) through the solved increment nodes
within `kIncCoastRadiusCells` = 8 cells, with weights `1 / (1 + (d/step)²)`.

The fit needs `kIncCoastMinNodes` = 6 nodes whose weighted spread is at least
one step along both principal axes, so it is constrained in both directions. A
coasting node is carried to the next reference but not reported. It is
dropped once the fitted increments have carried it
`kIncCoastMaxTravelSubsets` = 2 subsets, or once it leaves the image. Only a
node that has been measured coasts. A node that frame 1 cannot measure may sit
in a gap, or on texture too weak to correlate at all, and nothing could ever
confirm it, so it is not followed.

**Re-anchoring.** A plane fitted over a gap a subset wide drifts. Measured on a
synthetic band crossing, the drift is up to about 20% of the distance coasted,
0.5 to 2 px for a crossing. Once the node is measured again, that measurement
starts from where coasting left it, so the drift would remain in it. A node
that coasted more than `kIncCoastTrustedTravelSubsets` = 0.25 subset is
therefore reported again only after a direct check:

* While nodes are measured, each clean one within reach of a gap keeps an
  **anchor**: its subset, cut from a reference where every node inside that
  subset solved. "Within reach" means an unsolved or off-grid position within
  two subsets plus two steps. Only a ~1.5 KB grey patch is kept, with
  `kIncAnchorMarginPx` = 10 px of margin so its gradients do not see the
  patch edge.
* A measured node whose own subset is no longer clean keeps the anchor it has,
  which was cut before it reached the gap.
* When the coasted node can be sampled again and its subset here is clean, the
  anchor subset is solved by ICGN straight onto the current frame, with a
  simplex rescue as in Path B. It starts from where coasting put it.
* The match must pass `kCorrAccept`, and it may move the node by at most
  `max(1 px, 0.3 × distance coasted)`. Its position, through the matched
  affine map, replaces the coasted one.
* Until a match succeeds, the node moves on its measured increments,
  unreported. Those do not count against its coasting distance.

On the band crossing, nodes re-anchored on the far side are within 0.03 px
(median) and 0.20 px (worst) of the truth, against up to 2.3 px without
re-anchoring. `tracked_points` counts coasting and unconfirmed nodes; the
output holds only measured ones.

**Moving the reference.** Every `N` frames (`update_interval`), frame t
becomes the new reference. With N > 1 the reference also moves early, to the
frame just solved, when that frame reports `kIncEarlyUpdateDrop` = 5% fewer
points than the first frame solved against the current reference. Such a drop
means the increment has outgrown what the images still correlate over, as with
fast loading, necking or whitening. On the tensile video the early update
lifts the late-test minimum at N = 20 from 337 to 397 points, and changes
nothing at N = 5.

Neither update moves the reference to a frame that reports fewer than
`kIncUpdateMinReported` = half the points of that first frame. Before the
first frame, and always with N = 1, the comparison is with half of what the
reference itself reported when it was solved, so a failed first frame is
caught too. Such a frame is a failed solve, not a grown increment. On a 110 px
ROI at σ = 2 turned 6–10° from its reference, the seed occasionally locks onto
a wrong solution and reports one or two points 20–40 px off; moving onto that
frame would keep only those. The reference stays, the next frame is solved
against it, and an update that was due falls to that frame. In IncBench S4
(2°/frame, σ = 2) this keeps N = 5 at 0.19 px and 94% coverage at 90°, where
adopting the failed frames ended 12.7 px off.

The new reference's image is already prepared for correlation (intensities
and gradients), so it is handed over without decoding or preparing it again.

**Warping the mask.** The ROI mask has to follow the material, so the frame-0
mask is warped forward. Every material pixel of the frame-0 window (the grid
widened by `kIncMaskExtrapCells` = 3 cells plus half a subset) is sampled on a
sub-pixel lattice. Each sample `X` is moved to `X + U(X)`, and the pixel it
lands in is set. `U` is the bilinear total field with invalid nodes filled by
neighbour averaging. Outside the grid it is extended by the plane through the
nearest grid point, which is exact for any affine motion. The lattice uses 2
samples per pixel per axis, more where the field stretches, so neighbouring
samples land within 0.7 px of each other (up to `kIncMaskMaxSamplesPerPx` =
8). A 3×3 morphological close then fills the pinholes.

A forward warp has no inverse to solve, so it has no wrong branch to follow.
Solving `X + U(X) = p` per pixel by Newton does have one under large rotation.
A field held constant past the grid is a pure translation, which gives pixels
outside a turned body a second, false preimage. Where a few drifted nodes fold
the field there are false branches too, and a row's warm start can follow them
across the material. Extending the edge cells bilinearly, instead of by a
plane, is worse still: the bilinear cross term grows quadratically off the
corners. With the forward warp, a fold only makes samples overlap locally.
The new reference's grid spans the tracked nodes' current positions, plus one
step on the far sides.

Without a user mask, the frame-0 material is the ROI rectangle widened by
`ceil(half·√2) + 1` px, where `half` is half the subset size, rather than by
`half` alone. Subsets stay axis-aligned on every reference while the material
turns, so at 45° a node's subset reaches `half·√2` from its centre. With only
`half` of material around the ROI, the edge nodes of a rotating ROI fall out
of the scan, get sampled by Taylor fallback, and the losses spread inwards.
With a user mask the mask is used as drawn, so a rotating masked region loses
its edge ring the same way: draw it a subset wider if it will turn.

**Why the error grows, and what N does.** Each solve carries an interpolation
bias that depends on the sub-pixel part of the displacement (§10.1). Against a
fixed reference it is paid once. Incrementally it is paid once per reference
update. On slow, smooth loading the increments are small sub-pixel steps of
similar size, so the bias has one sign and adds up roughly linearly. Image
noise in each new reference adds a random-walk term on top. Measured on smooth
synthetic speckle over 160 frames of 0.1% stretch: rms 0.25 px at N = 1 and
0.15 px at N = 5 with Keys 4×4, 0.21 and 0.11 px with 6×6, against 0.03 px for
the fixed reference. On DICe's OHT ladder the fixed reference also wins
(0.003 vs 0.058 px after four updates; see TESTING.md, `DiceIncremental`).

In practice:

* For small or moderate deformation the **fixed reference is more accurate**.
  Use incremental only when it stops correlating: rigid motion past the seeding
  capture range, rotation past ~5–15° on noisy images with a small ROI (with a
  clean pattern, rotation-aware seeding takes the fixed reference through 90°),
  or speckle that changes appearance.
* A larger N means fewer updates, so less accumulated bias, as long as frame
  t still correlates with frame t − N.
* The 6×6 interpolator lowers the per-update bias.

[INCREMENTAL_BENCHMARK.md](INCREMENTAL_BENCHMARK.md) measures these
trade-offs across scenarios.

---

## 10. Error Analysis & Uncertainty

### 10.1 Sources of Error

* **Systematic Errors:**
  * **Interpolation Bias:** Bicubic interpolation introduces ~0.001–0.003 px bias, which depends on the sub-pixel part of the displacement.
  * **Discretization Error:** The finite-difference gradients of §4.3 have $O(h^4)$ truncation error.
  * **Subset Size Effect:** Smaller subsets give higher noise; larger subsets lose spatial resolution.
* **Random Errors:**
  * **Image Noise:** Sensor noise, of standard deviation $\sigma$ grey levels (§10.2).
  * **Speckle Quality:** Low contrast gives poor tracking.
  * **Out-of-Plane Motion:** Violates the 2D assumption.

### 10.2 Displacement Uncertainty

**Cramér-Rao Lower Bound:**
For a pure translation, with independent noise of standard deviation
$\sigma$ in both images, the first-order standard deviation of the
displacement error is
$$
\sigma_u \approx \frac{\sqrt{2}\,\sigma}{\sqrt{\sum_{\mathbf{x}_i \in S} f_x(\mathbf{x}_i)^2}}
$$
and likewise $\sigma_v$ with $f_y$. The factor $\sqrt{2}$ appears because
both images are noisy.

where:

* $\sigma$: Image noise standard deviation
* $|S|$: Number of pixels in subset, over which the sum runs
* $\sum f_x^2$: Sum of squared gradients (texture strength); at constant texture it grows with $|S|$, so $\sigma_u \propto |S|^{-1/2}$

**Experimental Validation:** Measured RMSE = 0.008 px on synthetic translation
tests. Current measurements are in [TESTING.md](TESTING.md) and
[SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md).

### 10.3 Strain Uncertainty

Strain is computed from displacement gradients (§9.3). Error propagation: for
a full window, which is symmetric about its centre
($\sum x_k = \sum y_k = \sum x_k y_k = 0$), $\mathbf{A}^{\top}\mathbf{A}$ is
diagonal. With independent displacement errors of standard deviation
$\sigma_u$, the gradient error is then
$$
\sigma_{\partial u/\partial x} = \frac{\sigma_u}{\sqrt{\sum_k x_k^2}} \approx \frac{8\,h\,\sigma_u}{\sqrt{\pi}\,L^2}
$$
where $L$ is the gauge length (strain window size) and $h$ the grid step. The
approximation replaces the sum by an integral over a disc of diameter $L$ and
holds once the window contains many nodes. Neighbouring subsets overlap, so
real displacement errors are correlated; treat the result as an estimate.

**Noise Reduction Strategy:** At a fixed step, strain noise falls as $L^{-2}$:
doubling the VSG window cuts it about fourfold, at the cost of spatial
resolution.

---

## 11. Implementation Validation: DIC Challenge

The engine was checked against datasets of the Society for Experimental
Mechanics (SEM) DIC Challenge, to test its ZNSSD and IC-GN implementations
under non-uniform deformation. Current measurements on these sets, per build,
are in [SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md) §4.3; the test suites are
described in [TESTING.md](TESTING.md).

### 11.1 Benchmark Dataset: Sample 14 L5

The check used **Sample 14 L5**. Its displacement is a sinusoid along X
(amplitude 0.1 px, period ~205 px), so the strain varies continuously across
the image. It tests subset matching and interpolation where the displacement
changes within a subset.

**Test 1: Sinusoidal Displacement Accuracy**
The computed U-displacement profile was compared with the analytical
sinusoidal ground truth, using the Keys bicubic interpolator of §4.2.

| Metric | Measured Value |
| :--- | :--- |
| **Test Case** | SEM Challenge Sample 14 L5 |
| **Displacement Field** | Sinusoidal ($U$ along $X$) |
| **Root Mean Square Error (RMSE)** | **0.0078 px** |
| **ZNSSD Correlation Threshold** | < 0.25 |

The RMSE is below 0.01 px. The 0.25 threshold is the one recorded for this
test; the engine's default result gate is 0.15 (§3.1).

---

**Test 2: Strain Accuracy ε_xx**
Strain was computed from the sinusoidal displacement field with the
**Virtual Strain Gauge (VSG)** of §9.3. This tests accuracy through numerical
differentiation.

| Metric | Result                       |
| :--- |:-----------------------------|
| **Strain Profile** | Sinusoidal Gradient **ε_xx** |
| **Mean Accuracy** | Within **± 400 µε**          |
| **Absolute Error** | ~0.0004                      |

The VSG strain follows the oscillating ε_xx of Sample 14 to within ±400 µε.

---

### 11.2 Experimental Summary

* **Benchmark:** Comparison against **DICe** (Digital Image Correlation Engine, Sandia National Laboratories).
* **Displacement:** 0.0078 px RMSE on the L5 sinusoid.
* **Hardware:** An Android NDK build.

## 12. References

**Core Algorithms**

1. Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 years on: A unifying framework. *International Journal of Computer Vision*, 56(3), 221-255.
2. Pan, B., et al. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review. *Measurement Science and Technology*, 20(6), 062001.
3. Blaber, J., Adair, B., & Antoniou, A. (2015). Ncorr: Open-source 2D digital image correlation matlab software. *Experimental Mechanics*, 55(6), 1105-1122.

**DIC Theory**

4.  Sutton, M. A., et al. (2009). *Image Correlation for Shape, Motion and Deformation Measurements*. Springer.
5.  Schreier, H., Orteu, J. J., & Sutton, M. A. (2009). *Image Correlation for Shape, Motion and Deformation Measurements: Basic Concepts, Theory and Applications*. Springer.

**Strain Calculation**

6.  Reu, P. L., et al. (2018). DIC Challenge: Developing images and guidelines for evaluating accuracy and resolution of 2D analyses. *Experimental Mechanics*, 58(7), 1067-1099.
7.  Lehoucq, R. B., & Silling, S. A. (2008). Force flux and the peridynamic stress tensor. *Journal of the Mechanics and Physics of Solids*, 56(4), 1566-1577.

**Interpolation**

8.  Keys, R. (1981). Cubic convolution interpolation for digital image processing. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 29(6), 1153-1160.

**Feature Matching**

9.  Alcantarilla, P. F., et al. (2013). Fast explicit diffusion for accelerated features in nonlinear scale spaces. *British Machine Vision Conference (BMVC)*.

**Optimization, Registration and Outlier Detection**

10. Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308-313.
11. Reddy, B. S., & Chatterji, B. N. (1996). An FFT-based technique for translation, rotation, and scale-invariant image registration. *IEEE Transactions on Image Processing*, 5(8), 1266-1271.
12. Westerweel, J., & Scarano, F. (2005). Universal outlier detection for PIV data. *Experiments in Fluids*, 39(6), 1096-1100.

---

### Appendix A: Notation Summary

| Symbol | LaTeX | Description |
| --- | --- | --- |
| $\mathbf{x}$ | `\mathbf{x}` | Position vector |
| $\mathbf{p}$ | `\mathbf{p}` | Parameter vector |
| $\nabla f$ | `\nabla f` | Gradient of $f$ |
| $W(\mathbf{x}; \mathbf{p})$ | `W(\mathbf{x}; \mathbf{p})` | Warp function |
| $\lVert\cdot\rVert$ | `\lVert\cdot\rVert` | Norm |
| $\epsilon_{ij}$ | `\epsilon_{ij}` | Strain tensor component |
| $\sum$ | `\sum` | Summation |
| $\int$ | `\int` | Integration |
| $\partial / \partial x$ | `\partial / \partial x` | Partial derivative |

### Appendix B: Unit Conversions

| Quantity | SI Unit | Engineering Unit |
| --- | --- | --- |
| **Displacement** | pixels | mm = pixels × pixel_size |
| **Strain** | dimensionless | με = strain × 10⁶ |
| **Stress** | N/m² = Pa | MPa = Pa × 10⁻⁶ |
| **Image intensity** | [0, 255] | 8-bit grayscale |

**Example:** For a camera with 5 μm pixel size:

* Measured displacement: 0.5 pixels = 2.5 μm
* Measured strain: 0.001 = 1000 με
