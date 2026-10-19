// =====================================================================
// SUITE: SubsetPrecomputer — native/preprocessing/SubsetPrecomputer.cpp
//
// The precomputer converts a reference-image patch into everything the
// ICGN solver needs: normalized intensities, gradients, the 6×6
// Gauss-Newton Hessian and its inverse, the steepest-descent images,
// and (post-SIMD-refactor) their SoA mirror `sdi_planes`.
//
// Key regression guarded here: the AoS steepest_descent_images and the
// SoA sdi_planes MUST stay bit-consistent — the SIMD hot loop reads the
// planes while the masked slow path reads the Eigen vectors. Drift
// between them silently corrupts solutions.
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"
#include <semper/subset.hpp>

using Semper::Image;
using Semper::SubsetData;
using Semper::SubsetPrecomputer;
using Semper::CachedHessianData;

namespace {
    constexpr int W = 128, H = 128, DIM = 21;

    Image test_image() {
        dictest::SpeckleField field(77, W, H);
        return dictest::make_reference_image(field, W, H);
    }
} // namespace

TEST_CASE(SubsetPrecomputer, MeanAndStdMatchManualComputation) {
    Image img = test_image();
    SubsetData data;
    SubsetPrecomputer::precompute_subset(data, img, 64, 64, DIM);
    REQUIRE(data.is_initialized);

    int n = DIM * DIM;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += data.ref_intensities[i];
    double mean = sum / n;
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = data.ref_intensities[i] - mean;
        ss += d * d;
    }
    double stddev = std::sqrt(ss / n);

    CHECK_REL(data.mean_intensity, mean, 1e-4);
    CHECK_REL(data.std_dev, stddev, 1e-4);

    // Normalized intensities: (I - mean) / std, zero-mean and unit-RMS
    double nsum = 0.0, nss = 0.0;
    for (int i = 0; i < n; ++i) {
        nsum += data.norm_ref_intensities[i];
        nss += data.norm_ref_intensities[i] * data.norm_ref_intensities[i];
    }
    CHECK_NEAR(nsum / n, 0.0, 1e-3);
    CHECK_NEAR(std::sqrt(nss / n), 1.0, 1e-3);
}

// 🚀 THE SIMD-REFACTOR REGRESSION GUARD
TEST_CASE(SubsetPrecomputer, SdiPlanesMirrorSteepestDescentImages) {
    Image img = test_image();
    SubsetData data;
    SubsetPrecomputer::precompute_subset(data, img, 64, 64, DIM);
    REQUIRE(data.is_initialized);

    size_t n = (size_t) DIM * DIM;
    REQUIRE(data.sdi_planes.size() == n * 6);
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 6; ++k)
            CHECK_NEAR(data.sdi_planes[k * n + i],
                       data.steepest_descent_images[i](k), 1e-6);
}

TEST_CASE(SubsetPrecomputer, SteepestDescentImagesFollowDefinition) {
    Image img = test_image();
    SubsetData data;
    SubsetPrecomputer::precompute_subset(data, img, 64, 64, DIM);
    REQUIRE(data.is_initialized);

    // sd = [gx, gy, gx·x, gx·y, gy·x, gy·y] / std  (gradients pre-divided)
    int n = DIM * DIM;
    for (int i = 0; i < n; i += 37) { // spot-check a spread of pixels
        float gx = data.gx_vec[i] / data.std_dev;
        float gy = data.gy_vec[i] / data.std_dev;
        float x = data.x_offsets_f[i], y = data.y_offsets_f[i];
        const auto &sd = data.steepest_descent_images[i];
        CHECK_NEAR(sd(0), gx, 1e-5);
        CHECK_NEAR(sd(1), gy, 1e-5);
        CHECK_NEAR(sd(2), gx * x, 1e-4);
        CHECK_NEAR(sd(3), gx * y, 1e-4);
        CHECK_NEAR(sd(4), gy * x, 1e-4);
        CHECK_NEAR(sd(5), gy * y, 1e-4);
    }
}

TEST_CASE(SubsetPrecomputer, HessianIsSymmetricAndInverseIsValid) {
    Image img = test_image();
    SubsetData data;
    SubsetPrecomputer::precompute_subset(data, img, 64, 64, DIM);
    REQUIRE(data.is_initialized);

    // Symmetry (H = Σ sd·sdᵀ is symmetric by construction)
    for (int r = 0; r < 6; ++r)
        for (int c = r + 1; c < 6; ++c)
            CHECK_REL(data.H(r, c), data.H(c, r), 1e-4);

    // H_inv really is the inverse: H · H_inv ≈ I
    Eigen::Matrix<float, 6, 6> prod = data.H * data.H_inv;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            CHECK_NEAR(prod(r, c), (r == c) ? 1.0f : 0.0f, 5e-2);
}

TEST_CASE(SubsetPrecomputer, RejectsSubsetOffImageBoundary) {
    Image img = test_image();
    SubsetData data;

    SubsetPrecomputer::precompute_subset(data, img, 5, 64, DIM); // half=10 > 5
    CHECK(!data.is_initialized);

    SubsetPrecomputer::precompute_subset(data, img, 64, H - 3, DIM);
    CHECK(!data.is_initialized);
}

// A subset is (2 * (dim / 2) + 1)^2 pixels, built around its centre. An even
// dim made the loops write (dim + 1)^2 values into dim^2 buffers, and dim 0
// wrote into empty ones. run_full_field rejects such a dim up front; these
// are the second line, for every direct caller (incremental re-anchoring
// calls precompute_subset itself).
TEST_CASE(SubsetPrecomputer, EvenDimRejected) {
    Image img = test_image();
    const CachedHessianData good = SubsetPrecomputer::compute_hessian_only(img, 64, 64, DIM);
    REQUIRE(good.valid);
    for (int dim : {20, 0, 1, 2, -21}) {
        CHECK(!SubsetPrecomputer::compute_hessian_only(img, 64, 64, dim).valid);

        SubsetData data;
        SubsetPrecomputer::precompute_subset(data, img, 64, 64, dim);
        CHECK(!data.is_initialized);

        // Even with a valid pool entry, which belongs to another dim.
        SubsetData fast;
        SubsetPrecomputer::precompute_subset_fast(fast, img, 64, 64, dim, good);
        CHECK(!fast.is_initialized);
    }
}

// The cached-Hessian fast path must produce the same state as the full
// precompute — it is used interchangeably by the JNI batch pipeline.
TEST_CASE(SubsetPrecomputer, FastPathMatchesFullPrecompute) {
    Image img = test_image();

    SubsetData full;
    SubsetPrecomputer::precompute_subset(full, img, 60, 70, DIM);
    REQUIRE(full.is_initialized);

    CachedHessianData cached =
            SubsetPrecomputer::compute_hessian_only(img, 60, 70, DIM);
    REQUIRE(cached.valid);

    CHECK_REL(cached.mean_intensity, full.mean_intensity, 1e-4);
    CHECK_REL(cached.std_dev, full.std_dev, 1e-4);

    SubsetData fast;
    SubsetPrecomputer::precompute_subset_fast(fast, img, 60, 70, DIM, cached);
    REQUIRE(fast.is_initialized);

    size_t n = (size_t) DIM * DIM;
    for (size_t i = 0; i < n; i += 13) {
        CHECK_NEAR(fast.norm_ref_intensities[i], full.norm_ref_intensities[i], 1e-3);
        for (int k = 0; k < 6; ++k) {
            CHECK_NEAR(fast.steepest_descent_images[i](k),
                       full.steepest_descent_images[i](k), 1e-3);
            CHECK_NEAR(fast.sdi_planes[k * n + i], full.sdi_planes[k * n + i], 1e-3);
        }
    }
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            CHECK_REL(fast.H_inv(r, c), full.H_inv(r, c), 5e-3);
}
