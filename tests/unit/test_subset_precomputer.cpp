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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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
    // precompute_subset is compute_hessian_only + precompute_subset_fast, so
    // the two are the same bytes, not merely close (before v0.5.0 it was a
    // separate implementation, equal to ~1e-3).
    Image img = test_image();

    SubsetData full;
    SubsetPrecomputer::precompute_subset(full, img, 60, 70, DIM);
    REQUIRE(full.is_initialized);

    CachedHessianData cached =
            SubsetPrecomputer::compute_hessian_only(img, 60, 70, DIM);
    REQUIRE(cached.valid);

    SubsetData fast;
    SubsetPrecomputer::precompute_subset_fast(fast, img, 60, 70, DIM, cached);
    REQUIRE(fast.is_initialized);

    const size_t n = (size_t) DIM * DIM;
    auto same = [](const void *a, const void *b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; };
    CHECK(fast.mean_intensity == full.mean_intensity);
    CHECK(fast.std_dev == full.std_dev);
    CHECK(same(fast.norm_ref_intensities.data(), full.norm_ref_intensities.data(), n * sizeof(float)));
    CHECK(same(fast.sdi_planes.data(), full.sdi_planes.data(), 6 * n * sizeof(float)));
    CHECK(same(fast.H_inv.data(), full.H_inv.data(), sizeof(float) * 36));
    CHECK(same(fast.H.data(), full.H.data(), sizeof(float) * 36));
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 6; ++k)
            CHECK(fast.steepest_descent_images[i](k) == full.steepest_descent_images[i](k));

    // An invalid pool entry is final for the fast path.
    SubsetData none;
    SubsetPrecomputer::precompute_subset_fast(none, img, 60, 70, DIM, CachedHessianData{});
    CHECK(!none.is_initialized);
}

// The conditioning test used to be |det H| < 1e-6, and det H scales with the
// subset: the four gradient rows and columns grow as (dim/2)^2. A singular
// Hessian's rounding noise alone cleared the floor by twenty orders of
// magnitude, and past ~170 px a healthy one overflowed float. It is now an
// rcond estimate with those rows and columns divided by dim/2.
TEST_CASE(SubsetPrecomputer, LargeSubset_HinvFinite) {
    dictest::SpeckleField field(5, 320, 320);
    const Image img = dictest::make_reference_image(field, 320, 320);
    SubsetData data;
    SubsetPrecomputer::precompute_subset(data, img, 160, 160, 301);
    REQUIRE(data.is_initialized);
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) CHECK(std::fabs(data.H_inv(r, c)) < 1e30f);
    const Eigen::Matrix<float, 6, 6> prod = data.H * data.H_inv;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            CHECK_NEAR(prod(r, c), (r == c) ? 1.0f : 0.0f, 5e-2);
}

TEST_CASE(SubsetPrecomputer, IllConditioned_Rejected) {
    // Vertical stripes: no y gradient, so the v and every d/dy column of the
    // steepest-descent images vanish. The 2x2 translation check catches it.
    {
        const int w = 96, h = 96;
        std::vector<uint8_t> zero((size_t)w * h, 0);
        Image img(w, h, zero.data());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                img.intensities[(size_t)y * w + x] = 128.0f + 80.0f * std::sin(x * 0.7f);
        img.prepare_data(false);
        SubsetData data;
        SubsetPrecomputer::precompute_subset(data, img, 48, 48, 21);
        CHECK(!data.is_initialized);
        CHECK(!SubsetPrecomputer::compute_hessian_only(img, 48, 48, 21).valid);
    }
    // A pattern symmetric about the subset centre cannot see a rotation about
    // it: on x^2 + y^2 the 5-point gradients are exact, gx*y == gy*x at every
    // pixel, and the d/dy-of-u and d/dx-of-v columns coincide. The translation
    // block is perfectly conditioned, so only the full 6x6 test can catch it;
    // det H came out at +1e9 (21 px) to +8e16 (201 px), or negative, and the
    // subset was accepted with a meaningless inverse.
    for (int dim : {21, 201}) {
        const int w = dim + 40, h = dim + 40, c = w / 2;
        const float k = 0.3f * (41.0f / dim) * (41.0f / dim);
        std::vector<uint8_t> zero((size_t)w * h, 0);
        Image img(w, h, zero.data());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float dx = (float)(x - c), dy = (float)(y - c);
                img.intensities[(size_t)y * w + x] = k * (dx * dx + dy * dy);
            }
        img.prepare_data(false);
        SubsetData data;
        SubsetPrecomputer::precompute_subset(data, img, c, c, dim);
        CHECK(!data.is_initialized);
        CHECK(!SubsetPrecomputer::compute_hessian_only(img, c, c, dim).valid);
    }
}
