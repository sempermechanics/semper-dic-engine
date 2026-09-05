// =====================================================================
// SUITE: Image — native/src/math/image_processor.cpp
//
// Validates the interpolation ladder (bilinear → bicubic → Keys 6×6),
// the 4th-order central-difference gradients, and the DICe 7-tap blur.
//
// Mathematical properties exploited:
//  • All three interpolators are INTERPOLATING kernels: at integer
//    coordinates they must reproduce the pixel value exactly.
//  • Catmull-Rom (bicubic) and Keys 4th-order reproduce polynomials up
//    to degree ≥ 1 exactly → a linear ramp must interpolate exactly at
//    ANY sub-pixel position.
//  • The 5-point central difference is exact for polynomials up to
//    degree 4 → gradient of a linear ramp is the exact slope.
//  • The 7-tap Gaussian kernel is normalized → blurring a constant
//    image must return the same constant.
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"
#include <semper/image.hpp>

#include <cstdio>
#include <vector>

using Semper::Image;

namespace {

    // Build an image whose intensity is the linear ramp a + b·x + c·y.
    Image make_ramp(int w, int h, float a, float b, float c) {
        std::vector<uint8_t> dummy((size_t) w * h, 0);
        Image img(w, h, dummy.data());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                img.intensities[(size_t) y * w + x] = a + b * x + c * y;
        img.prepare_data(false);
        return img;
    }

} // namespace

TEST_CASE(Image, InterpolatorsReproducePixelValuesAtIntegerCoords) {
    dictest::SpeckleField field(11, 64, 64);
    Image img = dictest::make_reference_image(field, 64, 64);

    for (int y = 8; y < 56; y += 5) {
        for (int x = 8; x < 56; x += 5) {
            float px = img.intensities[(size_t) y * 64 + x];
            CHECK_NEAR(img.interpolate_bilinear((float) x, (float) y), px, 1e-3);
            CHECK_NEAR(img.interpolate_bicubic((float) x, (float) y), px, 1e-2);
            CHECK_NEAR(img.interpolate_keys_fourth((float) x, (float) y), px, 1e-2);
        }
    }
}

TEST_CASE(Image, InterpolatorsExactOnLinearRamp) {
    Image img = make_ramp(40, 40, 10.0f, 2.0f, -1.5f);
    auto expect = [](float x, float y) { return 10.0f + 2.0f * x - 1.5f * y; };

    // Sub-pixel positions well inside every kernel's support
    const float pts[][2] = {{10.25f, 12.75f}, {17.5f, 20.5f}, {23.9f, 8.1f}};
    for (auto &p : pts) {
        CHECK_NEAR(img.interpolate_bilinear(p[0], p[1]), expect(p[0], p[1]), 1e-3);
        CHECK_NEAR(img.interpolate_bicubic(p[0], p[1]), expect(p[0], p[1]), 1e-2);
        CHECK_NEAR(img.interpolate_keys_fourth(p[0], p[1]), expect(p[0], p[1]), 1e-2);
    }
}

TEST_CASE(Image, GradientOfLinearRampIsExactSlope) {
    Image img = make_ramp(40, 40, 50.0f, 3.0f, -2.0f);
    // Central difference valid in [2, dim-3]; sample the interior
    for (int y = 5; y < 35; y += 7) {
        for (int x = 5; x < 35; x += 7) {
            CHECK_NEAR(img.grad_x[(size_t) y * 40 + x], 3.0f, 1e-3);
            CHECK_NEAR(img.grad_y[(size_t) y * 40 + x], -2.0f, 1e-3);
        }
    }
}

TEST_CASE(Image, BlurPreservesConstantImage) {
    int w = 32, h = 32;
    std::vector<uint8_t> raw((size_t) w * h, 100);
    Image img(w, h, raw.data());
    img.prepare_data(true); // apply the DICe 7-tap Gaussian

    // The kernel must be normalized: constant in → same constant out
    // (interior only; the border ring is untouched by design).
    for (int y = 4; y < h - 4; ++y)
        for (int x = 4; x < w - 4; ++x)
            CHECK_NEAR(img.intensities[(size_t) y * w + x], 100.0f, 0.05f);
}

TEST_CASE(Image, BilinearOutOfBoundsReturnsZero) {
    Image img = make_ramp(20, 20, 100.0f, 0.0f, 0.0f);
    CHECK_NEAR(img.interpolate_bilinear(-1.0f, 5.0f), 0.0f, 1e-6);
    CHECK_NEAR(img.interpolate_bilinear(5.0f, -0.5f), 0.0f, 1e-6);
    CHECK_NEAR(img.interpolate_bilinear(19.0f, 5.0f), 0.0f, 1e-6);  // >= w-1.5
    CHECK_NEAR(img.interpolate_bilinear(5.0f, 19.5f), 0.0f, 1e-6);
}

TEST_CASE(Image, BoundaryDemotionLadderIsContinuousInRange) {
    // Near the border the 6x6 kernel demotes to bilinear and the 4x4
    // demotes to bilinear: results must stay finite and within the
    // physical intensity range (no wild extrapolation).
    dictest::SpeckleField field(23, 48, 48);
    Image img = dictest::make_reference_image(field, 48, 48);

    const float probes[][2] = {
            {2.4f, 24.0f},  // inside keys demotion band
            {45.0f, 24.0f}, // right edge band
            {24.0f, 2.0f},  // top band
            {1.2f, 1.2f},   // corner
    };
    for (auto &p : probes) {
        float v6 = img.interpolate_keys_fourth(p[0], p[1]);
        float v4 = img.interpolate_bicubic(p[0], p[1]);
        CHECK(v6 >= 0.0f && v6 <= 255.0f);
        CHECK(v4 >= 0.0f && v4 <= 255.0f);
    }
}

// The batch-of-four interpolators are the one place in the engine where a
// sequence of floating-point operations is written out twice: once in
// semper_canon_sample_bicubic / _keys6 in the canonical header, and once
// per lane in interpolate_bicubic_x4 / interpolate_keys_fourth_x4, which
// exist so the compiler can pipeline four independent points.
//
// solve_icgn takes the batched path whenever four consecutive subset pixels
// are all inside the guard and the scalar path otherwise, so the two must
// agree to the LAST BIT or a subset's result would depend on where its
// out-of-guard pixels happened to fall. Nothing pinned that until the ICGN
// kernel needed the scalar sampler to be the definition of the arithmetic.
//
// Exact ==, deliberately: CHECK_NEAR here would pass while the property
// the batching relies on was already broken.
TEST_CASE(Image, BatchOfFourMatchesScalarExactly) {
    dictest::SpeckleField field(1234, 96, 96, 900);
    Image img = dictest::make_reference_image(field, 96, 96);

    // Interior only -- that is these functions' documented precondition, and
    // the guard in solve_icgn is tighter still.
    unsigned rng = 20240917u;
    auto next = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<float>((rng >> 8) & 0xFFFFu) / 65536.0f;
    };

    int bicubic_bad = 0, keys_bad = 0, groups = 0;
    for (int g = 0; g < 512; ++g) {
        float xs[4], ys[4];
        for (int k = 0; k < 4; ++k) {
            xs[k] = 8.0f + next() * 78.0f;
            ys[k] = 8.0f + next() * 78.0f;
        }
        float b4[4], k4[4];
        img.interpolate_bicubic_x4(xs, ys, b4);
        img.interpolate_keys_fourth_x4(xs, ys, k4);
        for (int k = 0; k < 4; ++k) {
            if (b4[k] != img.interpolate_bicubic(xs[k], ys[k])) ++bicubic_bad;
            if (k4[k] != img.interpolate_keys_fourth(xs[k], ys[k])) ++keys_bad;
        }
        ++groups;
    }
    std::printf("  Image: %d groups x 4 lanes, %d bicubic / %d keys mismatches\n",
                groups, bicubic_bad, keys_bad);
    CHECK(bicubic_bad == 0);
    CHECK(keys_bad == 0);
}
