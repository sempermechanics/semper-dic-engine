#ifndef SEMPER_TEST_SYNTHETIC_H
#define SEMPER_TEST_SYNTHETIC_H

// =====================================================================
// SYNTHETIC SPECKLE FIELD + ANALYTIC DEFORMATION
//
// The foundation of the regression suite. The trick that makes ground
// truth EXACT: both the reference and the deformed image are sampled
// from the same CONTINUOUS analytic speckle function g(x, y). The
// deformed image applies the exact inverse of the affine warp:
//
//     def(q) = g( c + A⁻¹ (q − c − t) )
//
// where c = deformation center, t = (u, v) translation and
// A = I + [[ux, uy], [vx, vy]] is the deformation gradient. No image
// resampling or interpolation is involved in generating ground truth,
// so any error the engine reports is entirely the engine's own.
//
// The speckle is a seeded sum of Gaussian blobs — smooth (matching the
// C² assumptions of bicubic/Keys interpolation), high-contrast, and
// fully deterministic across platforms.
// =====================================================================

#include <cstdint>
#include <random>
#include <vector>

#include <semper/image.hpp>

namespace dictest {

    // Continuous speckle pattern: g(x,y) = base + Σ amp_i · exp(−r²/2σ_i²)
    class SpeckleField {
    public:
        explicit SpeckleField(unsigned seed, int width, int height,
                              int blob_count = 400,
                              float sigma_min = 1.5f, float sigma_max = 3.5f) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> ux(0.0f, (float) width);
            std::uniform_real_distribution<float> uy(0.0f, (float) height);
            std::uniform_real_distribution<float> usigma(sigma_min, sigma_max);
            std::uniform_real_distribution<float> uamp(-90.0f, 90.0f);

            cx_.reserve(blob_count);
            cy_.reserve(blob_count);
            inv2s2_.reserve(blob_count);
            amp_.reserve(blob_count);
            for (int i = 0; i < blob_count; ++i) {
                cx_.push_back(ux(rng));
                cy_.push_back(uy(rng));
                float s = usigma(rng);
                inv2s2_.push_back(1.0f / (2.0f * s * s));
                amp_.push_back(uamp(rng));
            }
        }

        // Sample the continuous pattern at an arbitrary (sub-pixel) location.
        float sample(float x, float y) const {
            float v = 128.0f;
            for (size_t i = 0; i < cx_.size(); ++i) {
                float dx = x - cx_[i];
                float dy = y - cy_[i];
                float r2 = dx * dx + dy * dy;
                if (r2 < 40.0f) // blobs are compact; skip far-away ones
                    v += amp_[i] * std::exp(-r2 * inv2s2_[i]);
            }
            // Clamp to a valid 8-bit-like intensity range (engine masks < -5)
            if (v < 5.0f) v = 5.0f;
            if (v > 250.0f) v = 250.0f;
            return v;
        }

    private:
        std::vector<float> cx_, cy_, inv2s2_, amp_;
    };

    // 6-DOF affine deformation about a center point (matches the engine's
    // shape-function convention: point at c+(x,y) displaces by
    // (u + ux·x + uy·y,  v + vx·x + vy·y)).
    struct AffineDeformation {
        float u = 0.0f, v = 0.0f;
        float ux = 0.0f, uy = 0.0f, vx = 0.0f, vy = 0.0f;
        float cx = 0.0f, cy = 0.0f; // deformation center

        // Inverse-map a deformed-image coordinate back to reference space.
        void inverse_map(float qx, float qy, float &rx, float &ry) const {
            // Solve  q = c + A·p + t  for p (offsets relative to center)
            float bx = qx - cx - u;
            float by = qy - cy - v;
            float a11 = 1.0f + ux, a12 = uy;
            float a21 = vx, a22 = 1.0f + vy;
            float det = a11 * a22 - a12 * a21;
            float px = (a22 * bx - a12 * by) / det;
            float py = (-a21 * bx + a11 * by) / det;
            rx = cx + px;
            ry = cy + py;
        }
    };

    // Render the undeformed reference image from the continuous field.
    inline Semper::Image make_reference_image(const SpeckleField &field,
                                                   int w, int h) {
        std::vector<uint8_t> dummy((size_t) w * h, 0);
        Semper::Image img(w, h, dummy.data());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                img.intensities[(size_t) y * w + x] =
                        field.sample((float) x, (float) y);
        img.prepare_data(false);
        return img;
    }

    // Render the deformed image by exact analytic inverse warping.
    inline Semper::Image make_deformed_image(const SpeckleField &field,
                                                  int w, int h,
                                                  const AffineDeformation &def) {
        std::vector<uint8_t> dummy((size_t) w * h, 0);
        Semper::Image img(w, h, dummy.data());
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float rx, ry;
                def.inverse_map((float) x, (float) y, rx, ry);
                img.intensities[(size_t) y * w + x] = field.sample(rx, ry);
            }
        }
        img.prepare_data(false);
        return img;
    }

} // namespace dictest

#endif // SEMPER_TEST_SYNTHETIC_H
