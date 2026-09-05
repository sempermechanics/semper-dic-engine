#include <semper/image.hpp>
#include <semper/kernels/canonical_math.h>
#include <cmath>

namespace Semper {

    Image::Image(int_t w, int_t h, const uint8_t *raw_pixels) : width(w), height(h) {
        intensities.resize(w * h);
        for (int i = 0; i < w * h; ++i) {
            intensities[i] = static_cast<scalar_t>(raw_pixels[i]);
        }
    }

    // --- OPTIONAL: DICe 7-Tap Gaussian Blur Kernel ---
    static const float gf_coeffs_7[49] = {
            3.60000000e-05f,3.63600000e-04f,1.45080000e-03f,2.29860000e-03f,1.45080000e-03f,3.63600000e-04f,3.60000000e-05f,
            3.63600000e-04f,3.67236000e-03f,1.46530800e-02f,2.32158600e-02f,1.46530800e-02f,3.67236000e-03f,3.63600000e-04f,
            1.45080000e-03f,1.46530800e-02f,5.84672400e-02f,9.26335800e-02f,5.84672400e-02f,1.46530800e-02f,1.45080000e-03f,
            2.29860000e-03f,2.32158600e-02f,9.26335800e-02f,1.46765610e-01f,9.26335800e-02f,2.32158600e-02f,2.29860000e-03f,
            1.45080000e-03f,1.46530800e-02f,5.84672400e-02f,9.26335800e-02f,5.84672400e-02f,1.46530800e-02f,1.45080000e-03f,
            3.63600000e-04f,3.67236000e-03f,1.46530800e-02f,2.32158600e-02f,1.46530800e-02f,3.67236000e-03f,3.63600000e-04f,
            3.60000000e-05f,3.63600000e-04f,1.45080000e-03f,2.29860000e-03f,1.45080000e-03f,3.63600000e-04f,3.60000000e-05f
    };

    void Image::prepare_data(bool apply_dice_blur) {
        if (apply_dice_blur) {
            std::vector<scalar_t> temp = intensities;
            int half_mask = 3;
            for (int y = half_mask; y < height - half_mask; ++y) {
                for (int x = half_mask; x < width - half_mask; ++x) {
                    float val = 0.0f;
                    for (int i = 0; i < 7; ++i) {
                        for (int j = 0; j < 7; ++j) {
                            val += gf_coeffs_7[i * 7 + j] * temp[(y + (j - half_mask)) * width + x + (i - half_mask)];
                        }
                    }
                    intensities[y * width + x] = val;
                }
            }
        }

        grad_x.assign(width * height, 0.0f);
        grad_y.assign(width * height, 0.0f);

        for (int y = 2; y < height - 2; ++y) {
            for (int x = 2; x < width - 2; ++x) {
                int idx = y * width + x;
                grad_x[idx] = (-intensities[idx+2] + 8.0f*intensities[idx+1] - 8.0f*intensities[idx-1] + intensities[idx-2]) / 12.0f;
                grad_y[idx] = (-intensities[idx+2*width] + 8.0f*intensities[idx+width] - 8.0f*intensities[idx-width] + intensities[idx-2*width]) / 12.0f;
            }
        }
    }

    // ========================================================================
    // KEYS 4TH ORDER INTERPOLATION (6x6 KERNEL)
    // ========================================================================
    //
    // The weights and the 6x6 accumulation used to live here. They now live
    // in include/semper/kernels/canonical_math.h, because the ICGN kernel
    // samples the deformed image too and a second hand-written copy of an
    // interpolator is precisely the drift the canonical header exists to
    // prevent. The arithmetic is unchanged -- semper_canon_sample_keys6 is
    // the old body moved, not rewritten.

    scalar_t Image::interpolate_keys_fourth(scalar_t x, scalar_t y) const {
        return semper_canon_sample_keys6(intensities.data(), width, height, x, y);
    }

    // Lane k's block below is the same weights and the same 6x6 sum as
    // semper_canon_sample_keys6, indexed by k so four independent points are
    // computed together — see image.hpp's precondition comment. The weights
    // now come from the shared helper; the accumulation is still spelled out
    // here rather than calling the scalar sampler per lane, because the point
    // of this function is to let the compiler pipeline across the four lanes.
    // That leaves the row sum as the one duplicated sequence, which
    // Interp.BatchOfFourMatchesScalarExactly pins with ==.
    void Image::interpolate_keys_fourth_x4(const scalar_t x[4], const scalar_t y[4], scalar_t out[4]) const {
        int xi[4], yi[4];
        float dx[4], dy[4];
        float wx[4][6], wy[4][6];
        for (int k = 0; k < 4; ++k) {
            xi[k] = static_cast<int>(x[k]);
            yi[k] = static_cast<int>(y[k]);
            dx[k] = x[k] - xi[k];
            dy[k] = y[k] - yi[k];

            // Same weights the scalar sampler uses, from the same source --
            // semper_keys6_weights IS the six lines this used to spell out.
            semper_keys6_weights(dx[k], wx[k]);
            semper_keys6_weights(dy[k], wy[k]);
        }

        float val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = -2; j <= 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                const scalar_t* row_ptr = &intensities[(yi[k] + j) * width + xi[k]];
                float row_val = 0.0f;
                row_val += row_ptr[-2] * wx[k][0];
                row_val += row_ptr[-1] * wx[k][1];
                row_val += row_ptr[0]  * wx[k][2];
                row_val += row_ptr[1]  * wx[k][3];
                row_val += row_ptr[2]  * wx[k][4];
                row_val += row_ptr[3]  * wx[k][5];
                val[k] += row_val * wy[k][j + 2];
            }
        }
        for (int k = 0; k < 4; ++k) out[k] = val[k];
    }

    // ========================================================================
    // STANDARD 4x4 BICUBIC INTERPOLATION
    // ========================================================================

    scalar_t Image::interpolate_bilinear(scalar_t x, scalar_t y) const {
        return semper_canon_sample_bilinear(intensities.data(), width, height, x, y);
    }

    scalar_t Image::interpolate_bicubic(scalar_t x, scalar_t y) const {
        return semper_canon_sample_bicubic(intensities.data(), width, height, x, y);
    }

    // Per-lane copy of semper_canon_sample_bicubic's interior branch — see
    // image.hpp's precondition comment and interpolate_keys_fourth_x4's note
    // above, including which test pins the two together.
    void Image::interpolate_bicubic_x4(const scalar_t x[4], const scalar_t y[4], scalar_t out[4]) const {
        int xi[4], yi[4];
        float dx[4], dy[4];
        float wx[4][4], wy[4][4];
        for (int k = 0; k < 4; ++k) {
            xi[k] = static_cast<int>(x[k]);
            yi[k] = static_cast<int>(y[k]);
            dx[k] = x[k] - xi[k];
            dy[k] = y[k] - yi[k];
            semper_keys4_weights(dx[k], wx[k]);
            semper_keys4_weights(dy[k], wy[k]);
        }

        float val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = -1; j <= 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                const scalar_t* row_ptr = &intensities[(yi[k] + j) * width + xi[k]];
                float row_val = 0.0f;
                row_val += row_ptr[-1] * wx[k][0];
                row_val += row_ptr[0]  * wx[k][1];
                row_val += row_ptr[1]  * wx[k][2];
                row_val += row_ptr[2]  * wx[k][3];
                val[k] += row_val * wy[k][j + 1];
            }
        }
        for (int k = 0; k < 4; ++k) out[k] = val[k];
    }
}