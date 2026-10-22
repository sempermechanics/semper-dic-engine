#include <semper/image.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace Semper {

    Image::Image(int_t w, int_t h, const uint8_t *raw_pixels) : width(w), height(h) {
        // Every pixel index below and in the solver is an int: a w * h that
        // wraps would size the planes small and the row loops would then
        // write past them.
        if (w < 0 || h < 0 ||
            (int64_t)w * (int64_t)h > (int64_t)std::numeric_limits<int>::max()) {
            throw std::length_error("Image: width * height does not fit in an int");
        }
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

    // Zone 0: |s| <= 1
    inline float keys_f0(float s) {
        return (4.0f/3.0f)*s*s*s - (7.0f/3.0f)*s*s + 1.0f;
    }
    // Zone 1: 1 < |s| <= 2
    inline float keys_f1(float s) {
        return -(7.0f/12.0f)*s*s*s + 3.0f*s*s - (59.0f/12.0f)*s + 2.5f;
    }
    // Zone 2: 2 < |s| <= 3
    // 🚀 FIXED: True Keys 4th-Order Polynomial for the 3rd Zone
    inline float keys_f2(float s) {
        return (1.0f / 12.0f) * s * s * s - (2.0f / 3.0f) * s * s + (7.0f / 4.0f) * s - 1.5f;
    }

    scalar_t Image::interpolate_keys_fourth(scalar_t x, scalar_t y) const {
        // 🚀 DICe LAYER 1: Explicit demotion boundary for 6x6 kernel
        if (x <= 2.5f || x >= width - 3.5f || y <= 2.5f || y >= height - 3.5f) {
            return interpolate_bilinear(x, y); // 🚀 Demotes to Bilinear directly!
        }

        int xi = static_cast<int>(x);
        int yi = static_cast<int>(y);

        float dx = x - xi;
        float dy = y - yi;

        // Calculate the 6 weights for X
        float wx[6];
        wx[0] = keys_f2(dx + 2.0f);
        wx[1] = keys_f1(dx + 1.0f);
        wx[2] = keys_f0(dx);
        wx[3] = keys_f0(1.0f - dx);
        wx[4] = keys_f1(2.0f - dx);
        wx[5] = keys_f2(3.0f - dx);

        // Calculate the 6 weights for Y
        float wy[6];
        wy[0] = keys_f2(dy + 2.0f);
        wy[1] = keys_f1(dy + 1.0f);
        wy[2] = keys_f0(dy);
        wy[3] = keys_f0(1.0f - dy);
        wy[4] = keys_f1(2.0f - dy);
        wy[5] = keys_f2(3.0f - dy);

        float val = 0.0f;
        // Loop over the 6x6 neighborhood (j = row offset, i = column offset)
        for (int j = -2; j <= 3; ++j) {
            const scalar_t* row_ptr = &intensities[(yi + j) * width + xi];
            float row_val = 0.0f;

            // Unroll the inner loop for speed
            row_val += row_ptr[-2] * wx[0];
            row_val += row_ptr[-1] * wx[1];
            row_val += row_ptr[0]  * wx[2];
            row_val += row_ptr[1]  * wx[3];
            row_val += row_ptr[2]  * wx[4];
            row_val += row_ptr[3]  * wx[5];

            val += row_val * wy[j + 2];
        }

        return val;
    }

    // Lane k's block below is a verbatim copy of interpolate_keys_fourth's
    // weight computation and 6x6 sum, just indexed by k instead of being one
    // point per call — see image.hpp's precondition comment. Deliberately NOT
    // sharing a helper with the single-point version: keeping the operation
    // sequence textually identical here is what makes "this changes no
    // single point's arithmetic" easy to verify by inspection.
    void Image::interpolate_keys_fourth_x4(const scalar_t x[4], const scalar_t y[4], scalar_t out[4]) const {
        int xi[4], yi[4];
        float dx[4], dy[4];
        float wx[4][6], wy[4][6];
        for (int k = 0; k < 4; ++k) {
            xi[k] = static_cast<int>(x[k]);
            yi[k] = static_cast<int>(y[k]);
            dx[k] = x[k] - xi[k];
            dy[k] = y[k] - yi[k];

            wx[k][0] = keys_f2(dx[k] + 2.0f);
            wx[k][1] = keys_f1(dx[k] + 1.0f);
            wx[k][2] = keys_f0(dx[k]);
            wx[k][3] = keys_f0(1.0f - dx[k]);
            wx[k][4] = keys_f1(2.0f - dx[k]);
            wx[k][5] = keys_f2(3.0f - dx[k]);

            wy[k][0] = keys_f2(dy[k] + 2.0f);
            wy[k][1] = keys_f1(dy[k] + 1.0f);
            wy[k][2] = keys_f0(dy[k]);
            wy[k][3] = keys_f0(1.0f - dy[k]);
            wy[k][4] = keys_f1(2.0f - dy[k]);
            wy[k][5] = keys_f2(3.0f - dy[k]);
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

    void get_keys_weights(float s, float& w_m1, float& w_0, float& w_1, float& w_2) {
        float s2 = s * s;
        float s3 = s2 * s;
        w_m1 = -0.5f * s3 + s2 - 0.5f * s;
        w_0  =  1.5f * s3 - 2.5f * s2 + 1.0f;
        w_1  = -1.5f * s3 + 2.0f * s2 + 0.5f * s;
        w_2  =  0.5f * s3 - 0.5f * s2;
    }

    scalar_t Image::interpolate_bilinear(scalar_t x, scalar_t y) const {
        // DICe LAYER 1: Absolute hard boundary for Bilinear 2x2 kernel
        if (x < 0.0f || x >= width - 1.5f || y < 0.0f || y >= height - 1.5f) {
            return 0.0f; // Explicitly kills the pixel, returning a zero intensity
        }
        int xi = static_cast<int>(x); int yi = static_cast<int>(y);
        float dx = x - xi; float dy = y - yi;
        const scalar_t* row0 = &intensities[yi * width + xi];
        const scalar_t* row1 = &intensities[(yi + 1) * width + xi];
        float val0 = row0[0] * (1.0f - dx) + row0[1] * dx;
        float val1 = row1[0] * (1.0f - dx) + row1[1] * dx;
        return val0 * (1.0f - dy) + val1 * dy;
    }

    scalar_t Image::interpolate_bicubic(scalar_t x, scalar_t y) const {
        // DICe LAYER 1: Explicit demotion boundary for 4x4 kernel
        if (x < 1.0f || x >= width - 2.0f || y < 1.0f || y >= height - 2.0f) {
            return interpolate_bilinear(x, y); // Demotes to Bilinear
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

    // Verbatim per-lane copy of interpolate_bicubic — see image.hpp's
    // precondition comment and interpolate_keys_fourth_x4's note above.
    void Image::interpolate_bicubic_x4(const scalar_t x[4], const scalar_t y[4], scalar_t out[4]) const {
        int xi[4], yi[4];
        float dx[4], dy[4];
        float wx[4][4], wy[4][4];
        for (int k = 0; k < 4; ++k) {
            xi[k] = static_cast<int>(x[k]);
            yi[k] = static_cast<int>(y[k]);
            dx[k] = x[k] - xi[k];
            dy[k] = y[k] - yi[k];
            get_keys_weights(dx[k], wx[k][0], wx[k][1], wx[k][2], wx[k][3]);
            get_keys_weights(dy[k], wy[k][0], wy[k][1], wy[k][2], wy[k][3]);
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