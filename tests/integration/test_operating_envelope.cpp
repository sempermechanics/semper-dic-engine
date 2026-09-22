// Operating envelope of run_full_field under LARGE homogeneous deformation:
// large translation, large rigid rotation, large stretch and shear, and a
// rotation combined with stretch. Compiled only when DIC_HAVE_OPENCV is set
// (the AKAZE seeding stage is what these tests exercise).
//
// What is being pinned
// --------------------
// The shape function is first-order (6-DOF affine) and the reported strain is
// Green-Lagrange, E = ½(AᵀA − I). Both are exact for any homogeneous warp, so
// the engine has no kinematic limit here — the limit is whether AKAZE + RANSAC
// can seed the mesh. Measured on this speckle (seed 21, 512²), single-shot
// seeding holds to 40% stretch / γ = 0.4 and first fails at 50% / γ = 0.5.
//
// Why floors, not the cliff
// -------------------------
// These tests assert that the engine still works well INSIDE the envelope
// (30% stretch, γ = 0.3, 45° rotation, 100 px translation). They deliberately
// do not assert where it fails:
//   * an improvement to seeding would then fail CI;
//   * behaviour at the edge is non-monotonic — 70% stretch succeeds via the
//     Path C fallback while 50–60% return −1 (the seed search lands near the
//     deformation centre, where displacement ≈ 0).
// Every case passed on five independent speckle seeds with the tolerances
// below; the worst seed reached 0.058 px / 1.8e-3.
//
// Rotation invariance is the physics check: a small-strain (engineering)
// measure would report exx = cos45° − 1 = −0.29 for the rigid 45° case, which
// the 3e-3 tolerance rejects by two orders of magnitude.
#include "framework/test_framework.h"
#include "framework/synthetic.h"

#include <semper/pipeline.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/core.hpp>
#include <vector>

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::run_full_field;

#if defined(DIC_HAVE_OPENCV)

namespace {

constexpr int W = 512;
constexpr int H = 512;
constexpr int BLOBS = 2400;          // same blob density as FullField (600 / 256²)
constexpr unsigned SEED = 21;
constexpr int STEP = 20;
constexpr int SUBSET = 41;
constexpr int STRAIN_WIN = 41;       // VSG diameter: +-1 grid step (plus stencil)
constexpr float CX = W / 2.0f;
constexpr float CY = H / 2.0f;
constexpr double kPi = 3.14159265358979323846;

constexpr double TOL_DISP_PX = 0.10; // max over the field, not rms
constexpr double TOL_STRAIN = 3.0e-3;

// Rendering 512² analytic speckle dominates this file: ~4 s in Release, ~5 min
// in the -O0 coverage build. Results there were verified identical to Release,
// so the skip is for runtime only; the Release host job is the gate.
#if defined(SEMPER_COVERAGE_BUILD)
#define ENVELOPE_SKIP_UNDER_COVERAGE()                                          \
    do {                                                                        \
        std::printf("  skipped under coverage (runtime; Release host job gates)\n"); \
        return;                                                                 \
    } while (0)
#else
#define ENVELOPE_SKIP_UNDER_COVERAGE() do { } while (0)
#endif

cv::Mat gray8_from_image(const Semper::Image &img) {
    cv::Mat m(img.height, img.width, CV_8UC1);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float v = img.intensities[(size_t)y * img.width + x];
            if (v < 0.f) v = 0.f;
            if (v > 255.f) v = 255.f;
            m.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
        }
    }
    return m;
}

const dictest::SpeckleField &field() {
    static const dictest::SpeckleField f(SEED, W, H, BLOBS);
    return f;
}

// Rendering dominates this file's runtime, so the reference is rendered once.
const cv::Mat &reference() {
    static const cv::Mat ref = gray8_from_image(dictest::make_reference_image(field(), W, H));
    return ref;
}

dictest::AffineDeformation about_centre() {
    dictest::AffineDeformation d;
    d.cx = CX;
    d.cy = CY;
    return d;
}

// A = I + [[ux, uy], [vx, vy]]  from a full 2×2 deformation gradient.
void set_gradient(dictest::AffineDeformation &d, double a11, double a12, double a21, double a22) {
    d.ux = static_cast<float>(a11 - 1.0);
    d.uy = static_cast<float>(a12);
    d.vx = static_cast<float>(a21);
    d.vy = static_cast<float>(a22 - 1.0);
}

struct FieldError {
    int points = 0;
    int interior = 0;     // grid points the VSG window can fully support
    double max_disp = 0.0;
    double max_exx = 0.0, max_eyy = 0.0, max_exy = 0.0;
};

// Solve one deformed image against the cached reference and compare every
// packed point to the analytic displacement and Green-Lagrange strain.
FieldError solve_and_compare(const char *label, const dictest::AffineDeformation &d,
                             int rect_x, int rect_y, int rect_w, int rect_h) {
    const cv::Mat def = gray8_from_image(dictest::make_deformed_image(field(), W, H, d));

    ReferenceCache cache;
    cache.set_from_gray(reference(), cv::Mat());

    FullFieldParams p;
    p.rect_x = rect_x;
    p.rect_y = rect_y;
    p.rect_w = rect_w;
    p.rect_h = rect_h;
    p.step = STEP;
    p.subset_size = SUBSET;
    p.strain_window = STRAIN_WIN;
    p.use_6x6_interpolator = false;

    const int grid_w = rect_w / STEP;
    const int grid_h = rect_h / STEP;
    const int capacity = 8 * grid_w * grid_h;
    std::vector<float> out((size_t)capacity, 0.f);
    float metrics[17] = {};

    FieldError e;
    // The VSG plus stencil needs all four axial neighbours, so the outermost
    // ring of the grid is dropped by the strain post-filter by design.
    e.interior = (grid_w - 2) * (grid_h - 2);
    e.points = run_full_field(cache, def, cv::Mat(), p, out.data(), capacity, metrics, 17, nullptr);

    const double a11 = 1.0 + d.ux, a12 = d.uy, a21 = d.vx, a22 = 1.0 + d.vy;
    const double exx = 0.5 * (a11 * a11 + a21 * a21 - 1.0);
    const double eyy = 0.5 * (a12 * a12 + a22 * a22 - 1.0);
    const double exy = 0.5 * (a11 * a12 + a21 * a22);

    for (int i = 0; i < e.points; ++i) {
        const float *q = &out[(size_t)8 * i];
        const double X = q[0] - d.cx;
        const double Y = q[1] - d.cy;
        const double tu = d.u + d.ux * X + d.uy * Y;
        const double tv = d.v + d.vx * X + d.vy * Y;
        e.max_disp = std::max(e.max_disp, std::hypot(q[2] - tu, q[3] - tv));
        e.max_exx = std::max(e.max_exx, std::fabs(q[4] - exx));
        e.max_eyy = std::max(e.max_eyy, std::fabs(q[5] - eyy));
        e.max_exy = std::max(e.max_exy, std::fabs(q[6] - exy));
    }

    std::printf("  [%s] points %d (interior %d), seed flag %.0f, max|du| %.4f px, "
                "max|dE| xx %.5f yy %.5f xy %.5f\n",
                label, e.points, e.interior, metrics[16], e.max_disp,
                e.max_exx, e.max_eyy, e.max_exy);
    return e;
}

void check_within_tolerance(const FieldError &e) {
    REQUIRE(e.points > 0);            // -1: seeding failed; 0: nothing survived
    CHECK(e.points >= e.interior);    // every VSG-supported point is solved
    CHECK(e.max_disp <= TOL_DISP_PX);
    CHECK(e.max_exx <= TOL_STRAIN);
    CHECK(e.max_eyy <= TOL_STRAIN);
    CHECK(e.max_exy <= TOL_STRAIN);
}

} // namespace

TEST_CASE(Envelope, LargeTranslation_100px_Recovered) {
    ENVELOPE_SKIP_UNDER_COVERAGE();
    // Far outside the ±15 px brute-force search of the Path C fallback, so the
    // AKAZE mesh has to seed it. Non-integer, so the deformed image is not a
    // pixel-shifted copy of the reference.
    dictest::AffineDeformation d = about_centre();
    d.u = 100.3f;
    d.v = -40.6f;
    check_within_tolerance(solve_and_compare("translate 100 px", d, 96, 180, 200, 200));
}

TEST_CASE(Envelope, RigidRotation_45deg_IsStrainFree) {
    ENVELOPE_SKIP_UNDER_COVERAGE();
    const double th = 45.0 * kPi / 180.0;
    dictest::AffineDeformation d = about_centre();
    set_gradient(d, std::cos(th), -std::sin(th), std::sin(th), std::cos(th));
    // Truth E == 0 exactly; the displacement itself is up to ~130 px.
    check_within_tolerance(solve_and_compare("rotate 45 deg", d, 136, 136, 240, 240));
}

TEST_CASE(Envelope, UniaxialStretch_30pct_Recovered) {
    ENVELOPE_SKIP_UNDER_COVERAGE();
    dictest::AffineDeformation d = about_centre();
    set_gradient(d, 1.30, 0.0, 0.0, 1.0);   // Exx = 0.3 + 0.3²/2 = 0.345
    check_within_tolerance(solve_and_compare("stretch 30%", d, 156, 156, 200, 200));
}

TEST_CASE(Envelope, SimpleShear_Gamma0p3_Recovered) {
    ENVELOPE_SKIP_UNDER_COVERAGE();
    dictest::AffineDeformation d = about_centre();
    set_gradient(d, 1.0, 0.30, 0.0, 1.0);   // Exy = 0.15, Eyy = 0.045, Exx = 0
    check_within_tolerance(solve_and_compare("shear g=0.3", d, 156, 156, 200, 200));
}

TEST_CASE(Envelope, RotationPlusStretch_StrainIgnoresRotation) {
    ENVELOPE_SKIP_UNDER_COVERAGE();
    // A = R(30°) · diag(1.2, 1): Green-Lagrange sees only the stretch,
    // E = ½(diag(1.2,1)² − I) = diag(0.22, 0), exy = 0.
    const double th = 30.0 * kPi / 180.0;
    const double c = std::cos(th), s = std::sin(th);
    dictest::AffineDeformation d = about_centre();
    set_gradient(d, 1.2 * c, -s, 1.2 * s, c);
    check_within_tolerance(solve_and_compare("rot 30 + stretch 20%", d, 156, 156, 200, 200));
}

#endif
