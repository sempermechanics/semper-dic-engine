// Incremental DIC building blocks on exact fields: increment sampling, the
// node field built from a solve capture, and the material-mask warp.
#include "framework/test_framework.h"

#if defined(DIC_HAVE_OPENCV)

#include "pipeline/incremental_internal.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <vector>

using namespace Semper::pipeline::internal;

namespace {

// A lattice carrying u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y exactly, with
// the matching gradients.
NodeField linear_field(int gw, int gh, int rx, int ry, int step,
                       const double a[3], const double b[3]) {
    NodeField f;
    f.gridW = gw; f.gridH = gh; f.rect_x = rx; f.rect_y = ry; f.step = step;
    const size_t n = f.size();
    f.u.resize(n); f.v.resize(n); f.ux.resize(n); f.uy.resize(n);
    f.vx.resize(n); f.vy.resize(n); f.corr.assign(n, 0.01f); f.valid.assign(n, 1);
    for (int j = 0; j < gh; ++j)
        for (int i = 0; i < gw; ++i) {
            const size_t k = (size_t)j * gw + i;
            const double x = f.node_x(i), y = f.node_y(j);
            f.u[k] = (float)(a[0] + a[1] * x + a[2] * y);
            f.v[k] = (float)(b[0] + b[1] * x + b[2] * y);
            f.ux[k] = (float)a[1]; f.uy[k] = (float)a[2];
            f.vx[k] = (float)b[1]; f.vy[k] = (float)b[2];
        }
    return f;
}

const double kA[3] = {1.5, 0.02, -0.01};
const double kB[3] = {-0.75, 0.015, 0.03};

double lin(const double c[3], double x, double y) { return c[0] + c[1] * x + c[2] * y; }

} // namespace

TEST_CASE(IncrementalMath, Sample_OnNode_ReturnsNodeExactly) {
    NodeField f = linear_field(6, 5, 20, 30, 10, kA, kB);
    f.valid[(size_t)1 * 6 + 3] = 0;   // a missing neighbour must not matter on a node
    const IncrementSample s = sample_increment(f, f.node_x(2), f.node_y(1));
    REQUIRE(s.ok);
    CHECK(!s.taylor);
    CHECK(s.u == f.u[(size_t)1 * 6 + 2]);
    CHECK(s.v == f.v[(size_t)1 * 6 + 2]);
}

TEST_CASE(IncrementalMath, Sample_Bilinear_ReproducesLinearField) {
    const NodeField f = linear_field(6, 5, 20, 30, 10, kA, kB);
    for (double x : {21.3, 37.9, 64.5}) {
        for (double y : {30.2, 55.5, 69.9}) {
            const IncrementSample s = sample_increment(f, (float)x, (float)y);
            REQUIRE(s.ok);
            CHECK(!s.taylor);
            CHECK_NEAR(s.u, lin(kA, x, y), 1e-4);
            CHECK_NEAR(s.v, lin(kB, x, y), 1e-4);
        }
    }
}

TEST_CASE(IncrementalMath, Sample_TaylorFallback_ReproducesLinearField) {
    NodeField f = linear_field(6, 5, 20, 30, 10, kA, kB);
    // Knock out one corner of the cell around (44, 47): bilinear is impossible,
    // the nearest valid node's expansion still reproduces a linear field.
    f.valid[(size_t)2 * 6 + 3] = 0;   // node (50, 50)
    const IncrementSample s = sample_increment(f, 44.0f, 47.0f);
    REQUIRE(s.ok);
    CHECK(s.taylor);
    CHECK_NEAR(s.u, lin(kA, 44, 47), 1e-4);
    CHECK_NEAR(s.v, lin(kB, 44, 47), 1e-4);
}

TEST_CASE(IncrementalMath, Sample_TooFarFromAnyNode_IsInvalid) {
    const NodeField f = linear_field(4, 4, 100, 100, 10, kA, kB);
    // Within 1.5 cells of the corner node (100, 100): taylor.
    CHECK(sample_increment(f, 90.0f, 95.0f).ok);
    // 2 cells out: nothing within reach.
    CHECK(!sample_increment(f, 80.0f, 100.0f).ok);
    CHECK(!sample_increment(f, 100.0f, 160.0f).ok);
}

TEST_CASE(IncrementalMath, Sample_CorrIsWorstContributor) {
    NodeField f = linear_field(4, 4, 0, 0, 10, kA, kB);
    f.corr[(size_t)1 * 4 + 2] = 0.09f;
    const IncrementSample s = sample_increment(f, 15.0f, 15.0f);
    REQUIRE(s.ok);
    CHECK(s.corr == 0.09f);
}

TEST_CASE(IncrementalMath, Fit_ReproducesLinearFieldAcrossAHole) {
    // A band 4 nodes wide with nothing solved in it, as where the material
    // crosses a stationary background edge: sampling has no node within reach
    // of the band's middle, the plane through the nodes either side does.
    NodeField f = linear_field(12, 12, 20, 30, 10, kA, kB);
    for (int j = 4; j < 8; ++j)
        for (int i = 0; i < 12; ++i) f.valid[(size_t)j * 12 + i] = 0;
    for (double x : {20.0, 73.0, 130.0}) {
        const float y = 85.0f;   // between rows 5 and 6
        CHECK(!sample_increment(f, (float)x, y).ok);
        const IncrementSample s = fit_increment(f, (float)x, y, 80.0f);
        REQUIRE(s.ok);
        CHECK_NEAR(s.u, lin(kA, x, y), 1e-3);
        CHECK_NEAR(s.v, lin(kB, x, y), 1e-3);
    }
}

TEST_CASE(IncrementalMath, Fit_TooFewOrCollinearNodes_IsInvalid) {
    NodeField f = linear_field(10, 10, 0, 0, 10, kA, kB);
    // One row left: plenty of nodes, but no spread across it.
    for (size_t k = 0; k < f.size(); ++k) f.valid[k] = (k / 10 == 3) ? 1 : 0;
    CHECK(!fit_increment(f, 45.0f, 50.0f, 80.0f).ok);
    // Five nodes two steps around (50, 40): spread in both axes, one short of
    // the minimum; a sixth makes it.
    std::fill(f.valid.begin(), f.valid.end(), 0);
    for (int k : {43, 47, 25, 65, 23}) f.valid[(size_t)k] = 1;
    CHECK(!fit_increment(f, 50.0f, 40.0f, 80.0f).ok);
    f.valid[(size_t)67] = 1;
    const IncrementSample s = fit_increment(f, 50.0f, 40.0f, 80.0f);
    REQUIRE(s.ok);
    CHECK_NEAR(s.u, lin(kA, 50, 40), 1e-3);
    CHECK_NEAR(s.v, lin(kB, 50, 40), 1e-3);
    // Nothing within the radius.
    CHECK(!fit_increment(f, 50.0f, 40.0f, 5.0f).ok);
}

TEST_CASE(IncrementalMath, Fit_CorrIsWorstUsed) {
    NodeField f = linear_field(8, 8, 0, 0, 10, kA, kB);
    f.corr[(size_t)2 * 8 + 3] = 0.12f;   // within the radius of (35, 35)
    f.corr[(size_t)7 * 8 + 7] = 0.30f;   // outside it
    const IncrementSample s = fit_increment(f, 35.0f, 35.0f, 30.0f);
    REQUIRE(s.ok);
    CHECK(s.corr == 0.12f);
}

TEST_CASE(IncrementalMath, AnchorNeed_DropRefreshKeep) {
    // 30 x 30 nodes, step 5, over (0..145)^2; subset 21: its own reach 15 px,
    // a gap counts within 52 px.
    NodeField f = linear_field(30, 30, 0, 0, 5, kA, kB);
    auto need = [&](float x, float y) { return anchor_need(f, gap_distance(f), x, y, 21); };
    CHECK(need(75.0f, 75.0f) == AnchorNeed::kDrop);      // nothing missing near
    CHECK(need(75.0f, 20.0f) == AnchorNeed::kRefresh);   // the grid ends 25 px up
    CHECK(need(75.0f, 5.0f) == AnchorNeed::kRefresh);    // off the grid is not a hole
    f.valid[(size_t)23 * 30 + 15] = 0;                   // a gap 40 px below
    CHECK(need(75.0f, 75.0f) == AnchorNeed::kRefresh);
    f.valid[(size_t)17 * 30 + 15] = 0;                   // a hole in its own subset
    CHECK(need(75.0f, 75.0f) == AnchorNeed::kKeep);
}

namespace {

// Smooth random texture, 8-bit, the scale of a DIC speckle.
cv::Mat texture(int w, int h, unsigned seed) {
    cv::Mat noise(h, w, CV_32F);
    cv::RNG rng(seed);
    rng.fill(noise, cv::RNG::NORMAL, 0.0, 1.0);
    cv::GaussianBlur(noise, noise, cv::Size(0, 0), 2.0);
    cv::normalize(noise, noise, 30.0, 225.0, cv::NORM_MINMAX);
    cv::Mat out;
    noise.convertTo(out, CV_8U);
    return out;
}

} // namespace

TEST_CASE(IncrementalMath, Anchor_MatchesTheNodeWhereItMoved) {
    const cv::Mat ref = texture(200, 200, 7);
    // A whole-pixel shift, so resampling the texture adds no error of its own
    // (a cubic sub-pixel warp of it is off by ~0.05 px); the node's sub-pixel
    // offset from its anchor pixel is still exercised.
    const float tx = 12.0f, ty = -7.0f;
    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);
    cv::Mat def;
    cv::warpAffine(ref, def, M, ref.size(), cv::INTER_CUBIC, cv::BORDER_REFLECT);
    Semper::Image img(def.cols, def.rows, def.data);
    img.prepare_data(false);

    CoastAnchor a;
    REQUIRE(make_anchor(ref, 100.4f, 90.7f, 21, a));
    CHECK(a.ax == 100);
    CHECK(a.ay == 91);
    CHECK_NEAR(a.dx, 0.4, 1e-5);
    CHECK_NEAR(a.dy, -0.3, 1e-5);

    // Started where coasting left it, 1 px off: matched to where it is.
    float x = 0, y = 0, corr = 0;
    const float gx = 100.4f + tx + 0.8f, gy = 90.7f + ty - 0.6f;
    REQUIRE(match_anchor(a, img, 21, false, gx, gy, 2.0f, x, y, corr));
    CHECK_NEAR(x, 100.4 + tx, 0.05);
    CHECK_NEAR(y, 90.7 + ty, 0.05);
    CHECK(corr < 0.05f);
    // The same match, further from the guess than allowed.
    CHECK(!match_anchor(a, img, 21, false, gx, gy, 0.5f, x, y, corr));
    // Nothing like it here.
    const cv::Mat other = texture(200, 200, 8);
    Semper::Image img2(other.cols, other.rows, other.data);
    img2.prepare_data(false);
    CHECK(!match_anchor(a, img2, 21, false, gx, gy, 2.0f, x, y, corr));
    // A patch that would leave the image is not cut.
    CoastAnchor edge;
    CHECK(!make_anchor(ref, 12.0f, 100.0f, 21, edge));
}

TEST_CASE(IncrementalMath, NodeField_DropsUnsolvedAndOutliers) {
    FieldCapture cap;
    cap.gridW = 7; cap.gridH = 7; cap.rect_x = 0; cap.rect_y = 0; cap.step = 10;
    cap.grid.assign(7, std::vector<GridPoint>(7));
    for (int y = 0; y < 7; ++y)
        for (int x = 0; x < 7; ++x) {
            GridPoint &g = cap.grid[y][x];
            g.x = x * 10.0f; g.y = y * 10.0f;
            g.u = 1.0f + 0.01f * x; g.v = -0.5f + 0.01f * y;
            g.corr = 0.02f; g.solved = true;
        }
    cap.grid[1][1].solved = false;          // never solved
    cap.grid[2][5].corr = -1.0f;            // solved flag but no result
    cap.grid[4][3].u += 6.0f;               // a wrong match
    int rejected = -1;
    const NodeField f = node_field_from_capture(cap, &rejected);
    CHECK(!f.valid[1 * 7 + 1]);
    CHECK(!f.valid[2 * 7 + 5]);
    CHECK(!f.valid[4 * 7 + 3]);
    CHECK(rejected == 1);
    CHECK(f.valid[3 * 7 + 3]);
}

namespace {

cv::Mat rect_mask(int w, int h, cv::Rect r) {
    cv::Mat m = cv::Mat::zeros(h, w, CV_8U);
    m(r).setTo(255);
    return m;
}

GridGeom geom_over(cv::Rect r, int step) {
    return GridGeom{r.width / step, r.height / step, r.x, r.y, step};
}

} // namespace

TEST_CASE(IncrementalMath, MaskWarp_ZeroDisplacement_IsIdentity) {
    const cv::Rect r(60, 50, 100, 80);
    const cv::Mat m0 = rect_mask(240, 200, r);
    const GridGeom g = geom_over(r, 10);
    const size_t n = (size_t)g.gridW * g.gridH;
    const cv::Mat w = warp_material_mask(m0, g, std::vector<float>(n, 0.0f),
                                         std::vector<float>(n, 0.0f),
                                         std::vector<unsigned char>(n, 1), 40);
    CHECK(cv::norm(w, m0, cv::NORM_L1) == 0.0);
}

TEST_CASE(IncrementalMath, MaskWarp_Translation_ShiftsMask) {
    const cv::Rect r(60, 50, 100, 80);
    const cv::Mat m0 = rect_mask(240, 200, r);
    const GridGeom g = geom_over(r, 10);
    const size_t n = (size_t)g.gridW * g.gridH;
    std::vector<unsigned char> valid(n, 1);
    valid[5] = 0;   // a hole in the field is filled, not left empty
    const cv::Mat w = warp_material_mask(m0, g, std::vector<float>(n, 7.0f),
                                         std::vector<float>(n, -4.0f), valid, 40);
    const cv::Mat expected = rect_mask(240, 200, cv::Rect(r.x + 7, r.y - 4, r.width, r.height));
    CHECK(cv::norm(w, expected, cv::NORM_L1) == 0.0);
}

TEST_CASE(IncrementalMath, MaskWarp_Stretch_ScalesArea) {
    // u = 0.2 (x - cx): a 20% stretch about the mask centre.
    const cv::Rect r(80, 60, 120, 100);
    const cv::Mat m0 = rect_mask(320, 240, r);
    const GridGeom g = geom_over(r, 10);
    const size_t n = (size_t)g.gridW * g.gridH;
    std::vector<float> U(n), V(n, 0.0f);
    const double cx = r.x + r.width / 2.0;
    for (int j = 0; j < g.gridH; ++j)
        for (int i = 0; i < g.gridW; ++i)
            U[(size_t)j * g.gridW + i] = (float)(0.2 * (g.rect_x + i * g.step - cx));
    const cv::Mat w = warp_material_mask(m0, g, U, V, std::vector<unsigned char>(n, 1), 60);
    const double ratio = cv::countNonZero(w) / (double)cv::countNonZero(m0);
    CHECK_NEAR(ratio, 1.2, 0.012);
    // The mask edges went where the stretch sends them (to within a pixel).
    const cv::Rect bb = cv::boundingRect(w);
    CHECK(std::abs(bb.x - (cx - 1.2 * (cx - r.x))) <= 1.0);
    CHECK(std::abs(bb.x + bb.width - (cx + 1.2 * (r.x + r.width - cx))) <= 1.5);
}

TEST_CASE(IncrementalMath, MaskWarp_LargeRotation_IsTheRotatedMask) {
    // An 84 degree turn about the mask centre, with the material reaching past
    // the outer nodes as it does in a session (the field is extended there).
    // A Newton inverse once lost whole rows here; the forward warp must give
    // the rotated square: same area, no stray pixels, nothing missing inside.
    const cv::Rect r(110, 110, 100, 100);
    const cv::Mat m0 = rect_mask(320, 320, cv::Rect(r.x - 16, r.y - 16, r.width + 32, r.height + 32));
    const GridGeom g = geom_over(r, 5);
    const size_t n = (size_t)g.gridW * g.gridH;
    const double c = 160.0, th = 84.0 * 3.14159265358979323846 / 180.0;
    const double ct = std::cos(th), st = std::sin(th);
    std::vector<float> U(n), V(n);
    for (int j = 0; j < g.gridH; ++j)
        for (int i = 0; i < g.gridW; ++i) {
            const double X = g.rect_x + i * g.step - c, Y = g.rect_y + j * g.step - c;
            U[(size_t)j * g.gridW + i] = (float)(ct * X - st * Y - X);
            V[(size_t)j * g.gridW + i] = (float)(st * X + ct * Y - Y);
        }
    const cv::Mat w = warp_material_mask(m0, g, U, V, std::vector<unsigned char>(n, 1), 40);
    const double ratio = cv::countNonZero(w) / (double)cv::countNonZero(m0);
    CHECK_NEAR(ratio, 1.0, 0.02);
    // Pull every pixel back analytically: set pixels must come from the
    // material (within a pixel of its edge), and pixels well inside it must be set.
    int stray = 0, missing = 0;
    const double lo = r.x - 16, hi = r.x + r.width + 16;
    for (int y = 0; y < w.rows; ++y)
        for (int x = 0; x < w.cols; ++x) {
            const double dx = x - c, dy = y - c;
            const double X = ct * dx + st * dy + c, Y = -st * dx + ct * dy + c;
            const double inside = std::min(std::min(X - lo, hi - 1 - X), std::min(Y - lo, hi - 1 - Y));
            const bool set = w.at<unsigned char>(y, x) != 0;
            if (set && inside < -1.0) ++stray;
            if (!set && inside > 1.0) ++missing;
        }
    CHECK(stray == 0);
    CHECK(missing == 0);
}

TEST_CASE(IncrementalMath, MaskWarp_NoValidNode_IsEmpty) {
    const cv::Rect r(60, 50, 100, 80);
    const cv::Mat m0 = rect_mask(240, 200, r);
    const GridGeom g = geom_over(r, 10);
    const size_t n = (size_t)g.gridW * g.gridH;
    const cv::Mat w = warp_material_mask(m0, g, std::vector<float>(n, 0.0f),
                                         std::vector<float>(n, 0.0f),
                                         std::vector<unsigned char>(n, 0), 40);
    CHECK(cv::countNonZero(w) == 0);
}

#else

TEST_CASE(IncrementalMath, OpenCvRequired_SkippedWithoutOpenCV) {
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
