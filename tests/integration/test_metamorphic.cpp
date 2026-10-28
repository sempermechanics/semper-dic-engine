// Metamorphic relations of the full field (run_full_field).
//
// Accuracy tests compare a field with a ground truth; these compare two
// solves whose inputs are related by a transformation with a known effect on
// the answer, so they hold whatever the true field is and catch an error that
// a single solve against the truth could absorb into its tolerance (a sign
// slip in one path, an asymmetric guess, a seeding step that depends on where
// the speckle sits in the frame). Every pair of images is exact: rendered from
// one continuous speckle (framework/sequence.h), or moved by whole pixels, or
// mirrored, none resampled.
//
//   rigid translation t          -> every point moves by t
//   deformed image shifted by s  -> every displacement grows by s, strain same
//   reference and deformed swapped -> the inverse map
//   deformed intensity a*I + b   -> the same field (ZNSSD)
//   both images mirrored in x    -> u and exy change sign, the rest stays
//
// Bounds are 3-20x what was measured (noted at each test), above the IC-GN
// stop step (1e-3 px) that two solves of related inputs can differ by, and far
// below the error a broken relation gives. OpenCV-gated.
#include "framework/test_framework.h"

#if defined(DIC_HAVE_OPENCV)

#include "framework/sequence.h"

#include <semper/pipeline.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;

namespace {

constexpr int kW = 256, kH = 256;
// About 13 blobs per 21 px subset. At sequence.h's default (600 / 65536, about
// 4) a few subsets are nearly flat and their ZNSSD minimum is shallow: a rigid
// shift came back up to 0.23 px off there, with ZNSSD under 0.05, which is
// what any DIC does on missing texture, and would bury the relations below.
constexpr double kBlobsPerPx = 2000.0 / 65536.0;
// Each scene reports 900 of the 32 x 32 nodes (measured); a relation over
// far fewer common points would prove little.
constexpr size_t kMinPoints = 850;

using Point = std::array<float, 8>;              // x, y, u, v, exx, eyy, exy, corr
using Field = std::map<std::pair<int, int>, Point>;

FullFieldParams params() {
    FullFieldParams p;
    // Nodes at 50, 55, ..., 205 on both axes: symmetric about the centre of a
    // 256 px image, so mirroring maps nodes onto nodes.
    p.rect_x = 50;
    p.rect_y = 50;
    p.rect_w = 160;
    p.rect_h = 160;
    p.step = 5;
    p.subset_size = 21;
    p.strain_window = 15;
    p.use_6x6_interpolator = false;
    return p;
}

Field solve(const cv::Mat &ref, const cv::Mat &def) {
    const FullFieldParams p = params();
    ReferenceCache cache;
    cache.set_from_gray(ref, cv::Mat());
    const int cap = 8 * ((p.rect_w / p.step) * (p.rect_h / p.step) + 8);
    std::vector<float> out((size_t)cap, 0.0f);
    float metrics[Semper::pipeline::kMetricsLen] = {};
    const int n = Semper::pipeline::run_full_field(cache, def, cv::Mat(), p, out.data(), cap,
                                                   metrics, Semper::pipeline::kMetricsLen, nullptr);
    Field f;
    for (int i = 0; i < n; ++i) {
        Point pt;
        std::copy(&out[(size_t)i * 8], &out[(size_t)i * 8 + 8], pt.begin());
        f[{(int)std::lround(pt[0]), (int)std::lround(pt[1])}] = pt;
    }
    return f;
}

const dictest::SmoothSpeckleField &speckle() {
    static const dictest::SmoothSpeckleField field(57, kW, kH, kBlobsPerPx);
    return field;
}

cv::Mat render(const dictest::Motion &m) { return dictest::render_frame(speckle(), kW, kH, m); }

const dictest::Vec2 kCentre{kW / 2.0, kH / 2.0};

// A general affine load: 2% stretch, 0.03 rad turn, sub-pixel translation.
dictest::Motion affine_load() {
    return dictest::stretch_rotate(0.02, 0.3, 0.03, dictest::Vec2{1.6, -0.9}, kCentre);
}

// Worst differences over the nodes both fields report.
struct Diff {
    int common = 0;
    double u = 0, e = 0, corr = 0;
};

// Compares field b, at node map(x, y), with field a transformed by `expect`.
template <class Map, class Expect>
Diff compare(const Field &a, const Field &b, Map map, Expect expect) {
    Diff d;
    for (const auto &kv : a) {
        auto it = b.find(map(kv.first));
        if (it == b.end()) continue;
        const Point want = expect(kv.second);
        const Point &got = it->second;
        ++d.common;
        d.u = std::max({d.u, (double)std::fabs(got[2] - want[2]), (double)std::fabs(got[3] - want[3])});
        d.e = std::max({d.e, (double)std::fabs(got[4] - want[4]), (double)std::fabs(got[5] - want[5]),
                        (double)std::fabs(got[6] - want[6])});
        d.corr = std::max(d.corr, (double)std::fabs(got[7] - want[7]));
    }
    return d;
}

void report(const char *what, const Field &a, const Field &b, const Diff &d) {
    std::printf("    %s: %zu / %zu points, %d common; worst |du| %.2e px, |de| %.2e, |dcorr| %.2e\n",
                what, a.size(), b.size(), d.common, d.u, d.e, d.corr);
}

// Both solves report (almost) the same nodes: a relation over a handful of
// common points would prove nothing.
bool mostly_common(const Field &a, const Field &b, const Diff &d) {
    return d.common >= (int)(0.95 * (double)std::max(a.size(), b.size()));
}

std::pair<int, int> same_node(std::pair<int, int> k) { return k; }

// Least-squares affine fit of a displacement field: u = c0 + c1 x + c2 y.
struct AffineFit {
    double u[3] = {}, v[3] = {};
    dictest::Vec2 at(double x, double y) const {
        return {u[0] + u[1] * x + u[2] * y, v[0] + v[1] * x + v[2] * y};
    }
};
AffineFit fit(const Field &f) {
    cv::Mat A((int)f.size(), 3, CV_64F), bu((int)f.size(), 1, CV_64F), bv((int)f.size(), 1, CV_64F);
    int r = 0;
    for (const auto &kv : f) {
        A.at<double>(r, 0) = 1.0;
        A.at<double>(r, 1) = kv.second[0];
        A.at<double>(r, 2) = kv.second[1];
        bu.at<double>(r, 0) = kv.second[2];
        bv.at<double>(r, 0) = kv.second[3];
        ++r;
    }
    cv::Mat cu, cv_;
    cv::solve(A, bu, cu, cv::DECOMP_SVD);
    cv::solve(A, bv, cv_, cv::DECOMP_SVD);
    AffineFit a;
    for (int i = 0; i < 3; ++i) {
        a.u[i] = cu.at<double>(i, 0);
        a.v[i] = cv_.at<double>(i, 0);
    }
    return a;
}

} // namespace

// A rigid sub-pixel translation: every point reports the shift, and no strain.
// Measured: rms 7.4e-3 px, worst 1.6e-2 px, worst |strain| 9.4e-4.
TEST_CASE(Metamorphic, RigidTranslation_EveryPointMovesByTheShift) {
    const double tx = 3.37, ty = -2.18;
    const Field f = solve(render(dictest::identity_motion()),
                          render(dictest::affine_motion(dictest::Mat2{}, dictest::Vec2{tx, ty}, kCentre)));
    REQUIRE(f.size() >= kMinPoints);
    double worst = 0, sum2 = 0, de = 0;
    for (const auto &kv : f) {
        const Point &p = kv.second;
        const double err = std::hypot(p[2] - tx, p[3] - ty);
        worst = std::max(worst, err);
        sum2 += err * err;
        de = std::max({de, (double)std::fabs(p[4]), (double)std::fabs(p[5]), (double)std::fabs(p[6])});
    }
    const double rms = std::sqrt(sum2 / (double)f.size());
    std::printf("    %zu points; |u - t| rms %.2e px, worst %.2e px; worst |strain| %.2e\n", f.size(),
                rms, worst, de);
    CHECK(rms <= 0.02);
    CHECK(worst <= 0.05);
    CHECK(de <= 3e-3);
}

// Moving the deformed image by whole pixels (rendered, so the frame stays
// full; inside the image it is the same pixels moved, which the test checks)
// adds the shift to every displacement and leaves strain and correlation.
// Measured: |du| 1.9e-3 px, |de| 1.0e-4, |dcorr| 5.5e-6.
TEST_CASE(Metamorphic, DeformedShiftedByWholePixels_AddsTheShift) {
    const cv::Mat ref = render(dictest::identity_motion());
    const dictest::Motion m = affine_load();
    const int sx = 4, sy = -3;
    dictest::Motion shifted = m;
    shifted.forward = [m, sx, sy](double X, double Y) {
        const dictest::Vec2 x = m.forward(X, Y);
        return dictest::Vec2{x.x + sx, x.y + sy};
    };
    shifted.inverse = [m, sx, sy](double x, double y) { return m.inv(x - sx, y - sy); };

    const cv::Mat def = render(m), def_shifted = render(shifted);
    const cv::Mat moved = def_shifted(cv::Rect(8, 0, kW - 8, kH - 8));
    const cv::Mat source = def(cv::Rect(8 - sx, -sy, kW - 8, kH - 8));
    REQUIRE(cv::countNonZero(moved != source) == 0);

    const Field a = solve(ref, def), b = solve(ref, def_shifted);
    const Diff d = compare(a, b, same_node, [&](Point p) {
        p[2] += (float)sx;
        p[3] += (float)sy;
        return p;
    });
    report("shift", a, b, d);
    REQUIRE(a.size() >= kMinPoints);
    CHECK(mostly_common(a, b, d));
    CHECK(d.u <= 6e-3);
    CHECK(d.e <= 3e-4);
    CHECK(d.corr <= 1e-4);
}

// Swapping the images inverts the map: X goes to x = X + u(X) one way, and x
// back to x + u'(x) = X the other. For an affine load both fields are affine,
// so the backward field's least-squares fit is evaluated where each forward
// point lands, and the two fitted gradients compose to the identity.
// Measured: worst |u(X) + u'(X + u(X))| 1.9e-2 px (the fit residual is
// 1.7e-2: it is the solve's own scatter), |(I + G')(I + G) - I| 2.3e-6.
TEST_CASE(Metamorphic, SwappedImages_InvertTheMap) {
    const cv::Mat ref = render(dictest::identity_motion());
    const cv::Mat def = render(affine_load());
    const Field fwd = solve(ref, def), bwd = solve(def, ref);
    REQUIRE(fwd.size() >= kMinPoints);
    REQUIRE(bwd.size() >= kMinPoints);
    const AffineFit f = fit(fwd), b = fit(bwd);
    double worst = 0;
    for (const auto &kv : fwd) {
        const Point &p = kv.second;
        const dictest::Vec2 ub = b.at(p[0] + p[2], p[1] + p[3]);
        worst = std::max({worst, std::fabs(p[2] + ub.x), std::fabs(p[3] + ub.y)});
    }
    // (I + Gb)(I + Gf), with G = [[ux, uy], [vx, vy]].
    const double fa = 1 + f.u[1], fb = f.u[2], fc = f.v[1], fd = 1 + f.v[2];
    const double ba = 1 + b.u[1], bb = b.u[2], bc = b.v[1], bd = 1 + b.v[2];
    const double p00 = ba * fa + bb * fc, p01 = ba * fb + bb * fd;
    const double p10 = bc * fa + bd * fc, p11 = bc * fb + bd * fd;
    const double grad = std::max({std::fabs(p00 - 1), std::fabs(p01), std::fabs(p10), std::fabs(p11 - 1)});
    std::printf("    %zu / %zu points; worst |u(X) + u'(X + u(X))| %.2e px; "
                "|(I + G')(I + G) - I| %.2e (ux %.5f, ux' %.5f)\n",
                fwd.size(), bwd.size(), worst, grad, f.u[1], b.u[1]);
    CHECK(worst <= 0.05);
    CHECK(grad <= 5e-5);
}

// ZNSSD does not see a gain and offset on the deformed image (a lighting
// change between frames), so the field stays. 0.7 I + 30 rounds to 8 bits and
// loses grey levels, so the images are only nearly related and the bound is
// wider. Measured: |du| 1.0e-2 px, |de| 6.9e-4, every point kept.
TEST_CASE(Metamorphic, DeformedGainAndOffset_SameField) {
    const cv::Mat ref = render(dictest::identity_motion());
    const cv::Mat def = render(affine_load());
    cv::Mat dimmer;
    def.convertTo(dimmer, CV_8U, 0.7, 30.0);
    const Field a = solve(ref, def), b = solve(ref, dimmer);
    const Diff d = compare(a, b, same_node, [](Point p) { return p; });
    report("gain", a, b, d);
    REQUIRE(a.size() >= kMinPoints);
    CHECK(mostly_common(a, b, d));
    CHECK(d.u <= 0.03);
    CHECK(d.e <= 2e-3);
}

// Mirroring both images in x mirrors the field: at the mirrored node, u and
// exy change sign and v, exx, eyy and the correlation stay.
// Measured: |du| 1.1e-3 px, |de| 7.7e-5, |dcorr| 1.0e-5.
TEST_CASE(Metamorphic, MirroredImages_MirrorTheField) {
    const cv::Mat ref = render(dictest::identity_motion());
    const cv::Mat def = render(affine_load());
    cv::Mat ref_m, def_m;
    cv::flip(ref, ref_m, 1);
    cv::flip(def, def_m, 1);
    const Field a = solve(ref, def), b = solve(ref_m, def_m);
    const Diff d = compare(
            a, b, [](std::pair<int, int> k) { return std::make_pair(kW - 1 - k.first, k.second); },
            [](Point p) {
                p[2] = -p[2];
                p[6] = -p[6];
                return p;
            });
    report("mirror", a, b, d);
    REQUIRE(a.size() >= kMinPoints);
    CHECK(mostly_common(a, b, d));
    CHECK(d.u <= 5e-3);
    CHECK(d.e <= 3e-4);
    CHECK(d.corr <= 1e-4);
}

#else

TEST_CASE(Metamorphic, OpenCvRequired_SkippedWithoutOpenCV) {
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
