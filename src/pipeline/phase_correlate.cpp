// L0 of anchor-lattice seeding: the global motion, from every pixel in the ROI
// rather than from the median of sparse keypoint matches.
//
// Two estimates, both phase correlation and both imgproc only:
//
//   translation      cv::phaseCorrelate over the ROI.
//   rotation/scale   Fourier-Mellin: the magnitude spectrum ignores translation
//                    and rotates and scales with the image, so phase
//                    correlation of its log-polar resampling turns rotation
//                    into a shift along the angle axis and scale into a shift
//                    along log-radius. The translation is then re-measured on
//                    the deformed image with that rotation and scale undone.
//
// The L0/L1 design is documented at the top of full_field_anchors.cpp, which is
// this file's only pipeline caller, the derivation in docs/MATHEMATICS.md 8.1,
// and the evidence in docs/SEEDING_BENCHMARK.md.

#include <semper/seeding.hpp>
#include <semper/tuning.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace Semper {
namespace seeding {

namespace {

// Log-polar grid. Rows sample angle over [0, pi) -- the magnitude spectrum of a
// real image is point-symmetric, so the other half repeats it and the angle
// axis is exactly periodic, which is what the FFT inside phaseCorrelate
// assumes. 256 rows is 0.70 deg per row before sub-pixel peak refinement.
// Columns sample log-radius between kLpRadiusLo and kLpRadiusHi of the
// Nyquist radius: the low end drops the DC and window leakage, where angle is
// also barely resolved; the high end stays clear of the corners, which a
// circle of radius N/2 never reaches anyway.
constexpr int kLpAngles = 256;
constexpr int kLpRadii = 128;
constexpr double kLpRadiusLo = 1.0 / 8.0;
constexpr double kLpRadiusHi = 0.9;

// |F| of the windowed, mean-removed crop, quadrant-swapped so DC sits at
// (N/2, N/2). N is even, so the swap is its own inverse.
cv::Mat magnitude_spectrum(const cv::Mat &crop, const cv::Mat &hann) {
    cv::Mat w;
    crop.convertTo(w, CV_32F, 1.0, -cv::mean(crop)[0]);
    cv::multiply(w, hann, w);
    cv::Mat F;
    cv::dft(w, F, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat planes[2];
    cv::split(F, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);

    const int h = mag.rows / 2, wd = mag.cols / 2;
    cv::Mat q0(mag, cv::Rect(0, 0, wd, h)), q1(mag, cv::Rect(wd, 0, wd, h));
    cv::Mat q2(mag, cv::Rect(0, h, wd, h)), q3(mag, cv::Rect(wd, h, wd, h));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    return mag;
}

// log(1 + |F|) on the log-polar grid. The log compresses the spectrum's
// dynamic range so the low frequencies do not dominate the correlation; it is
// taken after resampling because the grid holds a third of the samples.
cv::Mat log_polar_of(const cv::Mat &img, const PhaseCorrRef &ref) {
    const cv::Mat spec = magnitude_spectrum(img(ref.lp_rect), ref.lp_hann);
    cv::Mat lp;
    cv::remap(spec, lp, ref.lp_mapx, ref.lp_mapy, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT, cv::Scalar(0));
    lp += 1.0;
    cv::log(lp, lp);
    return lp;
}

} // namespace

PhaseCorrRef prepare_phase_correlate_ref(const cv::Mat &ref, const cv::Rect &roi,
                                         bool with_log_polar) {
    PhaseCorrRef out;
    out.rect = roi & cv::Rect(0, 0, ref.cols, ref.rows);
    if (!out.usable()) return out;
    ref(out.rect).convertTo(out.ref32, CV_32F);
    cv::createHanningWindow(out.hann, out.ref32.size(), CV_32F);

    if (!with_log_polar) return out;
    int side = std::min({out.rect.width, out.rect.height, tuning::kLogPolarMaxSide});
    side &= ~1;
    if (side < tuning::kLogPolarMinSide) return out;

    out.lp_rect = cv::Rect(out.rect.x + (out.rect.width - side) / 2,
                           out.rect.y + (out.rect.height - side) / 2, side, side);
    cv::createHanningWindow(out.lp_hann, cv::Size(side, side), CV_32F);

    const double c = side / 2.0;
    const double r_lo = kLpRadiusLo * c, r_hi = kLpRadiusHi * c;
    out.lp_dlogr = std::log(r_hi / r_lo) / (double)(kLpRadii - 1);
    out.lp_mapx.create(kLpAngles, kLpRadii, CV_32F);
    out.lp_mapy.create(kLpAngles, kLpRadii, CV_32F);
    for (int a = 0; a < kLpAngles; ++a) {
        const double phi = CV_PI * (double)a / (double)kLpAngles;
        const double ca = std::cos(phi), sa = std::sin(phi);
        float *mx = out.lp_mapx.ptr<float>(a);
        float *my = out.lp_mapy.ptr<float>(a);
        for (int r = 0; r < kLpRadii; ++r) {
            const double rho = r_lo * std::exp(out.lp_dlogr * (double)r);
            mx[r] = (float)(c + rho * ca);
            my[r] = (float)(c + rho * sa);
        }
    }

    // Hann along log-radius, flat along angle: the angle axis is periodic, so
    // windowing it would only throw signal away.
    out.lp_win.create(kLpAngles, kLpRadii, CV_32F);
    for (int r = 0; r < kLpRadii; ++r) {
        const float w = (float)(0.5 * (1.0 - std::cos(2.0 * CV_PI * r / (kLpRadii - 1))));
        for (int a = 0; a < kLpAngles; ++a) out.lp_win.at<float>(a, r) = w;
    }

    out.lp_ref = log_polar_of(ref, out);
    return out;
}

bool phase_correlate_roi(const cv::Mat &ref, const cv::Mat &def,
                         const cv::Rect &roi, double &out_u, double &out_v,
                         double &out_response) {
    if (def.cols != ref.cols || def.rows != ref.rows) return false;
    return phase_correlate_roi(prepare_phase_correlate_ref(ref, roi, false), def,
                               out_u, out_v, out_response);
}

bool phase_correlate_roi(const PhaseCorrRef &ref, const cv::Mat &def,
                         double &out_u, double &out_v, double &out_response) {
    if (!ref.usable()) return false;
    const cv::Rect &r = ref.rect;
    if (r.x + r.width > def.cols || r.y + r.height > def.rows) return false;

    cv::Mat def32;
    def(r).convertTo(def32, CV_32F);
    // cv::phaseCorrelate multiplies the window into its inputs IN PLACE when
    // their size is already DFT-optimal (no padding copy), so the stored crop
    // must not be handed over directly or every reuse would window it again.
    cv::Mat ref32 = ref.ref32.clone();

    double response = 0.0;
    // Sign convention verified against the engine's warp on a synthetic shift
    // (SeedBench.PhaseCorrelateSignConvention): phaseCorrelate(ref, def)
    // returns (u, v) such that a point at reference p appears at p + (u, v).
    cv::Point2d shift = cv::phaseCorrelate(ref32, def32, ref.hann, &response);

    out_response = response;
    if (!(response > (double)tuning::kPhaseCorrMinResponse)) return false;
    if (!(std::abs(shift.x) < 1e6 && std::abs(shift.y) < 1e6)) return false;

    out_u = shift.x;
    out_v = shift.y;
    return true;
}

bool log_polar_rotation_scale(const PhaseCorrRef &ref, const cv::Mat &def,
                              double &out_theta_rad, double &out_scale,
                              double &out_response) {
    if (!ref.lp_usable()) return false;
    const cv::Rect &r = ref.lp_rect;
    if (r.x + r.width > def.cols || r.y + r.height > def.rows) return false;

    cv::Mat lp_def = log_polar_of(def, ref);
    cv::Mat lp_ref = ref.lp_ref.clone();   // phaseCorrelate windows in place
    double response = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(lp_ref, lp_def, ref.lp_win, &response);
    out_response = response;
    if (!(response > (double)tuning::kLogPolarMinResponse)) return false;

    // A turn by theta rotates the spectrum by theta: the deformed rows sit
    // theta / (pi / kLpAngles) further down. Scaling the image by s shrinks
    // the spectrum by s: the deformed columns sit log(s) / dlogr further left.
    const double theta = shift.y * CV_PI / (double)kLpAngles;
    const double scale = std::exp(-shift.x * ref.lp_dlogr);
    if (!(std::abs(theta) < CV_PI / 2.0) || !(scale > 0.5 && scale < 2.0)) return false;

    out_theta_rad = theta;
    out_scale = scale;
    return true;
}

void phase_correlate_similarity(const PhaseCorrRef &ref, const cv::Mat &def,
                                SimilarityShift &out) {
    out = SimilarityShift{};
    if (!ref.usable()) return;
    const cv::Rect &r = ref.rect;
    out.cx = r.x + 0.5 * (double)(r.width - 1);
    out.cy = r.y + 0.5 * (double)(r.height - 1);

    // The translation-only answer comes first and is kept unless the rotation
    // model beats it, so small motion sees exactly the values it always did.
    double u0 = 0.0, v0 = 0.0, r0 = 0.0;
    out.locked = phase_correlate_roi(ref, def, u0, v0, r0);
    out.u = u0; out.v = v0; out.response = r0;

    double theta = 0.0, s = 1.0, lp_resp = 0.0;
    out.lp_ok = log_polar_rotation_scale(ref, def, theta, s, lp_resp);
    out.lp_response = lp_resp;
    if (!out.lp_ok) return;
    out.theta_rad = theta;
    out.scale = s;

    // s R - I is itself a scaled rotation, so its operator norm is just the
    // length of its first column; times the half-diagonal, that is how far the
    // bare translation is off at the ROI corners.
    const double a = s * std::cos(theta) - 1.0, b = s * std::sin(theta);
    const double arm = 0.5 * std::hypot((double)r.width, (double)r.height);
    if (!(arm * std::hypot(a, b) > (double)tuning::kSimilarityMinCornerPx)) return;

    // Undo the rotation and scale about the ROI centre: sample the deformed
    // image at c + s R (x - c). What remains is a pure translation t' with
    // t = s R t'. The source is the whole image, so a turned ROI still finds
    // the pixels that rotated out of its rectangle.
    const double sc = s * std::cos(theta), ss = s * std::sin(theta);
    const double x0 = r.x - out.cx, y0 = r.y - out.cy;
    const cv::Mat M = (cv::Mat_<double>(2, 3) <<
            sc, -ss, out.cx + sc * x0 - ss * y0,
            ss,  sc, out.cy + ss * x0 + sc * y0);
    cv::Mat undone;
    cv::warpAffine(def, undone, M, r.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_REPLICATE);
    cv::Mat und32;
    undone.convertTo(und32, CV_32F);
    cv::Mat ref32 = ref.ref32.clone();
    double r1 = 0.0;
    const cv::Point2d tp = cv::phaseCorrelate(ref32, und32, ref.hann, &r1);
    if (!(r1 > (double)tuning::kPhaseCorrMinResponse) || !(r1 > r0)) return;
    if (!(std::abs(tp.x) < 1e6 && std::abs(tp.y) < 1e6)) return;

    out.u = sc * tp.x - ss * tp.y;
    out.v = ss * tp.x + sc * tp.y;
    out.ux = a;  out.uy = -b;
    out.vx = b;  out.vy = a;
    out.response = r1;
    out.locked = true;
    out.similarity = true;
}

} // namespace seeding
} // namespace Semper
