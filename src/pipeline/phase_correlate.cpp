// L0 of anchor-lattice seeding: the global rigid shift, from every pixel in
// the ROI rather than from the median of sparse keypoint matches.
//
// The L0/L1 design is documented at the top of full_field_anchors.cpp, which is
// this function's only caller, and the evidence for it in
// docs/SEEDING_BENCHMARK.md.

#include <semper/seeding.hpp>
#include <semper/tuning.hpp>

#include <opencv2/imgproc.hpp>

#include <cmath>

namespace Semper {
namespace seeding {

PhaseCorrRef prepare_phase_correlate_ref(const cv::Mat &ref, const cv::Rect &roi) {
    PhaseCorrRef out;
    out.rect = roi & cv::Rect(0, 0, ref.cols, ref.rows);
    if (!out.usable()) return out;
    ref(out.rect).convertTo(out.ref32, CV_32F);
    cv::createHanningWindow(out.hann, out.ref32.size(), CV_32F);
    return out;
}

bool phase_correlate_roi(const cv::Mat &ref, const cv::Mat &def,
                         const cv::Rect &roi, double &out_u, double &out_v,
                         double &out_response) {
    if (def.cols != ref.cols || def.rows != ref.rows) return false;
    return phase_correlate_roi(prepare_phase_correlate_ref(ref, roi), def,
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

} // namespace seeding
} // namespace Semper
