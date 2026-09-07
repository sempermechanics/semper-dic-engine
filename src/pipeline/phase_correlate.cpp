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

bool phase_correlate_roi(const cv::Mat &ref, const cv::Mat &def,
                         const cv::Rect &roi, double &out_u, double &out_v,
                         double &out_response) {
    cv::Rect r = roi & cv::Rect(0, 0, ref.cols, ref.rows);
    if (r.width < 16 || r.height < 16) return false;
    if (def.cols != ref.cols || def.rows != ref.rows) return false;

    cv::Mat ref32, def32;
    ref(r).convertTo(ref32, CV_32F);
    def(r).convertTo(def32, CV_32F);

    cv::Mat hann;
    cv::createHanningWindow(hann, ref32.size(), CV_32F);

    double response = 0.0;
    // Sign convention verified against the engine's warp on a synthetic shift
    // (SeedBench.PhaseCorrelateSignConvention): phaseCorrelate(ref, def)
    // returns (u, v) such that a point at reference p appears at p + (u, v).
    cv::Point2d shift = cv::phaseCorrelate(ref32, def32, hann, &response);

    out_response = response;
    if (!(response > (double)tuning::kPhaseCorrMinResponse)) return false;
    if (!(std::abs(shift.x) < 1e6 && std::abs(shift.y) < 1e6)) return false;

    out_u = shift.x;
    out_v = shift.y;
    return true;
}

} // namespace seeding
} // namespace Semper
