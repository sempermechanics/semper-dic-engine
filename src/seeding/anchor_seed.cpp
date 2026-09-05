// Anchor-lattice seeding — a seeder with no detector and no descriptor.
//
// Two layers:
//   L0  cv::phaseCorrelate over the ROI gives the global rigid shift from every
//       pixel, rather than from the median of sparse keypoint matches.
//   L1  a regular lattice of ROI grid nodes is solved with the engine's own
//       IC-GN, seeded from L0. The converged nodes become the Delaunay vertices.
//
// The point of the lattice is the guess *gradients*. build_mesh_guess_field
// derives (ux, uy, vx, vy) from cv::getAffineTransform over triangle vertices,
// so a vertex position error e over a triangle edge of length L produces a
// gradient error of order e/L. Keypoints give e ~ 1 px on short, clustered
// edges; IC-GN anchors give e ~ 0.01 px on edges fixed at stride * step.

#include <semper/seeding.hpp>
#include <semper/tuning.hpp>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <string>

namespace Semper {
namespace seeding {

void draw_outlined_text(cv::Mat &img, const std::string &text, cv::Point pt, double scale) {
    cv::putText(img, text, pt, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, text, pt, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

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
