#ifndef SEMPER_PIPELINE_INCREMENTAL_INTERNAL_HPP
#define SEMPER_PIPELINE_INCREMENTAL_INTERNAL_HPP

// Building blocks of incremental DIC (incremental_session.cpp), split out so
// they can be tested on exact fields without running a solve. Not public API.

#include "full_field_internal.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace Semper {
namespace pipeline {
namespace internal {

// A solved lattice in the form sampling wants: flat arrays, one validity flag.
struct NodeField {
    int gridW = 0, gridH = 0, rect_x = 0, rect_y = 0, step = 0;
    std::vector<float> u, v, ux, uy, vx, vy, corr;
    std::vector<unsigned char> valid;

    float node_x(int gx) const { return (float)(rect_x + gx * step); }
    float node_y(int gy) const { return (float)(rect_y + gy * step); }
    size_t size() const { return (size_t)gridW * gridH; }
};

// A node is valid when it holds a result (solved && corr >= 0) and survives
// the same displacement median test pack_full_field_output applies, so a
// wrong match that the output would drop is never composed onto a total.
// `rejected` (optional) receives how many nodes the median test removed.
NodeField node_field_from_capture(const FieldCapture &cap, int *rejected = nullptr);

struct IncrementSample {
    bool ok = false;
    bool taylor = false;   // first-order expansion from one node, not bilinear
    float u = 0.0f, v = 0.0f;
    float corr = 0.0f;     // worst (largest) ZNSSD of the nodes that contributed
};

// The increment at image point (x, y) of the field's reference.
//  - On a node: that node's value.
//  - Inside a cell whose four corners are valid: bilinear.
//  - Otherwise: u_n + ux_n*dx + uy_n*dy from the nearest valid node within
//    tuning::kIncTaylorMaxCells grid cells (ties go to the lowest row-major
//    index, so the choice is deterministic). The same expansion Path B uses
//    to guess a neighbour.
//  - Nothing within reach: ok = false.
IncrementSample sample_increment(const NodeField &f, float x, float y);

// Geometry of the frame-0 grid that total displacements are stored on.
struct GridGeom {
    int gridW = 0, gridH = 0, rect_x = 0, rect_y = 0, step = 0;
};

// Warp the frame-0 material mask into the configuration where frame-0 point X
// sits at X + U(X). For each output pixel p, solves X + U(X) = p by Newton on
// a continuous field built from the grid (invalid nodes filled by iterated
// neighbour averaging, bilinear inside, linear extrapolation up to
// kIncMaskExtrapCells cells outside), then reads material_mask0 at X.
// Output: CV_8U, 255 inside, 0 outside; all zero when no node is valid.
// `margin_px` widens the window searched around the forward-mapped grid.
// Rows are independent and each row is walked left to right with the previous
// pixel's solution as the starting point, so the result does not depend on
// thread scheduling.
cv::Mat warp_material_mask(const cv::Mat &material_mask0,
                           const GridGeom &geom,
                           const std::vector<float> &U,
                           const std::vector<float> &V,
                           const std::vector<unsigned char> &valid,
                           int margin_px);

} // namespace internal
} // namespace pipeline
} // namespace Semper

#endif
