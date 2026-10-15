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

// The increment at (x, y) estimated, not measured: a weighted least-squares
// plane u = a + b*dx + c*dy (and v likewise) through the valid nodes within
// `radius_px`, weights 1 / (1 + (d / step)^2). For a node that sits in a band
// no subset can match across (a stationary background edge under translucent
// material, a reflection) while its neighbours on either side still solve.
// Needs tuning::kIncCoastMinNodes nodes spread over both axes (the smaller
// principal spread at least one grid step); otherwise ok = false. corr is the
// worst ZNSSD of the nodes used. Exact for a linear increment field.
IncrementSample fit_increment(const NodeField &f, float x, float y, float radius_px);

// A coasting node's subset on the last reference it was measured on, kept so
// the node can be matched directly when it can be measured again: coasting
// leaves it where a fitted plane put it, and a measurement started there
// would carry that error for good.
struct CoastAnchor {
    cv::Mat patch;          // gray around (ax, ay), tuning::kIncAnchorMarginPx past the subset
    int ax = 0, ay = 0;     // the pixel the subset is centred on, on that reference
    float dx = 0, dy = 0;   // the node's position there minus (ax, ay)
};

// What to do with the anchor of a measured node at (x, y) on the field's
// reference. Its subset is clean when every grid node within it (half a
// subset plus a step) solved; a gap is within reach when a grid position
// within two subsets plus two steps did not solve, or is off the grid.
//  - kDrop:    clean, no gap within reach. It is not about to coast.
//  - kRefresh: clean, a gap within reach. Cut a new anchor here.
//  - kKeep:    its subset is not clean (it is at the gap's edge). The anchor
//              it already has, cut where it was clean, is the one to keep.
// Distances are taken from gap_distance(f), computed once per field.
enum class AnchorNeed { kDrop, kRefresh, kKeep };
struct GapDistance {
    cv::Mat hole;   // CV_32F per node: px to the nearest unsolved node
    cv::Mat gap;    // CV_32F per node: px to the nearest unsolved or off-grid position
};
GapDistance gap_distance(const NodeField &f);
AnchorNeed anchor_need(const NodeField &f, const GapDistance &g, float x, float y, int subset_size);

// Cut the anchor for a node at (x, y) on `ref_gray`. False when the patch
// would leave the image.
bool make_anchor(const cv::Mat &ref_gray, float x, float y, int subset_size, CoastAnchor &out);

// Match the anchor's subset on `def` (a prepared deformed image), starting
// from (guess_x, guess_y), where the node is thought to be now. True when the
// match converges, its ZNSSD is within tuning::kCorrAccept, and it puts the
// node within `max_shift` px of the guess; then (x, y) is where the node is
// and corr the match's ZNSSD.
bool match_anchor(const CoastAnchor &a, const Image &def, int subset_size, bool use_6x6,
                  float guess_x, float guess_y, float max_shift, float &x, float &y, float &corr);

// Geometry of the frame-0 grid that total displacements are stored on.
struct GridGeom {
    int gridW = 0, gridH = 0, rect_x = 0, rect_y = 0, step = 0;
};

// Warp the frame-0 material mask into the configuration where frame-0 point X
// sits at X + U(X). A forward warp: every material pixel of the frame-0 window
// (the grid widened by `margin_px`) is sampled 2+ times per axis (enough for
// the largest local stretch), each sample is pushed to X + U(X) on a
// continuous field built from the grid (invalid nodes filled by iterated
// neighbour averaging, bilinear inside, planar extension outside), and the
// landed pixels are closed with a 3x3 kernel to fill pinholes.
// Output: CV_8U, 255 inside, 0 outside; all zero when no node is valid.
// Every write stores 255, so the result does not depend on thread scheduling.
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
