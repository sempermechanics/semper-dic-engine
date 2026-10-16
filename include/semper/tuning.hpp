#ifndef SEMPER_TUNING_HPP
#define SEMPER_TUNING_HPP

// Named constants for the DIC pipeline / strain paths.
// Values match the pre-centralization literals exactly (rename-only). Do not
// change a number here without re-baselining throughput and field quality
// against docs/engine/PERF_BASELINE_bd44af0.md.

namespace Semper {
namespace tuning {

/** ZNSSD acceptance gate used by run_full_field Path A/B/prewarm. */
inline constexpr float kCorrAccept = 0.15f;

/** Correlation score above which calculate_deformation may try simplex. */
inline constexpr float kCorrSimplexTrigger = 0.4f;

/** Simplex residual above which the point is marked failed. */
inline constexpr float kSimplexFailScore = 0.1f;

/** Levenberg–Marquardt diagonal damping applied in the full-field paths. */
inline constexpr float kLmAlpha = 0.05f;

/** ICGN iteration budget (also the Path A/B "timeout" classify threshold). */
inline constexpr int kIcgnMaxIter = 50;

/** Reject a subset if more than this fraction of ref pixels hit the ghost wall. */
inline constexpr float kGhostRejectFraction = 0.05f;

/** Anchor convex-hull coverage for SPARSE vs clustered routing. */
inline constexpr float kAnchorSparseCoverage = 0.05f;

/** Anchor-lattice seeding: target number of lattice nodes over the ROI. */
inline constexpr int kAnchorTarget = 256;

/** Anchor-lattice seeding: minimum lattice stride in grid nodes. */
inline constexpr int kAnchorStrideMin = 1;

/** Anchor-lattice seeding: ZNSSD gate for keeping an anchor as a mesh vertex.
 *  Deliberately looser than kCorrAccept: a seed only has to be approximately
 *  right, and the universal median test below rejects the blunders. Gating
 *  seeds at the final acceptance threshold discards usable vertices on noisy
 *  images and collapses mesh coverage. */
inline constexpr float kAnchorAcceptScore = 0.40f;

/** Anchor-lattice seeding: universal-median-test rejection radius (px). */
inline constexpr float kAnchorMedianTol = 1.0f;

/** Anchor-lattice seeding: accepted/attempted fraction for a FULL mesh. */
inline constexpr float kAnchorFullFraction = 0.60f;

/** Anchor-lattice seeding: accepted vertices needed for a FULL mesh. Three is
 *  enough for one Delaunay triangle; ten is the point past which the mesh is
 *  dense enough to guide Path A over the whole ROI rather than a corner of it. */
inline constexpr int kAnchorFullMinVertices = 10;

/** Anchor-lattice seeding: convex-hull coverage of the ROI for a FULL mesh. */
inline constexpr float kAnchorFullCoverage = 0.30f;

/** Phase-correlation peak response below which the global shift is not trusted. */
inline constexpr float kPhaseCorrMinResponse = 0.05f;

/** Log-polar (Fourier-Mellin) rotation/scale estimate: peak response below
 *  which the estimate is not trusted and seeding stays translation-only.
 *  A turn the estimate resolves scores 0.27-0.86 on a 110 px ROI at sigma
 *  0-4 GL (0.24 at 80 px, sigma 2); a uniaxial stretch of 10% or more, which
 *  a similarity cannot describe, scores 0.11-0.20. Below 0.2 such frames
 *  would pass to the anchor lattice, fail kSimilarityKeepFraction and be
 *  solved twice (docs/SEEDING_BENCHMARK.md 4.5.2). */
inline constexpr float kLogPolarMinResponse = 0.2f;

/** Log-polar estimate: the square crop it is taken from. Below the minimum
 *  side the spectrum is too coarse in angle to be worth correlating; above the
 *  maximum, the central square is used, which bounds the cost on large ROIs. */
inline constexpr int kLogPolarMinSide = 64;
inline constexpr int kLogPolarMaxSide = 512;

/** Seed anchors with the rotation/scale model only when it moves the ROI
 *  corners by more than this (px) relative to the bare translation. Below it,
 *  IC-GN pulls in from the translation alone and seeding stays bit-identical
 *  to the translation-only path. Set above the 2.8 px the FullFieldGolden
 *  affine scenario implies, and below the ~8 px at which 2 deg on a 320 px
 *  ROI already costs CPU (docs/SEEDING_BENCHMARK.md 4.5). */
inline constexpr float kSimilarityMinCornerPx = 4.0f;

/** The similarity is trusted only if the anchor lattice it seeds routes FULL
 *  and keeps at least this fraction of its attempted anchors; otherwise the
 *  lattice is solved again from the translation alone and that one is used.
 *  The log-polar estimate is a similarity; under strong anisotropic stretch it
 *  can report a rotation that is not there (IncBench S5/S9: about 90 deg with
 *  noise, -5.8 deg x 0.92 without), and the lattice it seeds can then hold a
 *  self-consistent cluster of wrong anchors that no gate catches. Rotations
 *  the similarity fits keep ~100%. */
inline constexpr float kSimilarityKeepFraction = 0.90f;

/** Packed-field sentinel for "no valid correlation at this grid point". */
inline constexpr float kCorrInvalid = -1.0f;

/** Strain calculator write sentinel (VSG). Reader tests <= kStrainFailSentinel. */
inline constexpr float kStrainUninitSentinel = -1000.0f;

/** Strain drop-filter threshold in run_full_field (accepts the write sentinel). */
inline constexpr float kStrainFailSentinel = -999.0f;

/** Path B queue wait cancel-poll interval (ms). */
inline constexpr int kCancelPollMs = 20;

/**
 * Path B flood fill: solved nodes expanded per round (full_field_path_b.cpp).
 * A fixed number, not the core count, so the field does not depend on the
 * device. Larger rounds give the workers more cells at once but follow the
 * reliability order less strictly. Changing it changes the field.
 */
inline constexpr int kPathBBatchNodes = 32;

/** Floating-point radius fudge used in VSG window membership. */
inline constexpr double kVsgRadiusTiny = 1.0e-5;

/** Displacement median test: neighbourhood half-width in grid points (5x5). */
inline constexpr int kOutlierRadius = 2;

/** Displacement median test: noise floor added to the residual median, px. */
inline constexpr float kOutlierEpsPx = 0.1f;

/**
 * Displacement median test: normalized residual above which a point is
 * rejected. Westerweel & Scarano use 2 on 3x3; 3 on 5x5 rejects the same
 * wrong matches on real bending data and leaves more room for noise.
 */
inline constexpr float kOutlierThreshold = 3.0f;

/** Displacement median test: fewer accepted neighbours than this, no test. */
inline constexpr int kOutlierMinNeighbours = 3;

/** Displacement median test: pass limit (it stops early once stable). */
inline constexpr int kOutlierMaxPasses = 5;

/**
 * Incremental DIC: furthest a material point may sit from the nearest solved
 * node (in grid cells) and still take that node's first-order expansion when
 * bilinear interpolation lacks a corner. 1.5 covers a diagonal neighbour
 * (sqrt(2) cells); beyond it the point is dropped.
 */
inline constexpr float kIncTaylorMaxCells = 1.5f;

/**
 * Incremental DIC mask warp: cap on frame-0 samples per pixel per axis when
 * the material mask is pushed forward through X + U(X). The warp uses enough
 * samples that neighbours land within 0.7 px of each other at the largest
 * local stretch (2 per axis for a rigid or mildly strained body); 8 covers a
 * local stretch of 5.6 between references.
 */
inline constexpr int kIncMaskMaxSamplesPerPx = 8;

/**
 * Incremental DIC coasting: a tracked node whose increment cannot be sampled
 * (no solved node within kIncTaylorMaxCells) is carried on a plane fitted to
 * the solved increments within this many grid cells, instead of being dropped.
 * It is not reported while it coasts, and it is measured again as soon as the
 * solve reaches it. 8 cells spans a band one subset wide with solved nodes on
 * both sides at the default step-to-subset ratio.
 */
inline constexpr int kIncCoastRadiusCells = 8;

/** Incremental DIC coasting: solved nodes the plane fit needs. */
inline constexpr int kIncCoastMinNodes = 6;

/**
 * Incremental DIC coasting: how far (in subset sizes, image pixels) a node may
 * travel on fitted increments before it is dropped. Crossing a stationary
 * edge takes about one subset of travel; past two, the node is not coming back.
 */
inline constexpr float kIncCoastMaxTravelSubsets = 2.0f;

/**
 * Incremental DIC re-anchoring: a coasted node whose increment can be sampled
 * again is not reported until its subset from the last reference it was
 * measured on matches the current frame directly. The match may move the node
 * from where coasting left it by at most this fraction of the distance it
 * coasted (a plane fitted over a gap drifts by up to ~20% of the motion across
 * it), and never less than kIncReanchorMinShiftPx.
 */
/**
 * Incremental DIC re-anchoring: a node that coasted no further than this (in
 * subset sizes) is reported again as soon as it is measured, without a direct
 * match. A short coast, typically a frame or two at the edge of a gap, adds
 * an error well below the measurement noise; a node that has never had a
 * clean subset (it sat in the gap on frame 0) could not be confirmed anyway.
 */
inline constexpr float kIncCoastTrustedTravelSubsets = 0.25f;
inline constexpr float kIncReanchorMaxShiftFraction = 0.3f;
inline constexpr float kIncReanchorMinShiftPx = 1.0f;

/** Incremental DIC re-anchoring: pixels kept around a stored subset, past its
 *  edge, so its intensity gradients do not see the edge of the patch. */
inline constexpr int kIncAnchorMarginPx = 10;

/**
 * Incremental DIC early update: with an interval N > 1, the reference moves to
 * the frame just solved, without waiting out N, when the points it reports
 * fall this fraction below those reported on the first frame against the
 * current reference. The increment has grown past what the images still
 * correlate over (fast loading, necking, whitening), and the frames after it
 * correlate against something closer. The comparison is with this reference's
 * own first frame, so a band that always fails does not shorten N.
 */
inline constexpr float kIncEarlyUpdateDrop = 0.05f;

/**
 * Incremental DIC mask warp: how far past the outermost frame-0 nodes (in
 * grid cells, plus half a subset) the warped mask is evaluated. The field is
 * extended linearly from the edge cells over that window. The material
 * reaches up to ceil(half*sqrt(2)) + 1 px past the outer nodes (no user
 * mask), so this has to cover more than one cell.
 */
inline constexpr int kIncMaskExtrapCells = 3;

} // namespace tuning
} // namespace Semper

#endif // SEMPER_TUNING_HPP
