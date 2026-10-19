#ifndef SEMPER_METRICS_HPP
#define SEMPER_METRICS_HPP

// The metrics buffer run_full_field fills (docs/CONTRACT.md A.4). OpenCV-free,
// like cancel.hpp, so adapters and tests can use it without the pipeline.

namespace Semper {
namespace pipeline {

/** Slots the engine fills when the caller's buffer holds them all (v0.5.0:
 *  slot 23, points dropped by the displacement median test, was appended). */
constexpr int kMetricsLen = 24;

/**
 * How many slots a buffer of `metrics_len` floats receives: the largest
 * prefix the contract offers that fits. Below the 16-slot minimum nothing is
 * written; 18 gets 17, because slots 17 and 18 came in together. Every writer
 * (run_full_field, the JNI copy-back) goes through this one table.
 */
constexpr int metrics_prefix_len(int metrics_len) {
    return metrics_len < 16 ? 0
         : metrics_len >= kMetricsLen ? kMetricsLen
         : metrics_len == 18 ? 17
         : metrics_len;
}

} // namespace pipeline
} // namespace Semper

#endif // SEMPER_METRICS_HPP
