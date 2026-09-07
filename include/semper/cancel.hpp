#ifndef SEMPER_CANCEL_HPP
#define SEMPER_CANCEL_HPP

#include <atomic>

namespace Semper {
namespace pipeline {

/** Returned by run_full_field when a solve was stopped by [request_cancel]. */
constexpr int kCancelled = -99;

/**
 * Returned by run_full_field when `strain_window` is too small for `step` for
 * the VSG plane fit to have 3 grid nodes to work with. Added in 0.3.0; before
 * that this configuration returned 0 points and a success code, which no
 * caller could tell apart from a genuinely empty ROI.
 *
 * Added rather than folded into -2 so a caller can distinguish "your ROI is
 * wrong" from "your strain window is wrong" — they have different fixes.
 * Existing `if (n < 0)` error handling needs no change.
 */
constexpr int kBadStrainWindow = -4;

/**
 * A cancellation flag scoped to one solve — thread-safe, so a caller on any
 * thread can stop an in-flight run while the solver's workers poll it.
 *
 * Prefer this over the process-global free functions below when embedding the
 * engine as a library (SDK / Python bindings): each engine instance owns its own
 * token, so cancelling one solve never touches another. Relaxed atomics: a poll
 * that reads a stale `false` only costs one more point, and nothing downstream
 * orders against this flag.
 *
 * Non-copyable / non-movable (holds a std::atomic); pass by reference.
 */
class CancelToken {
public:
    CancelToken() = default;
    CancelToken(const CancelToken&) = delete;
    CancelToken& operator=(const CancelToken&) = delete;

    void request() noexcept { flag_.store(true, std::memory_order_relaxed); }
    void clear() noexcept { flag_.store(false, std::memory_order_relaxed); }
    bool requested() const noexcept { return flag_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> flag_{false};
};

/**
 * Cooperative cancellation for an in-flight solve — process-global convenience API.
 *
 * These delegate to the currently *active* token: the one bound by the
 * [CancelToken]-taking overload of run_full_field for the duration of that solve,
 * or a process-wide default token when none is bound. Only one solve runs at a
 * time (the JNI layer holds the reference-cache mutex for the duration of one),
 * so a single active target is sufficient and keeps the polling call sites inside
 * the solver unchanged.
 *
 * The solver polls the flag inside its point loops, so a cancel takes effect
 * within a point or two instead of at the end of the frame. run_full_field clears
 * the active flag on entry, so a stale cancel from a previous solve never aborts
 * the next one. [clear_cancel] remains public for callers that reset explicitly.
 *
 * Deliberately kept in a header of its own, with no OpenCV in it, so the flag can
 * be linked and tested without the rest of the pipeline.
 */
void request_cancel();
void clear_cancel();
bool cancel_requested();

namespace detail {

/**
 * Binds `token` as the active cancel target for the enclosing solve and restores
 * the previous target on scope exit. Used by the [CancelToken]-taking overload of
 * run_full_field so the solver's existing `cancel_requested()` polls route to the
 * per-solve token. Single-solve-at-a-time (see the note above).
 */
class ScopedCancelToken {
public:
    explicit ScopedCancelToken(CancelToken& token);
    ~ScopedCancelToken();
    ScopedCancelToken(const ScopedCancelToken&) = delete;
    ScopedCancelToken& operator=(const ScopedCancelToken&) = delete;

private:
    CancelToken* prev_;
};

} // namespace detail

} // namespace pipeline
} // namespace Semper

#endif
