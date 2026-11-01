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
 * Returned by run_full_field when nothing could seed the solve: the anchor
 * lattice solved no node, the global motion model (AKAZE) did not lock, and the
 * central-tendency fallback (Path C) did not converge either -- typically a
 * deformed frame unrelated to the reference, or motion beyond every search.
 * Added in 0.5.0; this case used to return an undocumented -1.
 */
constexpr int kSeedFailed = -5;

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
 * A solve clears its token on entry, so a stale cancel from a previous solve
 * never aborts the next one. A caller that does work of its own before the
 * solve starts (decoding the frame, waiting for a lock, allocating the output)
 * opens the window at its own entry instead, with [CancelScope]: a request
 * made during that work then stops the solve rather than being cleared by it.
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

    /** Opens the cancel window now: clears any earlier request, and the next
     *  solve's entry keeps the requests made from here on. Use [CancelScope]. */
    void begin() noexcept {
        flag_.store(false, std::memory_order_relaxed);
        window_open_.store(true, std::memory_order_release);
    }
    /** Closes a window that no solve entered (an early return). */
    void end() noexcept { window_open_.store(false, std::memory_order_release); }
    /** Solve entry (the engine's own entry points call it): clears a request
     *  left from before, unless [begin] opened the window for this call. */
    void enter() noexcept {
        if (!window_open_.exchange(false, std::memory_order_acq_rel)) clear();
    }

private:
    std::atomic<bool> flag_{false};
    std::atomic<bool> window_open_{false};
};

/**
 * One adapter call's cancel window: [CancelToken::begin] on construction, so
 * a cancel requested at any point of the call -- its own decoding and locking
 * included -- reaches the solve it starts; [CancelToken::end] on exit, so a
 * call that returned before reaching a solve leaves no window open.
 */
class CancelScope {
public:
    explicit CancelScope(CancelToken& token) noexcept : token_(token) { token_.begin(); }
    ~CancelScope() { token_.end(); }
    CancelScope(const CancelScope&) = delete;
    CancelScope& operator=(const CancelScope&) = delete;

private:
    CancelToken& token_;
};

/**
 * Cooperative cancellation for an in-flight solve — process-global convenience API.
 *
 * These act on one process-wide token, the one every solve started *without*
 * a [CancelToken] polls (the run_full_field and IncrementalSession overloads
 * that take none). The JNI adapter is that caller: it holds one lock around
 * one solve at a time. A solve started with its own token polls only that
 * token, so these functions never reach it, and its token never reaches
 * anything else (until 0.5.0 the token was bound into a process-global
 * pointer for the duration of the solve, so two concurrent solves polled one
 * token between them).
 *
 * The solver polls the flag inside its point loops, so a cancel takes effect
 * within a point or two instead of at the end of the frame. run_full_field clears
 * the token it polls on entry, so a stale cancel from a previous solve never
 * aborts the next one. [clear_cancel] remains public for callers that reset
 * explicitly. A solve that has seen a request finishes as cancelled even if
 * [clear_cancel] runs before it returns: clearing prepares the next solve, it
 * does not withdraw the cancel from the one in flight.
 *
 * Deliberately kept in a header of its own, with no OpenCV in it, so the flag can
 * be linked and tested without the rest of the pipeline.
 */
void request_cancel();
void clear_cancel();
bool cancel_requested();

namespace detail {

/** The process-wide token behind the free functions above. Internal: what the
 *  token-less overloads pass down to the solver. */
CancelToken& global_cancel_token() noexcept;

} // namespace detail

} // namespace pipeline
} // namespace Semper

#endif
