// =====================================================================
// SUITE: CancelToken — native/src/pipeline/cancel.cpp
//
// The flag the solver polls to stop a frame mid-solve. Before this
// existed a cancel only took effect between frames, so pressing Cancel
// on a slow frame left the engine running for seconds.
//
// The polling itself lives inside run_full_field and needs a real solve
// (OpenCV, threads) to exercise, so it is covered by the Android-side
// tests. What is pinned here is the contract those polls depend on:
//
//  • the flag is sticky — it survives the solve it stopped, so the
//    caller must clear it before starting a run that should proceed;
//  • it is visible across threads, since the flag is set on the UI
//    thread and read by the solver's workers.
// =====================================================================
#include "framework/test_framework.h"
#include <semper/cancel.hpp>
#include <semper/metrics.hpp>

#include <atomic>
#include <thread>

using Semper::pipeline::cancel_requested;
using Semper::pipeline::CancelToken;
using Semper::pipeline::clear_cancel;
using Semper::pipeline::request_cancel;

TEST_CASE(CancelToken, StartsClear) {
    clear_cancel();
    CHECK(!cancel_requested());
}

TEST_CASE(CancelToken, RequestIsObserved) {
    clear_cancel();
    request_cancel();
    CHECK(cancel_requested());
}

TEST_CASE(CancelToken, StaysSetUntilCleared) {
    clear_cancel();
    request_cancel();
    // Repeated polls are exactly what the solver does, once per point.
    for (int i = 0; i < 1000; ++i) {
        CHECK(cancel_requested());
    }
    clear_cancel();
    CHECK(!cancel_requested());
}

TEST_CASE(CancelToken, CrossesThreads) {
    clear_cancel();
    std::atomic<bool> saw_cancel(false);
    std::atomic<bool> ready(false);

    // Stands in for a solver worker spinning on the flag.
    std::thread worker([&]() {
        ready.store(true, std::memory_order_release);
        for (int i = 0; i < 10000000 && !saw_cancel.load(std::memory_order_relaxed); ++i) {
            if (cancel_requested()) saw_cancel.store(true, std::memory_order_relaxed);
        }
    });

    while (!ready.load(std::memory_order_acquire)) {
    }
    request_cancel();
    worker.join();

    CHECK(saw_cancel.load(std::memory_order_relaxed));
    clear_cancel();
}

TEST_CASE(CancelToken, CancelledCodeIsDistinctFromEngineErrors) {
    // -2 is a ROI error and -3 an init error; a cancel must not be mistaken
    // for either, and must match AnalysisRunCodes.ERROR_CANCELLED on the
    // Kotlin side, which is what the UI keys "cancelled" off.
    CHECK(Semper::pipeline::kCancelled == -99);
    CHECK(Semper::pipeline::kCancelled != -2);
    CHECK(Semper::pipeline::kCancelled != -3);
}

// A caller-owned token: cancelling one solve must never affect another. This is
// the property the SDK / Python bindings rely on when several engine instances
// exist in one process.
TEST_CASE(CancelToken, InstancesAreIndependent) {
    CancelToken a;
    CancelToken b;
    CHECK(!a.requested());
    CHECK(!b.requested());
    a.request();
    CHECK(a.requested());
    CHECK(!b.requested());  // isolation: one token never bleeds into another
    a.clear();
    CHECK(!a.requested());
}

// The free functions act on one process-wide token: the one the token-less
// run_full_field overload (the JNI path) hands to the solver. A caller-owned
// token is never touched by them, which is what lets two solves with their
// own tokens run at once (CancelToken.ConcurrentEnginesCancelIndependently).
TEST_CASE(CancelToken, FreeFunctionsActOnTheGlobalTokenOnly) {
    clear_cancel();
    CancelToken solve;
    CancelToken& global = Semper::pipeline::detail::global_cancel_token();
    request_cancel();          // stands in for the JNI setCancelRequested(true)
    CHECK(global.requested());
    CHECK(cancel_requested());
    CHECK(!solve.requested()); // a per-solve token never sees it
    solve.request();
    clear_cancel();
    CHECK(!global.requested());
    CHECK(solve.requested());  // ...nor is it cleared by it
}

// The metrics writer copies the largest prefix the caller's buffer holds;
// every adapter goes through this one table. 18 is not a prefix the contract
// offers (slots 17 and 18 came in together), so an 18-float buffer gets 17.
TEST_CASE(Metrics, PrefixLenTable) {
    using Semper::pipeline::kMetricsLen;
    using Semper::pipeline::metrics_prefix_len;
    CHECK(kMetricsLen == 24);
    CHECK(metrics_prefix_len(-1) == 0);
    CHECK(metrics_prefix_len(0) == 0);
    CHECK(metrics_prefix_len(15) == 0);
    CHECK(metrics_prefix_len(16) == 16);
    CHECK(metrics_prefix_len(17) == 17);
    CHECK(metrics_prefix_len(18) == 17);
    for (int n = 19; n <= 24; ++n) CHECK(metrics_prefix_len(n) == n);
    CHECK(metrics_prefix_len(25) == 24);
    CHECK(metrics_prefix_len(1000) == 24);
}
