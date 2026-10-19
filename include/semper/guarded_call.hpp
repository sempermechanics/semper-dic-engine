#ifndef SEMPER_GUARDED_CALL_HPP
#define SEMPER_GUARDED_CALL_HPP

// The exception barrier at the language boundaries (C ABI, JNI). Internal:
// not one of the Stable headers listed in docs/CONTRACT.md A.1.
//
// A C++ exception that reaches an extern "C" or JNI frame terminates the
// process: there is nothing on the other side that can catch it. The engine
// reports failure with return codes, but OpenCV and the standard library can
// still throw (std::bad_alloc on an allocation the input sized, cv::Exception
// from a codec), so every entry point runs its body through guarded_call and
// turns whatever escapes into the error value it already documents (-3, init
// failure, for the solve entry points).

#include <exception>
#include <utility>

namespace Semper {

// Runs `body` and returns its result, or `on_error` if it throws. `report`,
// when given, receives the exception's message (or "unknown exception").
template <typename R, typename F>
R guarded_call(R on_error, F&& body, void (*report)(const char*) = nullptr) noexcept {
    try {
        return std::forward<F>(body)();
    } catch (const std::exception& e) {
        if (report) report(e.what());
    } catch (...) {
        if (report) report("unknown exception");
    }
    return on_error;
}

// The same for entry points that return nothing.
template <typename F>
void guarded_call(F&& body, void (*report)(const char*) = nullptr) noexcept {
    try {
        std::forward<F>(body)();
    } catch (const std::exception& e) {
        if (report) report(e.what());
    } catch (...) {
        if (report) report("unknown exception");
    }
}

} // namespace Semper

#endif // SEMPER_GUARDED_CALL_HPP
