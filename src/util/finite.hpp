#ifndef SEMPER_UTIL_FINITE_HPP
#define SEMPER_UTIL_FINITE_HPP

// Inf/NaN checks that survive -ffast-math. The engine is built with it, and
// under it std::isfinite / std::isnan may be folded to a constant and a
// comparison may be assumed never to involve a NaN, so `!(x > eps)` stops
// catching one. These read the exponent bits instead.

#include <cstdint>
#include <cstring>

namespace Semper {
namespace util {

// True unless f is Inf or NaN.
inline bool finite_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return (u & 0x7f800000u) != 0x7f800000u;
}

inline bool finite_bits(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return (u & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
}

} // namespace util
} // namespace Semper

#endif // SEMPER_UTIL_FINITE_HPP
