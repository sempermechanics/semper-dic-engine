#ifndef SEMPER_UTIL_MEDIAN_HPP
#define SEMPER_UTIL_MEDIAN_HPP

// The engine's one median: the anchor lattice's median test and global shift,
// and the displacement outlier test before strain, all use it.

#include <algorithm>
#include <cstddef>

namespace Semper {
namespace util {

// Median of v[0..n), reordering v. For an even count it is the mean of the
// middle pair (as numpy's median), not the upper of the two: the anchor
// lattice used to take the upper one, which biases an 8-neighbourhood test
// towards the larger displacements. 0 for n == 0.
//
// NaNs are moved to the end and left out: nth_element needs a strict weak
// ordering, which `<` is not with a NaN in the range (undefined behaviour).
// The median of the rest is returned; 0 if nothing else is left.
inline float median_inplace(float *v, size_t n) {
    n = (size_t)(std::partition(v, v + n, [](float x) { return x == x; }) - v);
    if (n == 0) return 0.0f;
    const size_t mid = n / 2;
    std::nth_element(v, v + mid, v + n);
    const float hi = v[mid];
    if (n % 2 == 1) return hi;
    const float lo = *std::max_element(v, v + mid);
    return 0.5f * (lo + hi);
}

} // namespace util
} // namespace Semper

#endif // SEMPER_UTIL_MEDIAN_HPP
