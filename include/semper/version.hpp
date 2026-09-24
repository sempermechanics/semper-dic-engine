#ifndef SEMPER_VERSION_HPP
#define SEMPER_VERSION_HPP

// Semantic version of the public engine surface (see docs/CONTRACT.md §A.1):
//   major — a break to any Frozen/Stable contract (output packing, signatures, codes)
//   minor — additive Stable/Additive-only change, or a new capability
//   patch — internal-only improvement (accuracy, speed) with no surface change
#define SEMPER_VERSION_MAJOR 0
#define SEMPER_VERSION_MINOR 2
#define SEMPER_VERSION_PATCH 2

#define SEMPER_STRINGIFY_(x) #x
#define SEMPER_STRINGIFY(x) SEMPER_STRINGIFY_(x)
#define SEMPER_VERSION_STRING            \
    SEMPER_STRINGIFY(SEMPER_VERSION_MAJOR) "." \
    SEMPER_STRINGIFY(SEMPER_VERSION_MINOR) "." \
    SEMPER_STRINGIFY(SEMPER_VERSION_PATCH)

#endif
