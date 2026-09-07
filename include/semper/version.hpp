#ifndef SEMPER_VERSION_HPP
#define SEMPER_VERSION_HPP

// Semantic version of the public engine surface (see docs/CONTRACT.md §A.1):
//   major — a break to any Frozen/Stable contract (output packing, signatures, codes)
//   minor — additive Stable/Additive-only change, or a new capability
//   patch — internal-only improvement (accuracy, speed) with no surface change
//
// "Surface" is not the C ABI alone. It is whatever CONTRACT.md A.1 marks
// Stable, Frozen or Additive-only: the signatures in pipeline/io/cancel/
// version.hpp and semper_c.h, plus the metrics layout and the packed output.
// Source compatibility counts, not just ABI -- downstreams link statically,
// and there are no install() rules to hide a header change behind. Headers
// under include/semper/ that A.1 lists as Internal (seeding.hpp, tuning.hpp,
// types.hpp) are exempt.
#define SEMPER_VERSION_MAJOR 0
#define SEMPER_VERSION_MINOR 3
#define SEMPER_VERSION_PATCH 0

#define SEMPER_STRINGIFY_(x) #x
#define SEMPER_STRINGIFY(x) SEMPER_STRINGIFY_(x)
#define SEMPER_VERSION_STRING            \
    SEMPER_STRINGIFY(SEMPER_VERSION_MAJOR) "." \
    SEMPER_STRINGIFY(SEMPER_VERSION_MINOR) "." \
    SEMPER_STRINGIFY(SEMPER_VERSION_PATCH)

#endif
