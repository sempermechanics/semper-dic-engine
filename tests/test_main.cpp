// =====================================================================
// NATIVE TEST RUNNER — see docs/TESTING.md for the full test catalog.
//
// Usage:
//   dic_tests                 run all suites
//   dic_tests SimdKernels     run one suite
//   dic_tests Engine.PureTranslation_Subpixel   run a single test
//   dic_tests A:B-C        tests matching A or B, except those matching C
// =====================================================================
#include "framework/test_framework.h"

#include <cstdio>

int main(int argc, char **argv) {
    // Unbuffered output: on CI stdout is a pipe (fully buffered), so a
    // crash would otherwise swallow every [ RUN ] line printed before it,
    // hiding which test died.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    return dictest::run_all(argc, argv);
}
