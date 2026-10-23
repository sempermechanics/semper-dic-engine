#ifndef SEMPER_TEST_FRAMEWORK_H
#define SEMPER_TEST_FRAMEWORK_H

// =====================================================================
// MICRO TEST FRAMEWORK (zero dependencies)
//
// A deliberately tiny googletest-style harness so the native test
// suite builds anywhere with just a C++17 compiler — no downloads,
// no vendored frameworks, no NDK requirement.
//
// API:
//   TEST_CASE(Suite, Name) { ... }        register a test
//   CHECK(cond)                            non-fatal assertion
//   REQUIRE(cond)                          fatal assertion (aborts test)
//   CHECK_NEAR(a, b, tol)                  |a-b| <= tol
//   CHECK_REL(a, b, rel)                   relative error <= rel
//   FAIL_MSG("...")                        explicit failure
//
// Runner (test_main.cpp):
//   dic_tests            run everything
//   dic_tests Suite      run one suite
//   dic_tests Suite.Name run one test
//   dic_tests A:B-C:D    run tests matching A or B, except those matching
//                        C or D (substrings of Suite.Name, as gtest's
//                        --gtest_filter but without wildcards; CI splits
//                        the suite with a list and its negation)
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace dictest {

    struct TestCase {
        const char *suite;
        const char *name;
        std::function<void()> fn;
    };

    inline std::vector<TestCase> &registry() {
        static std::vector<TestCase> r;
        return r;
    }

    struct Registrar {
        Registrar(const char *suite, const char *name, std::function<void()> fn) {
            registry().push_back({suite, name, std::move(fn)});
        }
    };

    // Per-test failure state
    inline int &failures() { static int f = 0; return f; }
    inline bool &fatal_hit() { static bool b = false; return b; }

    inline void report_failure(const char *file, int line, const std::string &msg) {
        std::printf("    FAILED  %s:%d\n      %s\n", file, line, msg.c_str());
        failures()++;
    }

    struct FatalFailure {};

    inline bool approx_near(double a, double b, double tol) {
        return std::fabs(a - b) <= tol;
    }
    inline bool approx_rel(double a, double b, double rel) {
        double scale = std::fmax(std::fabs(a), std::fabs(b));
        if (scale < 1e-12) return true; // both effectively zero
        return std::fabs(a - b) / scale <= rel;
    }

    // Filter `pos1:pos2-neg1:neg2`: a test runs if its "Suite.Name" contains
    // one of the positive patterns (or there are none) and none of the
    // negative ones. Names never contain ':' or '-', so a plain substring
    // filters as it always did.
    inline std::vector<std::string> split_patterns(const std::string &s) {
        std::vector<std::string> out;
        size_t start = 0;
        while (start <= s.size()) {
            const size_t end = std::min(s.find(':', start), s.size());
            if (end > start) out.push_back(s.substr(start, end - start));
            start = end + 1;
        }
        return out;
    }

    inline bool filter_matches(const std::string &full, const char *filter) {
        if (!filter) return true;
        const std::string f(filter);
        const size_t dash = f.find('-');
        const std::vector<std::string> pos = split_patterns(f.substr(0, dash));
        const std::vector<std::string> neg =
                dash == std::string::npos ? std::vector<std::string>() : split_patterns(f.substr(dash + 1));
        bool in = pos.empty();
        for (const std::string &p : pos) in = in || full.find(p) != std::string::npos;
        for (const std::string &n : neg) in = in && full.find(n) == std::string::npos;
        return in;
    }

    // Runner: returns process exit code
    inline int run_all(int argc, char **argv) {
        const char *filter = (argc > 1) ? argv[1] : nullptr;
        int total = 0, passed = 0, failed = 0;

        for (auto &tc : registry()) {
            std::string full = std::string(tc.suite) + "." + tc.name;
            if (!filter_matches(full, filter)) continue;
            total++;

            std::printf("[ RUN      ] %s\n", full.c_str());
            failures() = 0;
            fatal_hit() = false;
            try {
                tc.fn();
            } catch (FatalFailure &) {
                // REQUIRE failed — failure already reported
            } catch (std::exception &e) {
                report_failure("<exception>", 0,
                               std::string("unhandled exception: ") + e.what());
            }

            if (failures() == 0) {
                std::printf("[       OK ] %s\n", full.c_str());
                passed++;
            } else {
                std::printf("[  FAILED  ] %s (%d assertion%s)\n", full.c_str(),
                            failures(), failures() == 1 ? "" : "s");
                failed++;
            }
        }

        std::printf("\n========================================\n");
        std::printf("  %d test(s) run: %d passed, %d failed\n", total, passed, failed);
        std::printf("========================================\n");
        return failed == 0 ? 0 : 1;
    }

} // namespace dictest

#define DICTEST_CONCAT_(a, b) a##b
#define DICTEST_CONCAT(a, b) DICTEST_CONCAT_(a, b)

#define TEST_CASE(Suite, Name)                                                 \
    static void DICTEST_CONCAT(dictest_fn_, DICTEST_CONCAT(Suite, Name))();    \
    static dictest::Registrar DICTEST_CONCAT(                                  \
            dictest_reg_, DICTEST_CONCAT(Suite, Name))(                        \
            #Suite, #Name, DICTEST_CONCAT(dictest_fn_,                         \
                                          DICTEST_CONCAT(Suite, Name)));       \
    static void DICTEST_CONCAT(dictest_fn_, DICTEST_CONCAT(Suite, Name))()

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond))                                                           \
            dictest::report_failure(__FILE__, __LINE__, "CHECK(" #cond ")");   \
    } while (0)

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            dictest::report_failure(__FILE__, __LINE__, "REQUIRE(" #cond ")"); \
            throw dictest::FatalFailure{};                                     \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        double va_ = (double)(a), vb_ = (double)(b), vt_ = (double)(tol);      \
        if (!dictest::approx_near(va_, vb_, vt_)) {                            \
            char buf_[256];                                                    \
            std::snprintf(buf_, sizeof(buf_),                                  \
                          "CHECK_NEAR(%s=%.8g, %s=%.8g, tol=%.3g) |d|=%.3g",   \
                          #a, va_, #b, vb_, vt_, std::fabs(va_ - vb_));        \
            dictest::report_failure(__FILE__, __LINE__, buf_);                 \
        }                                                                      \
    } while (0)

#define CHECK_REL(a, b, rel)                                                   \
    do {                                                                       \
        double va_ = (double)(a), vb_ = (double)(b);                           \
        if (!dictest::approx_rel(va_, vb_, (double)(rel))) {                   \
            char buf_[256];                                                    \
            std::snprintf(buf_, sizeof(buf_),                                  \
                          "CHECK_REL(%s=%.8g, %s=%.8g, rel=%.3g)",             \
                          #a, va_, #b, vb_, (double)(rel));                    \
            dictest::report_failure(__FILE__, __LINE__, buf_);                 \
        }                                                                      \
    } while (0)

#define FAIL_MSG(msg) dictest::report_failure(__FILE__, __LINE__, (msg))

#endif // SEMPER_TEST_FRAMEWORK_H
