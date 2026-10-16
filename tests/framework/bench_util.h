#ifndef DICTEST_BENCH_UTIL_H
#define DICTEST_BENCH_UTIL_H

// Cost accounting shared by the env-gated benchmarks (SeedBench, IncBench).
//
// Wall time alone can hide a CPU regression: an OpenMP-parallel phase can win
// on the clock and still cost more total CPU, which on a phone is battery. So
// the benchmarks report process CPU time and resident set size too. Those have
// no portable spelling. CI and every published measurement are Linux; Windows
// is a developer host, so it gets working equivalents rather than a stub -- an
// unbuildable test file takes the whole dic_tests binary down with it.

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX          // else the min/max macros shadow std::min/std::max
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#else
#  include <sys/resource.h>
#  include <unistd.h>
#endif

#include <fstream>

namespace dictest {

struct CpuClock {
    double user_ms = 0.0, sys_ms = 0.0;
    static CpuClock now() {
        CpuClock c;
#ifdef _WIN32
        // GetProcessTimes reports both in 100 ns units and, like getrusage,
        // sums across all threads of the process -- which is the property the
        // OpenMP-vs-serial comparison depends on.
        FILETIME created{}, exited{}, kernel{}, user{};
        if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
            const auto to_ms = [](const FILETIME &ft) {
                return (double)(((unsigned long long)ft.dwHighDateTime << 32) |
                                ft.dwLowDateTime) / 1e4;
            };
            c.user_ms = to_ms(user);
            c.sys_ms  = to_ms(kernel);
        }
#else
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        c.user_ms = ru.ru_utime.tv_sec * 1e3 + ru.ru_utime.tv_usec / 1e3;
        c.sys_ms  = ru.ru_stime.tv_sec * 1e3 + ru.ru_stime.tv_usec / 1e3;
#endif
        return c;
    }
    double total_ms() const { return user_ms + sys_ms; }
};

// ru_maxrss is a process-wide high-water mark: monotonic, so differencing it
// across candidates in one process reports the run order, not the candidate.
// Current RSS from /proc/self/statm can be differenced — it gives the memory a
// candidate *retains*, which is the comparable quantity. For a true
// per-candidate peak, run one candidate per process and read the peak.
inline double current_rss_mb() {
#ifdef _WIN32
    // WorkingSetSize is the Windows analogue of the statm resident count:
    // pages currently backed by physical memory, and it can be differenced.
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return -1.0;
    return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
#else
    std::ifstream f("/proc/self/statm");
    if (!f) return -1.0;
    long total_pages = 0, rss_pages = 0;
    f >> total_pages >> rss_pages;
    if (!f) return -1.0;
    return (double)rss_pages * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
#endif
}

inline double process_peak_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return -1.0;
    return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
#else
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#  ifdef __APPLE__
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#  else
    return (double)ru.ru_maxrss / 1024.0;
#  endif
#endif
}

} // namespace dictest

#endif // DICTEST_BENCH_UTIL_H
