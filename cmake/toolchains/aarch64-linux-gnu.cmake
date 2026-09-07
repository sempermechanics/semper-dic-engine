# aarch64 cross-compilation for size and code-generation measurements.
#
# This is NOT the Android build. The shipping arm64-v8a library is produced by
# the NDK (clang, Bionic, -flto, 16 KB page alignment) via SEMPER_ANDROID=ON.
# This toolchain exists so binary-size questions can be answered on a 64-bit ARM
# target — instruction encoding and WITH_CAROTENE/NEON both differ from x86_64 —
# without requiring an NDK. Treat absolute sizes as indicative; the before/after
# ratio it produces is the useful number.
#
#   cmake -S . -B build/arm64 -DSEMPER_BUILD_C_SDK=ON -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake
#
# Needs: apt install crossbuild-essential-arm64 (and qemu-user-static to run it).
#
# NO CI JOB RUNS THIS. It is manual-only, deliberately: a cross-build of the
# vendored OpenCV submodule for aarch64 costs far more CI minutes than a
# size measurement nobody reads per-commit is worth. The consequence is that
# this file can rot without anything failing, so if it stops working, fix it
# here rather than assuming the measurement it produced was wrong.
#
# The measurement it exists for is recorded in docs/SEEDING_BENCHMARK.md
# 6.1 (stripped .so 9.475 MB -> 6.720 MB, -29.1%, from dropping features2d /
# flann / calib3d) and its limits in 6.2. To reproduce, build the C SDK once
# per OpenCV module list and compare `aarch64-linux-gnu-strip`ped sizes; the
# smoke test runs under `qemu-aarch64-static`.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
