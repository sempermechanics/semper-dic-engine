#!/usr/bin/env bash
# Build LLVM's OpenMP runtime (libomp) for the macOS wheels.
#
# Homebrew's libomp is built for the runner's own macOS release (26.0 on
# macos-latest), so delocate refuses to bundle it into a wheel that targets
# macOS 11.0. This builds the same runtime from LLVM's release source for the
# wheel's deployment target and installs it under PREFIX; the wheel build
# finds it through OpenMP_ROOT.
#
# Usage: build-libomp-macos.sh [PREFIX]    (default /tmp/semper-libomp)
# Env:   MACOSX_DEPLOYMENT_TARGET (default 11.0), LIBOMP_ARCHS (default: host)
set -euo pipefail

LLVM_VERSION=21.1.8
# SHA-256 of the release files at github.com/llvm/llvm-project/releases.
OPENMP_SHA256=856b023748b41ac7b2c83fd8e9f765ff48a4df2fe6777d2811ef7c7ed8f2f977
CMAKE_SHA256=85735f20fd8c81ecb0a09abb0c267018475420e93b65050cc5b7634eab744de9

PREFIX="${1:-/tmp/semper-libomp}"
TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
ARCHS="${LIBOMP_ARCHS:-$(uname -m)}"
BASE="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fetch() {  # file sha256
    curl -fsSL --retry 3 -o "$work/$1" "$BASE/$1"
    echo "$2  $work/$1" | shasum -a 256 -c -
    tar -xf "$work/$1" -C "$work"
}
fetch "openmp-${LLVM_VERSION}.src.tar.xz" "$OPENMP_SHA256"
fetch "cmake-${LLVM_VERSION}.src.tar.xz" "$CMAKE_SHA256"
# A standalone OpenMP build reads LLVM's shared CMake modules from ../cmake.
mv "$work/openmp-${LLVM_VERSION}.src" "$work/openmp"
mv "$work/cmake-${LLVM_VERSION}.src" "$work/cmake"

# CMAKE_INSTALL_NAME_DIR gives the dylib an absolute install name, as
# Homebrew's has, so the extension records a path delocate can resolve.
cmake -S "$work/openmp" -B "$work/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_NAME_DIR="$PREFIX/lib" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$TARGET" \
    -DCMAKE_OSX_ARCHITECTURES="$ARCHS" \
    -DLIBOMP_INSTALL_ALIASES=OFF \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DLIBOMP_OMPD_SUPPORT=OFF \
    -DLIBOMP_ARCHER_SUPPORT=OFF
cmake --build "$work/build" --parallel
cmake --install "$work/build"

test -f "$PREFIX/include/omp.h"
ls "$PREFIX"/lib/libomp.*
