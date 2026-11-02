# Host test suite

Sources for `dic_tests`, the engine's C++ test binary, plus the C ABI test
programs in `c/`. [docs/TESTING.md](../docs/TESTING.md) covers build options,
sanitizers, CI jobs and what each suite proves.

```bash
# From the repository root; -DDIC_REQUIRE_OPENCV=ON needs libopencv-dev
git submodule update --init --recursive
cmake -S tests -B build/tests -DCMAKE_BUILD_TYPE=Release -DDIC_REQUIRE_OPENCV=ON
cmake --build build/tests -j"$(nproc)"
./build/tests/dic_tests            # all suites
./build/tests/dic_tests Engine.    # one suite (substring filter on Suite.Name)
```

The private Android app consumes this engine as a pinned submodule and does not
re-run these tests.
