#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/nanoquant demo --rows 1024 --cols 1024
ctest --test-dir build --output-on-failure
