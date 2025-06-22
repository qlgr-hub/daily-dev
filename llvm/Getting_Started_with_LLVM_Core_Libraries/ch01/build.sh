#!/usr/bin/env bash

set -e

ORG_PATH=$(pwd)
SCRIPT_BASE_DIR=$(cd "$(dirname $0)"; pwd)
cd "${SCRIPT_BASE_DIR}"

if [ ! -d build ]; then
    mkdir build
fi

cd build

BUILD_TYPE="Release"
if [ -n "$1" ]; then
    BUILD_TYPE="$1"
fi


cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="/usr/local" \
    -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU;RISCV" \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    # -DLLVM_ENABLE_LLD=ON \
    # -DLLVM_CCACHE_BUILD=ON \
    # -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    # -DLLVM_USE_SANITIZER="Address;Undefined" \

cmake --build . --target all --parallel 32
sudo ninja install

cd "${ORG_PATH}"
