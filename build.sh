#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
用法:
  ./build.sh [选项]

选项:
  --build-dir <dir>         构建目录 (默认: build)
  --type <Release|Debug>    构建类型 (默认: Release)
  --rocm-path <path>        ROCm/DTK 路径 (默认: 使用环境变量 ROCM_PATH)
  --rccl <ON|OFF>           是否启用 RCCL/NCCL (默认: OFF)
  --rccl-root <path>        RCCL_ROOT (默认: 使用环境变量 RCCL_ROOT; 若未设置则退回 ROCM_PATH)
  --hip <ON|OFF>            是否启用 HIP/GPU (默认: ON)
  --mpi <ON|OFF>            是否启用 MPI (默认: ON)
  --openmp <ON|OFF>         是否启用 OpenMP (默认: ON)
  --tests <ON|OFF>          是否编译测试 (默认: OFF)
  --tools <ON|OFF>          是否编译 tools (默认: ON)
  --pkg <ON|OFF>            是否执行 pkg fetch/install (默认: ON)
  --clean                   删除构建目录后重新配置
  --configure-only          仅执行 cmake 配置，不执行构建
  -h, --help                显示帮助

示例:
  ROCM_PATH=/opt/dtk-25.04.1 ./build.sh --rccl ON
  ./build.sh --type Debug --build-dir cmake-build-debug --tests ON --pkg OFF
EOF
}

BUILD_DIR="build"
BUILD_TYPE="Release"
ROCM_PATH_ARG=""
RCCL_ENABLE="OFF"
RCCL_ROOT_ARG=""
HIP_ENABLE="ON"
MPI_ENABLE="ON"
OPENMP_ENABLE="ON"
TESTS_ENABLE="OFF"
TOOLS_ENABLE="ON"
PKG_ENABLE="ON"
CLEAN="OFF"
CONFIGURE_ONLY="OFF"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="${2:?}"; shift 2 ;;
    --type) BUILD_TYPE="${2:?}"; shift 2 ;;
    --rocm-path) ROCM_PATH_ARG="${2:?}"; shift 2 ;;
    --rccl) RCCL_ENABLE="${2:?}"; shift 2 ;;
    --rccl-root) RCCL_ROOT_ARG="${2:?}"; shift 2 ;;
    --hip) HIP_ENABLE="${2:?}"; shift 2 ;;
    --mpi) MPI_ENABLE="${2:?}"; shift 2 ;;
    --openmp) OPENMP_ENABLE="${2:?}"; shift 2 ;;
    --tests) TESTS_ENABLE="${2:?}"; shift 2 ;;
    --tools) TOOLS_ENABLE="${2:?}"; shift 2 ;;
    --pkg) PKG_ENABLE="${2:?}"; shift 2 ;;
    --clean) CLEAN="ON"; shift 1 ;;
    --configure-only) CONFIGURE_ONLY="ON"; shift 1 ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; print_usage; exit 2 ;;
  esac
done

if [[ -f "./env.sh" ]]; then
  source ./env.sh
fi

if [[ -z "${ROCM_PATH_ARG}" ]]; then
  ROCM_PATH_ARG="${ROCM_PATH:-}"
fi
if [[ -z "${ROCM_PATH_ARG}" ]]; then
  echo "缺少 ROCM_PATH。请设置环境变量 ROCM_PATH 或传入 --rocm-path" >&2
  exit 2
fi

if [[ -z "${RCCL_ROOT_ARG}" ]]; then
  RCCL_ROOT_ARG="${RCCL_ROOT:-}"
fi
if [[ -z "${RCCL_ROOT_ARG}" ]]; then
  RCCL_ROOT_ARG="${ROCM_PATH_ARG}"
fi

if [[ "${CLEAN}" == "ON" ]]; then
  rm -rf "${BUILD_DIR}"
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "找不到命令: $1" >&2; exit 127; }
}

need_cmd cmake

if [[ "${PKG_ENABLE}" == "ON" ]]; then
  if command -v pkg >/dev/null 2>&1; then
    pkg fetch
    pkg install
  else
    echo "未找到 pkg 命令，跳过依赖安装。可用 --pkg OFF 关闭该步骤。" >&2
  fi
fi

JOBS="8"
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  JOBS="$(getconf _NPROCESSORS_ONLN || echo 8)"
elif command -v sysctl >/dev/null 2>&1; then
  JOBS="$(sysctl -n hw.ncpu || echo 8)"
fi
if [[ -z "${JOBS}" ]]; then
  JOBS="8"
fi

export ROCM_PATH="${ROCM_PATH_ARG}"
export RCCL_ROOT="${RCCL_ROOT_ARG}"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DROCM_PATH="${ROCM_PATH_ARG}" \
  -DKMC_HIP_ENABLE_FLAG="${HIP_ENABLE}" \
  -DKMC_RCCL_ENABLE_FLAG="${RCCL_ENABLE}" \
  -DKMC_MPI_ENABLE_FLAG="${MPI_ENABLE}" \
  -DKMC_OpenMP_ENABLE_FLAG="${OPENMP_ENABLE}" \
  -DKMC_TEST_BUILD_ENABLE_FLAG="${TESTS_ENABLE}" \
  -DKMC_TOOLS_BUILD_FLAG="${TOOLS_ENABLE}"

if [[ "${CONFIGURE_ONLY}" == "ON" ]]; then
  exit 0
fi

cmake --build "${BUILD_DIR}" -j "${JOBS}"
