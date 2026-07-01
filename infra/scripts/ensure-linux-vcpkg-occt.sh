#!/usr/bin/env bash
set -euo pipefail

VCPKG_REF="${RUNMAT_VCPKG_REF:-2026.06.01}"
VCPKG_ROOT="${RUNMAT_VCPKG_ROOT:-${HOME}/vcpkg-runmat}"
VCPKG_TRIPLET="${RUNMAT_VCPKG_TRIPLET:-x64-linux-runmat-dynamic-release}"
VCPKG_MARKER="${VCPKG_ROOT}/.runmat-vcpkg-ref"
LOCK_DIR="${VCPKG_ROOT}.lock"
LOCK_ACQUIRED=0
OVERLAY_TRIPLETS="${VCPKG_ROOT}/triplets/runmat"

mkdir -p "$(dirname "${VCPKG_ROOT}")"

for _ in $(seq 1 720); do
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    trap 'rmdir "${LOCK_DIR}"' EXIT
    LOCK_ACQUIRED=1
    break
  fi
  sleep 5
done

if [[ "${LOCK_ACQUIRED}" -ne 1 ]]; then
  echo "Timed out waiting for vcpkg provisioning lock at ${LOCK_DIR}" >&2
  exit 1
fi

if [[ -d "${VCPKG_ROOT}" ]]; then
  current_ref=""
  if [[ -f "${VCPKG_MARKER}" ]]; then
    current_ref="$(cat "${VCPKG_MARKER}")"
  fi
  if [[ "${current_ref}" != "${VCPKG_REF}" || ! -x "${VCPKG_ROOT}/vcpkg" ]]; then
    echo "Replacing stale vcpkg root at ${VCPKG_ROOT} with pinned ref ${VCPKG_REF}"
    rm -rf "${VCPKG_ROOT}"
  fi
fi

if [[ ! -x "${VCPKG_ROOT}/vcpkg" ]]; then
  echo "Cloning vcpkg ${VCPKG_REF} into ${VCPKG_ROOT}"
  git clone --depth 1 --branch "${VCPKG_REF}" https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"
  "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
  echo "${VCPKG_REF}" > "${VCPKG_MARKER}"
fi

mkdir -p "${OVERLAY_TRIPLETS}"
if [[ "${VCPKG_TRIPLET}" == "x64-linux-runmat-dynamic-release" ]]; then
  cat > "${OVERLAY_TRIPLETS}/${VCPKG_TRIPLET}.cmake" <<'EOF'
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)
EOF
fi

PREFIX="${VCPKG_ROOT}/installed/${VCPKG_TRIPLET}"
INCLUDE_DIR="${PREFIX}/include/opencascade"
LIB_DIR="${PREFIX}/lib"

has_occt() {
  [[ -d "${INCLUDE_DIR}" ]] &&
    [[ -e "${LIB_DIR}/libTKBRep.so" ]] &&
    [[ -e "${LIB_DIR}/libTKSTEP.so" || -e "${LIB_DIR}/libTKDESTEP.so" ]] &&
    [[ -e "${LIB_DIR}/libTKIGES.so" || -e "${LIB_DIR}/libTKDEIGES.so" || -e "${LIB_DIR}/libTKXDEIGES.so" ]]
}

if ! has_occt; then
  echo "Installing OpenCASCADE via vcpkg (${VCPKG_TRIPLET})"
  "${VCPKG_ROOT}/vcpkg" install "opencascade:${VCPKG_TRIPLET}" --overlay-triplets="${OVERLAY_TRIPLETS}"
fi

if ! has_occt; then
  echo "Pinned vcpkg OpenCASCADE install is missing expected headers or libraries under ${PREFIX}" >&2
  exit 1
fi

echo "Using pinned OpenCASCADE from ${PREFIX}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "VCPKG_ROOT=${VCPKG_ROOT}"
    echo "VCPKG_DEFAULT_TRIPLET=${VCPKG_TRIPLET}"
    echo "VCPKGRS_TRIPLET=${VCPKG_TRIPLET}"
    echo "RUNMAT_OCCT_ROOT=${PREFIX}"
    echo "RUNMAT_OCCT_INCLUDE_DIR=${INCLUDE_DIR}"
    echo "RUNMAT_OCCT_LIB_DIR=${LIB_DIR}"
    echo "RUNMAT_OCCT_BIN_DIR=${PREFIX}/bin"
    echo "RUNMAT_OCCT_LINK_MODE=dylib"
    echo "LD_LIBRARY_PATH=${LIB_DIR}:${LD_LIBRARY_PATH:-}"
  } >> "${GITHUB_ENV}"
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "${PREFIX}/bin" >> "${GITHUB_PATH}"
fi
