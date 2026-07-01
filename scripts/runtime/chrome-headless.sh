#!/usr/bin/env bash
set -euo pipefail

if [[ ${RUNMAT_HEADLESS_DEBUG:-0} == "1" ]]; then
  set -x
fi

if [[ -n "${RUNMAT_CHROME_BIN:-}" ]]; then
  CHROME_BIN="${RUNMAT_CHROME_BIN}"
else
  case "$(uname -s)" in
    Darwin)
      CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
      ;;
    Linux)
      CHROME_BIN="$(command -v google-chrome || command -v google-chrome-stable || command -v chromium || command -v chrome)"
      ;;
    *)
      echo "Unsupported OS for chrome-headless.sh: $(uname -s)" >&2
      exit 1
      ;;
  esac
fi

if [[ ! -x "${CHROME_BIN}" ]]; then
  echo "Chrome binary not found or not executable: ${CHROME_BIN}" >&2
  echo "Set RUNMAT_CHROME_BIN to a valid Chrome/Chromium binary." >&2
  exit 1
fi

if [[ "$(uname -s)" == "Linux" ]]; then
  # ubuntu-latest CI runners have no real GPU or Vulkan; use SwiftShader
  # (CPU-based software renderer) so Chrome doesn't hang on init.
  # --no-sandbox and --disable-dev-shm-usage are required in containers.
  exec "${CHROME_BIN}" \
    --headless=new \
    --no-sandbox \
    --disable-dev-shm-usage \
    --use-gl=angle \
    --use-angle=swiftshader \
    --enable-unsafe-webgpu \
    --disable-gpu-sandbox \
    --disable-webgpu-vsync \
    "$@"
else
  exec "${CHROME_BIN}" \
    --headless=new \
    --use-angle=metal \
    --enable-features=Vulkan,UseSkiaRenderer,WebGPUService \
    --enable-unsafe-webgpu \
    --disable-gpu-sandbox \
    --disable-webgpu-vsync \
    "$@"
fi

