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

ANGLE_FLAG="--use-angle=metal"
if [[ "$(uname -s)" == "Linux" ]]; then
  ANGLE_FLAG="--use-angle=vulkan"
fi

exec "${CHROME_BIN}" \
  --headless=new \
  --enable-features=Vulkan,UseSkiaRenderer,WebGPUService \
  --enable-unsafe-webgpu \
  --disable-gpu-sandbox \
  --disable-webgpu-vsync \
  "${ANGLE_FLAG}" \
  "$@"

