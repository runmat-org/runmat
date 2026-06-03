#!/usr/bin/env bash
set -euo pipefail

detect_cache_root() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "${HOME}/Library/Caches/.wasm-pack"
  else
    local xdg_root="${XDG_CACHE_HOME:-${HOME}/.cache}"
    echo "${xdg_root}/.wasm-pack"
  fi
}

detect_chrome_bin() {
  if [[ -n "${RUNMAT_CHROME_BIN:-}" ]]; then
    echo "${RUNMAT_CHROME_BIN}"
    return 0
  fi

  case "$(uname -s)" in
    Darwin)
      echo "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
      ;;
    Linux)
      command -v google-chrome \
        || command -v google-chrome-stable \
        || command -v chromium \
        || command -v chrome
      ;;
    *)
      return 1
      ;;
  esac
}

extract_version_key() {
  local version="$1"
  local a=0
  local b=0
  local c=0
  local d=0
  IFS='.' read -r a b c d <<< "${version}"
  printf "%05d%05d%05d%05d" "${a:-0}" "${b:-0}" "${c:-0}" "${d:-0}"
}

version_is_newer() {
  local lhs="$1"
  local rhs="$2"
  [[ "$(extract_version_key "${lhs}")" > "$(extract_version_key "${rhs}")" ]]
}

driver_version() {
  local driver_bin="$1"
  "${driver_bin}" --version 2>/dev/null | awk '{print $2}'
}

driver_match_score() {
  local chrome_version="$1"
  local drv_version="$2"
  local chrome_major="${chrome_version%%.*}"
  local drv_major="${drv_version%%.*}"
  local chrome_triplet
  local drv_triplet
  chrome_triplet="$(echo "${chrome_version}" | cut -d. -f1-3)"
  drv_triplet="$(echo "${drv_version}" | cut -d. -f1-3)"

  if [[ "${drv_version}" == "${chrome_version}" ]]; then
    echo "3"
  elif [[ "${drv_triplet}" == "${chrome_triplet}" ]]; then
    echo "2"
  elif [[ "${drv_major}" == "${chrome_major}" ]]; then
    echo "1"
  else
    echo "0"
  fi
}

find_cached_driver() {
  local chrome_version="$1"
  local cache_root="$2"
  local best_path=""
  local best_version=""
  local best_score=0
  local drv_path
  local drv_version
  local score

  for drv_path in "${cache_root}"/chromedriver-*/chromedriver; do
    [[ -x "${drv_path}" ]] || continue
    drv_version="$(driver_version "${drv_path}")"
    [[ -n "${drv_version}" ]] || continue
    score="$(driver_match_score "${chrome_version}" "${drv_version}")"
    (( score > 0 )) || continue
    if (( score > best_score )); then
      best_path="${drv_path}"
      best_version="${drv_version}"
      best_score="${score}"
      continue
    fi
    if (( score == best_score )) && version_is_newer "${drv_version}" "${best_version}"; then
      best_path="${drv_path}"
      best_version="${drv_version}"
      best_score="${score}"
    fi
  done

  if [[ -n "${best_path}" ]]; then
    echo "${best_path}"
    return 0
  fi
  return 1
}

download_matching_driver() {
  local chrome_version="$1"
  local cache_root="$2"
  local chrome_major="${chrome_version%%.*}"
  local release_version
  local platform
  local zip_name
  local download_url
  local target_dir
  local zip_path
  local out_bin

  [[ "${RUNMAT_CHROMEDRIVER_ALLOW_DOWNLOAD:-1}" == "1" ]] || return 1
  command -v curl >/dev/null || return 1
  command -v unzip >/dev/null || return 1

  release_version="$(
    curl -fsSL "https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_${chrome_major}" \
      || true
  )"
  [[ -n "${release_version}" ]] || return 1

  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
      platform="mac-arm64"
      ;;
    Darwin-x86_64)
      platform="mac-x64"
      ;;
    Linux-x86_64)
      platform="linux64"
      ;;
    *)
      return 1
      ;;
  esac

  zip_name="chromedriver-${platform}.zip"
  download_url="https://storage.googleapis.com/chrome-for-testing-public/${release_version}/${platform}/${zip_name}"
  target_dir="${cache_root}/runmat-chromedriver-${release_version}-${platform}"
  out_bin="${target_dir}/chromedriver-${platform}/chromedriver"
  if [[ -x "${out_bin}" ]]; then
    echo "${out_bin}"
    return 0
  fi

  mkdir -p "${target_dir}"
  zip_path="${target_dir}/${zip_name}"
  curl -fsSL -o "${zip_path}" "${download_url}" || return 1
  unzip -qo "${zip_path}" -d "${target_dir}" || return 1
  [[ -x "${out_bin}" ]] || return 1
  echo "${out_bin}"
}

main() {
  if [[ -n "${RUNMAT_CHROMEDRIVER_BIN:-}" ]]; then
    [[ -x "${RUNMAT_CHROMEDRIVER_BIN}" ]] || {
      echo "RUNMAT_CHROMEDRIVER_BIN is set but not executable: ${RUNMAT_CHROMEDRIVER_BIN}" >&2
      exit 1
    }
    echo "${RUNMAT_CHROMEDRIVER_BIN}"
    return 0
  fi

  local chrome_bin
  chrome_bin="$(detect_chrome_bin || true)"
  [[ -x "${chrome_bin}" ]] || exit 1

  local chrome_version
  chrome_version="$("${chrome_bin}" --version 2>/dev/null | awk '{print $NF}')"
  [[ -n "${chrome_version}" ]] || exit 1

  local cache_root
  cache_root="$(detect_cache_root)"
  mkdir -p "${cache_root}"

  local resolved
  resolved="$(find_cached_driver "${chrome_version}" "${cache_root}" || true)"
  if [[ -n "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi

  resolved="$(download_matching_driver "${chrome_version}" "${cache_root}" || true)"
  if [[ -n "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi

  exit 1
}

main "$@"
