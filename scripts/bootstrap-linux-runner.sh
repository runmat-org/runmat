#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as the Linux runner user, not as root. It will use sudo for system packages." >&2
  exit 1
fi

EXPECTED_HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
if [[ -z "${EXPECTED_HOME}" ]]; then
  echo "Could not determine passwd home for $(id -un)" >&2
  exit 1
fi
export HOME="${EXPECTED_HOME}"

RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.90.0-x86_64-unknown-linux-gnu}"
CARGO_HOME="${CARGO_HOME:-${HOME}/.cargo}"
RUSTUP_HOME="${RUSTUP_HOME:-${HOME}/.rustup}"

PACKAGES=(
  build-essential
  ca-certificates
  curl
  gdb
  git
  libdbus-1-dev
  libegl1-mesa-dev
  libgl1-mesa-dev
  liblapack-dev
  libopenblas-dev
  libssl-dev
  libudev-dev
  libwayland-dev
  libx11-dev
  libxcursor-dev
  libxi-dev
  libxinerama-dev
  libxkbcommon-dev
  libxkbcommon-x11-0
  libxrandr-dev
  libzmq3-dev
  pkg-config
)

echo "Installing Linux system packages used by the self-hosted build-and-test job"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${PACKAGES[@]}"

mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"
export CARGO_HOME
export RUSTUP_HOME
export PATH="${CARGO_HOME}/bin:${PATH}"

LDCONFIG_BIN="$(command -v ldconfig || true)"
if [[ -z "${LDCONFIG_BIN}" && -x /usr/sbin/ldconfig ]]; then
  LDCONFIG_BIN=/usr/sbin/ldconfig
fi

if [[ ! -x "${CARGO_HOME}/bin/rustup" ]]; then
  echo "Installing rustup and toolchain ${RUST_TOOLCHAIN}"
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"
fi

if [[ ! -x "${CARGO_HOME}/bin/rustup" ]]; then
  echo "rustup was not installed to ${CARGO_HOME}/bin/rustup" >&2
  exit 1
fi

echo "Ensuring Rust toolchain ${RUST_TOOLCHAIN} and required components are present"
"${CARGO_HOME}/bin/rustup" toolchain install "${RUST_TOOLCHAIN}" --profile minimal
"${CARGO_HOME}/bin/rustup" component add --toolchain "${RUST_TOOLCHAIN}" rustfmt clippy

echo "Validating Linux runner toolchain"
echo "Runner user: $(whoami)"
echo "Runner home: ${HOME}"
"${CARGO_HOME}/bin/rustup" --version
"${CARGO_HOME}/bin/rustc" -V
"${CARGO_HOME}/bin/cargo" -V
if [[ "$(command -v rustup || true)" != "${CARGO_HOME}/bin/rustup" ]]; then
  echo "rustup resolution is not deterministic. Expected $(command -v rustup || true) to equal ${CARGO_HOME}/bin/rustup" >&2
  exit 1
fi
if [[ "$(command -v cargo || true)" != "${CARGO_HOME}/bin/cargo" ]]; then
  echo "cargo resolution is not deterministic. Expected $(command -v cargo || true) to equal ${CARGO_HOME}/bin/cargo" >&2
  exit 1
fi
pkg-config --modversion libzmq
if [[ -z "${LDCONFIG_BIN}" ]]; then
  echo "ldconfig was not found on PATH or at /usr/sbin/ldconfig" >&2
  exit 1
fi
"${LDCONFIG_BIN}" -p | grep -q 'libopenblas\.so'
"${LDCONFIG_BIN}" -p | grep -q 'liblapack\.so'
gdb --version >/dev/null

echo
echo "Linux runner bootstrap complete."
echo "If the GitHub Actions runner service was already running, restart it before rerunning CI."
