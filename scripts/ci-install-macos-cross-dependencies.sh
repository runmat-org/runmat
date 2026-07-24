#!/usr/bin/env bash
set -euo pipefail

target="${1:?usage: ci-install-macos-cross-dependencies.sh <rust-target>}"

brew update
brew list cmake >/dev/null 2>&1 || brew install cmake
brew list zeromq >/dev/null 2>&1 || brew install zeromq

if [[ "$target" == "aarch64-apple-darwin" ]]; then
    brew list hdf5 >/dev/null 2>&1 || brew install hdf5
    exit 0
fi

if [[ "$target" != "x86_64-apple-darwin" ]]; then
    echo "unsupported macOS target: $target" >&2
    exit 1
fi

# GitHub's macOS runners are Apple Silicon hosts. Homebrew therefore installs
# arm64 HDF5 libraries under /opt/homebrew, which an x86_64 Rust binary cannot
# link. Build the dependency for the Rust target instead of allowing hdf5-sys
# to discover the host-architecture Homebrew installation.
vcpkg_root="$GITHUB_WORKSPACE/vcpkg"
vcpkg_revision="a7eda31dc16994fcaa8587982eb833a8695f1b6f"
overlay_ports="$GITHUB_WORKSPACE/infra/vcpkg-ports"

git clone https://github.com/microsoft/vcpkg.git "$vcpkg_root"
git -C "$vcpkg_root" checkout "$vcpkg_revision"
"$vcpkg_root/bootstrap-vcpkg.sh" -disableMetrics

install_hdf5() {
    "$vcpkg_root/vcpkg" install hdf5:x64-osx \
        "--overlay-ports=$overlay_ports" \
        --clean-after-build
}

for attempt in 1 2 3; do
    if install_hdf5; then
        break
    fi
    if [[ "$attempt" == "3" ]]; then
        echo "vcpkg failed to install hdf5:x64-osx after three attempts" >&2
        exit 1
    fi
    echo "vcpkg install attempt $attempt failed; retrying" >&2
done

hdf5_prefix="$vcpkg_root/installed/x64-osx"
hdf5_header="$hdf5_prefix/include/H5pubconf.h"
hdf5_library="$hdf5_prefix/lib/libhdf5.a"

if [[ ! -f "$hdf5_header" || ! -f "$hdf5_library" ]]; then
    echo "vcpkg did not produce the expected x86_64 HDF5 installation" >&2
    exit 1
fi

if ! lipo -verify_arch x86_64 "$hdf5_library"; then
    echo "the installed HDF5 library does not contain x86_64 code" >&2
    exit 1
fi

echo "HDF5_DIR=$hdf5_prefix" >> "$GITHUB_ENV"
