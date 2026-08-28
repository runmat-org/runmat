#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <runmat-binary>" >&2
  exit 2
fi

runmat_binary="$1"
if [[ ! -x "$runmat_binary" ]]; then
  echo "RunMat binary is not executable: $runmat_binary" >&2
  exit 1
fi

# Linux distributions can install the unversioned HDF5 linker name outside the
# linker's default search directories. The AOT runtime intentionally records
# portable library names rather than build-machine paths, so expose the local
# development package's canonical library directory while verifying it.
if [[ "$(uname -s)" == "Linux" ]]; then
  command -v pkg-config >/dev/null 2>&1 || {
    echo "pkg-config is required to locate the HDF5 development libraries" >&2
    exit 1
  }
  if ! hdf5_link_search_output="$(pkg-config --libs-only-L hdf5)"; then
    echo "pkg-config could not locate the HDF5 development package" >&2
    exit 1
  fi

  read -r -a hdf5_link_search_flags <<< "$hdf5_link_search_output"
  hdf5_library_path=""
  for flag in "${hdf5_link_search_flags[@]}"; do
    [[ "$flag" == -L* ]] || continue
    hdf5_library_directory="${flag#-L}"
    if [[ -z "$hdf5_library_directory" || ! -d "$hdf5_library_directory" ]]; then
      echo "pkg-config returned an invalid HDF5 library directory: $hdf5_library_directory" >&2
      exit 1
    fi
    hdf5_library_path="${hdf5_library_path:+$hdf5_library_path:}$hdf5_library_directory"
  done

  if [[ -z "$hdf5_library_path" ]]; then
    echo "pkg-config returned no HDF5 linker search directories" >&2
    exit 1
  fi
  export LIBRARY_PATH="$hdf5_library_path${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

smoke_directory="$(mktemp -d "${TMPDIR:-/tmp}/runmat-aot-smoke.XXXXXX")"
trap 'rm -rf "$smoke_directory"' EXIT
printf 'x = 2 + 3;\ndisp(x);\n' > "$smoke_directory/smoke.m"

"$runmat_binary" compile "$smoke_directory/smoke.m" -o "$smoke_directory/smoke"
"$smoke_directory/smoke" | grep -Fx '5'
