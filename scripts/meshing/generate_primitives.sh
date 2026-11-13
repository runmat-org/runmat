#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
occt_root="${repo_root}/target/OCCT"
source_file="${repo_root}/crates/runmat-geometry/io/tests/fixtures/generate_meshing_primitives.cc"
fixture_root="${repo_root}/crates/runmat-geometry/io/tests/fixtures"
generator="${TMPDIR:-/tmp}/runmat-generate-meshing-primitives"

if [[ ! -f "${occt_root}/include/BRepTools.hxx" ]]; then
  echo "OCCT headers are missing at ${occt_root}; build RunMat with occt-native first" >&2
  exit 2
fi

c++ -std=c++17 \
  -I "${occt_root}/include" \
  "${source_file}" \
  -L "${occt_root}/lib" \
  -lTKShHealing -lTKPrim -lTKTopAlgo -lTKGeomAlgo -lTKGeomBase -lTKG3d -lTKG2d -lTKBRep -lTKMath -lTKernel \
  -o "${generator}"
"${generator}" "${fixture_root}"
