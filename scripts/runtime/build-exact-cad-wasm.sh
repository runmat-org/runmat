#!/usr/bin/env bash
set -euo pipefail
# Emscripten 4.0.23 and Rust 1.93.0 are the supported toolchain pair.
# Activate emsdk before invoking this script; no global SDK installation is made.
runmat_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v emcc >/dev/null || { echo "Activate Emscripten 4.0.23 before building the exact CAD browser host." >&2; exit 1; }
runmat_emcc_version="$(emcc -dumpversion)"
[[ "$runmat_emcc_version" == "4.0.23" ]] || { echo "Exact CAD requires Emscripten 4.0.23; found $runmat_emcc_version." >&2; exit 1; }
: "${CARGO_TARGET_DIR:=$runmat_root/target-exact-cad-wasm}"
export CARGO_TARGET_DIR
export CXXFLAGS="${CXXFLAGS:-} -fwasm-exceptions -UOCC_CONVERT_SIGNALS"
# OCCT enables Unix signal-to-longjmp conversion for every non-MSVC build.
# Browsers have no POSIX signal handler. Keep C++ exception recovery, but omit
# that native-only shim (its setjmp/EH control flow also produces invalid Wasm).
# Rust errors cross this ABI as Results; Rust panics terminate the worker.
# Rust 1.93 uses Wasm exception handling. Compile OCCT with the same ABI so
# its caught C++ import errors remain recoverable across repeated requests.
export CARGO_TARGET_WASM32_UNKNOWN_EMSCRIPTEN_RUSTFLAGS="-C panic=abort -C link-arg=-sMODULARIZE=1 -C link-arg=-sEXPORT_ES6=1 -C link-arg=-sALLOW_MEMORY_GROWTH=1 -C link-arg=-sSTACK_SIZE=8388608 -C link-arg=-sINITIAL_MEMORY=33554432 -C link-arg=-sENVIRONMENT=web,worker,node -C link-arg=-sEXPORTED_RUNTIME_METHODS=cwrap,UTF8ToString,HEAPU8 -C link-arg=-sEXPORTED_FUNCTIONS=_main,_malloc,_free,_runmat_occt_exact_import,_runmat_occt_exact_free,_runmat_occt_exact_execute,_runmat_cad_call,_runmat_cad_call_framed,_runmat_cad_free_framed,_runmat_cad_seed -C link-arg=-fwasm-exceptions -C link-arg=-sNO_EXIT_RUNTIME=1"
cd "$runmat_root"
# This is a leaf-link constraint, so changing the foreign-kernel heap ceiling
# does not invalidate every Rust dependency's compilation cache.
cargo +1.93.0 rustc --locked -p runmat-geometry-wasm-host --bin runmat_geometry_wasm_host --target wasm32-unknown-emscripten --release -j "${RUNMAT_CAD_BUILD_JOBS:-4}" -- -C link-arg=-sMAXIMUM_MEMORY=536870912
runmat_output="$runmat_root/bindings/ts/dist/occt"
mkdir -p "$runmat_output"
cp "$CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/runmat_geometry_wasm_host.js" "$runmat_output/"
cp "$CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/runmat_geometry_wasm_host.wasm" "$runmat_output/"
runmat_occt_root="${RUNMAT_OCCT_ROOT:-$CARGO_TARGET_DIR/wasm32-unknown-emscripten/OCCT}"
for runmat_license in LICENSE_LGPL_21.txt OCCT_LGPL_EXCEPTION.txt; do
  if [[ -f "$runmat_occt_root/share/doc/opencascade/$runmat_license" ]]; then
    cp "$runmat_occt_root/share/doc/opencascade/$runmat_license" "$runmat_output/"
  else
    cp "$runmat_occt_root/$runmat_license" "$runmat_output/"
  fi
done
