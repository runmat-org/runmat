#!/usr/bin/env bash
set -euo pipefail

profile="${RUNMAT_BUILD_PROFILE:-release}"
profile_directory="$profile"
if [[ "$profile" == "dev" ]]; then
  profile_directory="debug"
fi
build_root="$(mktemp -d "${TMPDIR:-/tmp}/runmat-aot-build.XXXXXX")"
trap 'rm -rf "$build_root"' EXIT
rustc_log="$build_root/native-static-libs.log"

cargo rustc -p runmat-aot-runtime --profile "$profile" --lib --crate-type staticlib -- --print native-static-libs 2>&1 | tee "$rustc_log"

case "$(uname -s)" in
  Darwin|Linux) archive="target/$profile_directory/librunmat_aot_runtime.a" ;;
  *) echo "scripts/build-runmat-with-aot-runtime.sh supports macOS and Linux hosts; use the PowerShell build on Windows" >&2; exit 1 ;;
esac

native_line="$(sed -n 's/^note: native-static-libs: //p' "$rustc_log" | tail -n 1)"
if [[ -z "$native_line" ]]; then
  echo "Cargo did not report native-static-libs for the AOT runtime archive" >&2
  exit 1
fi

pack_args=(
  --archive "$archive"
  --payload-out "$build_root/runtime-archive.payload"
  --manifest-out "$build_root/runtime-archive.json"
)
read -r -a native_tokens <<< "$native_line"
for token in "${native_tokens[@]}"; do
  pack_args+=(--native-link-token "$token")
done
cargo run -p runmat-aot --bin runmat-aot-pack -- "${pack_args[@]}"

RUNMAT_AOT_RUNTIME_ARCHIVE="$build_root/runtime-archive.payload" \
RUNMAT_AOT_RUNTIME_MANIFEST="$build_root/runtime-archive.json" \
  cargo build -p runmat --profile "$profile" "$@"
