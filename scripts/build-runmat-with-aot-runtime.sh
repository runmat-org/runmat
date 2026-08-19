#!/usr/bin/env bash
set -euo pipefail

profile="${RUNMAT_BUILD_PROFILE:-release}"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/runmat-aot-build.XXXXXX")"
trap 'rm -rf "$build_root"' EXIT
rustc_log="$build_root/native-static-libs.log"
target="${RUNMAT_BUILD_TARGET:-}"
locked_args=()
build_args=()

while (($#)); do
  argument="$1"
  case "$argument" in
    --release)
      profile="release"
      ;;
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }
      profile="$2"
      shift
      ;;
    --profile=*)
      profile="${argument#--profile=}"
      ;;
    --target)
      [[ $# -ge 2 ]] || { echo "--target requires a value" >&2; exit 2; }
      target="$2"
      build_args+=("$argument" "$2")
      shift
      ;;
    --target=*)
      target="${argument#--target=}"
      build_args+=("$argument")
      ;;
    --locked|--offline|--frozen)
      locked_args+=("$argument")
      build_args+=("$argument")
      ;;
    *)
      build_args+=("$argument")
      ;;
  esac
  shift
done

profile_directory="$profile"
if [[ "$profile" == "dev" ]]; then
  profile_directory="debug"
fi

target_directory="target/$profile_directory"
runtime_build=(cargo rustc -p runmat-aot-runtime --profile "$profile")
pack_command=(cargo run)
for locked_argument in "${locked_args[@]-}"; do
  [[ -n "$locked_argument" ]] || continue
  runtime_build+=("$locked_argument")
  pack_command+=("$locked_argument")
done
if [[ -n "$target" ]]; then
  runtime_build+=(--target "$target")
  pack_command+=(--target "$target")
  target_directory="target/$target/$profile_directory"
fi
runtime_build+=(--lib --crate-type staticlib -- --print native-static-libs)

"${runtime_build[@]}" 2>&1 | tee "$rustc_log"

case "$(uname -s)" in
  Darwin|Linux) archive="$target_directory/librunmat_aot_runtime.a" ;;
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
pack_command+=(-p runmat-aot --bin runmat-aot-pack --)
"${pack_command[@]}" "${pack_args[@]}"

RUNMAT_AOT_RUNTIME_ARCHIVE="$build_root/runtime-archive.payload" \
RUNMAT_AOT_RUNTIME_MANIFEST="$build_root/runtime-archive.json" \
  cargo build -p runmat --profile "$profile" "${build_args[@]}"
