#!/usr/bin/env bash
set -euo pipefail

profile="${RUNMAT_BUILD_PROFILE:-release}"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/runmat-aot-build.XXXXXX")"
trap 'rm -rf "$build_root"' EXIT
rustc_log="$build_root/native-static-libs.log"
target="${RUNMAT_BUILD_TARGET:-${CARGO_BUILD_TARGET:-}}"
locked_args=()
build_args=()
runtime_no_default=false
runtime_wgpu=false
runtime_cuda=false

record_runtime_features() {
  local feature
  local requested=${1//,/ }
  for feature in $requested; do
    case "$feature" in
      wgpu|runmat/wgpu) runtime_wgpu=true ;;
      cuda|runmat/cuda) runtime_cuda=true ;;
    esac
  done
}

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
      [[ -n "$target" ]] || { echo "--target requires a value" >&2; exit 2; }
      shift
      ;;
    --target=*)
      target="${argument#--target=}"
      [[ -n "$target" ]] || { echo "--target requires a value" >&2; exit 2; }
      ;;
    --locked|--offline|--frozen)
      locked_args+=("$argument")
      build_args+=("$argument")
      ;;
    --no-default-features)
      runtime_no_default=true
      build_args+=("$argument")
      ;;
    --features)
      [[ $# -ge 2 ]] || { echo "--features requires a value" >&2; exit 2; }
      record_runtime_features "$2"
      build_args+=("$argument" "$2")
      shift
      ;;
    --features=*)
      record_runtime_features "${argument#--features=}"
      build_args+=("$argument")
      ;;
    --all-features)
      runtime_wgpu=true
      runtime_cuda=true
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

target_root="${CARGO_TARGET_DIR:-target}"
target_directory="$target_root/$profile_directory"
runtime_build=(cargo rustc -p runmat-aot-runtime --profile "$profile")
if "$runtime_no_default"; then
  runtime_build+=(--no-default-features)
fi
runtime_features=()
if "$runtime_wgpu"; then runtime_features+=(wgpu); fi
if "$runtime_cuda"; then runtime_features+=(cuda); fi
if ((${#runtime_features[@]})); then
  runtime_feature_list="$(IFS=,; echo "${runtime_features[*]}")"
  runtime_build+=(--features "$runtime_feature_list")
fi
# The packer reads target object bytes but executes on the build host. Keep
# its target explicit so Cargo's deployment-target environment cannot select
# a foreign executable for this phase.
host_target="$("${RUSTC:-rustc}" -vV | sed -n 's/^host: //p')"
[[ -n "$host_target" ]] || { echo "Rust compiler did not report its build-host target" >&2; exit 1; }
pack_command=(cargo run --target "$host_target")
for locked_argument in "${locked_args[@]-}"; do
  [[ -n "$locked_argument" ]] || continue
  runtime_build+=("$locked_argument")
  pack_command+=("$locked_argument")
done
if [[ -n "$target" ]]; then
  runtime_build+=(--target "$target")
  target_directory="$target_root/$target/$profile_directory"
fi
runtime_build+=(--lib --crate-type staticlib -- --print native-static-libs)

"${runtime_build[@]}" 2>&1 | tee "$rustc_log"

# Static-library naming belongs to the deployment target, not the build host.
archive_name_command=("${RUSTC:-rustc}" --crate-name runmat_aot_runtime --crate-type staticlib --print file-names)
if [[ -n "$target" ]]; then
  archive_name_command+=(--target "$target")
fi
archive_name="$("${archive_name_command[@]}" - < /dev/null)"
[[ -n "$archive_name" && "$archive_name" != *$'\n'* ]] || { echo "Rust compiler did not report one AOT runtime archive name" >&2; exit 1; }
archive="$target_directory/$archive_name"
[[ -f "$archive" ]] || { echo "AOT runtime archive was not produced at $archive" >&2; exit 1; }

native_line="$(sed -n 's/^note: native-static-libs: //p' "$rustc_log" | tail -n 1)"
if [[ -z "$native_line" ]]; then
  echo "Cargo did not report native-static-libs for the AOT runtime archive" >&2
  exit 1
fi

pack_args=(
  --archive "$archive"
  --payload-out "$build_root/runtime-archive.payload"
  --manifest-out "$build_root/runtime-archive.json"
  --notices-out "$build_root/THIRD_PARTY_NOTICES.txt"
)
read -r -a native_tokens <<< "$native_line"
for token in "${native_tokens[@]}"; do
  pack_args+=(--native-link-token "$token")
done
pack_command+=(-p runmat-aot --bin runmat-aot-pack --)
"${pack_command[@]}" "${pack_args[@]}"

final_build=(cargo build -p runmat --profile "$profile")
if [[ -n "$target" ]]; then
  final_build+=(--target "$target")
fi
for build_argument in "${build_args[@]-}"; do
  [[ -n "$build_argument" ]] || continue
  final_build+=("$build_argument")
done
RUNMAT_AOT_RUNTIME_ARCHIVE="$build_root/runtime-archive.payload" \
RUNMAT_AOT_RUNTIME_MANIFEST="$build_root/runtime-archive.json" \
  "${final_build[@]}"

notice_destination="$target_directory/THIRD_PARTY_NOTICES.txt"
if [[ -L "$notice_destination" ]]; then
  echo "refusing symlink notice destination $notice_destination" >&2
  exit 1
fi
if [[ -e "$notice_destination" ]]; then
  [[ -f "$notice_destination" ]] && cmp -s "$build_root/THIRD_PARTY_NOTICES.txt" "$notice_destination" || {
    echo "existing AOT notice differs from canonical Runtime bytes: $notice_destination" >&2
    exit 1
  }
else
  cp "$build_root/THIRD_PARTY_NOTICES.txt" "$notice_destination"
fi
