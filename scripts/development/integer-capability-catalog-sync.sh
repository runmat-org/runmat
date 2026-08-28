#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)
catalog="${repo_root}/docs/builtins/meta.json"
live=''
output=''
in_place=false
names=()

usage() {
  printf '%s\n' \
    'Usage:' \
    '  integer-capability-catalog-sync.sh --live LIVE.json --output OUTPUT.json NAME...' \
    '  integer-capability-catalog-sync.sh --live LIVE.json --in-place NAME...'
}

die() {
  printf 'integer-capability-catalog-sync: %s\n' "$1" >&2
  exit 2
}

command -v jq >/dev/null 2>&1 || die 'jq is required'
command -v shasum >/dev/null 2>&1 || die 'shasum is required'

while (($# > 0)); do
  case "$1" in
    --live)
      (($# >= 2)) || die '--live requires a path'
      live=$2
      shift 2
      ;;
    --catalog)
      (($# >= 2)) || die '--catalog requires a path'
      catalog=$2
      shift 2
      ;;
    --output)
      (($# >= 2)) || die '--output requires a path'
      output=$2
      shift 2
      ;;
    --in-place)
      in_place=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*) die "unknown option: $1" ;;
    *)
      names[${#names[@]}]=$1
      shift
      ;;
  esac
done

[[ -n "${live}" ]] || die '--live is required'
[[ -f "${live}" ]] || die "live catalog does not exist: ${live}"
[[ -f "${catalog}" ]] || die "checked catalog does not exist: ${catalog}"
if ${in_place}; then
  [[ -z "${output}" ]] || die '--output and --in-place are mutually exclusive'
  output=${catalog}
else
  [[ -n "${output}" ]] || die '--output or --in-place is required'
  [[ ! -e "${output}" && ! -L "${output}" ]] || die "refusing to overwrite output: ${output}"
fi
[[ -d "$(dirname -- "${output}")" ]] || die "output directory does not exist: $(dirname -- "${output}")"

((${#names[@]} >= 8 && ${#names[@]} <= 25)) || die 'a cohort must contain 8-25 names'
names_json=$(printf '%s\n' "${names[@]}" | jq -R . | jq -s .)
unique_count=$(jq 'unique | length' <<<"${names_json}")
[[ "${unique_count}" == "${#names[@]}" ]] || die 'cohort contains duplicate names'

jq -e --argjson names "${names_json}" '[$names[] as $name | ([.builtins[] | select(.name == $name)] | length) == 1] | all' "${catalog}" >/dev/null || die 'every cohort name must occur exactly once in the checked catalog'
jq -e --argjson names "${names_json}" '[$names[] as $name | ([.builtins[] | select(.name == $name)] | length) == 1] | all' "${live}" >/dev/null || die 'every cohort name must occur exactly once in the live catalog'

canonical_hash() {
  local path=$1
  local selection=$2
  jq -Sc --argjson names "${names_json}" "${selection}" "${path}" | shasum -a 256 | awk '{print $1}'
}

target_before=$(canonical_hash "${catalog}" '[$names[] as $name | .builtins[] | select(.name == $name)]')
target_live=$(canonical_hash "${live}" '[$names[] as $name | .builtins[] | select(.name == $name)]')
nontarget_before=$(canonical_hash "${catalog}" '[.builtins[] | .name as $name | select(($names | index($name)) == null)]')

candidate=$(mktemp "${output}.sync.XXXXXX")
trap 'rm -f "${candidate}"' EXIT
cp -p "${catalog}" "${candidate}"
jq --slurpfile live "${live}" --argjson names "${names_json}" '
  ($live[0].builtins | map({key: .name, value: .}) | from_entries) as $live_by_name
  | .builtins |= map(.name as $name | if ($names | index($name)) != null then $live_by_name[$name] else . end)
' "${catalog}" >"${candidate}"

jq -e . "${candidate}" >/dev/null
target_after=$(canonical_hash "${candidate}" '[$names[] as $name | .builtins[] | select(.name == $name)]')
nontarget_after=$(canonical_hash "${candidate}" '[.builtins[] | .name as $name | select(($names | index($name)) == null)]')
[[ "${target_after}" == "${target_live}" ]] || die 'target records do not exactly match the live catalog'
[[ "${nontarget_after}" == "${nontarget_before}" ]] || die 'non-target checked records changed'

mv "${candidate}" "${output}"
printf 'metric\tvalue\n'
printf 'cohort_names\t%s\n' "$(IFS=,; printf '%s' "${names[*]}")"
printf 'target_hash_before\t%s\n' "${target_before}"
printf 'target_hash_live\t%s\n' "${target_live}"
printf 'target_hash_after\t%s\n' "${target_after}"
printf 'nontarget_hash_before\t%s\n' "${nontarget_before}"
printf 'nontarget_hash_after\t%s\n' "${nontarget_after}"
printf 'output\t%s\n' "${output}"
