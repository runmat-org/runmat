#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sync_script="${script_dir}/integer-capability-catalog-sync.sh"
temp_dir=$(mktemp -d)
trap 'rm -rf "${temp_dir}"' EXIT
checked="${temp_dir}/checked.json"
live="${temp_dir}/live.json"
output="${temp_dir}/output.json"
names=(builtin0 builtin1 builtin2 builtin3 builtin4 builtin5 builtin6 builtin7)
all_names=()
for ((index = 0; index < 30; index++)); do
  all_names[${#all_names[@]}]="builtin${index}"
done

jq -n '{builtins: ([range(0; 30) | {name: ("builtin" + tostring), marker: "checked"}] + [{name: "missing-live", marker: "checked"}, {name: "untouched", marker: "checked"}])}' >"${checked}"
jq -n '{builtins: ([range(0; 30) | {name: ("builtin" + tostring), marker: "live"}] + [{name: "missing-checked", marker: "live"}, {name: "untouched", marker: "live"}, {name: "live-only", marker: "live"}])}' >"${live}"

report=$("${sync_script}" --catalog "${checked}" --live "${live}" --output "${output}" "${names[@]}")
[[ -f "${output}" ]]
jq -e '([.builtins[:8][]] | all(.marker == "live")) and ([.builtins[8:30][]] | all(.marker == "checked")) and (.builtins[] | select(.name == "untouched") | .marker == "checked") and ([.builtins[] | select(.name == "live-only")] | length == 0)' "${output}" >/dev/null
jq -e --slurpfile checked "${checked}" '[.builtins[].name] == [$checked[0].builtins[].name]' "${output}" >/dev/null
target_live=$(awk -F '\t' '$1 == "target_hash_live" {print $2}' <<<"${report}")
target_after=$(awk -F '\t' '$1 == "target_hash_after" {print $2}' <<<"${report}")
nontarget_before=$(awk -F '\t' '$1 == "nontarget_hash_before" {print $2}' <<<"${report}")
nontarget_after=$(awk -F '\t' '$1 == "nontarget_hash_after" {print $2}' <<<"${report}")
[[ "${target_live}" == "${target_after}" ]]
[[ "${nontarget_before}" == "${nontarget_after}" ]]

cp "${checked}" "${temp_dir}/in-place.json"
"${sync_script}" --catalog "${temp_dir}/in-place.json" --live "${live}" --in-place "${names[@]}" >/dev/null
jq -e '([.builtins[:8][]] | all(.marker == "live")) and ([.builtins[8:30][]] | all(.marker == "checked")) and (.builtins[] | select(.name == "untouched") | .marker == "checked")' "${temp_dir}/in-place.json" >/dev/null

expect_sync_rejection() {
  if "${sync_script}" "$@" >/dev/null 2>&1; then
    printf '%s\n' 'expected catalog-sync rejection' >&2
    exit 1
  fi
}

expect_sync_rejection --catalog "${checked}" --live "${live}" --output "${temp_dir}/too-small.json" "${names[@]:0:7}"
expect_sync_rejection --catalog "${checked}" --live "${live}" --output "${temp_dir}/too-large.json" "${all_names[@]:0:26}"
expect_sync_rejection --catalog "${checked}" --live "${live}" --output "${temp_dir}/duplicate.json" "${names[0]}" "${names[0]}" "${names[@]:1:6}"
expect_sync_rejection --catalog "${checked}" --live "${live}" --output "${temp_dir}/missing-checked.json" missing-checked "${names[@]:1:7}"
expect_sync_rejection --catalog "${checked}" --live "${live}" --output "${temp_dir}/missing-live.json" missing-live "${names[@]:1:7}"
jq -n '{}' >"${temp_dir}/existing.json"
expect_sync_rejection --catalog "${checked}" --live "${live}" --output "${temp_dir}/existing.json" "${names[@]}"
expect_sync_rejection --catalog "${checked}" --live "${live}" --output "${temp_dir}/mutually-exclusive.json" --in-place "${names[@]}"

cp "${checked}" "${temp_dir}/failed-in-place.json"
cp "${temp_dir}/failed-in-place.json" "${temp_dir}/failed-in-place-before.json"
expect_sync_rejection --catalog "${temp_dir}/failed-in-place.json" --live "${live}" --in-place missing-live "${names[@]:1:7}"
cmp "${temp_dir}/failed-in-place-before.json" "${temp_dir}/failed-in-place.json"

printf '%s\n' 'integer-capability-catalog-sync tests passed'
