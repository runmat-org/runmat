#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)
audit="${script_dir}/integer-capability-audit.sh"
temp_dir=$(mktemp -d)
trap 'rm -rf "${temp_dir}"' EXIT
fixture="${temp_dir}/catalog.json"

jq -n '{builtins: ([range(0; 30) | {name: ("candidate" + tostring), signatures: [{label: ("candidate" + tostring + "(X)"), inputs: [{ty: (if . % 2 == 0 then "Any" else "SizeArg" end)}]}], integer_capabilities: [], integer_audit: null}] + [{name: "settled", signatures: [{label: "settled(X)", inputs: [{ty: "Any"}]}], integer_capabilities: [], integer_audit: {kind: "NotApplicable"}}, {name: "open-form", signatures: [{label: "open-form(X)", inputs: [{ty: "Any"}]}], integer_capabilities: [{form: "open-form(X)", notes: "[integer-audit-open] documented form remains unresolved"}], integer_audit: null}])}' >"${fixture}"

run_audit() {
  RUNMAT_INTEGER_AUDIT_CATALOG="${fixture}" "${audit}" "$@"
}

summary_count=$(run_audit summary | awk -F '\t' '$1 == "signature_screen_untriaged" {print $2}')
queue_count=$(run_audit queue --limit 2000 --format json | jq 'length')
[[ "${summary_count}" == "${queue_count}" ]]
[[ "$(run_audit summary | awk -F '\t' '$1 == "integer_capability_open_builtin_names" {print $2}')" == "1" ]]
run_audit queue --limit 2000 --format names | grep -qx 'open-form'

first_queue=$(run_audit queue --limit 30 --format names)
second_queue=$(run_audit queue --limit 30 --format names)
[[ "${first_queue}" == "${second_queue}" ]]

cohort_names=()
extended_names=()
while IFS= read -r name; do
  extended_names[${#extended_names[@]}]="${name}"
  if ((${#cohort_names[@]} < 8)); then
    cohort_names[${#cohort_names[@]}]="${name}"
  fi
done <<<"${first_queue}"
[[ ${#cohort_names[@]} -eq 8 ]]
[[ ${#extended_names[@]} -eq 30 ]]

type_filtered=$(run_audit queue --limit 2000 --input-type SizeArg --format json)
jq -e 'length > 0 and all(.[]; .screened_input_types | index("SizeArg") != null)' >/dev/null <<<"${type_filtered}"

run_audit queue --limit 2000 --format json | jq -r '.[] | select(.reference != null) | .reference' | while IFS= read -r reference; do
  [[ -f "${repo_root}/${reference}" ]]
done

packet=$(run_audit packet "${cohort_names[@]}")
jq -e '.selected_count == 8 and (.errors | length) == 0 and (.records | length) == 8' >/dev/null <<<"${packet}"

expect_packet_rejection() {
  local output
  if output=$(run_audit packet "$@" 2>/dev/null); then
    printf '%s\n' 'expected packet rejection' >&2
    exit 1
  fi
  printf '%s\n' "${output}"
}

expect_packet_rejection settled "${cohort_names[@]:1:7}" | jq -e '.errors | any(contains("not currently untriaged"))' >/dev/null
expect_packet_rejection not.a.real.builtin "${cohort_names[@]:1:7}" | jq -e '.errors | any(contains("unknown builtin"))' >/dev/null
expect_packet_rejection "${cohort_names[@]:0:7}" | jq -e '.errors | any(contains("8-25 unique names"))' >/dev/null
expect_packet_rejection "${cohort_names[0]}" "${cohort_names[0]}" "${cohort_names[@]:1:6}" | jq -e '.errors | any(contains("duplicate names"))' >/dev/null
expect_packet_rejection "${extended_names[@]:0:26}" | jq -e '.errors | any(contains("8-25 unique names"))' >/dev/null

printf '%s\n' 'integer-capability-audit tests passed'
