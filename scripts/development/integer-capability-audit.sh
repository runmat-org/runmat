#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)
catalog=${RUNMAT_INTEGER_AUDIT_CATALOG:-"${repo_root}/docs/builtins/meta.json"}
program="${script_dir}/integer-capability-audit.jq"

command -v jq >/dev/null 2>&1 || {
  printf '%s\n' 'integer-capability-audit: jq is required' >&2
  exit 1
}
command -v rg >/dev/null 2>&1 || {
  printf '%s\n' 'integer-capability-audit: rg is required' >&2
  exit 1
}
[[ -f "${catalog}" ]] || {
  printf 'integer-capability-audit: catalog does not exist: %s\n' "${catalog}" >&2
  exit 1
}

reference_names=$(rg --files "${repo_root}/docs/builtins/reference" -g '*.json' | sed -E 's#^.*/##; s#\.json$##' | jq -R . | jq -s .)

usage() {
  printf '%s\n' \
    'Usage:' \
    '  integer-capability-audit.sh summary' \
    '  integer-capability-audit.sh queue [--limit N] [--offset N] [--name-regex REGEX] [--input-type TYPE] [--format tsv|names|json]' \
    '  integer-capability-audit.sh packet NAME...'
}

die() {
  printf 'integer-capability-audit: %s\n' "$1" >&2
  exit 2
}

require_nonnegative_integer() {
  local label=$1
  local value=$2
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${label} must be a nonnegative integer"
}

command=${1:-}
[[ -n "${command}" ]] || {
  usage >&2
  exit 2
}
shift

case "${command}" in
  summary)
    (($# == 0)) || die 'summary accepts no arguments'
    jq -r --arg command summary --arg name_regex '' --arg input_type '' --argjson offset 0 --argjson limit 0 --argjson requested_names '[]' --argjson reference_names "${reference_names}" -f "${program}" "${catalog}" | jq -r '.[] | @tsv'
    ;;
  queue)
    limit=25
    offset=0
    name_regex=''
    input_type=''
    format=tsv
    while (($# > 0)); do
      case "$1" in
        --limit)
          (($# >= 2)) || die '--limit requires a value'
          limit=$2
          shift 2
          ;;
        --offset)
          (($# >= 2)) || die '--offset requires a value'
          offset=$2
          shift 2
          ;;
        --name-regex)
          (($# >= 2)) || die '--name-regex requires a value'
          name_regex=$2
          shift 2
          ;;
        --input-type)
          (($# >= 2)) || die '--input-type requires a value'
          input_type=$2
          shift 2
          ;;
        --format)
          (($# >= 2)) || die '--format requires a value'
          format=$2
          shift 2
          ;;
        *) die "unknown queue argument: $1" ;;
      esac
    done
    require_nonnegative_integer limit "${limit}"
    require_nonnegative_integer offset "${offset}"
    result=$(jq -c --arg command queue --arg name_regex "${name_regex}" --arg input_type "${input_type}" --argjson offset "${offset}" --argjson limit "${limit}" --argjson requested_names '[]' --argjson reference_names "${reference_names}" -f "${program}" "${catalog}")
    case "${format}" in
      json) jq . <<<"${result}" ;;
      names) jq -r '.[].name' <<<"${result}" ;;
      tsv)
        printf 'queue_position\tcatalog_position\tname\tsignatures\tscreened_input_types\treference\n'
        jq -r '.[] | [.queue_position, .catalog_position, .name, .signature_count, (.screened_input_types | join(",")), .reference] | @tsv' <<<"${result}"
        ;;
      *) die "unsupported format: ${format}" ;;
    esac
    ;;
  packet)
    (($# > 0)) || die 'packet requires 8-25 builtin names'
    requested_names=$(printf '%s\n' "$@" | jq -R . | jq -s .)
    result=$(jq -c --arg command packet --arg name_regex '' --arg input_type '' --argjson offset 0 --argjson limit 0 --argjson requested_names "${requested_names}" --argjson reference_names "${reference_names}" -f "${program}" "${catalog}")
    jq . <<<"${result}"
    if ! jq -e '.errors | length == 0' >/dev/null <<<"${result}"; then
      exit 2
    fi
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    die "unknown command: ${command}"
    ;;
esac
