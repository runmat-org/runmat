#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

node "${script_dir}/check-architecture-boundaries.mjs"
exec node "${repo_dir}/scripts/fea/check-language-surface-ledger.mjs"
