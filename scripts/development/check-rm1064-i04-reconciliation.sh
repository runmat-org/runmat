#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

node scripts/development/rm1064-i04-reconciliation.mjs \
  f24908ddc 25ab190277 \
  docs/development/rm1064-i04-terminal-reconciliation.json \
  83879a509ac5117a8d066383a0593f9af4d5ea3f
git diff --exit-code -- docs/development/rm1064-i04-terminal-reconciliation.json
