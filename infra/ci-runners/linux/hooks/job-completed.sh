#!/usr/bin/env bash
# Runner job-completed hook (ACTIONS_RUNNER_HOOK_JOB_COMPLETED). Releases this
# slot's warm lanes, wipes the slot's work folder, HOME and TMPDIR, and applies
# the lane and cache caps. Build outputs never stay on the work disk.
set -euo pipefail
sudo -n --preserve-env=CI_SLOT,GITHUB_WORKSPACE,GITHUB_EVENT_NAME,GITHUB_EVENT_PATH,GITHUB_REPOSITORY \
  /usr/bin/python3 /opt/ci/common/ci_cache.py job-completed
sudo -n chown -R runner:runner "/ci/work/${CI_SLOT}"
