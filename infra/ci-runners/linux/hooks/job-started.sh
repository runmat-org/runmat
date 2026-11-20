#!/usr/bin/env bash
# Runner job-started hook (ACTIONS_RUNNER_HOOK_JOB_STARTED). Refuses fork code,
# empties this slot's workspace, HOME and TMPDIR, and refuses the job when a
# disk is below its free-space floor. It runs as root so it can remove files a
# job created with sudo or through Docker.
set -euo pipefail
sudo -n --preserve-env=CI_SLOT,GITHUB_WORKSPACE,GITHUB_EVENT_NAME,GITHUB_EVENT_PATH,GITHUB_REPOSITORY \
  /usr/bin/python3 /opt/ci/common/ci_cache.py job-started
sudo -n chown -R runner:runner "/ci/work/${CI_SLOT}/home" "/ci/work/${CI_SLOT}/tmp" "/ci/work/${CI_SLOT}/cold-target"
