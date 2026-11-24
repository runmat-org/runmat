# Runner job-completed hook (ACTIONS_RUNNER_HOOK_JOB_COMPLETED). Releases this
# slot's warm lanes, wipes the slot's work folder and TEMP, and applies the
# lane and cache caps. Build outputs never stay on the work disk.
$ErrorActionPreference = 'Stop'
& 'C:\Python313\python.exe' 'C:\ci\common\ci_cache.py' job-completed
exit $LASTEXITCODE
