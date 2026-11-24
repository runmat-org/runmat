# Runner job-started hook (ACTIONS_RUNNER_HOOK_JOB_STARTED). Refuses fork code,
# empties this slot's workspace and TEMP, and refuses the job when a disk is
# below its free-space floor.
$ErrorActionPreference = 'Stop'
& 'C:\Python313\python.exe' 'C:\ci\common\ci_cache.py' job-started
exit $LASTEXITCODE
