use tokio::process::{Child, Command};

#[cfg(target_vendor = "apple")]
use super::retirement_observation::group_probe_error;
use super::retirement_observation::{retirement_observed, RetirementObservation};
use super::ResourceLimits;
use crate::command::ProcessGroupPlacement;

#[cfg(not(windows))]
pub(super) struct ProcessContainment;

#[cfg(windows)]
pub(super) struct ProcessContainment {
    job: std::os::windows::io::OwnedHandle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TerminationStart {
    Signaled,
    #[cfg(target_vendor = "apple")]
    GroupSignalDenied,
}

#[cfg(unix)]
pub(super) fn configure(command: &mut Command, placement: ProcessGroupPlacement) {
    if placement == ProcessGroupPlacement::OwnGroup {
        command.process_group(0);
    }
}

#[cfg(windows)]
pub(super) fn configure(command: &mut Command, _placement: ProcessGroupPlacement) {
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    command.creation_flags(CREATE_NEW_PROCESS_GROUP);
}

#[cfg(not(any(unix, windows)))]
pub(super) fn configure(_command: &mut Command, _placement: ProcessGroupPlacement) {}

#[cfg(unix)]
pub(super) fn validate_join_parent(placement: ProcessGroupPlacement) -> std::io::Result<()> {
    if placement == ProcessGroupPlacement::JoinAncestorGroup
        && unsafe { libc::getpid() } != unsafe { libc::getpgrp() }
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "joined helper requires the current process to lead its owned group",
        ));
    }
    Ok(())
}

#[cfg(windows)]
pub(super) fn validate_join_parent(placement: ProcessGroupPlacement) -> std::io::Result<()> {
    if placement != ProcessGroupPlacement::JoinAncestorGroup {
        return Ok(());
    }
    use windows_sys::Win32::System::JobObjects::IsProcessInJob;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut in_job = 0;
    if unsafe { IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut in_job) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    if in_job == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "joined helper requires inherited JobObject containment",
        ));
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
pub(super) fn validate_join_parent(placement: ProcessGroupPlacement) -> std::io::Result<()> {
    if placement == ProcessGroupPlacement::JoinAncestorGroup {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "joined process containment is unsupported on this platform",
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
pub(super) fn contain(
    _: &Child,
    _: ResourceLimits,
    _: ProcessGroupPlacement,
) -> std::io::Result<Option<ProcessContainment>> {
    Ok(None)
}

#[cfg(windows)]
pub(super) fn contain(
    child: &Child,
    limits: ResourceLimits,
    placement: ProcessGroupPlacement,
) -> std::io::Result<Option<ProcessContainment>> {
    if placement == ProcessGroupPlacement::JoinAncestorGroup {
        // A child of the Graph worker remains in its ancestor JobObject;
        // no breakaway flag is set. Do not assign a second job that could
        // fail on hosts without nested-job support.
        return Ok(None);
    }
    use std::mem::size_of;
    use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _};
    use std::ptr;

    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOB_OBJECT_LIMIT_PROCESS_TIME,
    };

    let raw_job = unsafe { CreateJobObjectW(ptr::null(), ptr::null()) };
    if raw_job.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    let job = unsafe {
        std::os::windows::io::OwnedHandle::from_raw_handle(
            raw_job as std::os::windows::io::RawHandle,
        )
    };
    let mut information: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    information.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if let Some(memory_bytes) = limits.memory_bytes {
        information.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        information.ProcessMemoryLimit =
            usize::try_from(memory_bytes).map_err(|_| std::io::Error::other("memory limit"))?;
    }
    if let Some(cpu_seconds) = limits.cpu_seconds {
        information.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_TIME;
        information.BasicLimitInformation.PerProcessUserTimeLimit = i64::try_from(
            cpu_seconds
                .checked_mul(10_000_000)
                .ok_or_else(|| std::io::Error::other("CPU limit"))?,
        )
        .map_err(|_| std::io::Error::other("CPU limit"))?;
    }
    if let Some(process_count) = limits.process_count {
        information.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
        information.BasicLimitInformation.ActiveProcessLimit =
            u32::try_from(process_count).map_err(|_| std::io::Error::other("process limit"))?;
    }
    let configured = unsafe {
        SetInformationJobObject(
            job.as_raw_handle() as _,
            JobObjectExtendedLimitInformation,
            &information as *const _ as *const _,
            u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
                .map_err(|_| std::io::Error::other("job information size"))?,
        )
    };
    if configured == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let child_handle = child
        .raw_handle()
        .ok_or_else(|| std::io::Error::other("child process handle is unavailable"))?;
    let assigned = unsafe { AssignProcessToJobObject(job.as_raw_handle() as _, child_handle as _) };
    if assigned == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(Some(ProcessContainment { job }))
}

pub(super) async fn await_retirement(
    child: &mut Child,
    process_id: Option<u32>,
    containment: Option<&ProcessContainment>,
    placement: ProcessGroupPlacement,
    initial_probe_error: Option<std::io::Error>,
) -> std::io::Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut first_probe_error = initial_probe_error;
    loop {
        let observation = tree_retired(child, process_id, containment, placement)?;
        if retirement_observed(
            observation,
            std::time::Instant::now() >= deadline,
            &mut first_probe_error,
        )? {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

pub(super) fn await_retirement_blocking(
    child: &mut Child,
    process_id: Option<u32>,
    containment: Option<&ProcessContainment>,
    placement: ProcessGroupPlacement,
    initial_probe_error: Option<std::io::Error>,
) -> std::io::Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut first_probe_error = initial_probe_error;
    loop {
        let observation = tree_retired(child, process_id, containment, placement)?;
        if retirement_observed(
            observation,
            std::time::Instant::now() >= deadline,
            &mut first_probe_error,
        )? {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

pub(super) fn begin_termination(
    child: &mut Child,
    process_id: Option<u32>,
    containment: Option<&ProcessContainment>,
    placement: ProcessGroupPlacement,
    unreaped_exit_observed: bool,
) -> std::io::Result<TerminationStart> {
    #[cfg(unix)]
    {
        let _ = containment;
        if placement == ProcessGroupPlacement::JoinAncestorGroup {
            if child.try_wait()?.is_none() {
                child.start_kill()?;
            }
            return Ok(TerminationStart::Signaled);
        }
        let group = owned_group(process_id)?;
        let result = unsafe { libc::kill(-group, libc::SIGKILL) };
        let signal = if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        };
        group_signal_outcome(signal, || {
            Ok(unreaped_exit_observed
                || super::exit_observation::try_wait_unreaped(process_id)?.is_some())
        })
    }
    #[cfg(windows)]
    {
        let _ = unreaped_exit_observed;
        let _ = process_id;
        if placement == ProcessGroupPlacement::JoinAncestorGroup {
            if child.try_wait()?.is_none() {
                child.start_kill()?;
            }
            return Ok(TerminationStart::Signaled);
        }
        use std::os::windows::io::AsRawHandle as _;
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        let job = containment.ok_or_else(|| {
            std::io::Error::other("owned process tree has no JobObject containment")
        })?;
        if unsafe { TerminateJobObject(job.job.as_raw_handle() as _, 1) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(TerminationStart::Signaled)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (
            child,
            process_id,
            containment,
            placement,
            unreaped_exit_observed,
        );
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "owned process-tree retirement is unsupported on this platform",
        ))
    }
}

#[cfg(unix)]
fn group_signal_outcome(
    signal: std::io::Result<()>,
    terminal: impl FnOnce() -> std::io::Result<bool>,
) -> std::io::Result<TerminationStart> {
    let error = match signal {
        Ok(()) => return Ok(TerminationStart::Signaled),
        Err(error) => error,
    };
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(TerminationStart::Signaled),
        Some(libc::EPERM) => {
            #[cfg(target_vendor = "apple")]
            {
                let _ = terminal;
                // Darwin can omit an exiting member before waitid reports its
                // terminal status. Commit to observation without re-signaling;
                // the owned leader and whole group must still retire.
                Ok(TerminationStart::GroupSignalDenied)
            }
            #[cfg(not(target_vendor = "apple"))]
            if terminal()? {
                Ok(TerminationStart::Signaled)
            } else {
                Err(error)
            }
        }
        _ => Err(error),
    }
}

fn tree_retired(
    child: &mut Child,
    process_id: Option<u32>,
    containment: Option<&ProcessContainment>,
    placement: ProcessGroupPlacement,
) -> std::io::Result<RetirementObservation> {
    let leader_reaped = child.try_wait()?.is_some();
    #[cfg(unix)]
    {
        let _ = containment;
        if placement == ProcessGroupPlacement::JoinAncestorGroup {
            return Ok(RetirementObservation::from_retired(leader_reaped));
        }
        let group = owned_group(process_id)?;
        let descendants_gone = if unsafe { libc::kill(-group, 0) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                true
            } else {
                // Darwin's permission verdict leaves group absence unconfirmed,
                // regardless of leader reap. Retry only this read-only probe;
                // the lifecycle has already committed never to signal again.
                #[cfg(target_vendor = "apple")]
                {
                    return group_probe_error(error);
                }
                #[cfg(not(target_vendor = "apple"))]
                return Err(error);
            }
        } else {
            false
        };
        Ok(RetirementObservation::from_retired(
            leader_reaped && descendants_gone,
        ))
    }
    #[cfg(windows)]
    {
        let _ = process_id;
        if placement == ProcessGroupPlacement::JoinAncestorGroup {
            return Ok(RetirementObservation::from_retired(leader_reaped));
        }
        use std::mem::size_of;
        use std::os::windows::io::AsRawHandle as _;

        use windows_sys::Win32::System::JobObjects::{
            JobObjectBasicAccountingInformation, QueryInformationJobObject,
            JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
        };

        let job = containment.ok_or_else(|| {
            std::io::Error::other("owned process tree has no JobObject containment")
        })?;
        let handle = job.job.as_raw_handle() as _;
        let mut accounting: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { std::mem::zeroed() };
        let inspected = unsafe {
            QueryInformationJobObject(
                handle,
                JobObjectBasicAccountingInformation,
                &mut accounting as *mut _ as *mut _,
                size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                std::ptr::null_mut(),
            )
        };
        if inspected == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(RetirementObservation::from_retired(
            leader_reaped && accounting.ActiveProcesses == 0,
        ))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (leader_reaped, process_id, containment, placement);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "owned process-tree retirement is unsupported on this platform",
        ))
    }
}

#[cfg(unix)]
fn owned_group(process_id: Option<u32>) -> std::io::Result<i32> {
    let process_id = process_id
        .ok_or_else(|| std::io::Error::other("owned process tree has no process-group identity"))?;
    unix_process_id(process_id)
}

pub async fn terminate_id(process_id: u32) -> std::io::Result<()> {
    if process_id == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "process identity must be positive",
        ));
    }
    #[cfg(unix)]
    {
        let group = unix_process_id(process_id)?;
        let result = unsafe { libc::kill(-group, libc::SIGKILL) };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }

    #[cfg(windows)]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &process_id.to_string(), "/T", "/F"])
            .status()
            .await?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!(
                "taskkill /T /F for process {process_id} failed with {status}"
            )))
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = process_id;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "process-tree termination is unsupported on this platform",
        ))
    }
}

pub fn is_alive(process_id: u32) -> bool {
    if process_id == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        let Ok(process_id) = unix_process_id(process_id) else {
            return false;
        };
        let result = unsafe { libc::kill(process_id, 0) };
        if result == 0 {
            #[cfg(target_os = "linux")]
            if linux_process_is_zombie(process_id) {
                return false;
            }
            return true;
        }
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }

    #[cfg(windows)]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {process_id}"), "/FO", "CSV", "/NH"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .is_some_and(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .any(|line| line.contains(&format!("\"{process_id}\"")))
            })
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = process_id;
        false
    }
}

#[cfg(target_os = "linux")]
fn linux_process_is_zombie(process_id: i32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{process_id}/stat")) else {
        return false;
    };
    stat.rsplit_once(')')
        .and_then(|(_, fields)| fields.split_whitespace().next())
        == Some("Z")
}

#[cfg(unix)]
fn unix_process_id(process_id: u32) -> std::io::Result<i32> {
    i32::try_from(process_id)
        .ok()
        .filter(|id| *id > 0)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "process identity is outside the positive native PID range",
            )
        })
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
mod tests {

    #[tokio::test]
    async fn cancellation_racing_with_leader_exit_preserves_owned_retirement() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        for attempt in 0..96 {
            let mut command = crate::HostCommand::new("/bin/sh");
            command.arguments = vec!["-c".into(), "printf ready; read finish; exit 7".into()];
            let mut child = command.spawn().await.expect("owned child");
            let mut stdio = child.take_stdio().expect("owned pipes");
            let mut ready = [0_u8; 5];
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                stdio.stdout.read_exact(&mut ready),
            )
            .await
            .expect("child reaches its exit gate")
            .expect("child readiness");
            assert_eq!(&ready, b"ready");
            stdio.stdin.write_all(b"finish\n").await.unwrap();
            // Sample the interval between exit beginning and waitable status
            // publication, as well as cancellation before and after that exit.
            for _ in 0..attempt % 32 {
                std::hint::spin_loop();
            }
            tokio::time::timeout(std::time::Duration::from_secs(2), child.terminate_tree())
                .await
                .unwrap_or_else(|error| panic!("retirement attempt {attempt}: {error}"))
                .unwrap_or_else(|error| panic!("retirement attempt {attempt}: {error}"));
            let exit = child.wait().await.expect("retained terminal status");
            assert!(
                (exit.code == Some(7) && exit.signal.is_none())
                    || (exit.code.is_none() && exit.signal == Some(libc::SIGKILL)),
                "unexpected terminal outcome at {attempt}: {exit:?}"
            );
            child.terminate_tree().await.expect("idempotent retirement");
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn denied_group_signal_commits_to_observation_before_waitable_exit() {
        let start = super::group_signal_outcome(
            Err(std::io::Error::from_raw_os_error(libc::EPERM)),
            || panic!("an exiting leader need not be waitable yet"),
        )
        .unwrap();
        assert_eq!(start, super::TerminationStart::GroupSignalDenied);
        for code in [libc::EIO, libc::EACCES] {
            let error =
                super::group_signal_outcome(Err(std::io::Error::from_raw_os_error(code)), || {
                    panic!("unrelated errors do not inspect leader exit")
                })
                .unwrap_err();
            assert_eq!(error.raw_os_error(), Some(code));
        }
    }

    #[tokio::test]
    async fn terminal_leader_can_retire_before_lifecycle_observes_its_exit() {
        for (script, expected) in [
            ("exit 0", (Some(0), None)),
            ("exit 42", (Some(42), None)),
            ("kill -TERM $$", (None, Some(libc::SIGTERM))),
        ] {
            let mut command = crate::HostCommand::new("/bin/sh");
            command.arguments = vec!["-c".into(), script.into()];
            let mut child = command.spawn().await.expect("owned child");
            let id = child.id().expect("owned child identity");
            // Leave ChildProcess's lifecycle Live while deterministically
            // waiting for its terminal, still-unreaped OS child record.
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                super::super::exit_observation::wait_unreaped(Some(id)),
            )
            .await
            .expect("child exits")
            .expect("non-reaping observation");
            child
                .terminate_tree()
                .await
                .expect("owned group retirement");
            let exit = child.wait().await.expect("retained original exit");
            assert_eq!((exit.code, exit.signal), expected);
            assert_eq!(unsafe { libc::kill(-(id as i32), 0) }, -1);
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ESRCH)
            );
        }
    }
}
