use tokio::process::{Child, Command};

use super::ResourceLimits;
use crate::command::ProcessGroupPlacement;

#[cfg(not(windows))]
pub(super) struct ProcessContainment;

#[cfg(windows)]
pub(super) struct ProcessContainment {
    job: std::os::windows::io::OwnedHandle,
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
) -> std::io::Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if tree_retired(child, process_id, containment, placement)? {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(retirement_timeout());
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

pub(super) fn await_retirement_blocking(
    child: &mut Child,
    process_id: Option<u32>,
    containment: Option<&ProcessContainment>,
    placement: ProcessGroupPlacement,
) -> std::io::Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if tree_retired(child, process_id, containment, placement)? {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(retirement_timeout());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn retirement_timeout() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "owned process tree did not retire within the deadline",
    )
}

pub(super) fn begin_termination(
    child: &mut Child,
    process_id: Option<u32>,
    containment: Option<&ProcessContainment>,
    placement: ProcessGroupPlacement,
    unreaped_exit_observed: bool,
) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let _ = containment;
        if placement == ProcessGroupPlacement::JoinAncestorGroup {
            if child.try_wait()?.is_none() {
                child.start_kill()?;
            }
            return Ok(());
        }
        let group = owned_group(process_id)?;
        let result = unsafe { libc::kill(-group, libc::SIGKILL) };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            // Darwin can refuse signaling a group containing only an exited,
            // unreaped leader. Its PID is still pinned by our exit observation.
            // Proceed only to reaping/absence checks, never another signal;
            // surviving members still prevent a retirement certificate.
            if error.raw_os_error() != Some(libc::ESRCH)
                && !(unreaped_exit_observed && error.raw_os_error() == Some(libc::EPERM))
            {
                return Err(error);
            }
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        let _ = unreaped_exit_observed;
        let _ = process_id;
        if placement == ProcessGroupPlacement::JoinAncestorGroup {
            if child.try_wait()?.is_none() {
                child.start_kill()?;
            }
            return Ok(());
        }
        use std::os::windows::io::AsRawHandle as _;
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        let job = containment.ok_or_else(|| {
            std::io::Error::other("owned process tree has no JobObject containment")
        })?;
        if unsafe { TerminateJobObject(job.job.as_raw_handle() as _, 1) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
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

fn tree_retired(
    child: &mut Child,
    process_id: Option<u32>,
    containment: Option<&ProcessContainment>,
    placement: ProcessGroupPlacement,
) -> std::io::Result<bool> {
    let leader_reaped = child.try_wait()?.is_some();
    #[cfg(unix)]
    {
        let _ = containment;
        if placement == ProcessGroupPlacement::JoinAncestorGroup {
            return Ok(leader_reaped);
        }
        let group = owned_group(process_id)?;
        let descendants_gone = if unsafe { libc::kill(-group, 0) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                true
            } else {
                return Err(error);
            }
        } else {
            false
        };
        Ok(leader_reaped && descendants_gone)
    }
    #[cfg(windows)]
    {
        let _ = process_id;
        if placement == ProcessGroupPlacement::JoinAncestorGroup {
            return Ok(leader_reaped);
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
        Ok(leader_reaped && accounting.ActiveProcesses == 0)
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
