use tokio::process::{Child, Command};

use super::ResourceLimits;

#[cfg(not(windows))]
pub(super) struct ProcessContainment;

#[cfg(windows)]
pub(super) struct ProcessContainment {
    job: std::os::windows::io::OwnedHandle,
}

#[cfg(unix)]
pub(super) fn configure(command: &mut Command) {
    command.process_group(0);
}

#[cfg(windows)]
pub(super) fn configure(command: &mut Command) {
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    command.creation_flags(CREATE_NEW_PROCESS_GROUP);
}

#[cfg(not(any(unix, windows)))]
pub(super) fn configure(_command: &mut Command) {}

#[cfg(not(windows))]
pub(super) fn contain(_: &Child, _: ResourceLimits) -> std::io::Result<Option<ProcessContainment>> {
    Ok(None)
}

#[cfg(windows)]
pub(super) fn contain(
    child: &Child,
    limits: ResourceLimits,
) -> std::io::Result<Option<ProcessContainment>> {
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

pub(super) async fn terminate(
    child: &mut Child,
    process_id: Option<u32>,
    containment: Option<&ProcessContainment>,
) -> std::io::Result<()> {
    #[cfg(any(unix, windows))]
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    #[cfg(not(windows))]
    let _ = containment;
    #[cfg(not(any(unix, windows)))]
    let _ = process_id;

    #[cfg(unix)]
    {
        let process_id = process_id.ok_or_else(|| {
            std::io::Error::other("owned process tree has no process-group identity")
        })?;
        let group = i32::try_from(process_id)
            .ok()
            .filter(|group| *group > 0)
            .ok_or_else(|| std::io::Error::other("invalid owned process-group identity"))?;
        let result = unsafe { libc::kill(-group, libc::SIGKILL) };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        let _ = tokio::time::timeout_at(deadline, child.wait())
            .await
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "owned process-group leader did not retire",
                )
            })??;
        // Reaping the group leader alone does not prove that its descendants
        // have released private artifacts. ESRCH is the kernel's proof that
        // no process remains in this owned process group. Bound the wait so
        // unreaped descendants fail closed instead of hanging retirement.
        loop {
            if unsafe { libc::kill(-group, 0) } != 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() == Some(libc::ESRCH) {
                    return Ok(());
                }
                return Err(error);
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(std::io::Error::other(
                    "owned process group still has processes after termination",
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    #[cfg(windows)]
    {
        let _ = process_id;
        let job = containment.ok_or_else(|| {
            std::io::Error::other("owned process tree has no JobObject containment")
        })?;
        job.terminate(deadline).await?;
        let _ = tokio::time::timeout_at(deadline, child.wait())
            .await
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "owned JobObject leader did not retire",
                )
            })??;
        return Ok(());
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = child;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "owned process-tree retirement is unsupported on this platform",
        ))
    }
}

#[cfg(windows)]
impl ProcessContainment {
    async fn terminate(&self, deadline: tokio::time::Instant) -> std::io::Result<()> {
        use std::mem::size_of;
        use std::os::windows::io::AsRawHandle as _;

        use windows_sys::Win32::System::JobObjects::{
            JobObjectBasicAccountingInformation, QueryInformationJobObject, TerminateJobObject,
            JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
        };

        let handle = self.job.as_raw_handle() as _;
        if unsafe { TerminateJobObject(handle, 1) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        // The root may have exited before this call. Query the owned job,
        // rather than treating a missing root PID as proof that its children
        // have stopped using selected private artifact paths.
        loop {
            let mut accounting: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION =
                unsafe { std::mem::zeroed() };
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
            if accounting.ActiveProcesses == 0 {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(std::io::Error::other(
                    "owned JobObject still has active processes after termination",
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

pub async fn terminate_id(process_id: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let group = i32::try_from(process_id).unwrap_or(i32::MAX);
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
    #[cfg(unix)]
    {
        let process_id = i32::try_from(process_id).unwrap_or(i32::MAX);
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
