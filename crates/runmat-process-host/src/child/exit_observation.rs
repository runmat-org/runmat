//! Observe a terminal Unix child without releasing its PID before group
//! retirement. Only the owning ChildProcess may wait/reap this child.

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
pub(super) async fn wait_unreaped(process_id: Option<u32>) -> std::io::Result<()> {
    loop {
        if try_wait_unreaped(process_id)?.is_some() {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
pub(super) fn try_wait_unreaped(
    process_id: Option<u32>,
) -> std::io::Result<Option<super::ProcessExit>> {
    let process_id = process_id
        .filter(|id| *id != 0)
        .ok_or_else(|| std::io::Error::other("owned child has no process identity"))?;
    loop {
        // SAFETY: waitid initializes this plain C record. P_PID selects only
        // our still-owned direct child; WNOWAIT preserves its waitable state.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                process_id as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
            )
        };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            // In particular, ECHILD is not proof of retirement or permission
            // to signal a potentially recycled process-group number.
            return Err(error);
        }
        // SAFETY: a successful waitid filled the child-status union, or left
        // its zero PID when no terminal status is available with WNOHANG.
        if unsafe { info.si_pid() } == 0 {
            return Ok(None);
        }
        // SAFETY: WEXITED reports only a terminal child-status record.
        let status = unsafe { info.si_status() };
        return match info.si_code {
            libc::CLD_EXITED => Ok(Some(super::ProcessExit {
                code: Some(status),
                signal: None,
                success: status == 0,
            })),
            libc::CLD_KILLED | libc::CLD_DUMPED => Ok(Some(super::ProcessExit {
                code: None,
                signal: Some(status),
                success: false,
            })),
            _ => Err(std::io::Error::other(
                "child exit observation returned a nonterminal state",
            )),
        };
    }
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
pub(super) async fn wait_unreaped(_: Option<u32>) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "non-reaping child observation is unsupported on this platform",
    ))
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
pub(super) fn try_wait_unreaped(_: Option<u32>) -> std::io::Result<Option<super::ProcessExit>> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "non-reaping child observation is unsupported on this platform",
    ))
}
