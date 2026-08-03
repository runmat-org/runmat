use tokio::process::{Child, Command};

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

pub(super) async fn terminate(child: &mut Child, process_id: Option<u32>) -> std::io::Result<()> {
    #[cfg(unix)]
    if let Some(process_id) = process_id {
        let group = i32::try_from(process_id).unwrap_or(i32::MAX);
        let result = unsafe { libc::kill(-group, libc::SIGKILL) };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        let _ = child.wait().await?;
        return Ok(());
    }

    #[cfg(windows)]
    if let Some(process_id) = process_id {
        let process_id = process_id.to_string();
        let status = Command::new("taskkill")
            .args(["/PID", &process_id, "/T", "/F"])
            .status()
            .await?;
        if status.success() {
            let _ = child.wait().await?;
            return Ok(());
        }
    }

    match child.kill().await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => Ok(()),
        Err(error) => Err(error),
    }
}
