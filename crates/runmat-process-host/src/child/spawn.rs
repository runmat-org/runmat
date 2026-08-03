use std::process::Stdio;

use tokio::process::Command;

use super::{CapturedStderr, ChildProcess, ChildStdio};
use crate::command::{ChildLifetime, HostCommand, StdioPolicy};
use crate::environment::apply_environment;
use crate::{ProcessHostError, ProcessHostResult};

pub async fn spawn(spec: HostCommand) -> ProcessHostResult<ChildProcess> {
    let mut command = Command::new(&spec.executable);
    command
        .args(&spec.arguments)
        .kill_on_drop(spec.lifetime == ChildLifetime::Owned);
    if let Some(working_directory) = &spec.working_directory {
        command.current_dir(working_directory);
    }
    let piped = match &spec.stdio {
        StdioPolicy::Piped => {
            command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            true
        }
        StdioPolicy::Inherit => {
            command
                .stdin(Stdio::inherit())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());
            false
        }
        StdioPolicy::Null => {
            command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            false
        }
        StdioPolicy::Files { stdout, stderr } => {
            let stdout = append_file(stdout)?;
            let stderr = append_file(stderr)?;
            command
                .stdin(Stdio::null())
                .stdout(Stdio::from(stdout))
                .stderr(Stdio::from(stderr));
            false
        }
    };
    apply_environment(&mut command, &spec.environment_policy, &spec.environment);
    super::process_tree::configure(&mut command);
    super::limits::configure(&mut command, spec.resource_limits);
    let child = command.spawn()?;
    let process_id = child.id();
    let containment = super::process_tree::contain(&child, spec.resource_limits)?;
    let stderr = CapturedStderr::new(spec.max_stderr_bytes);
    let mut process = ChildProcess::new(child, process_id, stderr, containment);
    if piped {
        let stdin = process.child_stdin().ok_or_else(|| {
            ProcessHostError::Protocol("child stdin was unavailable after piped spawn".into())
        })?;
        let stdout = process.child_stdout().ok_or_else(|| {
            ProcessHostError::Protocol("child stdout was unavailable after piped spawn".into())
        })?;
        let stderr_reader = process.child_stderr().ok_or_else(|| {
            ProcessHostError::Protocol("child stderr was unavailable after piped spawn".into())
        })?;
        process.captured_stderr().drain(stderr_reader);
        process.install_stdio(ChildStdio { stdin, stdout });
    }
    Ok(process)
}

fn append_file(path: &std::path::Path) -> ProcessHostResult<std::fs::File> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(file)
}
