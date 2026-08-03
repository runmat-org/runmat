use std::process::Stdio;

use tokio::process::Command;

use super::{CapturedStderr, ChildProcess, ChildStdio};
use crate::command::{HostCommand, StdioPolicy};
use crate::environment::apply_environment;
use crate::{ProcessHostError, ProcessHostResult};

pub async fn spawn(spec: HostCommand) -> ProcessHostResult<ChildProcess> {
    let mut command = Command::new(&spec.executable);
    command.args(&spec.arguments).kill_on_drop(true);
    match spec.stdio {
        StdioPolicy::Piped => {
            command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
        }
    }
    apply_environment(&mut command, &spec.environment_policy, &spec.environment);
    super::process_tree::configure(&mut command);
    let mut child = command.spawn()?;
    let process_id = child.id();
    let stdin = child.stdin.take().ok_or_else(|| {
        ProcessHostError::Protocol("child stdin was unavailable after piped spawn".into())
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        ProcessHostError::Protocol("child stdout was unavailable after piped spawn".into())
    })?;
    let stderr_reader = child.stderr.take().ok_or_else(|| {
        ProcessHostError::Protocol("child stderr was unavailable after piped spawn".into())
    })?;
    let stderr = CapturedStderr::new(spec.max_stderr_bytes);
    stderr.drain(stderr_reader);
    let mut process = ChildProcess::new(child, process_id, stderr);
    process.install_stdio(ChildStdio { stdin, stdout });
    Ok(process)
}
