use std::process::Stdio;

use tokio::process::Command;

use super::retirement::ProcessSpawnEvidence;
use super::{CapturedProcessOutput, ChildProcess, ChildStdio};
use crate::command::{ChildLifetime, HostCommand, StdioPolicy};
use crate::environment::apply_environment;
use crate::{ProcessHostError, ProcessHostResult};

pub async fn spawn(spec: HostCommand) -> ProcessHostResult<ChildProcess> {
    spawn_internal(spec, None).await
}

pub async fn spawn_observed(
    spec: HostCommand,
    evidence: &ProcessSpawnEvidence,
) -> ProcessHostResult<ChildProcess> {
    spawn_internal(spec, Some(evidence)).await
}

async fn spawn_internal(
    spec: HostCommand,
    evidence: Option<&ProcessSpawnEvidence>,
) -> ProcessHostResult<ChildProcess> {
    super::process_tree::validate_join_parent(spec.process_group_placement)?;
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
        StdioPolicy::CompleteOutput => {
            command
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            false
        }
        #[cfg(any(unix, windows))]
        StdioPolicy::PrivateControl(_) => {
            command
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            false
        }
        StdioPolicy::CompleteOutputWithInput => {
            command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            false
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
        StdioPolicy::StartupDiagnostics => {
            command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
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
    #[cfg(any(unix, windows))]
    if let StdioPolicy::PrivateControl(endpoint) = &spec.stdio {
        command.env(
            crate::ipc::private_control::BOOTSTRAP_ENV,
            endpoint.encode()?,
        );
    }
    super::process_tree::configure(&mut command, spec.process_group_placement);
    super::limits::configure(&mut command, spec.resource_limits);
    let child = command.spawn()?;
    if let Some(evidence) = evidence {
        evidence.mark_child_created();
    }
    let process_id = child.id();
    // Owned children stay inside the host's process-tree containment so they
    // are terminated with their owner. Detached children deliberately outlive
    // this `ChildProcess`; assigning them to kill-on-close containment would
    // either terminate them when this value is dropped or fail on Windows
    // hosts whose runner already owns the process through a Job Object.
    let containment = if spec.lifetime == ChildLifetime::Owned {
        super::process_tree::contain(&child, spec.resource_limits, spec.process_group_placement)?
    } else {
        None
    };
    let stderr = CapturedProcessOutput::new(spec.max_stderr_bytes);
    let mut process = ChildProcess::new(
        child,
        process_id,
        stderr,
        containment,
        spec.process_group_placement,
    );
    if piped {
        let stdin = process.child_stdin().ok_or_else(|| {
            ProcessHostError::Protocol("child stdin was unavailable after piped spawn".into())
        })?;
        let stdout = process.child_stdout().ok_or_else(|| {
            ProcessHostError::Protocol("child stdout was unavailable after piped spawn".into())
        })?;
        process.install_stdio(ChildStdio { stdin, stdout });
    }
    if piped || spec.stdio == StdioPolicy::StartupDiagnostics {
        let stderr_reader = process.child_stderr().ok_or_else(|| {
            ProcessHostError::Protocol("child stderr was unavailable after piped spawn".into())
        })?;
        process.captured_stderr().drain(stderr_reader)?;
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
