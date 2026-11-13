use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::child::{self, ChildProcess, ResourceLimits};
use crate::environment::EnvironmentPolicy;
use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChildLifetime {
    Owned,
    Detached,
}

/// Unix process-group ownership for a child launched by RunMat. A joined
/// member is owned by an already-contained ancestor; its own shutdown may
/// reap that member, but only the ancestor can certify group retirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessGroupPlacement {
    OwnGroup,
    JoinAncestorGroup,
}

impl Default for ProcessGroupPlacement {
    fn default() -> Self {
        Self::OwnGroup
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StdioPolicy {
    Piped,
    /// Application output is piped separately from the authenticated control
    /// channels. stdin is closed; the caller owns both output drains.
    #[cfg(any(unix, windows))]
    PrivateControl(crate::ipc::private_control::PrivateControlEndpoint),
    /// Both output pipes belong to one complete-output call; neither is capped.
    CompleteOutput,
    /// Complete output with one owned input body written concurrently with both drains.
    CompleteOutputWithInput,
    Inherit,
    Null,
    /// Null stdin/stdout, bounded piped stderr. A detached child must release
    /// this startup stream before readiness, so it can outlive the starter.
    StartupDiagnostics,
    Files {
        stdout: PathBuf,
        stderr: PathBuf,
    },
}

#[derive(Clone, Debug)]
pub struct HostCommand {
    pub executable: PathBuf,
    pub arguments: Vec<String>,
    pub working_directory: Option<PathBuf>,
    pub environment: BTreeMap<String, String>,
    pub environment_policy: EnvironmentPolicy,
    pub lifetime: ChildLifetime,
    pub process_group_placement: ProcessGroupPlacement,
    pub stdio: StdioPolicy,
    pub max_stderr_bytes: usize,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug)]
pub struct CompleteProcessOutput {
    pub exit: crate::child::ProcessExit,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl HostCommand {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            arguments: Vec::new(),
            working_directory: None,
            environment: BTreeMap::new(),
            environment_policy: EnvironmentPolicy::Clear,
            lifetime: ChildLifetime::Owned,
            process_group_placement: ProcessGroupPlacement::OwnGroup,
            stdio: StdioPolicy::Piped,
            max_stderr_bytes: 1024 * 1024,
            resource_limits: ResourceLimits::default(),
        }
    }

    pub fn validate(&self) -> ProcessHostResult<()> {
        if self.executable.as_os_str().is_empty() {
            return Err(ProcessHostError::Configuration(
                "child executable must not be empty".into(),
            ));
        }
        if self
            .working_directory
            .as_ref()
            .is_some_and(|path| !path.is_absolute() || !path.is_dir())
        {
            return Err(ProcessHostError::Configuration(
                "child working directory must be an existing absolute directory".into(),
            ));
        }
        if self.max_stderr_bytes == 0 {
            return Err(ProcessHostError::Configuration(
                "child stderr bound must be greater than zero".into(),
            ));
        }
        self.resource_limits.validate()?;
        let joined_stdio = self.stdio == StdioPolicy::Piped;
        #[cfg(any(unix, windows))]
        let joined_stdio = joined_stdio || matches!(self.stdio, StdioPolicy::PrivateControl(_));
        if self.process_group_placement == ProcessGroupPlacement::JoinAncestorGroup
            && (self.lifetime != ChildLifetime::Owned || !joined_stdio)
        {
            return Err(ProcessHostError::Configuration(
                "ancestor-group helpers require owned protocol and output channels".into(),
            ));
        }
        #[cfg(windows)]
        if self.process_group_placement == ProcessGroupPlacement::JoinAncestorGroup
            && self.resource_limits != ResourceLimits::default()
        {
            return Err(ProcessHostError::Configuration(
                "joined JobObject helpers cannot override ancestor resource limits".into(),
            ));
        }
        if self.lifetime == ChildLifetime::Detached
            && matches!(
                self.stdio,
                StdioPolicy::Piped
                    | StdioPolicy::CompleteOutput
                    | StdioPolicy::CompleteOutputWithInput
                    | StdioPolicy::Inherit
            )
        {
            return Err(ProcessHostError::Configuration(
                "detached children require null, file-backed or startup diagnostic stdio".into(),
            ));
        }
        #[cfg(any(unix, windows))]
        if self.lifetime == ChildLifetime::Detached
            && matches!(self.stdio, StdioPolicy::PrivateControl(_))
        {
            return Err(ProcessHostError::Configuration(
                "private control channels require an owned child".into(),
            ));
        }
        Ok(())
    }

    pub async fn spawn(self) -> ProcessHostResult<ChildProcess> {
        self.validate()?;
        child::spawn(self).await
    }

    /// Track the exact OS-spawn phase for callers that must retain private
    /// bytes after a post-spawn containment or stdio failure.
    pub async fn spawn_observed(
        self,
        evidence: &crate::ProcessSpawnEvidence,
    ) -> ProcessHostResult<ChildProcess> {
        self.validate()?;
        child::spawn_observed(self, evidence).await
    }

    /// Collect both pipes to EOF without truncation, retaining the owned child
    /// and its process-tree teardown authority until the final wait completes.
    pub async fn output_complete(self) -> ProcessHostResult<CompleteProcessOutput> {
        self.complete_output(None).await
    }

    /// The input body remains owned by this call. Dropping the future closes
    /// stdin and terminates the owned child while both output pipes drain.
    pub async fn output_complete_with_input(
        self,
        input: Vec<u8>,
    ) -> ProcessHostResult<CompleteProcessOutput> {
        self.complete_output(Some(input)).await
    }

    async fn complete_output(
        mut self,
        input: Option<Vec<u8>>,
    ) -> ProcessHostResult<CompleteProcessOutput> {
        use tokio::io::AsyncReadExt as _;
        use tokio::io::AsyncWriteExt as _;

        if self.lifetime != ChildLifetime::Owned {
            return Err(ProcessHostError::Configuration(
                "complete output requires an owned child".into(),
            ));
        }
        self.stdio = if input.is_some() {
            StdioPolicy::CompleteOutputWithInput
        } else {
            StdioPolicy::CompleteOutput
        };
        let mut process = self.spawn().await?;
        let guard = CompleteOutputDropGuard::new(process.id());
        let (stdin_writer, mut stdout_reader, mut stderr_reader) =
            process.take_complete_output()?;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        // Keep the direct child unreaped while draining: its PID remains
        // reserved, so cancellation cannot signal a recycled process group.
        let write_input = async {
            match (stdin_writer, input) {
                (Some(mut writer), Some(bytes)) => {
                    writer.write_all(&bytes).await?;
                    writer.shutdown().await?;
                }
                (None, Some(_)) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "owned child stdin is unavailable",
                    ));
                }
                _ => {}
            }
            Ok::<(), std::io::Error>(())
        };
        tokio::try_join!(
            write_input,
            stdout_reader.read_to_end(&mut stdout),
            stderr_reader.read_to_end(&mut stderr),
        )?;
        let exit = process.wait().await?;
        guard.disarm();
        Ok(CompleteProcessOutput {
            exit,
            stdout,
            stderr,
        })
    }
}

struct CompleteOutputDropGuard {
    process_id: Option<u32>,
}

impl CompleteOutputDropGuard {
    fn new(process_id: Option<u32>) -> Self {
        Self { process_id }
    }

    fn disarm(mut self) {
        self.process_id = None;
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn complete_output_drains_both_pipes_beyond_pipe_capacity() {
        let mut command = HostCommand::new("/bin/sh");
        command.arguments = vec![
            "-c".into(),
            "i=0; while [ \"$i\" -lt 8192 ]; do printf '0123456789abcdef'; printf 'fedcba9876543210' >&2; i=$((i+1)); done".into(),
        ];
        let output = tokio::time::timeout(Duration::from_secs(20), command.output_complete())
            .await
            .expect("both pipes must be drained concurrently")
            .expect("owned child must complete");
        assert!(output.exit.success);
        assert_eq!(output.exit.code, Some(0));
        assert_eq!(output.exit.signal, None);
        assert_eq!(output.stdout.len(), 8192 * 16);
        assert_eq!(output.stderr.len(), 8192 * 16);
        assert!(output
            .stdout
            .chunks_exact(16)
            .all(|chunk| chunk == b"0123456789abcdef"));
        assert!(output
            .stderr
            .chunks_exact(16)
            .all(|chunk| chunk == b"fedcba9876543210"));
    }

    #[tokio::test]
    async fn owned_input_and_both_output_pipes_progress_concurrently() {
        let mut command = HostCommand::new("/bin/sh");
        command.arguments = vec![
            "-c".into(),
            "i=0; while [ \"$i\" -lt 8192 ]; do printf 'out'; printf 'err' >&2; i=$((i+1)); done; cat".into(),
        ];
        let input = vec![0x5au8; 256 * 1024];
        let output = tokio::time::timeout(
            Duration::from_secs(20),
            command.output_complete_with_input(input.clone()),
        )
        .await
        .expect("stdin and both outputs must make progress")
        .expect("owned child must complete");
        assert!(output.exit.success);
        assert_eq!(output.stderr.len(), 8192 * 3);
        let expected_prefix = b"out".repeat(8192);
        assert_eq!(&output.stdout[..8192 * 3], expected_prefix.as_slice());
        assert_eq!(&output.stdout[8192 * 3..], input);
    }

    #[tokio::test]
    async fn early_stdin_close_retains_typed_io_cause() {
        let mut command = HostCommand::new("/bin/sh");
        command.arguments = vec!["-c".into(), "exec 0<&-; sleep 1".into()];
        let error = command
            .output_complete_with_input(vec![0x5a; 256 * 1024])
            .await
            .expect_err("closed child stdin must fail owned write");
        assert!(matches!(error, ProcessHostError::Io(_)));
    }

    #[tokio::test]
    async fn unpolled_input_call_starts_no_child() {
        let directory = tempfile::tempdir().expect("test directory");
        let marker = directory.path().join("unpolled-marker");
        let mut command = HostCommand::new("/bin/sh");
        command.arguments = vec![
            "-c".into(),
            "printf started > \"$1\"".into(),
            "sh".into(),
            marker.to_string_lossy().into_owned(),
        ];
        drop(command.output_complete_with_input(vec![1, 2, 3]));
        assert!(!marker.exists());
    }

    #[tokio::test]
    async fn aborted_input_call_terminates_owned_process_group() {
        let directory = tempfile::tempdir().expect("test directory");
        let marker = directory.path().join("input-process-id");
        let mut command = HostCommand::new("/bin/sh");
        command.arguments = vec![
            "-c".into(),
            "printf '%s' \"$$\" > \"$1\"; exec /bin/sleep 30".into(),
            "sh".into(),
            marker.to_string_lossy().into_owned(),
        ];
        let task = tokio::spawn(command.output_complete_with_input(vec![0x5a; 256 * 1024]));
        let pid = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(text) = std::fs::read_to_string(&marker) {
                    if let Ok(pid) = text.parse::<u32>() {
                        break pid;
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("child must start");
        task.abort();
        let _ = task.await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while crate::is_process_alive(pid) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("cancelled input/output child must terminate");
    }

    #[tokio::test]
    async fn complete_output_preserves_exit_code_and_signal_distinction() {
        let mut failed = HostCommand::new("/bin/sh");
        failed.arguments = vec!["-c".into(), "exit 23".into()];
        let failed = failed.output_complete().await.expect("exited child");
        assert_eq!(failed.exit.code, Some(23));
        assert_eq!(failed.exit.signal, None);
        assert!(!failed.exit.success);

        let mut signaled = HostCommand::new("/bin/sh");
        signaled.arguments = vec!["-c".into(), "kill -TERM $$".into()];
        let signaled = signaled.output_complete().await.expect("signaled child");
        assert_eq!(signaled.exit.code, None);
        assert_eq!(signaled.exit.signal, Some(libc::SIGTERM));
        assert!(!signaled.exit.success);
    }

    #[tokio::test]
    async fn aborted_complete_output_terminates_owned_process_group() {
        let directory = tempfile::tempdir().expect("test directory");
        let marker = directory.path().join("owned-process-id");
        let mut command = HostCommand::new("/bin/sh");
        command.arguments = vec![
            "-c".into(),
            "printf '%s' \"$$\" > \"$1\"; exec /bin/sleep 30".into(),
            "sh".into(),
            marker.to_string_lossy().into_owned(),
        ];
        let task = tokio::spawn(command.output_complete());
        let pid = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(text) = std::fs::read_to_string(&marker) {
                    if let Ok(pid) = text.parse::<u32>() {
                        break pid;
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("child must start");
        task.abort();
        let _ = task.await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while crate::is_process_alive(pid) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("cancelled complete-output child must terminate");
    }
}

impl Drop for CompleteOutputDropGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(process_id) = self.process_id {
            if let Ok(group) = i32::try_from(process_id) {
                // spawn() places this child in its own group. A dropped future
                // cannot await termination, so kill the owned group now; the
                // ChildProcess drop still kills the direct child on all hosts.
                unsafe { libc::kill(-group, libc::SIGKILL) };
            }
        }
    }
}
