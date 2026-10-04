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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StdioPolicy {
    Piped,
    /// Both output pipes belong to one complete-output call; neither is capped.
    CompleteOutput,
    Inherit,
    Null,
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
        if self.lifetime == ChildLifetime::Detached
            && matches!(
                self.stdio,
                StdioPolicy::Piped | StdioPolicy::CompleteOutput | StdioPolicy::Inherit
            )
        {
            return Err(ProcessHostError::Configuration(
                "detached children require null or file-backed stdio".into(),
            ));
        }
        Ok(())
    }

    pub async fn spawn(self) -> ProcessHostResult<ChildProcess> {
        self.validate()?;
        child::spawn(self).await
    }

    /// Collect both pipes to EOF without truncation, retaining the owned child
    /// and its process-tree teardown authority until the final wait completes.
    pub async fn output_complete(mut self) -> ProcessHostResult<CompleteProcessOutput> {
        use tokio::io::AsyncReadExt as _;

        if self.lifetime != ChildLifetime::Owned {
            return Err(ProcessHostError::Configuration(
                "complete output requires an owned child".into(),
            ));
        }
        self.stdio = StdioPolicy::CompleteOutput;
        let mut process = self.spawn().await?;
        let guard = CompleteOutputDropGuard::new(process.id());
        let (mut stdout_reader, mut stderr_reader) = process.take_complete_output()?;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        // Keep the direct child unreaped while draining: its PID remains
        // reserved, so cancellation cannot signal a recycled process group.
        tokio::try_join!(
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
