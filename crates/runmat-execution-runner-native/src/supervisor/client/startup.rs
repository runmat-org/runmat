//! Original bounded startup diagnostics, before the existing authority serves readiness.
use super::*;
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::{
    ChildLifetime, HiddenMode, HostCommand, ProcessOutputCompletion, StdioPolicy,
};

impl LocalSupervisorClient {
    pub(super) async fn start_original_supervisor(&self) -> NativeExecutionResult<()> {
        let profile = crate::supervisor::bootstrap::publish_profile(&self.config)?;
        let mut command = HostCommand::new(&self.config.execution.executable);
        command.arguments = vec![HiddenMode::LocalSupervisor.marker().into()];
        command.environment_policy = EnvironmentPolicy::Inherit;
        command.environment.insert(
            crate::supervisor::bootstrap::PROFILE_ENV.into(),
            profile
                .to_str()
                .expect("validated bootstrap environment path")
                .into(),
        );
        command.lifetime = ChildLifetime::Detached;
        command.stdio = StdioPolicy::StartupDiagnostics;
        command.max_stderr_bytes = self.config.execution.max_stderr_bytes;
        let mut child = command.spawn().await?;
        let stderr = child.captured_stderr();
        let mut exit = None;
        // The original five-second readiness budget also bounds stderr drainage.
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match self.ping().await {
                    Ok(()) => {
                        stderr.wait_closed().await;
                        if stderr.completion() != ProcessOutputCompletion::EndOfStream {
                            return Err(NativeExecutionError::Protocol(
                                "original supervisor startup diagnostic stream did not close cleanly".into()));
                        }
                        return Ok(());
                    }
                    Err(error) if endpoint_absent(&error) => {}
                    Err(error) => return Err(error),
                }
                if let Some(observed) = child.try_wait()? {
                    exit = Some(observed);
                    stderr.wait_closed().await;
                    return Err(NativeExecutionError::Protocol(
                        "original supervisor exited before authenticated readiness".into()));
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }).await.unwrap_or_else(|_| Err(readiness_timeout()));
        if result.is_err() && exit.is_none() {
            if let Ok(observed) = child.try_wait() {
                exit = observed;
            }
        }
        // Detached lifetime is unchanged. A timeout does not prove child/descendant
        // quiescence, and cannot authorize replacing an existing endpoint.
        result.map_err(|cause| NativeExecutionError::SupervisorStartup {
            cause: Box::new(cause),
            exit,
            stderr: Box::new(stderr.snapshot()),
        })
    }
}
