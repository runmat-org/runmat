//! One invocation owner for complete native process calls. Builtins supply
//! commands and interpret exits; this module owns service and path admission.

use std::io;

use futures::future::{select, Either};
use runmat_filesystem::{CapturedNativePath, NativePathIntent, NativeProcessRevocation};
use runmat_process_host::{CompleteProcessOutput, HostCommand, ProcessHostError};

use super::{
    complete_process_output_requires_evidence, RuntimeCapability, RuntimeCapabilityError,
    RuntimeContext,
};
use crate::execution::ExecutionServiceError;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ProcessInvocationError {
    #[error("complete child output has no certified finite resource bound")]
    ResourceEvidence,
    #[error("cannot capture process directory: {0}")]
    DirectoryCapture(#[source] io::Error),
    #[error("selected filesystem cannot issue a native process directory")]
    NoNativeDirectory,
    #[error("selected process directory has no revocation observation")]
    NoRevocation,
    #[error("process directory authority revoked: {0}")]
    StaleDirectory(#[source] io::Error),
    #[error("local executable belongs to another native path owner")]
    ForeignExecutableOwner,
    #[error("local executable grant does not match the selected command")]
    WrongExecutableGrant,
    #[error("selected local executable has no revocation observation")]
    NoExecutableRevocation,
    #[error("local executable authority revoked: {0}")]
    StaleExecutable(#[source] io::Error),
    #[error("{0}")]
    Capability(#[source] RuntimeCapabilityError),
    #[error("{0}")]
    Host(#[source] ProcessHostError),
    #[error("{0}")]
    Execution(#[source] ExecutionServiceError),
}

pub(crate) async fn output_complete(
    runtime: Option<&RuntimeContext>,
    mut command: HostCommand,
    input: Option<Vec<u8>>,
    executable_authority: Option<&CapturedNativePath>,
) -> Result<CompleteProcessOutput, ProcessInvocationError> {
    // Both output pipes are unbounded until the host supplies an enforceable
    // physical producer. A finite quota cannot admit this route by estimate.
    if complete_process_output_requires_evidence(runtime) {
        return Err(ProcessInvocationError::ResourceEvidence);
    }
    let cwd =
        runmat_filesystem::capture_native_path(".", NativePathIntent::ProcessWorkingDirectory)
            .map_err(ProcessInvocationError::DirectoryCapture)?
            .ok_or(ProcessInvocationError::NoNativeDirectory)?;
    cwd.check_current()
        .map_err(ProcessInvocationError::StaleDirectory)?;
    let revocation = cwd.process_revocation();
    if matches!(revocation, NativeProcessRevocation::Unavailable) {
        return Err(ProcessInvocationError::NoRevocation);
    }
    cwd.check_current()
        .map_err(ProcessInvocationError::StaleDirectory)?;
    command.working_directory = Some(cwd.path().to_path_buf());
    if let Some(executable) = executable_authority {
        if !std::sync::Arc::ptr_eq(cwd.provider(), executable.provider()) {
            return Err(ProcessInvocationError::ForeignExecutableOwner);
        }
        if executable.path() != command.executable.as_path() {
            return Err(ProcessInvocationError::WrongExecutableGrant);
        }
        executable
            .check_current()
            .map_err(ProcessInvocationError::StaleExecutable)?;
    }
    let executable_revocation = executable_authority.map(CapturedNativePath::process_revocation);
    if matches!(
        executable_revocation,
        Some(NativeProcessRevocation::Unavailable)
    ) {
        return Err(ProcessInvocationError::NoExecutableRevocation);
    }
    // Keep the prior system admission order: the selected filesystem route
    // is resolved before an absent process port is reported.
    let runtime = runtime.ok_or_else(missing_service)?;
    let process = runtime
        .service_ports()
        .require_process("execute complete native process call")
        .map_err(ProcessInvocationError::Capability)?;
    select_environment(runtime, &mut command);

    let execute = async {
        let request = match input {
            Some(input) => process.output_complete_with_input(command, input),
            None => process.output_complete(command),
        };
        let observed_executable = async {
            match executable_revocation {
                None | Some(NativeProcessRevocation::Stable) => {
                    request.await.map_err(ProcessInvocationError::Host)
                }
                Some(NativeProcessRevocation::Watch(watch)) => match select(watch, request).await {
                    Either::Right((result, _watch)) => result.map_err(ProcessInvocationError::Host),
                    Either::Left(((), _request)) => {
                        Err(ProcessInvocationError::StaleExecutable(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "local executable authority revoked",
                        )))
                    }
                },
                Some(NativeProcessRevocation::Unavailable) => {
                    unreachable!("rejected before effect")
                }
            }
        };
        match revocation {
            NativeProcessRevocation::Stable => observed_executable.await,
            NativeProcessRevocation::Watch(watch) => {
                match select(watch, Box::pin(observed_executable)).await {
                    Either::Right((result, _watch)) => result,
                    Either::Left(((), _request)) => {
                        Err(ProcessInvocationError::StaleDirectory(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "process directory authority revoked",
                        )))
                    }
                }
            }
            NativeProcessRevocation::Unavailable => unreachable!("rejected before effect"),
        }
    };
    let output = runtime
        .cpu_cooperation()
        .await_cancellable(execute)
        .await
        .map_err(ProcessInvocationError::Execution)??;
    cwd.check_current()
        .map_err(ProcessInvocationError::StaleDirectory)?;
    if let Some(executable) = executable_authority {
        executable
            .check_current()
            .map_err(ProcessInvocationError::StaleExecutable)?;
    }
    Ok(output)
}

/// Inheritance at a language process boundary refers to the selected session,
/// never the embedding process. Explicit command values retain precedence.
fn select_environment(runtime: &RuntimeContext, command: &mut HostCommand) {
    use runmat_process_host::environment::EnvironmentPolicy;

    if command.environment_policy == EnvironmentPolicy::Clear {
        return;
    }
    let mut selected = runtime
        .environment_snapshot()
        .into_iter()
        .filter(|(name, _)| match &command.environment_policy {
            EnvironmentPolicy::Inherit => true,
            EnvironmentPolicy::InheritExcept(excluded) => !excluded.iter().any(|excluded| {
                #[cfg(windows)]
                {
                    excluded.eq_ignore_ascii_case(name)
                }
                #[cfg(not(windows))]
                {
                    excluded == name
                }
            }),
            EnvironmentPolicy::Clear => false,
            EnvironmentPolicy::Allow(allowed) => allowed.contains(name),
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    selected.append(&mut command.environment);
    command.environment = selected;
    command.environment_policy = EnvironmentPolicy::Clear;
}

fn missing_service() -> ProcessInvocationError {
    ProcessInvocationError::Capability(RuntimeCapabilityError::new(
        RuntimeCapability::Process,
        "execute complete native process call",
    ))
}

#[cfg(test)]
#[path = "process_invocation/tests.rs"]
mod tests;
