//! One invocation owner for complete native process calls. Builtins supply
//! commands and interpret exits; this module owns service and path admission.

use std::io;

use futures::future::{select, Either};
use runmat_filesystem::{NativePathIntent, NativeProcessRevocation};
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
    // Keep the prior system admission order: the selected filesystem route
    // is resolved before an absent process port is reported.
    let runtime = runtime.ok_or_else(missing_service)?;
    let process = runtime
        .service_ports()
        .require_process("execute complete native process call")
        .map_err(ProcessInvocationError::Capability)?;

    let execute = async {
        let request = match input {
            Some(input) => process.output_complete_with_input(command, input),
            None => process.output_complete(command),
        };
        match revocation {
            NativeProcessRevocation::Stable => request.await.map_err(ProcessInvocationError::Host),
            NativeProcessRevocation::Watch(watch) => match select(watch, request).await {
                Either::Right((result, _watch)) => result.map_err(ProcessInvocationError::Host),
                Either::Left(((), _request)) => {
                    Err(ProcessInvocationError::StaleDirectory(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "process directory authority revoked",
                    )))
                }
            },
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
    Ok(output)
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
