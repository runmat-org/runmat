//! The `system` process route. The shared invocation owner holds native child I/O.

use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};

#[cfg(not(target_arch = "wasm32"))]
use crate::context::process_invocation::{self, ProcessInvocationError};

#[cfg(not(target_arch = "wasm32"))]
fn process_error(source: ProcessInvocationError) -> crate::RuntimeError {
    match source {
        ProcessInvocationError::ResourceEvidence => build_runtime_error(
            "system: complete child output has no certified finite resource bound",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .with_builtin("system")
        .build(),
        ProcessInvocationError::NoNativeDirectory | ProcessInvocationError::NoRevocation => {
            build_runtime_error(format!("system: {source}"))
                .with_identifier("RunMat:system:UnsupportedFilesystem")
                .with_builtin("system")
                .build()
        }
        ProcessInvocationError::StaleDirectory(source) => build_runtime_error(format!(
            "system: working directory authority revoked: {source}"
        ))
        .with_identifier("RunMat:system:StaleFilesystem")
        .with_builtin("system")
        .with_source(source)
        .build(),
        ProcessInvocationError::DirectoryCapture(source) => build_runtime_error(format!(
            "system: cannot capture working directory: {source}"
        ))
        .with_builtin("system")
        .with_source(source)
        .build(),
        ProcessInvocationError::Capability(source) => {
            build_runtime_error("system: no native process service is installed")
                .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
                .with_builtin("system")
                .with_source(source)
                .build()
        }
        ProcessInvocationError::Host(source) => build_runtime_error(format!("system: {source}"))
            .with_builtin("system")
            .with_source(source)
            .build(),
        ProcessInvocationError::ForeignExecutableOwner
        | ProcessInvocationError::WrongExecutableGrant
        | ProcessInvocationError::NoExecutableRevocation => {
            build_runtime_error(format!("system: {source}"))
                .with_identifier("RunMat:system:UnsupportedFilesystem")
                .with_builtin("system")
                .build()
        }
        ProcessInvocationError::StaleExecutable(source) => build_runtime_error(format!(
            "system: local executable authority revoked: {source}"
        ))
        .with_identifier("RunMat:system:StaleFilesystem")
        .with_builtin("system")
        .with_source(source)
        .build(),
        ProcessInvocationError::Execution(source) => source.into_runtime_error(),
    }
}

pub(super) async fn run(
    runtime: Option<&RuntimeContext>,
    command: &str,
) -> BuiltinResult<(i32, String)> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = command;
        if crate::context::complete_process_output_requires_evidence(runtime) {
            return Err(build_runtime_error(
                "system: complete child output has no certified finite resource bound",
            )
            .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
            .with_builtin("system")
            .build());
        }
        Err(
            build_runtime_error("system: process execution is unavailable in WebAssembly")
                .with_identifier("RunMat:system:UnsupportedOnWasm")
                .with_builtin("system")
                .build(),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use runmat_process_host::environment::EnvironmentPolicy;
        use runmat_process_host::HostCommand;

        #[cfg(windows)]
        let mut child = HostCommand::new("cmd");
        #[cfg(not(windows))]
        let mut child = HostCommand::new("/bin/sh");
        #[cfg(windows)]
        {
            child.arguments = vec!["/C".into(), command.into()];
        }
        #[cfg(not(windows))]
        {
            child.arguments = vec!["-c".into(), command.into()];
        }
        child.environment_policy = EnvironmentPolicy::Inherit;
        let output = process_invocation::output_complete(runtime, child, None, None)
            .await
            .map_err(process_error)?;
        let code = output
            .exit
            .code
            .unwrap_or_else(|| output.exit.signal.map(|signal| 128 + signal).unwrap_or(1));
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        Ok((code, text))
    }
}

#[cfg(test)]
#[path = "system/tests.rs"]
mod tests;
