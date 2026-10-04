//! The `system` process route. The process host alone owns native child I/O.

use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(not(target_arch = "wasm32"))]
fn process_error(source: runmat_process_host::ProcessHostError) -> RuntimeError {
    build_runtime_error(format!("system: {source}"))
        .with_builtin("system")
        .with_source(source)
        .build()
}

fn insufficient_resource_evidence() -> RuntimeError {
    build_runtime_error("system: complete child output has no certified finite resource bound")
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .with_builtin("system")
        .build()
}

#[cfg(not(target_arch = "wasm32"))]
fn stale_directory(source: std::io::Error) -> RuntimeError {
    build_runtime_error(format!(
        "system: working directory authority revoked: {source}"
    ))
    .with_identifier("RunMat:system:StaleFilesystem")
    .with_builtin("system")
    .with_source(source)
    .build()
}

pub(super) async fn run(
    runtime: Option<&RuntimeContext>,
    command: &str,
) -> BuiltinResult<(i32, String)> {
    // Child output is unbounded until the process-host route has an enforced
    // producer. An installed finite memory policy must reject before spawn.
    if runtime
        .and_then(|runtime| runtime.service_ports().parallel())
        .is_some_and(|port| port.placement_resources().memory_available_bytes.is_some())
    {
        return Err(insufficient_resource_evidence());
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = command;
        return Err(
            build_runtime_error("system: process execution is unavailable in WebAssembly")
                .with_identifier("RunMat:system:UnsupportedOnWasm")
                .with_builtin("system")
                .build(),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use futures::future::{select, Either};
        use runmat_filesystem::{NativePathIntent, NativeProcessRevocation};
        use runmat_process_host::environment::EnvironmentPolicy;
        use runmat_process_host::HostCommand;

        let cwd =
            runmat_filesystem::capture_native_path(".", NativePathIntent::ProcessWorkingDirectory)
                .map_err(|source| {
                    build_runtime_error(format!(
                        "system: cannot capture working directory: {source}"
                    ))
                    .with_builtin("system")
                    .with_source(source)
                    .build()
                })?
                .ok_or_else(|| {
                    build_runtime_error(
                        "system: selected filesystem cannot issue a native process directory",
                    )
                    .with_identifier("RunMat:system:UnsupportedFilesystem")
                    .with_builtin("system")
                    .build()
                })?;
        cwd.check_current().map_err(stale_directory)?;
        let revocation = cwd.process_revocation();
        if matches!(revocation, NativeProcessRevocation::Unavailable) {
            return Err(build_runtime_error(
                "system: selected process directory has no revocation observation",
            )
            .with_identifier("RunMat:system:UnsupportedFilesystem")
            .with_builtin("system")
            .build());
        }
        cwd.check_current().map_err(stale_directory)?;
        let process = runtime
            .and_then(|runtime| runtime.service_ports().process())
            .ok_or_else(|| {
                build_runtime_error("system: no native process service is installed")
                    .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
                    .with_builtin("system")
                    .build()
            })?;

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
        child.working_directory = Some(cwd.path().to_path_buf());
        child.environment_policy = EnvironmentPolicy::Inherit;

        let execute = async {
            match revocation {
                NativeProcessRevocation::Stable => {
                    process.output_complete(child).await.map_err(process_error)
                }
                NativeProcessRevocation::Watch(watch) => {
                    match select(watch, process.output_complete(child)).await {
                        Either::Right((result, _watch)) => result.map_err(process_error),
                        Either::Left(((), _process)) => Err(build_runtime_error(
                            "system: selected process directory authority was revoked",
                        )
                        .with_identifier("RunMat:system:StaleFilesystem")
                        .with_builtin("system")
                        .with_source(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "process directory authority revoked",
                        ))
                        .build()),
                    }
                }
                NativeProcessRevocation::Unavailable => unreachable!("rejected before spawn"),
            }
        };
        let output = if let Some(runtime) = runtime {
            runtime
                .cpu_cooperation()
                .await_cancellable(execute)
                .await
                .map_err(|source| source.into_runtime_error())??
        } else {
            execute.await?
        };
        cwd.check_current().map_err(stale_directory)?;
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
