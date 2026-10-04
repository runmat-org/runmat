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
        use runmat_filesystem::NativePathIntent;
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
        cwd.check_current().map_err(|source| {
            build_runtime_error(format!(
                "system: working directory authority revoked: {source}"
            ))
            .with_builtin("system")
            .with_source(source)
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

        let output = if let Some(runtime) = runtime {
            runtime
                .cpu_cooperation()
                .await_cancellable(child.output_complete())
                .await
                .map_err(|source| source.into_runtime_error())?
                .map_err(process_error)?
        } else {
            child.output_complete().await.map_err(process_error)?
        };
        cwd.check_current().map_err(|source| {
            build_runtime_error(format!(
                "system: working directory authority revoked: {source}"
            ))
            .with_builtin("system")
            .with_source(source)
            .build()
        })?;
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
mod tests {
    use super::*;
    use crate::context::{ParallelCapability, RuntimeParallelResources, RuntimeParallelService};
    use std::rc::Rc;

    struct Finite;

    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            RuntimeParallelResources {
                memory_available_bytes: Some(1),
                ..Default::default()
            }
        }
    }

    #[test]
    fn finite_resource_budget_rejects_unbounded_process_before_spawn() {
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    crate::context::RuntimeServicePorts::default().with_parallel(Rc::new(Finite)),
                );
        let error = futures::executor::block_on(run(Some(&runtime), "printf no-process"))
            .expect_err("unbounded route cannot claim a finite certificate");
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn virtual_provider_cannot_be_used_as_native_process_cwd() {
        let _serial = runmat_filesystem::provider_override_lock();
        let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
            "/virtual",
        ));
        let _selected = runmat_filesystem::replace_provider(provider);
        let error = futures::executor::block_on(run(None, "printf no-process"))
            .expect_err("virtual cwd cannot escape to an OS process");
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:system:UnsupportedFilesystem")
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    async fn browser_process_rejects_without_spawning() {
        let error = run(None, "printf no-process")
            .await
            .expect_err("browser has no process host");
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:system:UnsupportedOnWasm")
        );
    }
}
