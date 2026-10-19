use std::process::ExitCode;

pub async fn dispatch_hidden_mode(args: &[std::ffi::OsString]) -> Result<Option<ExitCode>, String> {
    match runmat_process_host::HiddenModeRegistry::standard()
        .detect(args.to_vec())
        .map_err(|error| format!("invalid RunMat host mode: {error}"))?
    {
        Some(runmat_process_host::HiddenMode::MeshingWorker) => {
            Ok(Some(run_meshing_worker().await))
        }
        Some(_) => {
            Err("this meshing verifier does not support the requested hidden host mode".into())
        }
        None => Ok(None),
    }
}

async fn run_meshing_worker() -> ExitCode {
    let result =
        match std::env::var_os(runmat_execution_runner_native::NATIVE_OBJECT_STORE_ROOT_ENV) {
            Some(root) => {
                runmat_execution_runner_native::run_meshing_worker_stdio(
                    std::sync::Arc::new(
                        runmat_execution_runner_native::native_meshing_kernel_dispatcher(),
                    ),
                    std::path::Path::new(&root),
                    runmat_execution_runner_native::NativeMeshingHostLimits::default(),
                )
                .await
            }
            None => Err(
                runmat_execution_runner_native::NativeExecutionError::Configuration(
                    "meshing worker object-store root is missing".into(),
                ),
            ),
        };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("RunMat meshing verification worker failed: {error}");
            ExitCode::from(2)
        }
    }
}
