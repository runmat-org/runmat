pub(super) fn stage(
    stage: &'static str,
    error: impl std::fmt::Display,
) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(error.to_string())
        .with_identifier(format!("RunMat:{stage}"))
        .with_phase("native")
        .build()
}

pub(super) fn from_jit_error(error: runmat_jit::JitError) -> runmat_runtime::RuntimeError {
    match error {
        runmat_jit::JitError::Runtime(error) => *error,
        runmat_jit::JitError::Cancelled => {
            runmat_runtime::build_runtime_error("native execution was cancelled")
                .with_identifier("RunMat:ExecutionCancelled")
                .with_phase("native")
                .build()
        }
        runmat_jit::JitError::UnsupportedSite(message) => stage("NativeUnsupportedSite", message),
        runmat_jit::JitError::Unavailable => stage(
            "NativeUnavailable",
            "native execution is unavailable on this platform",
        ),
        other => stage("NativeExecution", other),
    }
}
