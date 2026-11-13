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
        runmat_jit::JitError::Executor(error) => from_native_executor_error(error),
        runmat_jit::JitError::Codegen(error) if error.allocation.is_some() => {
            from_native_executor_error(runmat_native_executor::NativeExecutorError::InvalidProgram(
                error,
            ))
        }
        error => stage("NativeCompilation", error),
    }
}

pub(super) fn from_native_executor_error(
    error: runmat_native_executor::NativeExecutorError,
) -> runmat_runtime::RuntimeError {
    error.into_runtime_error()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn missing_native_source_keeps_typed_admission_cause() {
        let missing = runmat_runtime::context::ProgramSourceSelectionError::Missing(
            runmat_types::ProgramSourceId(7),
        );
        let error = from_native_executor_error(
            runmat_native_executor::NativeExecutorError::SourceAdmission(missing),
        );
        assert_eq!(error.identifier(), Some("RunMat:NativeSourceAdmission"));
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<runmat_runtime::context::ProgramSourceSelectionError>(),
            Some(&missing)
        );
    }
}
