use runmat_runtime::{build_runtime_error, RuntimeError};

use super::CallableArtifactSelectionError;

pub(super) fn contract_error(source: runmat_execution::ContractError) -> RuntimeError {
    build_runtime_error(source.to_string())
        .with_identifier("RunMat:CallableArtifact:Invalid")
        .with_source(source)
        .build()
}

pub(super) fn artifact_index_error(
    source: super::super::generation::CallableArtifactIndexError,
) -> RuntimeError {
    build_runtime_error(source.to_string())
        .with_identifier("RunMat:CallableArtifact:Invalid")
        .with_source(source)
        .build()
}

pub(super) fn codec_error(
    source: runmat_runtime::execution::value_codec::ValueCodecError,
) -> RuntimeError {
    build_runtime_error(source.to_string())
        .with_identifier("RunMat:CallableArtifact:ValueCodec")
        .with_source(source)
        .build()
}

pub(super) fn cooperative_codec_error(
    source: runmat_runtime::execution::value_codec::CooperativeValueCodecError,
) -> RuntimeError {
    use runmat_runtime::execution::value_codec::CooperativeValueCodecError;
    match source {
        CooperativeValueCodecError::Value(source) => codec_error(source),
        CooperativeValueCodecError::Execution(source) => {
            let retained = source.clone();
            let mut error = source.into_runtime_error();
            if error.source.is_none() {
                error.source = Some(Box::new(retained));
            }
            error
        }
    }
}

pub(super) fn executable_admission_error(
    source: runmat_vm::ExecutableUnitAdmissionError,
) -> RuntimeError {
    build_runtime_error(source.to_string())
        .with_identifier("RunMat:CallableArtifact:ExecutableAdmission")
        .with_source(source)
        .build()
}

pub(super) fn staging_error(source: runmat_vm::RetainedCallableStageError) -> RuntimeError {
    build_runtime_error(source.to_string())
        .with_identifier("RunMat:CallableArtifact:DefinitionAdmission")
        .with_source(source)
        .build()
}

pub(super) fn artifact_selection_error(source: CallableArtifactSelectionError) -> RuntimeError {
    match source {
        CallableArtifactSelectionError::Missing(program) => build_runtime_error(format!(
            "callable definition program {program} has no retained executable artifact"
        ))
        .with_identifier("RunMat:CallableArtifact:MissingExecutable")
        .build(),
        CallableArtifactSelectionError::Limit { field, limit } => {
            build_runtime_error(format!("{field} exceeds its limit of {limit}"))
                .with_identifier("RunMat:CallableArtifact:Invalid")
                .build()
        }
    }
}

pub(super) fn unavailable_binding(function: usize) -> RuntimeError {
    build_runtime_error(format!(
        "bound function {function} is not owned by the selected definition generation"
    ))
    .with_identifier("RunMat:BoundFunction:Unavailable")
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::execution::ExecutionServiceError;
    use std::error::Error;

    #[test]
    fn cooperative_execution_failures_keep_canonical_identifiers_and_typed_causes() {
        let cases = [
            (
                ExecutionServiceError::Cancelled,
                "RunMat:parallel:Cancelled",
            ),
            (
                ExecutionServiceError::Infrastructure("turn failed".into()),
                "RunMat:parallel:InfrastructureFailure",
            ),
        ];
        for (source, identifier) in cases {
            let expected = source.clone();
            let error = cooperative_codec_error(
                runmat_runtime::execution::value_codec::CooperativeValueCodecError::Execution(
                    source,
                ),
            );
            assert_eq!(error.identifier(), Some(identifier));
            assert_eq!(
                Error::source(&error)
                    .and_then(|cause| { cause.downcast_ref::<ExecutionServiceError>().cloned() }),
                Some(expected)
            );
        }
    }
}
