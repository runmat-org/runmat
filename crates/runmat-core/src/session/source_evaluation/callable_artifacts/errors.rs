use runmat_runtime::{build_runtime_error, RuntimeError};

use super::CallableArtifactSelectionError;

pub(super) fn contract_error(source: runmat_execution::ContractError) -> RuntimeError {
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
