use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use serde::Serialize;
use std::path::PathBuf;

use crate::analysis::{
    analysis_plan_study_op, analysis_plan_study_sweep_op, analysis_run_study_op,
    analysis_run_study_sweep_op, analysis_validate_study_op, analysis_validate_study_sweep_op,
    load_analysis_study_document_from_path_async, AnalysisStudyDocument,
};
use crate::builtins::io::json::jsondecode::value_from_json;
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const VALIDATE_NAME: &str = "analysis_validate_study";
const PLAN_NAME: &str = "analysis_plan_study";
const RUN_NAME: &str = "analysis_run_study";

const STRUCT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operation result as a struct.",
}];
const STUDY_PATH_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to a .study.json, .study.yaml, or .study.yml file.",
}];

const VALIDATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = analysis_validate_study(path)",
    inputs: &STUDY_PATH_INPUT,
    outputs: &STRUCT_OUTPUT,
}];
const PLAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "plan = analysis_plan_study(path)",
    inputs: &STUDY_PATH_INPUT,
    outputs: &STRUCT_OUTPUT,
}];
const RUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "run = analysis_run_study(path)",
    inputs: &STUDY_PATH_INPUT,
    outputs: &STRUCT_OUTPUT,
}];

const VALIDATE_ERROR_LOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_VALIDATE_STUDY.LOAD_FAILED",
    identifier: Some("RunMat:analysis_validate_study:LoadFailed"),
    when: "The study document cannot be read or parsed.",
    message: "analysis_validate_study: failed to load study document",
};
const VALIDATE_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_VALIDATE_STUDY.OPERATION_FAILED",
    identifier: Some("RunMat:analysis_validate_study:OperationFailed"),
    when: "The validation operation fails.",
    message: "analysis_validate_study: operation failed",
};
const VALIDATE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_VALIDATE_STUDY.INTERNAL",
    identifier: Some("RunMat:analysis_validate_study:Internal"),
    when: "The validation result cannot be converted to a RunMat value.",
    message: "analysis_validate_study: internal error",
};
const VALIDATE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    VALIDATE_ERROR_LOAD,
    VALIDATE_ERROR_OPERATION,
    VALIDATE_ERROR_INTERNAL,
];
pub const ANALYSIS_VALIDATE_STUDY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VALIDATE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VALIDATE_ERRORS,
};

const PLAN_ERROR_LOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_PLAN_STUDY.LOAD_FAILED",
    identifier: Some("RunMat:analysis_plan_study:LoadFailed"),
    when: "The study document cannot be read or parsed.",
    message: "analysis_plan_study: failed to load study document",
};
const PLAN_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_PLAN_STUDY.OPERATION_FAILED",
    identifier: Some("RunMat:analysis_plan_study:OperationFailed"),
    when: "The planning operation fails.",
    message: "analysis_plan_study: operation failed",
};
const PLAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_PLAN_STUDY.INTERNAL",
    identifier: Some("RunMat:analysis_plan_study:Internal"),
    when: "The plan result cannot be converted to a RunMat value.",
    message: "analysis_plan_study: internal error",
};
const PLAN_ERRORS: [BuiltinErrorDescriptor; 3] =
    [PLAN_ERROR_LOAD, PLAN_ERROR_OPERATION, PLAN_ERROR_INTERNAL];
pub const ANALYSIS_PLAN_STUDY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PLAN_ERRORS,
};

const RUN_ERROR_LOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_RUN_STUDY.LOAD_FAILED",
    identifier: Some("RunMat:analysis_run_study:LoadFailed"),
    when: "The study document cannot be read or parsed.",
    message: "analysis_run_study: failed to load study document",
};
const RUN_ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_RUN_STUDY.OPERATION_FAILED",
    identifier: Some("RunMat:analysis_run_study:OperationFailed"),
    when: "The run operation fails.",
    message: "analysis_run_study: operation failed",
};
const RUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANALYSIS_RUN_STUDY.INTERNAL",
    identifier: Some("RunMat:analysis_run_study:Internal"),
    when: "The run result cannot be converted to a RunMat value.",
    message: "analysis_run_study: internal error",
};
const RUN_ERRORS: [BuiltinErrorDescriptor; 3] =
    [RUN_ERROR_LOAD, RUN_ERROR_OPERATION, RUN_ERROR_INTERNAL];
pub const ANALYSIS_RUN_STUDY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RUN_ERRORS,
};

#[runtime_builtin(
    name = "analysis_validate_study",
    category = "analysis",
    summary = "Validate a study or sweep file before running it.",
    keywords = "analysis,study,validate,sweep",
    descriptor(crate::builtins::analysis::ANALYSIS_VALIDATE_STUDY_DESCRIPTOR),
    builtin_path = "crate::builtins::analysis"
)]
pub async fn analysis_validate_study_builtin(path: String) -> BuiltinResult<Value> {
    let document = load_document(VALIDATE_NAME, &VALIDATE_ERROR_LOAD, path).await?;
    let context = OperationContext::new(None, None);
    match document {
        AnalysisStudyDocument::Study(spec) => operation_result_to_value(
            VALIDATE_NAME,
            &VALIDATE_ERROR_OPERATION,
            &VALIDATE_ERROR_INTERNAL,
            analysis_validate_study_op(&spec, context),
        ),
        AnalysisStudyDocument::Sweep(spec) => operation_result_to_value(
            VALIDATE_NAME,
            &VALIDATE_ERROR_OPERATION,
            &VALIDATE_ERROR_INTERNAL,
            analysis_validate_study_sweep_op(&spec, context),
        ),
    }
}

#[runtime_builtin(
    name = "analysis_plan_study",
    category = "analysis",
    summary = "Plan a study or sweep file without running the solve.",
    keywords = "analysis,study,plan,sweep",
    descriptor(crate::builtins::analysis::ANALYSIS_PLAN_STUDY_DESCRIPTOR),
    builtin_path = "crate::builtins::analysis"
)]
pub async fn analysis_plan_study_builtin(path: String) -> BuiltinResult<Value> {
    let document = load_document(PLAN_NAME, &PLAN_ERROR_LOAD, path).await?;
    let context = OperationContext::new(None, None);
    match document {
        AnalysisStudyDocument::Study(spec) => operation_result_to_value(
            PLAN_NAME,
            &PLAN_ERROR_OPERATION,
            &PLAN_ERROR_INTERNAL,
            analysis_plan_study_op(&spec, context),
        ),
        AnalysisStudyDocument::Sweep(spec) => operation_result_to_value(
            PLAN_NAME,
            &PLAN_ERROR_OPERATION,
            &PLAN_ERROR_INTERNAL,
            analysis_plan_study_sweep_op(&spec, context),
        ),
    }
}

#[runtime_builtin(
    name = "analysis_run_study",
    category = "analysis",
    summary = "Run a study or sweep file.",
    keywords = "analysis,study,run,solve,sweep",
    descriptor(crate::builtins::analysis::ANALYSIS_RUN_STUDY_DESCRIPTOR),
    builtin_path = "crate::builtins::analysis"
)]
pub async fn analysis_run_study_builtin(path: String) -> BuiltinResult<Value> {
    let document = load_document(RUN_NAME, &RUN_ERROR_LOAD, path).await?;
    let context = OperationContext::new(None, None);
    match document {
        AnalysisStudyDocument::Study(spec) => operation_result_to_value(
            RUN_NAME,
            &RUN_ERROR_OPERATION,
            &RUN_ERROR_INTERNAL,
            analysis_run_study_op(&spec, context),
        ),
        AnalysisStudyDocument::Sweep(spec) => operation_result_to_value(
            RUN_NAME,
            &RUN_ERROR_OPERATION,
            &RUN_ERROR_INTERNAL,
            analysis_run_study_sweep_op(&spec, context),
        ),
    }
}

async fn load_document(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    path: String,
) -> BuiltinResult<AnalysisStudyDocument> {
    load_analysis_study_document_from_path_async(&PathBuf::from(path))
        .await
        .map_err(|err| builtin_error(builtin, error, err))
}

fn operation_result_to_value<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    serializable_to_value(builtin, internal_error_descriptor, &envelope.data)
}

fn serializable_to_value<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &T,
) -> BuiltinResult<Value> {
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    value_from_json(&json)
}

fn operation_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    source: OperationErrorEnvelope,
) -> RuntimeError {
    let message = format!(
        "{}: {}: {}",
        error.message, source.error_code, source.message
    );
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(
            error
                .identifier
                .unwrap_or("RunMat:analysis:OperationFailed"),
        )
        .build()
}

fn builtin_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(error.identifier.unwrap_or("RunMat:analysis:Internal"))
        .build()
}

fn builtin_error_with_source<E>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(error.identifier.unwrap_or("RunMat:analysis:Internal"))
        .with_source(source)
        .build()
}
