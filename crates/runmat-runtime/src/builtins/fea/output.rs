use crate::analysis::AnalysisStudySpec;
use crate::analysis::AnalysisStudySweepData;
use crate::analysis::AnalysisStudySweepFailureEntry;
use crate::analysis::AnalysisStudySweepPlanData;
use crate::analysis::AnalysisStudySweepSpec;
use crate::analysis::FeaResolvedDocument;
use crate::builtins::fea::contracts::descriptors::{ERROR_INTERNAL, ERROR_OPERATION};
use crate::builtins::fea::contracts::identities::{
    FEA_PAYLOAD_JSON_PROPERTY, FEA_PLAN_CLASS, FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS,
    FEA_STUDY_SPEC_JSON_PROPERTY, FEA_SWEEP_CLASS, FEA_SWEEP_SPEC_JSON_PROPERTY, PLAN_NAME,
    RUN_NAME, STUDY_NAME, SWEEP_NAME,
};
use crate::builtins::fea::errors::{builtin_error, operation_error};
use crate::builtins::fea::integer_serialization::{
    serializable_to_object, serializable_to_object_preserving_integers,
};
use crate::operations::OperationEnvelope;
use crate::operations::OperationErrorEnvelope;
use crate::BuiltinResult;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::Value;
use serde::Serialize;

pub(in crate::builtins::fea) fn resolved_document_to_object(
    document: FeaResolvedDocument,
) -> BuiltinResult<Value> {
    match document {
        FeaResolvedDocument::Study(spec) => study_to_object(*spec),
        FeaResolvedDocument::Sweep(spec) => sweep_to_object(spec),
    }
}

pub(in crate::builtins::fea) fn study_to_object(spec: AnalysisStudySpec) -> BuiltinResult<Value> {
    let mut object = serializable_to_object(
        STUDY_NAME,
        &ERROR_INTERNAL,
        FEA_STUDY_CLASS,
        &spec,
        Some(FEA_STUDY_SPEC_JSON_PROPERTY),
    )?;
    if let Value::Object(ref mut object) = object {
        object
            .properties
            .insert("id".to_string(), Value::String(spec.study_id));
    }
    Ok(object)
}

pub(in crate::builtins::fea) fn sweep_to_object(
    spec: AnalysisStudySweepSpec,
) -> BuiltinResult<Value> {
    let mut object = serializable_to_object(
        SWEEP_NAME,
        &ERROR_INTERNAL,
        FEA_SWEEP_CLASS,
        &spec,
        Some(FEA_SWEEP_SPEC_JSON_PROPERTY),
    )?;
    if let Value::Object(ref mut object) = object {
        object
            .properties
            .insert("id".to_string(), Value::String(spec.sweep_id));
    }
    Ok(object)
}

pub(in crate::builtins::fea) fn operation_result_to_object<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    class_name: runmat_types::StaticClassIdentity,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    serializable_to_object(
        builtin,
        internal_error_descriptor,
        class_name,
        &envelope.data,
        hidden_json_property,
    )
}

pub(in crate::builtins::fea) fn operation_result_to_object_preserving_integers<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    class_name: runmat_types::StaticClassIdentity,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
    hidden_json_property: Option<&'static str>,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    serializable_to_object_preserving_integers(
        builtin,
        internal_error_descriptor,
        class_name,
        &envelope.data,
        hidden_json_property,
        signed_fields,
        unsigned_fields,
    )
}

pub(in crate::builtins::fea) fn sweep_plan_result_to_object(
    result: Result<OperationEnvelope<AnalysisStudySweepPlanData>, OperationErrorEnvelope>,
) -> BuiltinResult<Value> {
    let mut envelope = result
        .map_err(|error| operation_error(PLAN_NAME, &ERROR_OPERATION, public_sweep_error(error)))?;
    one_base_failure_entries(PLAN_NAME, &mut envelope.data.failure_entries)?;
    serializable_to_object_preserving_integers(
        PLAN_NAME,
        &ERROR_INTERNAL,
        FEA_PLAN_CLASS,
        &envelope.data,
        None,
        &[],
        &[
            "study_count",
            "planned_count",
            "failed_count",
            "study_index",
        ],
    )
}

pub(in crate::builtins::fea) fn sweep_run_result_to_object(
    result: Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope>,
) -> BuiltinResult<Value> {
    let mut envelope = result
        .map_err(|error| operation_error(RUN_NAME, &ERROR_OPERATION, public_sweep_error(error)))?;
    one_base_failure_entries(RUN_NAME, &mut envelope.data.failure_entries)?;
    serializable_to_object_preserving_integers(
        RUN_NAME,
        &ERROR_INTERNAL,
        FEA_RUN_RESULT_CLASS,
        &envelope.data,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[],
        &[
            "study_count",
            "success_count",
            "failed_count",
            "study_index",
        ],
    )
}

pub(in crate::builtins::fea) fn one_base_failure_entries(
    builtin: &'static str,
    entries: &mut [AnalysisStudySweepFailureEntry],
) -> BuiltinResult<()> {
    for entry in entries {
        entry.study_index = entry.study_index.checked_add(1).ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INTERNAL,
                "study index cannot be represented at the one-based public boundary",
            )
        })?;
    }
    Ok(())
}

pub(in crate::builtins::fea) fn public_sweep_error(
    mut error: OperationErrorEnvelope,
) -> OperationErrorEnvelope {
    let Some(index) = error
        .context
        .get("study_index")
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return error;
    };
    let Some(public_index) = index.checked_add(1) else {
        return error;
    };
    error
        .context
        .insert("study_index".to_string(), public_index.to_string());
    error.message = error.message.replacen(
        &format!("at index {index} "),
        &format!("at index {public_index} "),
        1,
    );
    error
}
