use crate::analysis::AnalysisStudySpec;
use crate::builtins::fea::contracts::descriptors::ERROR_INPUT;
use crate::builtins::fea::contracts::identities::{
    FEA_BOUNDARY_CONDITION_CLASS, FEA_DOMAIN_CLASS, FEA_INTERFACE_CLASS, FEA_LOAD_CASE_CLASS,
    FEA_MATERIAL_ASSIGNMENT_CLASS, FEA_MATERIAL_CLASS, FEA_MODEL_CLASS, FEA_PAYLOAD_JSON_PROPERTY,
    FEA_STEP_CLASS, FEA_STUDY_CLASS, FEA_STUDY_SPEC_JSON_PROPERTY,
};
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::object_json_property;
use crate::builtins::fea::study::DomainPayload;
use crate::BuiltinResult;
use runmat_analysis_core::AnalysisInterface;
use runmat_analysis_core::AnalysisModel;
use runmat_analysis_core::AnalysisStep;
use runmat_analysis_core::BoundaryCondition;
use runmat_analysis_core::LoadCase;
use runmat_analysis_core::MaterialAssignment;
use runmat_analysis_core::MaterialModel;
use runmat_value::Value;
use serde::de::DeserializeOwned;

pub(in crate::builtins::fea) fn model_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<AnalysisModel> {
    object_payload(builtin, value, FEA_MODEL_CLASS)
}

pub(in crate::builtins::fea) fn study_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<AnalysisStudySpec>> {
    object_vec_from_value_with_property(
        builtin,
        value,
        FEA_STUDY_CLASS,
        FEA_STUDY_SPEC_JSON_PROPERTY,
    )
}

pub(in crate::builtins::fea) fn material_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<MaterialModel>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_CLASS)
}

pub(in crate::builtins::fea) fn material_assignment_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<MaterialAssignment>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_ASSIGNMENT_CLASS)
}

pub(in crate::builtins::fea) fn boundary_condition_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<BoundaryCondition>> {
    object_vec_from_value(builtin, value, FEA_BOUNDARY_CONDITION_CLASS)
}

pub(in crate::builtins::fea) fn load_case_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<LoadCase>> {
    object_vec_from_value(builtin, value, FEA_LOAD_CASE_CLASS)
}

pub(in crate::builtins::fea) fn step_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<AnalysisStep>> {
    object_vec_from_value(builtin, value, FEA_STEP_CLASS)
}

pub(in crate::builtins::fea) fn domain_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<DomainPayload>> {
    object_vec_from_value(builtin, value, FEA_DOMAIN_CLASS)
}

pub(in crate::builtins::fea) fn interface_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<AnalysisInterface>> {
    object_vec_from_value(builtin, value, FEA_INTERFACE_CLASS)
}

fn object_vec_from_value<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: runmat_types::StaticClassIdentity,
) -> BuiltinResult<Vec<T>> {
    object_vec_from_value_with_property(builtin, value, expected_class, FEA_PAYLOAD_JSON_PROPERTY)
}

fn object_vec_from_value_with_property<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: runmat_types::StaticClassIdentity,
    payload_property: &'static str,
) -> BuiltinResult<Vec<T>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| {
                object_payload_with_property(builtin, item, expected_class, payload_property)
            })
            .collect(),
        Value::Object(_) => Ok(vec![object_payload_with_property(
            builtin,
            value,
            expected_class,
            payload_property,
        )?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class} object or cell array; got {other:?}"),
        )),
    }
}

pub(in crate::builtins::fea) fn object_payload<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: runmat_types::StaticClassIdentity,
) -> BuiltinResult<T> {
    object_payload_with_property(builtin, value, expected_class, FEA_PAYLOAD_JSON_PROPERTY)
}

fn object_payload_with_property<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: runmat_types::StaticClassIdentity,
    payload_property: &'static str,
) -> BuiltinResult<T> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class} object"),
        ));
    };
    if !object.class_name.is(expected_class) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class}, got {}", object.class_name),
        ));
    }
    object_json_property(builtin, object, payload_property, &ERROR_INPUT)
}
