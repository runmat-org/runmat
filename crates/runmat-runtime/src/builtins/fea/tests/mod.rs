use super::boundary::boundary_integer_to_f64;
use super::comparison::create_compare_object_from_args;
use super::contracts::descriptors::{ERROR_INPUT, ERROR_INTERNAL};
use super::contracts::identities::{
    BOUNDARY_CONDITION_NAME, COMPARE_NAME, DOMAIN_NAME, FEA_BOUNDARY_CONDITION_CLASS,
    FEA_FIELD_CLASS, FEA_LOAD_CASE_CLASS, FEA_MATERIAL_ASSIGNMENT_CLASS, FEA_MATERIAL_CLASS,
    FEA_MODEL_CLASS, FEA_PAYLOAD_JSON_PROPERTY, FEA_PLAN_CLASS, FEA_RESULTS_CLASS,
    FEA_RUN_ID_CONTEXT_PROPERTY, FEA_RUN_OPTIONS_CLASS, FEA_RUN_RESULT_CLASS, FEA_STEP_CLASS,
    FEA_STUDY_CLASS, FEA_STUDY_CONTEXT_JSON_PROPERTY, FEA_STUDY_SPEC_JSON_PROPERTY,
    FEA_SWEEP_CLASS, FEA_VALIDATION_CLASS, FIELD_NAME, INTERFACE_NAME, MATERIAL_NAME, MODEL_NAME,
    PLAN_NAME, RESULTS_NAME, RUN_NAME, RUN_OPTIONS_NAME, STUDY_NAME,
};
use super::contracts::integer::FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES;
use super::entrypoints::{
    fea_boundary_condition_builtin, fea_domain_builtin, fea_field_builtin, fea_interface_builtin,
    fea_load_builtin, fea_load_case_builtin, fea_material_assignment_builtin, fea_material_builtin,
    fea_model_builtin, fea_plan_builtin, fea_plot_builtin, fea_results_builtin,
    fea_run_options_builtin, fea_step_builtin, fea_study_builtin, fea_sweep_builtin,
    fea_trends_builtin, fea_validate_builtin,
};
use super::geometry::geometry_asset_from_value;
use super::integer_serialization::{
    promote_named_integer_fields, serializable_to_object_preserving_integers,
    serializable_to_object_value,
};
use super::options_json::{canonical_field_name, json_deserialize, value_to_json};
use super::output::{one_base_failure_entries, public_sweep_error};
use super::plot::{plot_request_from_args, select_generated_figure};
use super::results::constructors::create_field_object_from_args;
use super::results::field::{field_to_object, field_values_value};
use super::study::DomainPayload;
use super::value_decode::{
    exact_bool_from_value, one_based_usize_vec_from_value, usize_from_value, usize_vec_from_value,
};
use crate::analysis::{AnalysisFieldDescriptor, AnalysisStudySweepFailureEntry};
use crate::operations::OperationErrorEnvelope;
use futures::executor::block_on;
use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, AnalysisInterface, AnalysisInterfaceKind, AnalysisModel,
    BoundaryCondition, BoundaryConditionKind, LoadCase, LoadKind, MaterialModel,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_value::{
    CellArray, IntValue, IntegerStorage, ObjectInstance, StructArray, StructValue, Tensor, Value,
};
use serde::de::DeserializeOwned;
use serde::Serialize;

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

fn cell(values: Vec<Value>) -> Value {
    let cols = values.len().max(1);
    Value::Cell(CellArray::new(values, 1, cols).expect("cell should build"))
}

fn force_vector() -> Value {
    Value::Tensor(Tensor::new_2d(vec![0.0, -1000.0, 0.0], 1, 3).expect("tensor should build"))
}

fn moment_vector() -> Value {
    Value::Tensor(Tensor::new_2d(vec![10.0, 20.0, 30.0], 1, 3).expect("tensor should build"))
}

fn boundary_payload(value: Value) -> BoundaryCondition {
    let Value::Object(object) = value else {
        panic!("expected boundary condition object");
    };
    let Some(Value::String(payload)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) else {
        panic!("expected boundary condition JSON payload");
    };
    serde_json::from_str(payload).expect("boundary condition payload should decode")
}

fn object_payload<T: DeserializeOwned>(value: &Value) -> T {
    let Value::Object(object) = value else {
        panic!("expected FEA object");
    };
    let Some(Value::String(payload)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) else {
        panic!("expected FEA JSON payload");
    };
    serde_json::from_str(payload).expect("FEA payload should decode")
}

fn boundary_args(kind: &str, fields: Vec<(&str, Value)>) -> Vec<Value> {
    let mut args = vec![
        Value::String("bc".into()),
        Value::String("region".into()),
        Value::String(kind.into()),
    ];
    for (name, value) in fields {
        args.push(Value::String(name.into()));
        args.push(value);
    }
    args
}

mod construction;
mod numeric_integrity;
mod results_and_plotting;
mod support;
