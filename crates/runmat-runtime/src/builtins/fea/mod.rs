use runmat_analysis_core::{
    AnalysisInterface, AnalysisInterfaceKind, AnalysisModel, AnalysisModelId, AnalysisStep,
    AnalysisStepKind, BoundaryCondition, BoundaryConditionKind, EvidenceConfidence, LoadCase,
    LoadKind, MaterialAcousticModel, MaterialAssignment, MaterialElectricalModel,
    MaterialMechanicalModel, MaterialModel, MaterialPlasticModel, MaterialThermalModel,
    ReferenceFrame,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, MethodDef, ObjectInstance, Tensor, Value,
};
use runmat_geometry_core::GeometryAsset;
use runmat_macros::runtime_builtin;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::analysis::{
    analysis_create_model_op, analysis_plan_study_op, analysis_plan_study_sweep_op,
    analysis_results_by_run_id_op, analysis_results_compare_op, analysis_run_study_op,
    analysis_run_study_sweep_op, analysis_trends_op, analysis_validate_study_op,
    analysis_validate_study_sweep_op, load_fea_document_from_path_async,
    AnalysisAcousticRunOptions, AnalysisCfdRunOptions, AnalysisChtRunOptions,
    AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile, AnalysisElectromagneticRunOptions,
    AnalysisFsiRunOptions, AnalysisModalRunOptions, AnalysisNonlinearRunOptions,
    AnalysisResultsCompareQuery, AnalysisResultsQuery, AnalysisRunKind, AnalysisRunOptions,
    AnalysisStudySpec, AnalysisStudySweepSpec, AnalysisThermalRunOptions,
    AnalysisTransientRunOptions, AnalysisTrendsQuery, FeaResolvedDocument,
};
use crate::builtins::geometry::{GEOMETRY_ASSET_CLASS, GEOMETRY_ASSET_JSON_PROPERTY};
use crate::builtins::io::json::jsondecode::value_from_json;
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const FEA_STUDY_CLASS: &str = "fea.Study";
const FEA_SWEEP_CLASS: &str = "fea.Sweep";
const FEA_VALIDATION_CLASS: &str = "fea.Validation";
const FEA_PLAN_CLASS: &str = "fea.Plan";
const FEA_RUN_RESULT_CLASS: &str = "fea.RunResult";
const FEA_MODEL_CLASS: &str = "fea.Model";
const FEA_MATERIAL_CLASS: &str = "fea.Material";
const FEA_MATERIAL_ASSIGNMENT_CLASS: &str = "fea.MaterialAssignment";
const FEA_BOUNDARY_CONDITION_CLASS: &str = "fea.BoundaryCondition";
const FEA_LOAD_CASE_CLASS: &str = "fea.LoadCase";
const FEA_STEP_CLASS: &str = "fea.Step";
const FEA_DOMAIN_CLASS: &str = "fea.Domain";
const FEA_INTERFACE_CLASS: &str = "fea.Interface";
const FEA_RUN_OPTIONS_CLASS: &str = "fea.RunOptions";
const FEA_RESULTS_CLASS: &str = "fea.Results";
const FEA_FIELD_CLASS: &str = "fea.Field";
const FEA_COMPARE_CLASS: &str = "fea.Compare";
const FEA_TRENDS_CLASS: &str = "fea.Trends";
const FEA_STUDY_SPEC_JSON_PROPERTY: &str = "__runmat_fea_study_spec_json";
const FEA_SWEEP_SPEC_JSON_PROPERTY: &str = "__runmat_fea_sweep_spec_json";
const FEA_PAYLOAD_JSON_PROPERTY: &str = "__runmat_fea_payload_json";

const LOAD_NAME: &str = "fea.load";
const STUDY_NAME: &str = "fea.study";
const SWEEP_NAME: &str = "fea.sweep";
const MODEL_NAME: &str = "fea.model";
const MATERIAL_NAME: &str = "fea.material";
const MATERIAL_ASSIGNMENT_NAME: &str = "fea.materialAssignment";
const BOUNDARY_CONDITION_NAME: &str = "fea.boundaryCondition";
const LOAD_CASE_NAME: &str = "fea.loadCase";
const STEP_NAME: &str = "fea.step";
const DOMAIN_NAME: &str = "fea.domain";
const INTERFACE_NAME: &str = "fea.interface";
const RUN_OPTIONS_NAME: &str = "fea.runOptions";
const VALIDATE_NAME: &str = "fea.validate";
const PLAN_NAME: &str = "fea.plan";
const RUN_NAME: &str = "fea.run";
const RESULTS_NAME: &str = "fea.results";
const FIELD_NAME: &str = "fea.field";
const COMPARE_NAME: &str = "fea.compare";
const TRENDS_NAME: &str = "fea.trends";

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "FEA object or operation result.",
}];
const IN_PATH: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to a .fea file.",
}];
const IN_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "study",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A .fea path, fea.Study object, or fea.Sweep object.",
}];
const IN_STUDY_ARGS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Study id.",
    },
    BuiltinParamDescriptor {
        name: "geometry",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "geometry.Asset returned by geometry.load.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Profile, Backend, ModelId, and model setup options.",
    },
];
const IN_VARIADIC_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Constructor or query arguments.",
}];

const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "doc = fea.load(path)",
    inputs: &IN_PATH,
    outputs: &OUT_ANY,
}];
const STUDY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "study = fea.study(id, geometry, Name, Value, ...)",
    inputs: &IN_STUDY_ARGS,
    outputs: &OUT_ANY,
}];
const VALIDATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = fea.validate(study)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
const PLAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "plan = fea.plan(study)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
const RUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "run = fea.run(study)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
const SWEEP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "sweep = fea.sweep(id, studies, Name, Value, ...)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
const MODEL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "model = fea.model(id, geometry, Name, Value, ...)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
const MATERIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "material = fea.material(id, Name, Value, ...)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
const COMPONENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "component = fea.component(args, ...)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
const RESULTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "results = fea.results(runOrRunId, Name, Value, ...)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
const FIELD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.field(resultsOrRun, fieldId)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
const COMPARE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "comparison = fea.compare(baselineRunId, candidateRunId)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
const TRENDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "trends = fea.trends(Name, Value, ...)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];

const ERROR_LOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.LOAD_FAILED",
    identifier: Some("RunMat:fea:LoadFailed"),
    when: "A .fea document cannot be read, parsed, or resolved.",
    message: "fea: failed to load FEA document",
};
const ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INVALID_INPUT",
    identifier: Some("RunMat:fea:InvalidInput"),
    when: "A builtin receives an unsupported argument pattern or object type.",
    message: "fea: invalid input",
};
const ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.OPERATION_FAILED",
    identifier: Some("RunMat:fea:OperationFailed"),
    when: "The validation, planning, or run operation fails.",
    message: "fea: operation failed",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INTERNAL",
    identifier: Some("RunMat:fea:Internal"),
    when: "An FEA object or operation result cannot be converted to a RunMat value.",
    message: "fea: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] =
    [ERROR_LOAD, ERROR_INPUT, ERROR_OPERATION, ERROR_INTERNAL];

pub const FEA_LOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_STUDY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STUDY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_VALIDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VALIDATE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_PLAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_RUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_SWEEP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SWEEP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_MODEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MODEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_MATERIAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATERIAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_COMPONENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPONENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_RESULTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESULTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_FIELD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIELD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_COMPARE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPARE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_TRENDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRENDS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.load",
    category = "fea",
    summary = "Load a .fea study or sweep document.",
    keywords = "fea,study,sweep,load,yaml",
    descriptor(crate::builtins::fea::FEA_LOAD_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_load_builtin(path: String) -> BuiltinResult<Value> {
    load_document_object(PathBuf::from(path)).await
}

#[runtime_builtin(
    name = "fea.study",
    category = "fea",
    summary = "Create a typed FEA study from geometry, model data, and run settings.",
    keywords = "fea,study,geometry,run",
    descriptor(crate::builtins::fea::FEA_STUDY_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_study_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() == 1 {
        let path = scalar_string(&args[0], STUDY_NAME, &ERROR_INPUT)?;
        return load_document_object(PathBuf::from(path)).await;
    }
    create_study_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.sweep",
    category = "fea",
    summary = "Create a FEA study sweep from study objects.",
    keywords = "fea,sweep,study,run",
    descriptor(crate::builtins::fea::FEA_SWEEP_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_sweep_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_sweep_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.model",
    category = "fea",
    summary = "Create a typed FEA model object from geometry and model components.",
    keywords = "fea,model,materials,boundary,loads,domains",
    descriptor(crate::builtins::fea::FEA_MODEL_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_model_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_model_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.material",
    category = "fea",
    summary = "Create a typed FEA material object.",
    keywords = "fea,material,mechanical,thermal,electrical,plastic",
    descriptor(crate::builtins::fea::FEA_MATERIAL_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_material_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_material_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.materialAssignment",
    category = "fea",
    summary = "Create a typed FEA material assignment.",
    keywords = "fea,material,assignment,region",
    descriptor(crate::builtins::fea::FEA_COMPONENT_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_material_assignment_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_material_assignment_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.boundaryCondition",
    category = "fea",
    summary = "Create a typed FEA boundary condition.",
    keywords = "fea,boundary,condition,region",
    descriptor(crate::builtins::fea::FEA_COMPONENT_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_boundary_condition_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_boundary_condition_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.loadCase",
    category = "fea",
    summary = "Create a typed FEA load case.",
    keywords = "fea,load,force,pressure,current",
    descriptor(crate::builtins::fea::FEA_COMPONENT_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_load_case_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_load_case_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.step",
    category = "fea",
    summary = "Create a typed FEA analysis step.",
    keywords = "fea,step,static,modal,transient",
    descriptor(crate::builtins::fea::FEA_COMPONENT_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_step_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_step_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.domain",
    category = "fea",
    summary = "Create a typed FEA physics domain object.",
    keywords = "fea,domain,thermal,electromagnetic,cfd",
    descriptor(crate::builtins::fea::FEA_COMPONENT_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_domain_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_domain_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.interface",
    category = "fea",
    summary = "Create a typed FEA interface object.",
    keywords = "fea,interface,contact,region",
    descriptor(crate::builtins::fea::FEA_COMPONENT_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_interface_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_interface_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.runOptions",
    category = "fea",
    summary = "Create typed FEA run options for a solver.",
    keywords = "fea,run,options,solver,quality",
    descriptor(crate::builtins::fea::FEA_COMPONENT_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_run_options_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_run_options_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.validate",
    category = "fea",
    summary = "Validate a FEA study or sweep without planning or solving.",
    keywords = "fea,validate,study,sweep",
    descriptor(crate::builtins::fea::FEA_VALIDATE_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_validate_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, VALIDATE_NAME).await? {
        FeaResolvedDocument::Study(spec) => operation_result_to_object(
            VALIDATE_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_VALIDATION_CLASS,
            analysis_validate_study_op(&spec, OperationContext::new(None, None)),
            None,
        ),
        FeaResolvedDocument::Sweep(spec) => operation_result_to_object(
            VALIDATE_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_VALIDATION_CLASS,
            analysis_validate_study_sweep_op(&spec, OperationContext::new(None, None)),
            None,
        ),
    }
}

#[runtime_builtin(
    name = "fea.plan",
    category = "fea",
    summary = "Plan a FEA study or sweep without solving it.",
    keywords = "fea,plan,study,sweep",
    descriptor(crate::builtins::fea::FEA_PLAN_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_plan_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, PLAN_NAME).await? {
        FeaResolvedDocument::Study(spec) => operation_result_to_object(
            PLAN_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_PLAN_CLASS,
            analysis_plan_study_op(&spec, OperationContext::new(None, None)),
            None,
        ),
        FeaResolvedDocument::Sweep(spec) => operation_result_to_object(
            PLAN_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_PLAN_CLASS,
            analysis_plan_study_sweep_op(&spec, OperationContext::new(None, None)),
            None,
        ),
    }
}

#[runtime_builtin(
    name = "fea.run",
    category = "fea",
    summary = "Run a FEA study or sweep.",
    keywords = "fea,run,study,sweep,solve",
    descriptor(crate::builtins::fea::FEA_RUN_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_run_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, RUN_NAME).await? {
        FeaResolvedDocument::Study(spec) => operation_result_to_object(
            RUN_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_RUN_RESULT_CLASS,
            analysis_run_study_op(&spec, OperationContext::new(None, None)),
            Some(FEA_PAYLOAD_JSON_PROPERTY),
        ),
        FeaResolvedDocument::Sweep(spec) => operation_result_to_object(
            RUN_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_RUN_RESULT_CLASS,
            analysis_run_study_sweep_op(&spec, OperationContext::new(None, None)),
            Some(FEA_PAYLOAD_JSON_PROPERTY),
        ),
    }
}

#[runtime_builtin(
    name = "fea.results",
    category = "fea",
    summary = "Load or project FEA run results for post-processing.",
    keywords = "fea,results,run_id,fields,diagnostics",
    descriptor(crate::builtins::fea::FEA_RESULTS_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_results_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_results_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.field",
    category = "fea",
    summary = "Extract a field from FEA results or a run result.",
    keywords = "fea,field,displacement,von_mises,post",
    descriptor(crate::builtins::fea::FEA_FIELD_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_field_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_field_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.compare",
    category = "fea",
    summary = "Compare two persisted FEA runs by run id.",
    keywords = "fea,compare,run_id,quality",
    descriptor(crate::builtins::fea::FEA_COMPARE_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_compare_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_compare_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.trends",
    category = "fea",
    summary = "Summarize recent persisted FEA run trends.",
    keywords = "fea,trends,history,quality",
    descriptor(crate::builtins::fea::FEA_TRENDS_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_trends_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_trends_object_from_args(args)
}

async fn load_document_object(path: PathBuf) -> BuiltinResult<Value> {
    let document = load_fea_document_from_path_async(&path)
        .await
        .map_err(|err| builtin_error(LOAD_NAME, &ERROR_LOAD, err))?;
    resolved_document_to_object(document)
}

async fn resolve_document_input(
    input: Value,
    builtin: &'static str,
) -> BuiltinResult<FeaResolvedDocument> {
    match input {
        Value::Object(object) if object.class_name == FEA_STUDY_CLASS => {
            let spec: AnalysisStudySpec =
                object_json_property(builtin, &object, FEA_STUDY_SPEC_JSON_PROPERTY, &ERROR_INPUT)?;
            Ok(FeaResolvedDocument::Study(Box::new(spec)))
        }
        Value::Object(object) if object.class_name == FEA_SWEEP_CLASS => {
            let spec: AnalysisStudySweepSpec =
                object_json_property(builtin, &object, FEA_SWEEP_SPEC_JSON_PROPERTY, &ERROR_INPUT)?;
            Ok(FeaResolvedDocument::Sweep(spec))
        }
        Value::String(path) => load_fea_document_from_path_async(&PathBuf::from(path))
            .await
            .map_err(|err| builtin_error(builtin, &ERROR_LOAD, err)),
        Value::CharArray(chars) if chars.rows == 1 => {
            let path: String = chars.data.iter().collect();
            load_fea_document_from_path_async(&PathBuf::from(path))
                .await
                .map_err(|err| builtin_error(builtin, &ERROR_LOAD, err))
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected .fea path, {FEA_STUDY_CLASS}, or {FEA_SWEEP_CLASS}; got {other:?}"),
        )),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunOptionsPayload {
    run_kind: AnalysisRunKind,
    options: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DomainPayload {
    kind: String,
    data: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelDefaultsMode {
    ProfileScaffold,
    None,
}

impl Default for ModelDefaultsMode {
    fn default() -> Self {
        Self::ProfileScaffold
    }
}

#[derive(Debug, Default)]
struct StudyConstructorOptions {
    run_kind: Option<AnalysisRunKind>,
    profile: Option<AnalysisCreateModelProfile>,
    backend: Option<ComputeBackend>,
    model_id: Option<String>,
    model: Option<AnalysisModel>,
    frame: Option<ReferenceFrame>,
    model_defaults: ModelDefaultsMode,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    boundary_conditions: Vec<BoundaryCondition>,
    loads: Vec<LoadCase>,
    steps: Vec<AnalysisStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
    run_options: Option<RunOptionsPayload>,
}

fn create_study_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study requires id and geometry arguments",
        ));
    }
    let study_id = scalar_string(&args[0], STUDY_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(&args[1])?;
    let options = StudyConstructorOptions::parse(&args[2..])?;
    let (profile, run_kind) = resolve_study_profile_and_run_kind(&options)?;
    let model_id = options.model_id.clone().unwrap_or_else(|| {
        options
            .model
            .as_ref()
            .map(|model| model.model_id.0.clone())
            .unwrap_or_else(|| format!("{}_model", sanitize_id(&study_id)))
    });
    let model = match options.model {
        Some(model) => Some(model),
        None if options.has_model_components() => Some(build_model_from_parts(
            STUDY_NAME,
            &geometry,
            model_id.clone(),
            profile,
            options.model_defaults,
            options.frame,
            options.materials,
            options.material_assignments,
            options.boundary_conditions,
            options.loads,
            options.steps,
            options.domains,
            options.interfaces,
        )?),
        None => None,
    };
    let run_options = options
        .run_options
        .map(|payload| resolved_run_options_from_payload(STUDY_NAME, payload, run_kind))
        .transpose()?
        .unwrap_or_default();
    let spec = AnalysisStudySpec {
        study_id,
        geometry,
        create_model_intent: AnalysisCreateModelIntentSpec {
            model_id,
            profile,
            prep_context: None,
        },
        model,
        run_kind,
        backend: options.backend.unwrap_or(ComputeBackend::Cpu),
        linear_static_run_options: run_options.linear_static,
        modal_run_options: run_options.modal,
        acoustic_run_options: run_options.acoustic,
        thermal_run_options: run_options.thermal,
        transient_run_options: run_options.transient,
        cfd_run_options: run_options.cfd,
        cht_run_options: run_options.cht,
        fsi_run_options: run_options.fsi,
        nonlinear_run_options: run_options.nonlinear,
        electromagnetic_run_options: run_options.electromagnetic,
    };
    study_to_object(spec)
}

impl StudyConstructorOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(builtin_error(
                STUDY_NAME,
                &ERROR_INPUT,
                "fea.study options must be Name, Value pairs",
            ));
        }
        let mut options = Self::default();
        for pair in args.chunks(2) {
            let key = option_key(&pair[0], STUDY_NAME)?;
            match key.as_str() {
                "runkind" | "kind" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.run_kind = Some(parse_scalar_enum(&text, "RunKind")?);
                }
                "profile" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.profile = Some(parse_scalar_enum(&text, "Profile")?);
                }
                "backend" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.backend = Some(parse_scalar_enum(&text, "Backend")?);
                }
                "modelid" => {
                    options.model_id = Some(scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?);
                }
                "model" => {
                    options.model = Some(model_from_value(STUDY_NAME, &pair[1])?);
                }
                "frame" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.frame = Some(parse_scalar_enum(&text, "Frame")?);
                }
                "defaults" => {
                    options.model_defaults = parse_model_defaults_mode(&scalar_string(
                        &pair[1],
                        STUDY_NAME,
                        &ERROR_INPUT,
                    )?)?;
                }
                "materials" => options.materials = material_vec_from_value(STUDY_NAME, &pair[1])?,
                "materialassignments" | "assignments" => {
                    options.material_assignments =
                        material_assignment_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "boundaryconditions" | "bcs" => {
                    options.boundary_conditions =
                        boundary_condition_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "loads" | "loadcases" => {
                    options.loads = load_case_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "steps" => options.steps = step_vec_from_value(STUDY_NAME, &pair[1])?,
                "domains" => options.domains = domain_vec_from_value(STUDY_NAME, &pair[1])?,
                "interfaces" => {
                    options.interfaces = interface_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "runoptions" | "options" => {
                    options.run_options =
                        Some(run_options_payload_from_value(STUDY_NAME, &pair[1])?);
                }
                other => {
                    return Err(builtin_error(
                        STUDY_NAME,
                        &ERROR_INPUT,
                        format!("unsupported fea.study option `{other}`"),
                    ));
                }
            }
        }
        Ok(options)
    }

    fn has_model_components(&self) -> bool {
        self.frame.is_some()
            || !self.materials.is_empty()
            || !self.material_assignments.is_empty()
            || !self.boundary_conditions.is_empty()
            || !self.loads.is_empty()
            || !self.steps.is_empty()
            || !self.domains.is_empty()
            || !self.interfaces.is_empty()
    }
}

#[derive(Debug, Default)]
struct ModelConstructorOptions {
    profile: Option<AnalysisCreateModelProfile>,
    frame: Option<ReferenceFrame>,
    defaults: ModelDefaultsMode,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    boundary_conditions: Vec<BoundaryCondition>,
    loads: Vec<LoadCase>,
    steps: Vec<AnalysisStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
}

#[derive(Debug, Default)]
struct ResolvedRunOptions {
    linear_static: Option<AnalysisRunOptions>,
    modal: Option<AnalysisModalRunOptions>,
    acoustic: Option<AnalysisAcousticRunOptions>,
    thermal: Option<AnalysisThermalRunOptions>,
    transient: Option<AnalysisTransientRunOptions>,
    cfd: Option<AnalysisCfdRunOptions>,
    cht: Option<AnalysisChtRunOptions>,
    fsi: Option<AnalysisFsiRunOptions>,
    nonlinear: Option<AnalysisNonlinearRunOptions>,
    electromagnetic: Option<AnalysisElectromagneticRunOptions>,
}

fn create_sweep_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            SWEEP_NAME,
            &ERROR_INPUT,
            "fea.sweep requires id and studies arguments",
        ));
    }
    let sweep_id = scalar_string(&args[0], SWEEP_NAME, &ERROR_INPUT)?;
    let studies = study_vec_from_value(SWEEP_NAME, &args[1])?;
    let mut fail_fast = true;
    for pair in expect_name_value_tail(SWEEP_NAME, &args[2..])? {
        match pair.key.as_str() {
            "failfast" => fail_fast = bool_from_value(SWEEP_NAME, pair.value)?,
            other => {
                return Err(builtin_error(
                    SWEEP_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.sweep option `{other}`"),
                ));
            }
        }
    }
    sweep_to_object(AnalysisStudySweepSpec {
        sweep_id,
        studies,
        fail_fast,
    })
}

fn create_model_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            "fea.model requires id and geometry arguments",
        ));
    }
    let model_id = scalar_string(&args[0], MODEL_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(&args[1])?;
    let options = parse_model_constructor_options(MODEL_NAME, &args[2..])?;
    let profile = options
        .profile
        .unwrap_or(AnalysisCreateModelProfile::LinearStaticStructural);
    let model = build_model_from_parts(
        MODEL_NAME,
        &geometry,
        model_id,
        profile,
        options.defaults,
        options.frame,
        options.materials,
        options.material_assignments,
        options.boundary_conditions,
        options.loads,
        options.steps,
        options.domains,
        options.interfaces,
    )?;
    serializable_to_object(
        MODEL_NAME,
        &ERROR_INTERNAL,
        FEA_MODEL_CLASS,
        &model,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn parse_model_constructor_options(
    builtin: &'static str,
    args: &[Value],
) -> BuiltinResult<ModelConstructorOptions> {
    let mut options = ModelConstructorOptions::default();
    for pair in expect_name_value_tail(builtin, args)? {
        match pair.key.as_str() {
            "profile" => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.profile = Some(parse_scalar_enum(&text, "Profile")?);
            }
            "frame" => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.frame = Some(parse_scalar_enum(&text, "Frame")?);
            }
            "defaults" => {
                options.defaults =
                    parse_model_defaults_mode(&scalar_string(pair.value, builtin, &ERROR_INPUT)?)?;
            }
            "materials" => options.materials = material_vec_from_value(builtin, pair.value)?,
            "materialassignments" | "assignments" => {
                options.material_assignments =
                    material_assignment_vec_from_value(builtin, pair.value)?;
            }
            "boundaryconditions" | "bcs" => {
                options.boundary_conditions =
                    boundary_condition_vec_from_value(builtin, pair.value)?;
            }
            "loads" | "loadcases" => options.loads = load_case_vec_from_value(builtin, pair.value)?,
            "steps" => options.steps = step_vec_from_value(builtin, pair.value)?,
            "domains" => options.domains = domain_vec_from_value(builtin, pair.value)?,
            "interfaces" => options.interfaces = interface_vec_from_value(builtin, pair.value)?,
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported {builtin} option `{other}`"),
                ));
            }
        }
    }
    Ok(options)
}

fn create_material_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            MATERIAL_NAME,
            &ERROR_INPUT,
            "fea.material requires a material id",
        ));
    }
    let material_id = scalar_string(&args[0], MATERIAL_NAME, &ERROR_INPUT)?;
    let mut fields = json_fields_from_name_values(MATERIAL_NAME, &args[1..])?;
    let name = fields
        .remove("name")
        .map(json_to_string)
        .transpose()?
        .unwrap_or_else(|| material_id.clone());
    let mechanical = if let Some(value) = fields.remove("mechanical") {
        json_deserialize(MATERIAL_NAME, value, "mechanical material model")?
    } else {
        let youngs = remove_required_f64(&mut fields, MATERIAL_NAME, "youngs_modulus_pa")?;
        let poisson = remove_required_f64(&mut fields, MATERIAL_NAME, "poisson_ratio")?;
        MaterialMechanicalModel {
            youngs_modulus_pa: youngs,
            poisson_ratio: poisson,
        }
    };
    let thermal = if let Some(value) = fields.remove("thermal") {
        json_deserialize(MATERIAL_NAME, value, "thermal material model")?
    } else {
        let mut thermal = serde_json::to_value(MaterialThermalModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        move_known_fields(
            &mut fields,
            thermal.as_object_mut().expect("thermal model is object"),
            &[
                "reference_temperature_k",
                "modulus_temp_coeff_per_k",
                "conductivity_w_per_mk",
                "specific_heat_j_per_kgk",
                "expansion_coefficient_per_k",
            ],
        );
        json_deserialize(MATERIAL_NAME, thermal, "thermal material model")?
    };
    let electrical = if let Some(value) = fields.remove("electrical") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "electrical material model",
        )?)
    } else {
        let mut electrical = serde_json::to_value(MaterialElectricalModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        let moved = move_known_fields(
            &mut fields,
            electrical
                .as_object_mut()
                .expect("electrical material model is object"),
            &[
                "reference_temperature_k",
                "conductivity_s_per_m",
                "resistive_heating_coefficient",
                "relative_permittivity",
                "relative_permeability",
                "conductivity_frequency_response",
            ],
        );
        if moved {
            Some(json_deserialize(
                MATERIAL_NAME,
                electrical,
                "electrical material model",
            )?)
        } else {
            None
        }
    };
    let acoustic = if let Some(value) = fields.remove("acoustic") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "acoustic material model",
        )?)
    } else {
        let mut acoustic = serde_json::to_value(MaterialAcousticModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        let moved = move_known_fields(
            &mut fields,
            acoustic
                .as_object_mut()
                .expect("acoustic material model is object"),
            &[
                "density_kg_per_m3",
                "speed_of_sound_m_per_s",
                "damping_ratio",
            ],
        );
        if moved {
            Some(json_deserialize(
                MATERIAL_NAME,
                acoustic,
                "acoustic material model",
            )?)
        } else {
            None
        }
    };
    let plastic = if let Some(value) = fields.remove("plastic") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "plastic material model",
        )?)
    } else if fields.contains_key("yield_strain")
        || fields.contains_key("hardening_modulus_ratio")
        || fields.contains_key("saturation_exponent")
    {
        Some(MaterialPlasticModel {
            yield_strain: remove_required_f64(&mut fields, MATERIAL_NAME, "yield_strain")?,
            hardening_modulus_ratio: remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "hardening_modulus_ratio",
            )?,
            saturation_exponent: remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "saturation_exponent",
            )?,
        })
    } else {
        None
    };
    reject_unknown_fields(MATERIAL_NAME, fields)?;
    material_to_object(MaterialModel {
        material_id,
        name,
        mechanical,
        thermal,
        acoustic,
        electrical,
        plastic,
    })
}

fn create_material_assignment_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MATERIAL_ASSIGNMENT_NAME,
            &ERROR_INPUT,
            "fea.materialAssignment requires region and material arguments",
        ));
    }
    let region_id = scalar_string(&args[0], MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
    let assigned_material_id = scalar_string(&args[1], MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
    let mut expected_material_id = assigned_material_id.clone();
    let mut confidence = EvidenceConfidence::Verified;
    for pair in expect_name_value_tail(MATERIAL_ASSIGNMENT_NAME, &args[2..])? {
        match pair.key.as_str() {
            "expectedmaterial" | "expectedmaterialid" => {
                expected_material_id =
                    scalar_string(pair.value, MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
            }
            "confidence" => {
                let text = scalar_string(pair.value, MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
                confidence = parse_scalar_enum(&text, "Confidence")?;
            }
            other => {
                return Err(builtin_error(
                    MATERIAL_ASSIGNMENT_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.materialAssignment option `{other}`"),
                ));
            }
        }
    }
    material_assignment_to_object(MaterialAssignment {
        region_id,
        expected_material_id,
        assigned_material_id,
        confidence,
    })
}

fn create_boundary_condition_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            "fea.boundaryCondition requires id, region, and kind arguments",
        ));
    }
    let bc_id = scalar_string(&args[0], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let region_id = scalar_string(&args[1], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[2], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let mut fields = json_fields_from_name_values(BOUNDARY_CONDITION_NAME, &args[3..])?;
    let kind = match normalize_token(&kind_text).as_str() {
        "acousticimpedance" => BoundaryConditionKind::AcousticImpedance {
            specific_impedance_pa_s_per_m: remove_required_f64(
                &mut fields,
                BOUNDARY_CONDITION_NAME,
                "specific_impedance_pa_s_per_m",
            )?,
        },
        "thermalprescribedtemperature" => BoundaryConditionKind::ThermalPrescribedTemperature {
            temperature_k: remove_required_f64(
                &mut fields,
                BOUNDARY_CONDITION_NAME,
                "temperature_k",
            )?,
        },
        "thermalheatflux" => BoundaryConditionKind::ThermalHeatFlux {
            heat_flux_w_per_m2: remove_required_f64(
                &mut fields,
                BOUNDARY_CONDITION_NAME,
                "heat_flux_w_per_m2",
            )?,
        },
        "thermalconvection" => BoundaryConditionKind::ThermalConvection {
            ambient_temperature_k: remove_required_f64(
                &mut fields,
                BOUNDARY_CONDITION_NAME,
                "ambient_temperature_k",
            )?,
            coefficient_w_per_m2k: remove_required_f64(
                &mut fields,
                BOUNDARY_CONDITION_NAME,
                "coefficient_w_per_m2k",
            )?,
        },
        _ => parse_scalar_enum::<BoundaryConditionKind>(&kind_text, "BoundaryConditionKind")?,
    };
    reject_unknown_fields(BOUNDARY_CONDITION_NAME, fields)?;
    boundary_condition_to_object(BoundaryCondition {
        bc_id,
        region_id,
        kind,
    })
}

fn create_load_case_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            LOAD_CASE_NAME,
            &ERROR_INPUT,
            "fea.loadCase requires id, region, and kind arguments",
        ));
    }
    let load_id = scalar_string(&args[0], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let region_id = scalar_string(&args[1], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[2], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let mut fields = json_fields_from_name_values(LOAD_CASE_NAME, &args[3..])?;
    let kind = match normalize_token(&kind_text).as_str() {
        "force" => {
            let [fx, fy, fz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::Force { fx, fy, fz }
        }
        "pressure" => LoadKind::Pressure {
            magnitude_pa: remove_required_f64(&mut fields, LOAD_CASE_NAME, "magnitude_pa")?,
        },
        "bodyforce" => {
            let [gx, gy, gz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::BodyForce { gx, gy, gz }
        }
        "currentdensity" => {
            let [jx, jy, jz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::CurrentDensity {
                jx,
                jy,
                jz,
                phase_rad: remove_optional_f64(&mut fields, "phase_rad")?.unwrap_or_default(),
                amplitude_scale: remove_optional_f64(&mut fields, "amplitude_scale")?
                    .unwrap_or(1.0),
            }
        }
        "coilcurrent" => LoadKind::CoilCurrent {
            current_a: remove_required_f64(&mut fields, LOAD_CASE_NAME, "current_a")?,
            phase_rad: remove_optional_f64(&mut fields, "phase_rad")?.unwrap_or_default(),
            amplitude_scale: remove_optional_f64(&mut fields, "amplitude_scale")?.unwrap_or(1.0),
        },
        "heatsource" => LoadKind::HeatSource {
            volumetric_w_per_m3: remove_required_f64(
                &mut fields,
                LOAD_CASE_NAME,
                "volumetric_w_per_m3",
            )?,
        },
        other => {
            return Err(builtin_error(
                LOAD_CASE_NAME,
                &ERROR_INPUT,
                format!("unsupported load kind `{other}`"),
            ));
        }
    };
    reject_unknown_fields(LOAD_CASE_NAME, fields)?;
    load_case_to_object(LoadCase {
        load_id,
        region_id,
        kind,
    })
}

fn create_step_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            STEP_NAME,
            &ERROR_INPUT,
            "fea.step requires id and kind arguments",
        ));
    }
    let step_id = scalar_string(&args[0], STEP_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[1], STEP_NAME, &ERROR_INPUT)?;
    let kind = parse_scalar_enum::<AnalysisStepKind>(&kind_text, "AnalysisStepKind")?;
    step_to_object(AnalysisStep { step_id, kind })
}

fn create_domain_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            DOMAIN_NAME,
            &ERROR_INPUT,
            "fea.domain requires a domain kind",
        ));
    }
    let kind_text = scalar_string(&args[0], DOMAIN_NAME, &ERROR_INPUT)?;
    let kind = normalize_token(&kind_text);
    let fields = json_fields_from_name_values(DOMAIN_NAME, &args[1..])?;
    let payload = match kind.as_str() {
        "thermomechanical" => DomainPayload {
            kind: "thermo_mechanical".to_string(),
            data: json_with_overrides(
                DOMAIN_NAME,
                serde_json::json!({
                    "enabled": true,
                    "reference_temperature_k": 293.15,
                    "applied_temperature_delta_k": 0.0,
                    "field_artifact_id": null,
                    "field_source": null,
                    "region_temperature_deltas": [],
                    "time_profile": []
                }),
                fields,
                "thermo_mechanical domain",
            )?,
        },
        "electrothermal" => DomainPayload {
            kind: "electro_thermal".to_string(),
            data: json_with_overrides(
                DOMAIN_NAME,
                serde_json::json!({
                    "enabled": true,
                    "reference_temperature_k": 293.15,
                    "applied_voltage_v": 0.0,
                    "region_conductivity_scales": [],
                    "time_profile": []
                }),
                fields,
                "electro_thermal domain",
            )?,
        },
        "electromagnetic" => DomainPayload {
            kind: "electromagnetic".to_string(),
            data: json_with_overrides(
                DOMAIN_NAME,
                serde_json::json!({
                    "enabled": true,
                    "reference_frequency_hz": 0.0,
                    "applied_current_a": 0.0
                }),
                fields,
                "electromagnetic domain",
            )?,
        },
        "cfd" => DomainPayload {
            kind: "cfd".to_string(),
            data: json_with_overrides(
                DOMAIN_NAME,
                serde_json::json!({
                    "enabled": true,
                    "solve_family": "steady_state",
                    "reference_density_kg_per_m3": 1.225,
                    "dynamic_viscosity_pa_s": 1.8e-5,
                    "inlet_velocity_m_per_s": 0.0,
                    "turbulence_intensity": 0.0,
                    "time_profile": []
                }),
                fields,
                "cfd domain",
            )?,
        },
        other => {
            return Err(builtin_error(
                DOMAIN_NAME,
                &ERROR_INPUT,
                format!("unsupported FEA domain kind `{other}`"),
            ));
        }
    };
    domain_to_object(payload)
}

fn create_interface_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            INTERFACE_NAME,
            &ERROR_INPUT,
            "fea.interface requires id, primary region, and secondary region arguments",
        ));
    }
    let interface_id = scalar_string(&args[0], INTERFACE_NAME, &ERROR_INPUT)?;
    let primary_region_id = scalar_string(&args[1], INTERFACE_NAME, &ERROR_INPUT)?;
    let secondary_region_id = scalar_string(&args[2], INTERFACE_NAME, &ERROR_INPUT)?;
    let mut kind = "contact".to_string();
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(INTERFACE_NAME, &args[3..])? {
        if pair.key == "kind" {
            kind = scalar_string(pair.value, INTERFACE_NAME, &ERROR_INPUT)?;
        } else {
            fields.insert(
                canonical_field_name(&scalar_string(pair.name, INTERFACE_NAME, &ERROR_INPUT)?),
                value_to_json(INTERFACE_NAME, pair.value)?,
            );
        }
    }
    let kind = match normalize_token(&kind).as_str() {
        "contact" => AnalysisInterfaceKind::Contact(json_deserialize(
            INTERFACE_NAME,
            json_with_overrides(
                INTERFACE_NAME,
                serde_json::json!({
                    "penalty_stiffness_scale": 1.0,
                    "max_penetration_ratio": 0.0,
                    "friction_coefficient": 0.0
                }),
                fields,
                "contact interface",
            )?,
            "contact interface",
        )?),
        "fluid_structure" | "fluidstructure" | "fsi" => {
            AnalysisInterfaceKind::FluidStructure(json_deserialize(
                INTERFACE_NAME,
                json_with_overrides(
                    INTERFACE_NAME,
                    serde_json::json!({
                        "normal_stiffness_pa_per_m": 1.0e9,
                        "damping_ratio": 0.0,
                        "relaxation_factor": 0.5
                    }),
                    fields,
                    "fluid-structure interface",
                )?,
                "fluid-structure interface",
            )?)
        }
        other => {
            return Err(builtin_error(
                INTERFACE_NAME,
                &ERROR_INPUT,
                format!("unsupported interface kind `{other}`"),
            ));
        }
    };
    interface_to_object(AnalysisInterface {
        interface_id,
        primary_region_id,
        secondary_region_id,
        kind,
    })
}

fn create_run_options_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            RUN_OPTIONS_NAME,
            &ERROR_INPUT,
            "fea.runOptions requires a solver",
        ));
    }
    let kind_text = scalar_string(&args[0], RUN_OPTIONS_NAME, &ERROR_INPUT)?;
    let run_kind = parse_scalar_enum::<AnalysisRunKind>(&kind_text, "solver")?;
    let fields = json_fields_from_name_values(RUN_OPTIONS_NAME, &args[1..])?;
    let data = run_options_json_for_kind(RUN_OPTIONS_NAME, run_kind, fields)?;
    run_options_to_object(RunOptionsPayload {
        run_kind,
        options: data,
    })
}

fn create_results_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            RESULTS_NAME,
            &ERROR_INPUT,
            "fea.results requires a run id or fea.RunResult",
        ));
    }
    if let Value::Object(object) = &args[0] {
        if object.class_name == FEA_RESULTS_CLASS && args.len() == 1 {
            return Ok(args[0].clone());
        }
    }
    let run_id = run_id_from_value(RESULTS_NAME, &args[0])?;
    let query = results_query_from_args(&args[1..])?;
    operation_result_to_object(
        RESULTS_NAME,
        &ERROR_OPERATION,
        &ERROR_INTERNAL,
        FEA_RESULTS_CLASS,
        analysis_results_by_run_id_op(&run_id, query, OperationContext::new(None, None)),
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn create_field_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            FIELD_NAME,
            &ERROR_INPUT,
            "fea.field requires results/run input and field id",
        ));
    }
    let field_id = scalar_string(&args[1], FIELD_NAME, &ERROR_INPUT)?;
    let results = results_data_from_value(FIELD_NAME, &args[0])?;
    let field = results
        .fields
        .into_iter()
        .find(|field| field.field_id == field_id)
        .ok_or_else(|| {
            builtin_error(
                FIELD_NAME,
                &ERROR_INPUT,
                format!("FEA field `{field_id}` was not found in results"),
            )
        })?;
    serializable_to_object(
        FIELD_NAME,
        &ERROR_INTERNAL,
        FEA_FIELD_CLASS,
        &field,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn create_compare_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            COMPARE_NAME,
            &ERROR_INPUT,
            "fea.compare requires baseline and candidate run ids",
        ));
    }
    let baseline_run_id = scalar_string(&args[0], COMPARE_NAME, &ERROR_INPUT)?;
    let candidate_run_id = scalar_string(&args[1], COMPARE_NAME, &ERROR_INPUT)?;
    operation_result_to_object(
        COMPARE_NAME,
        &ERROR_OPERATION,
        &ERROR_INTERNAL,
        FEA_COMPARE_CLASS,
        analysis_results_compare_op(
            AnalysisResultsCompareQuery {
                baseline_run_id,
                candidate_run_id,
            },
            OperationContext::new(None, None),
        ),
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn create_trends_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    let mut window_size = AnalysisTrendsQuery::default().window_size;
    for pair in expect_name_value_tail(TRENDS_NAME, args.as_slice())? {
        match pair.key.as_str() {
            "windowsize" => window_size = usize_from_value(TRENDS_NAME, pair.value)?,
            other => {
                return Err(builtin_error(
                    TRENDS_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.trends option `{other}`"),
                ));
            }
        }
    }
    operation_result_to_object(
        TRENDS_NAME,
        &ERROR_OPERATION,
        &ERROR_INTERNAL,
        FEA_TRENDS_CLASS,
        analysis_trends_op(
            AnalysisTrendsQuery { window_size },
            OperationContext::new(None, None),
        ),
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn build_model_from_parts(
    builtin: &'static str,
    geometry: &GeometryAsset,
    model_id: String,
    profile: AnalysisCreateModelProfile,
    defaults: ModelDefaultsMode,
    frame: Option<ReferenceFrame>,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    boundary_conditions: Vec<BoundaryCondition>,
    loads: Vec<LoadCase>,
    steps: Vec<AnalysisStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
) -> BuiltinResult<AnalysisModel> {
    let mut model = match defaults {
        ModelDefaultsMode::ProfileScaffold => analysis_create_model_op(
            geometry,
            AnalysisCreateModelIntentSpec {
                model_id: model_id.clone(),
                profile,
                prep_context: None,
            },
            OperationContext::new(None, None),
        )
        .map(|envelope| envelope.data)
        .map_err(|err| operation_error(builtin, &ERROR_OPERATION, err))?,
        ModelDefaultsMode::None => empty_model(model_id, geometry),
    };

    if let Some(frame) = frame {
        model.frame = frame;
    }
    if !materials.is_empty() {
        model.materials = materials;
    }
    if !material_assignments.is_empty() {
        model.material_assignments = material_assignments
            .into_iter()
            .map(|mut assignment| {
                assignment.region_id = resolve_region_selector(&assignment.region_id, geometry)?;
                Ok(assignment)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !boundary_conditions.is_empty() {
        model.boundary_conditions = boundary_conditions
            .into_iter()
            .map(|mut bc| {
                bc.region_id = resolve_region_selector(&bc.region_id, geometry)?;
                Ok(bc)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !loads.is_empty() {
        model.loads = loads
            .into_iter()
            .map(|mut load| {
                load.region_id = resolve_region_selector(&load.region_id, geometry)?;
                Ok(load)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !steps.is_empty() {
        model.steps = steps;
    }
    for domain in domains {
        match domain.kind.as_str() {
            "thermo_mechanical" => {
                model.thermo_mechanical = Some(json_deserialize(
                    builtin,
                    domain.data,
                    "thermo_mechanical domain",
                )?);
            }
            "electro_thermal" => {
                model.electro_thermal = Some(json_deserialize(
                    builtin,
                    domain.data,
                    "electro_thermal domain",
                )?);
            }
            "electromagnetic" => {
                model.electromagnetic = Some(json_deserialize(
                    builtin,
                    domain.data,
                    "electromagnetic domain",
                )?);
            }
            "cfd" => {
                model.cfd = Some(json_deserialize(builtin, domain.data, "cfd domain")?);
            }
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported domain payload `{other}`"),
                ));
            }
        }
    }
    if !interfaces.is_empty() {
        model.interfaces = interfaces
            .into_iter()
            .map(|mut interface| {
                interface.primary_region_id =
                    resolve_region_selector(&interface.primary_region_id, geometry)?;
                interface.secondary_region_id =
                    resolve_region_selector(&interface.secondary_region_id, geometry)?;
                Ok(interface)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    Ok(model)
}

fn empty_model(model_id: String, geometry: &GeometryAsset) -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId(model_id),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: ReferenceFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        boundary_conditions: Vec::new(),
        loads: Vec::new(),
        steps: Vec::new(),
    }
}

fn resolve_region_selector(selector: &str, geometry: &GeometryAsset) -> BuiltinResult<String> {
    if let Some(id) = selector
        .strip_prefix("id:")
        .or_else(|| selector.strip_prefix("region:"))
    {
        return require_region_id(id, geometry);
    }
    if let Some(tag) = selector.strip_prefix("tag:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.tag.as_deref() == Some(tag))
            .map(|region| region.region_id.clone())
            .ok_or_else(|| {
                builtin_error(
                    MODEL_NAME,
                    &ERROR_INPUT,
                    format!("region tag `{tag}` was not found in geometry"),
                )
            });
    }
    if let Some(name) = selector.strip_prefix("name:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.name == name)
            .map(|region| region.region_id.clone())
            .ok_or_else(|| {
                builtin_error(
                    MODEL_NAME,
                    &ERROR_INPUT,
                    format!("region name `{name}` was not found in geometry"),
                )
            });
    }
    require_region_id(selector, geometry)
}

fn require_region_id(region_id: &str, geometry: &GeometryAsset) -> BuiltinResult<String> {
    geometry
        .regions
        .iter()
        .find(|region| region.region_id == region_id)
        .map(|region| region.region_id.clone())
        .ok_or_else(|| {
            builtin_error(
                MODEL_NAME,
                &ERROR_INPUT,
                format!("region id `{region_id}` was not found in geometry"),
            )
        })
}

fn material_to_object(material: MaterialModel) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_CLASS,
        &material,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn material_assignment_to_object(assignment: MaterialAssignment) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_ASSIGNMENT_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_ASSIGNMENT_CLASS,
        &assignment,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn boundary_condition_to_object(bc: BoundaryCondition) -> BuiltinResult<Value> {
    serializable_to_object(
        BOUNDARY_CONDITION_NAME,
        &ERROR_INTERNAL,
        FEA_BOUNDARY_CONDITION_CLASS,
        &bc,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn load_case_to_object(load: LoadCase) -> BuiltinResult<Value> {
    serializable_to_object(
        LOAD_CASE_NAME,
        &ERROR_INTERNAL,
        FEA_LOAD_CASE_CLASS,
        &load,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn step_to_object(step: AnalysisStep) -> BuiltinResult<Value> {
    serializable_to_object(
        STEP_NAME,
        &ERROR_INTERNAL,
        FEA_STEP_CLASS,
        &step,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn domain_to_object(domain: DomainPayload) -> BuiltinResult<Value> {
    serializable_to_object(
        DOMAIN_NAME,
        &ERROR_INTERNAL,
        FEA_DOMAIN_CLASS,
        &domain,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn interface_to_object(interface: AnalysisInterface) -> BuiltinResult<Value> {
    serializable_to_object(
        INTERFACE_NAME,
        &ERROR_INTERNAL,
        FEA_INTERFACE_CLASS,
        &interface,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn run_options_to_object(payload: RunOptionsPayload) -> BuiltinResult<Value> {
    serializable_to_object(
        RUN_OPTIONS_NAME,
        &ERROR_INTERNAL,
        FEA_RUN_OPTIONS_CLASS,
        &payload,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn model_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<AnalysisModel> {
    object_payload(builtin, value, FEA_MODEL_CLASS)
}

fn study_vec_from_value(
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

fn material_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<MaterialModel>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_CLASS)
}

fn material_assignment_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<MaterialAssignment>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_ASSIGNMENT_CLASS)
}

fn boundary_condition_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<BoundaryCondition>> {
    object_vec_from_value(builtin, value, FEA_BOUNDARY_CONDITION_CLASS)
}

fn load_case_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<LoadCase>> {
    object_vec_from_value(builtin, value, FEA_LOAD_CASE_CLASS)
}

fn step_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<AnalysisStep>> {
    object_vec_from_value(builtin, value, FEA_STEP_CLASS)
}

fn domain_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<DomainPayload>> {
    object_vec_from_value(builtin, value, FEA_DOMAIN_CLASS)
}

fn interface_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<AnalysisInterface>> {
    object_vec_from_value(builtin, value, FEA_INTERFACE_CLASS)
}

fn object_vec_from_value<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
) -> BuiltinResult<Vec<T>> {
    object_vec_from_value_with_property(builtin, value, expected_class, FEA_PAYLOAD_JSON_PROPERTY)
}

fn object_vec_from_value_with_property<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
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

fn object_payload<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
) -> BuiltinResult<T> {
    object_payload_with_property(builtin, value, expected_class, FEA_PAYLOAD_JSON_PROPERTY)
}

fn object_payload_with_property<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
    payload_property: &'static str,
) -> BuiltinResult<T> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class} object"),
        ));
    };
    if object.class_name != expected_class {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class}, got {}", object.class_name),
        ));
    }
    object_json_property(builtin, object, payload_property, &ERROR_INPUT)
}

fn run_options_payload_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<RunOptionsPayload> {
    object_payload(builtin, value, FEA_RUN_OPTIONS_CLASS)
}

fn resolved_run_options_from_payload(
    builtin: &'static str,
    payload: RunOptionsPayload,
    expected_kind: AnalysisRunKind,
) -> BuiltinResult<ResolvedRunOptions> {
    if payload.run_kind != expected_kind {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "run options kind {:?} does not match selected study solver {:?}",
                payload.run_kind, expected_kind
            ),
        ));
    }
    let mut resolved = ResolvedRunOptions::default();
    match payload.run_kind {
        AnalysisRunKind::LinearStatic => {
            resolved.linear_static = Some(json_deserialize(
                builtin,
                payload.options,
                "linear_static run options",
            )?);
        }
        AnalysisRunKind::Modal => {
            resolved.modal = Some(json_deserialize(
                builtin,
                payload.options,
                "modal run options",
            )?);
        }
        AnalysisRunKind::Acoustic => {
            resolved.acoustic = Some(json_deserialize(
                builtin,
                payload.options,
                "acoustic run options",
            )?);
        }
        AnalysisRunKind::Thermal => {
            resolved.thermal = Some(json_deserialize(
                builtin,
                payload.options,
                "thermal run options",
            )?);
        }
        AnalysisRunKind::Transient => {
            resolved.transient = Some(json_deserialize(
                builtin,
                payload.options,
                "transient run options",
            )?);
        }
        AnalysisRunKind::Cfd => {
            resolved.cfd = Some(json_deserialize(
                builtin,
                payload.options,
                "cfd run options",
            )?);
        }
        AnalysisRunKind::Cht => {
            resolved.cht = Some(json_deserialize(
                builtin,
                payload.options,
                "cht run options",
            )?);
        }
        AnalysisRunKind::Fsi => {
            resolved.fsi = Some(json_deserialize(
                builtin,
                payload.options,
                "fsi run options",
            )?);
        }
        AnalysisRunKind::Nonlinear => {
            resolved.nonlinear = Some(json_deserialize(
                builtin,
                payload.options,
                "nonlinear run options",
            )?);
        }
        AnalysisRunKind::Electromagnetic => {
            resolved.electromagnetic = Some(json_deserialize(
                builtin,
                payload.options,
                "electromagnetic run options",
            )?);
        }
    }
    Ok(resolved)
}

fn run_options_json_for_kind(
    builtin: &'static str,
    run_kind: AnalysisRunKind,
    fields: serde_json::Map<String, serde_json::Value>,
) -> BuiltinResult<serde_json::Value> {
    match run_kind {
        AnalysisRunKind::LinearStatic => typed_json_with_overrides::<AnalysisRunOptions>(
            builtin,
            AnalysisRunOptions::default(),
            fields,
            "linear_static run options",
        ),
        AnalysisRunKind::Modal => typed_json_with_overrides::<AnalysisModalRunOptions>(
            builtin,
            AnalysisModalRunOptions::default(),
            fields,
            "modal run options",
        ),
        AnalysisRunKind::Acoustic => typed_json_with_overrides::<AnalysisAcousticRunOptions>(
            builtin,
            AnalysisAcousticRunOptions::default(),
            fields,
            "acoustic run options",
        ),
        AnalysisRunKind::Thermal => typed_json_with_overrides::<AnalysisThermalRunOptions>(
            builtin,
            AnalysisThermalRunOptions::default(),
            fields,
            "thermal run options",
        ),
        AnalysisRunKind::Transient => typed_json_with_overrides::<AnalysisTransientRunOptions>(
            builtin,
            AnalysisTransientRunOptions::default(),
            fields,
            "transient run options",
        ),
        AnalysisRunKind::Cfd => typed_json_with_overrides::<AnalysisCfdRunOptions>(
            builtin,
            AnalysisCfdRunOptions::default(),
            fields,
            "cfd run options",
        ),
        AnalysisRunKind::Cht => typed_json_with_overrides::<AnalysisChtRunOptions>(
            builtin,
            AnalysisChtRunOptions::default(),
            fields,
            "cht run options",
        ),
        AnalysisRunKind::Fsi => typed_json_with_overrides::<AnalysisFsiRunOptions>(
            builtin,
            AnalysisFsiRunOptions::default(),
            fields,
            "fsi run options",
        ),
        AnalysisRunKind::Nonlinear => typed_json_with_overrides::<AnalysisNonlinearRunOptions>(
            builtin,
            AnalysisNonlinearRunOptions::default(),
            fields,
            "nonlinear run options",
        ),
        AnalysisRunKind::Electromagnetic => {
            typed_json_with_overrides::<AnalysisElectromagneticRunOptions>(
                builtin,
                AnalysisElectromagneticRunOptions::default(),
                fields,
                "electromagnetic run options",
            )
        }
    }
}

fn results_query_from_args(args: &[Value]) -> BuiltinResult<AnalysisResultsQuery> {
    let mut query = AnalysisResultsQuery::default();
    for pair in expect_name_value_tail(RESULTS_NAME, args)? {
        match pair.key.as_str() {
            "includefields" | "fields" => {
                query.include_fields = string_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includefieldvalues" | "fieldvalues" => {
                query.include_field_values = bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "includediagnostics" => {
                query.include_diagnostics = bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "diagnosticcodes" => {
                query.diagnostic_codes = string_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includemodalresults" => {
                query.include_modal_results = bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "modeindices" => {
                query.mode_indices = usize_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includetransientresults" => {
                query.include_transient_results = bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "transientsnapshotindices" => {
                query.transient_snapshot_indices = usize_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includenonlinearresults" => {
                query.include_nonlinear_results = bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "includeelectromagneticresults" => {
                query.include_electromagnetic_results = bool_from_value(RESULTS_NAME, pair.value)?;
            }
            other => {
                return Err(builtin_error(
                    RESULTS_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.results option `{other}`"),
                ));
            }
        }
    }
    Ok(query)
}

fn run_id_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::Object(object) if object.class_name == FEA_RUN_RESULT_CLASS => {
            if let Some(Value::String(run_id)) = object.properties.get("run_id") {
                Ok(run_id.clone())
            } else {
                Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "fea.RunResult does not contain a run_id; sweep results expose run_entries",
                ))
            }
        }
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            scalar_string(value, builtin, &ERROR_INPUT)
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected run id string or fea.RunResult; got {other:?}"),
        )),
    }
}

fn results_data_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<crate::analysis::AnalysisResultsData> {
    match value {
        Value::Object(object) if object.class_name == FEA_RESULTS_CLASS => {
            object_json_property(builtin, object, FEA_PAYLOAD_JSON_PROPERTY, &ERROR_INPUT)
        }
        _ => {
            let run_id = run_id_from_value(builtin, value)?;
            analysis_results_by_run_id_op(
                &run_id,
                AnalysisResultsQuery::default(),
                OperationContext::new(None, None),
            )
            .map(|envelope| envelope.data)
            .map_err(|err| operation_error(builtin, &ERROR_OPERATION, err))
        }
    }
}

struct NameValuePair<'a> {
    name: &'a Value,
    key: String,
    value: &'a Value,
}

fn expect_name_value_tail<'a>(
    builtin: &'static str,
    args: &'a [Value],
) -> BuiltinResult<Vec<NameValuePair<'a>>> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin} options must be Name, Value pairs"),
        ));
    }
    args.chunks(2)
        .map(|pair| {
            let key = option_key(&pair[0], builtin)?;
            Ok(NameValuePair {
                name: &pair[0],
                key,
                value: &pair[1],
            })
        })
        .collect()
}

fn json_fields_from_name_values(
    builtin: &'static str,
    args: &[Value],
) -> BuiltinResult<serde_json::Map<String, serde_json::Value>> {
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(builtin, args)? {
        let raw = scalar_string(pair.name, builtin, &ERROR_INPUT)?;
        fields.insert(
            canonical_field_name(&raw),
            value_to_json(builtin, pair.value)?,
        );
    }
    Ok(fields)
}

fn option_key(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    Ok(normalize_token(&scalar_string(
        value,
        builtin,
        &ERROR_INPUT,
    )?))
}

fn normalize_token(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn canonical_field_name(text: &str) -> String {
    let mut out = String::new();
    let mut previous_lower_or_digit = false;
    for ch in text.chars() {
        if ch == '-' || ch == ' ' {
            if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
            }
            previous_lower_or_digit = false;
            continue;
        }
        if ch == '_' {
            if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
            }
            previous_lower_or_digit = false;
            continue;
        }
        if ch.is_ascii_uppercase() {
            if previous_lower_or_digit && !out.ends_with('_') {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            previous_lower_or_digit = false;
        } else if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            previous_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    match normalize_token(&out).as_str() {
        "youngsmoduluspa" => "youngs_modulus_pa".to_string(),
        "poissonratio" => "poisson_ratio".to_string(),
        "magnitude" | "magnitudepa" => "magnitude_pa".to_string(),
        "current" | "currenta" => "current_a".to_string(),
        "phase" | "phaserad" => "phase_rad".to_string(),
        "amplitudescale" => "amplitude_scale".to_string(),
        "deterministicmode" => "deterministic_mode".to_string(),
        "precisionmode" => "precision_mode".to_string(),
        "preconditionermode" => "preconditioner_mode".to_string(),
        "qualitypolicy" => "quality_policy".to_string(),
        "prepcalibrationprofile" => "prep_calibration_profile".to_string(),
        "prepartifactid" => "prep_artifact_id".to_string(),
        "sweepfrequencyhz" => "sweep_frequency_hz".to_string(),
        "sweepenabled" => "sweep_enabled".to_string(),
        _ => out.trim_matches('_').to_string(),
    }
}

fn value_to_json(builtin: &'static str, value: &Value) -> BuiltinResult<serde_json::Value> {
    match value {
        Value::Num(n) => json_number(builtin, *n),
        Value::Int(i) => Ok(serde_json::Value::Number(i.to_i64().into())),
        Value::Bool(b) => Ok(serde_json::Value::Bool(*b)),
        Value::String(s) => Ok(serde_json::Value::String(s.clone())),
        Value::CharArray(chars) if chars.rows == 1 => {
            Ok(serde_json::Value::String(chars.data.iter().collect()))
        }
        Value::StringArray(array) if array.data.len() == 1 => {
            Ok(serde_json::Value::String(array.data[0].clone()))
        }
        Value::StringArray(array) => Ok(serde_json::Value::Array(
            array
                .data
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        )),
        Value::Tensor(tensor) if tensor.data.len() == 1 => json_number(builtin, tensor.data[0]),
        Value::Tensor(tensor) => Ok(serde_json::Value::Array(
            tensor
                .data
                .iter()
                .map(|value| json_number(builtin, *value))
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Value::Cell(cell) => Ok(serde_json::Value::Array(
            cell.data
                .iter()
                .map(|item| value_to_json(builtin, item))
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Value::Struct(fields) => {
            let mut object = serde_json::Map::new();
            for (key, value) in &fields.fields {
                object.insert(canonical_field_name(key), value_to_json(builtin, value)?);
            }
            Ok(serde_json::Value::Object(object))
        }
        Value::Object(object) => {
            if let Some(Value::String(json)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) {
                serde_json::from_str(json).map_err(|err| {
                    builtin_error_with_source(builtin, &ERROR_INPUT, err.to_string(), err)
                })
            } else {
                let mut object_json = serde_json::Map::new();
                for (key, value) in &object.properties {
                    if key.starts_with("__runmat_") {
                        continue;
                    }
                    object_json.insert(canonical_field_name(key), value_to_json(builtin, value)?);
                }
                Ok(serde_json::Value::Object(object_json))
            }
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("cannot convert value to FEA JSON payload: {other:?}"),
        )),
    }
}

fn json_number(builtin: &'static str, value: f64) -> BuiltinResult<serde_json::Value> {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "FEA numeric option values must be finite JSON numbers",
            )
        })
}

fn typed_json_with_overrides<T: Serialize + DeserializeOwned>(
    builtin: &'static str,
    default: T,
    fields: serde_json::Map<String, serde_json::Value>,
    label: &str,
) -> BuiltinResult<serde_json::Value> {
    let base = serde_json::to_value(default)
        .map_err(|err| builtin_error(builtin, &ERROR_INTERNAL, err.to_string()))?;
    json_with_overrides(builtin, base, fields, label)
}

fn json_with_overrides(
    builtin: &'static str,
    mut base: serde_json::Value,
    fields: serde_json::Map<String, serde_json::Value>,
    label: &str,
) -> BuiltinResult<serde_json::Value> {
    let Some(object) = base.as_object_mut() else {
        return Err(builtin_error(
            builtin,
            &ERROR_INTERNAL,
            format!("{label} default payload is not an object"),
        ));
    };
    for (key, value) in fields {
        if !object.contains_key(&key) {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("unsupported {label} option `{key}`"),
            ));
        }
        object.insert(key, value);
    }
    Ok(base)
}

fn json_deserialize<T: DeserializeOwned>(
    builtin: &'static str,
    value: serde_json::Value,
    label: &str,
) -> BuiltinResult<T> {
    serde_json::from_value(value)
        .map_err(|err| builtin_error(builtin, &ERROR_INPUT, format!("invalid {label}: {err}")))
}

fn json_to_string(value: serde_json::Value) -> BuiltinResult<String> {
    serde_json::from_value(value).map_err(|err| {
        builtin_error(
            MATERIAL_NAME,
            &ERROR_INPUT,
            format!("invalid string option: {err}"),
        )
    })
}

fn remove_required_f64(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    builtin: &'static str,
    key: &str,
) -> BuiltinResult<f64> {
    let Some(value) = fields.remove(key) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("missing required option `{key}`"),
        ));
    };
    serde_json::from_value(value).map_err(|err| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("invalid numeric option `{key}`: {err}"),
        )
    })
}

fn remove_optional_f64(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> BuiltinResult<Option<f64>> {
    fields
        .remove(key)
        .map(|value| {
            serde_json::from_value(value).map_err(|err| {
                builtin_error(
                    LOAD_CASE_NAME,
                    &ERROR_INPUT,
                    format!("invalid numeric option `{key}`: {err}"),
                )
            })
        })
        .transpose()
}

fn remove_required_vector3(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    builtin: &'static str,
    key: &str,
) -> BuiltinResult<[f64; 3]> {
    let Some(value) = fields.remove(key) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("missing required vector option `{key}`"),
        ));
    };
    let values: Vec<f64> = serde_json::from_value(value).map_err(|err| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("invalid vector option `{key}`: {err}"),
        )
    })?;
    if values.len() != 3 {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("vector option `{key}` must contain exactly 3 values"),
        ));
    }
    Ok([values[0], values[1], values[2]])
}

fn move_known_fields(
    source: &mut serde_json::Map<String, serde_json::Value>,
    target: &mut serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> bool {
    let mut moved = false;
    for key in keys {
        if let Some(value) = source.remove(*key) {
            target.insert((*key).to_string(), value);
            moved = true;
        }
    }
    moved
}

fn reject_unknown_fields(
    builtin: &'static str,
    fields: serde_json::Map<String, serde_json::Value>,
) -> BuiltinResult<()> {
    if fields.is_empty() {
        return Ok(());
    }
    let keys = fields.keys().cloned().collect::<Vec<_>>().join(", ");
    Err(builtin_error(
        builtin,
        &ERROR_INPUT,
        format!("unsupported option field(s): {keys}"),
    ))
}

fn bool_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<bool> {
    bool::try_from(value).map_err(|err| builtin_error(builtin, &ERROR_INPUT, err))
}

fn usize_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(int) => Ok(int.to_i64().max(0) as usize),
        Value::Num(n) if *n >= 0.0 => Ok(*n as usize),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected non-negative integer value; got {other:?}"),
        )),
    }
}

fn string_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| scalar_string(item, builtin, &ERROR_INPUT))
            .collect(),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::String(_) | Value::CharArray(_) => {
            Ok(vec![scalar_string(value, builtin, &ERROR_INPUT)?])
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected string, string array, or cell array of strings; got {other:?}"),
        )),
    }
}

fn usize_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(Tensor { data, .. }) => {
            Ok(data.iter().map(|value| *value as usize).collect())
        }
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| usize_from_value(builtin, item))
            .collect(),
        Value::Int(_) | Value::Num(_) => Ok(vec![usize_from_value(builtin, value)?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected numeric vector or cell array of indices; got {other:?}"),
        )),
    }
}

fn parse_model_defaults_mode(text: &str) -> BuiltinResult<ModelDefaultsMode> {
    match normalize_token(text).as_str() {
        "profilescaffold" | "scaffold" | "profile" => Ok(ModelDefaultsMode::ProfileScaffold),
        "none" | "empty" => Ok(ModelDefaultsMode::None),
        other => Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            format!("unsupported model defaults mode `{other}`"),
        )),
    }
}

fn resolved_document_to_object(document: FeaResolvedDocument) -> BuiltinResult<Value> {
    match document {
        FeaResolvedDocument::Study(spec) => study_to_object(*spec),
        FeaResolvedDocument::Sweep(spec) => sweep_to_object(spec),
    }
}

fn study_to_object(spec: AnalysisStudySpec) -> BuiltinResult<Value> {
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

fn sweep_to_object(spec: AnalysisStudySweepSpec) -> BuiltinResult<Value> {
    let mut object = serializable_to_object(
        LOAD_NAME,
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

fn operation_result_to_object<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
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

fn serializable_to_object<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    value: &T,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    ensure_fea_classes_registered();
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let converted = value_from_json(&json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.message().to_string(), err))?;
    let mut object = ObjectInstance::new(class_name.to_string());
    if let Value::Struct(fields) = converted {
        object.properties = fields.fields.into_iter().collect();
    } else {
        object.properties.insert("value".to_string(), converted);
    }
    if let Some(property) = hidden_json_property {
        object
            .properties
            .insert(property.to_string(), Value::String(json.to_string()));
    }
    Ok(Value::Object(object))
}

fn geometry_asset_from_value(value: &Value) -> BuiltinResult<GeometryAsset> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            format!("fea.study geometry must be {GEOMETRY_ASSET_CLASS}"),
        ));
    };
    if object.class_name != GEOMETRY_ASSET_CLASS {
        return Err(builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            format!(
                "fea.study geometry must be {GEOMETRY_ASSET_CLASS}, got {}",
                object.class_name
            ),
        ));
    }
    object_json_property(
        STUDY_NAME,
        object,
        GEOMETRY_ASSET_JSON_PROPERTY,
        &ERROR_INPUT,
    )
}

fn object_json_property<T: DeserializeOwned>(
    builtin: &'static str,
    object: &ObjectInstance,
    property: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    let Some(Value::String(json)) = object.properties.get(property) else {
        return Err(builtin_error(
            builtin,
            error,
            format!(
                "{} is missing required runtime payload property `{property}`",
                object.class_name
            ),
        ));
    };
    serde_json::from_str(json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))
}

fn scalar_string(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    String::try_from(value).map_err(|err| builtin_error(builtin, error, err))
}

fn parse_scalar_enum<T: DeserializeOwned>(text: &str, label: &str) -> BuiltinResult<T> {
    serde_yaml::from_str::<T>(&text.to_ascii_lowercase()).map_err(|err| {
        builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            format!("invalid {label} value `{text}`: {err}"),
        )
    })
}

fn default_profile_for_run_kind(run_kind: AnalysisRunKind) -> AnalysisCreateModelProfile {
    match run_kind {
        AnalysisRunKind::LinearStatic => AnalysisCreateModelProfile::LinearStaticStructural,
        AnalysisRunKind::Modal => AnalysisCreateModelProfile::ModalStructural,
        AnalysisRunKind::Acoustic => AnalysisCreateModelProfile::AcousticHarmonic,
        AnalysisRunKind::Thermal => AnalysisCreateModelProfile::ThermalStandalone,
        AnalysisRunKind::Transient => AnalysisCreateModelProfile::TransientStructural,
        AnalysisRunKind::Cfd => AnalysisCreateModelProfile::CfdSteadyState,
        AnalysisRunKind::Cht => AnalysisCreateModelProfile::ChtCoupled,
        AnalysisRunKind::Fsi => AnalysisCreateModelProfile::FsiCoupled,
        AnalysisRunKind::Nonlinear => AnalysisCreateModelProfile::NonlinearStructural,
        AnalysisRunKind::Electromagnetic => AnalysisCreateModelProfile::ElectromagneticStatic,
    }
}

fn resolve_study_profile_and_run_kind(
    options: &StudyConstructorOptions,
) -> BuiltinResult<(AnalysisCreateModelProfile, AnalysisRunKind)> {
    let profile = match (options.profile, options.run_kind) {
        (Some(profile), _) => profile,
        (None, Some(run_kind)) => default_profile_for_run_kind(run_kind),
        (None, None) => AnalysisCreateModelProfile::LinearStaticStructural,
    };
    let run_kind = profile.derived_run_kind();
    if let Some(explicit_run_kind) = options.run_kind {
        if explicit_run_kind != run_kind {
            return Err(builtin_error(
                STUDY_NAME,
                &ERROR_INPUT,
                format!(
                    "explicit solver {:?} does not match Profile {:?}; omit RunKind or choose a matching Profile",
                    explicit_run_kind, profile
                ),
            ));
        }
    }
    Ok((profile, run_kind))
}

fn ensure_fea_classes_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        let workflow_methods = workflow_methods();
        for class_name in [FEA_STUDY_CLASS, FEA_SWEEP_CLASS] {
            runmat_builtins::register_class(ClassDef {
                name: class_name.to_string(),
                parent: None,
                properties: HashMap::new(),
                methods: workflow_methods.clone(),
            });
        }
        runmat_builtins::register_class(ClassDef {
            name: FEA_RUN_RESULT_CLASS.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: run_result_methods(),
        });
        runmat_builtins::register_class(ClassDef {
            name: FEA_RESULTS_CLASS.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: results_methods(),
        });
        for class_name in [FEA_VALIDATION_CLASS, FEA_PLAN_CLASS, FEA_RUN_RESULT_CLASS] {
            if class_name == FEA_RUN_RESULT_CLASS {
                continue;
            }
            runmat_builtins::register_class(ClassDef {
                name: class_name.to_string(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::new(),
            });
        }
        for class_name in [
            FEA_MODEL_CLASS,
            FEA_MATERIAL_CLASS,
            FEA_MATERIAL_ASSIGNMENT_CLASS,
            FEA_BOUNDARY_CONDITION_CLASS,
            FEA_LOAD_CASE_CLASS,
            FEA_STEP_CLASS,
            FEA_DOMAIN_CLASS,
            FEA_INTERFACE_CLASS,
            FEA_RUN_OPTIONS_CLASS,
            FEA_FIELD_CLASS,
            FEA_COMPARE_CLASS,
            FEA_TRENDS_CLASS,
        ] {
            runmat_builtins::register_class(ClassDef {
                name: class_name.to_string(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::new(),
            });
        }
    });
}

fn workflow_methods() -> HashMap<String, MethodDef> {
    [
        ("validate", VALIDATE_NAME),
        ("plan", PLAN_NAME),
        ("run", RUN_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.to_string(),
            MethodDef {
                name: name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: function_name.to_string(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}

fn run_result_methods() -> HashMap<String, MethodDef> {
    [("results", RESULTS_NAME), ("field", FIELD_NAME)]
        .into_iter()
        .map(|(name, function_name)| {
            (
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: function_name.to_string(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
}

fn results_methods() -> HashMap<String, MethodDef> {
    [("field", FIELD_NAME)]
        .into_iter()
        .map(|(name, function_name)| {
            (
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: function_name.to_string(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
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
                .unwrap_or(ERROR_OPERATION.identifier.expect("descriptor identifier")),
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
        .with_identifier(
            error
                .identifier
                .unwrap_or(ERROR_INTERNAL.identifier.expect("descriptor identifier")),
        )
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
        .with_identifier(
            error
                .identifier
                .unwrap_or(ERROR_INTERNAL.identifier.expect("descriptor identifier")),
        )
        .with_source(source)
        .build()
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::CellArray;

    const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
    const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

    fn cell(values: Vec<Value>) -> Value {
        let cols = values.len().max(1);
        Value::Cell(CellArray::new(values, 1, cols).expect("cell should build"))
    }

    fn force_vector() -> Value {
        Value::Tensor(Tensor::new_2d(vec![0.0, -1000.0, 0.0], 1, 3).expect("tensor should build"))
    }

    #[test]
    fn fea_study_requires_geometry_asset() {
        let err = block_on(fea_study_builtin(vec![
            Value::String("demo".to_string()),
            Value::Num(1.0),
        ]))
        .expect_err("invalid geometry should fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    }

    #[test]
    fn fea_load_validate_and_plan_document_workflow() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
            .expect("geometry fixture should write");
        let fea_path = tmp.path().join("bracket.fea");
        std::fs::write(
            &fea_path,
            r#"
version: 1
kind: study
id: bracket_static
geometry:
  path: part.stl
  units: meter
model:
  profile: linear_static_structural
run:
  backend: cpu
"#,
        )
        .expect("FEA fixture should write");

        let study = block_on(fea_load_builtin(fea_path.to_string_lossy().to_string()))
            .expect("FEA document should load");
        let Value::Object(study_object) = study.clone() else {
            panic!("expected loaded FEA study object");
        };
        assert_eq!(study_object.class_name, FEA_STUDY_CLASS);
        assert!(study_object
            .properties
            .contains_key(FEA_STUDY_SPEC_JSON_PROPERTY));

        let validation =
            block_on(fea_validate_builtin(study.clone())).expect("FEA study should validate");
        let Value::Object(validation_object) = validation else {
            panic!("expected validation object");
        };
        assert_eq!(validation_object.class_name, FEA_VALIDATION_CLASS);
        assert_eq!(
            validation_object.properties.get("valid"),
            Some(&Value::Bool(true))
        );

        let plan = block_on(fea_plan_builtin(study)).expect("FEA study should plan");
        let Value::Object(plan_object) = plan else {
            panic!("expected plan object");
        };
        assert_eq!(plan_object.class_name, FEA_PLAN_CLASS);
        assert!(plan_object.properties.contains_key("operation_sequence"));
    }

    #[test]
    fn typed_constructors_build_full_study_and_sweep_objects() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");

        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            geometry_path.to_string_lossy().to_string(),
        ))
        .expect("geometry should load");
        let asset = geometry_asset_from_value(&geometry).expect("geometry payload should decode");
        let region_id = asset
            .regions
            .first()
            .expect("fixture should import a region")
            .region_id
            .clone();

        let material = block_on(fea_material_builtin(vec![
            Value::String("steel".to_string()),
            Value::String("YoungsModulusPa".to_string()),
            Value::Num(200e9),
            Value::String("PoissonRatio".to_string()),
            Value::Num(0.30),
        ]))
        .expect("material should build");
        assert_object_class(&material, FEA_MATERIAL_CLASS);

        let assignment = block_on(fea_material_assignment_builtin(vec![
            Value::String(region_id.clone()),
            Value::String("steel".to_string()),
        ]))
        .expect("material assignment should build");
        assert_object_class(&assignment, FEA_MATERIAL_ASSIGNMENT_CLASS);

        let fixed = block_on(fea_boundary_condition_builtin(vec![
            Value::String("fixed_base".to_string()),
            Value::String(region_id.clone()),
            Value::String("fixed".to_string()),
        ]))
        .expect("boundary condition should build");
        assert_object_class(&fixed, FEA_BOUNDARY_CONDITION_CLASS);

        let load = block_on(fea_load_case_builtin(vec![
            Value::String("tip_force".to_string()),
            Value::String(region_id),
            Value::String("force".to_string()),
            Value::String("Vector".to_string()),
            force_vector(),
        ]))
        .expect("load case should build");
        assert_object_class(&load, FEA_LOAD_CASE_CLASS);

        let step = block_on(fea_step_builtin(vec![
            Value::String("static_step".to_string()),
            Value::String("static".to_string()),
        ]))
        .expect("analysis step should build");
        assert_object_class(&step, FEA_STEP_CLASS);

        let model = block_on(fea_model_builtin(vec![
            Value::String("bracket_static_model".to_string()),
            geometry.clone(),
            Value::String("Defaults".to_string()),
            Value::String("none".to_string()),
            Value::String("Profile".to_string()),
            Value::String("linear_static_structural".to_string()),
            Value::String("Materials".to_string()),
            cell(vec![material]),
            Value::String("MaterialAssignments".to_string()),
            cell(vec![assignment]),
            Value::String("BoundaryConditions".to_string()),
            cell(vec![fixed]),
            Value::String("Loads".to_string()),
            cell(vec![load]),
            Value::String("Steps".to_string()),
            cell(vec![step]),
        ]))
        .expect("model should build");
        assert_object_class(&model, FEA_MODEL_CLASS);

        let run_options = block_on(fea_run_options_builtin(vec![
            Value::String("linear_static".to_string()),
            Value::String("DeterministicMode".to_string()),
            Value::Bool(true),
            Value::String("PrecisionMode".to_string()),
            Value::String("fp64".to_string()),
            Value::String("QualityPolicy".to_string()),
            Value::String("balanced".to_string()),
        ]))
        .expect("run options should build");
        assert_object_class(&run_options, FEA_RUN_OPTIONS_CLASS);

        let study = block_on(fea_study_builtin(vec![
            Value::String("bracket_static".to_string()),
            geometry,
            Value::String("Profile".to_string()),
            Value::String("linear_static_structural".to_string()),
            Value::String("Backend".to_string()),
            Value::String("cpu".to_string()),
            Value::String("Model".to_string()),
            model,
            Value::String("RunOptions".to_string()),
            run_options,
        ]))
        .expect("study should build");
        assert_object_class(&study, FEA_STUDY_CLASS);

        let sweep = block_on(fea_sweep_builtin(vec![
            Value::String("bracket_sweep".to_string()),
            cell(vec![study]),
            Value::String("FailFast".to_string()),
            Value::Bool(false),
        ]))
        .expect("sweep should build");
        assert_object_class(&sweep, FEA_SWEEP_CLASS);
    }

    fn assert_object_class(value: &Value, expected: &str) {
        let Value::Object(object) = value else {
            panic!("expected object value");
        };
        assert_eq!(object.class_name, expected);
        assert!(
            object.properties.contains_key(FEA_PAYLOAD_JSON_PROPERTY)
                || object.properties.contains_key(FEA_STUDY_SPEC_JSON_PROPERTY)
                || object.properties.contains_key(FEA_SWEEP_SPEC_JSON_PROPERTY)
        );
    }
}
