use crate::analysis::analysis_plan_study_op;
use crate::analysis::analysis_plan_study_sweep_op;
use crate::analysis::analysis_run_study_sweep_op;
use crate::analysis::analysis_validate_study_op;
use crate::analysis::analysis_validate_study_sweep_op;
use crate::analysis::FeaResolvedDocument;
use crate::builtins::fea::author_study;
use crate::builtins::fea::boundary::create_boundary_condition_object_from_args;
use crate::builtins::fea::comparison::{
    create_compare_object_from_args, create_trends_object_from_args,
};
use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_INTERNAL, ERROR_OPERATION};
use crate::builtins::fea::contracts::identities::{
    FEA_PLAN_CLASS, FEA_VALIDATION_CLASS, PLAN_NAME, RUN_NAME, STUDY_NAME, VALIDATE_NAME,
};
use crate::builtins::fea::document::{load_document_object, resolve_document_input};
use crate::builtins::fea::domain_interface::{
    create_domain_object_from_args, create_interface_object_from_args,
};
use crate::builtins::fea::geometry::scalar_string;
use crate::builtins::fea::load_step::{
    create_load_case_object_from_args, create_step_object_from_args,
};
use crate::builtins::fea::material::{
    create_material_assignment_object_from_args, create_material_object_from_args,
};
use crate::builtins::fea::model::constructors::create_model_object_from_args;
use crate::builtins::fea::output::{
    operation_result_to_object, sweep_plan_result_to_object, sweep_run_result_to_object,
};
use crate::builtins::fea::plot::create_plot_from_args;
use crate::builtins::fea::results::constructors::{
    create_field_object_from_args, create_results_object_from_args,
};
use crate::builtins::fea::results::query::run_study_result_to_object;
use crate::builtins::fea::run_options::create_run_options_object_from_args;
use crate::builtins::fea::study::{create_study_object_from_args, create_sweep_object_from_args};
use crate::operations::OperationContext;
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;
use std::path::PathBuf;

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
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
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
    name = "fea.authorStudy",
    category = "fea",
    summary = "Author a typed FEA study from compact mesh authoring evidence.",
    keywords = "fea,study,author,mesh,evidence,agent",
    descriptor(crate::builtins::fea::FEA_AUTHOR_STUDY_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_author_study_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    author_study::create_author_study_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.sweep",
    category = "fea",
    summary = "Create a FEA study sweep from study objects.",
    keywords = "fea,sweep,study,run",
    descriptor(crate::builtins::fea::FEA_SWEEP_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
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
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
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
    integer_capabilities(crate::builtins::fea::FEA_MATERIAL_INTEGER_CAPABILITIES),
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
    descriptor(crate::builtins::fea::FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_material_assignment_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_material_assignment_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.boundaryCondition",
    category = "fea",
    summary = "Create a typed FEA boundary condition.",
    keywords = "fea,boundary,condition,region,prescribed,rotation",
    descriptor(crate::builtins::fea::FEA_BOUNDARY_CONDITION_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_boundary_condition_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_boundary_condition_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.loadCase",
    category = "fea",
    summary = "Create a typed FEA load case.",
    keywords = "fea,load,force,moment,torque,pressure,current",
    descriptor(crate::builtins::fea::FEA_LOAD_CASE_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_LOAD_CASE_INTEGER_CAPABILITIES),
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
    descriptor(crate::builtins::fea::FEA_STEP_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
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
    descriptor(crate::builtins::fea::FEA_DOMAIN_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_DOMAIN_INTEGER_CAPABILITIES),
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
    descriptor(crate::builtins::fea::FEA_INTERFACE_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_INTERFACE_INTEGER_CAPABILITIES),
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
    descriptor(crate::builtins::fea::FEA_RUN_OPTIONS_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_RUN_OPTIONS_INTEGER_CAPABILITIES),
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
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
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
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
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
        FeaResolvedDocument::Sweep(spec) => sweep_plan_result_to_object(
            analysis_plan_study_sweep_op(&spec, OperationContext::new(None, None)),
        ),
    }
}

#[runtime_builtin(
    name = "fea.run",
    category = "fea",
    summary = "Run a FEA study or sweep.",
    keywords = "fea,run,study,sweep,solve",
    descriptor(crate::builtins::fea::FEA_RUN_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_run_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, RUN_NAME).await? {
        FeaResolvedDocument::Study(spec) => run_study_result_to_object(&spec),
        FeaResolvedDocument::Sweep(spec) => sweep_run_result_to_object(
            analysis_run_study_sweep_op(&spec, OperationContext::new(None, None)),
        ),
    }
}

#[runtime_builtin(
    name = "fea.results",
    category = "fea",
    summary = "Load or project FEA run results for post-processing.",
    keywords = "fea,results,run_id,fields,diagnostics",
    descriptor(crate::builtins::fea::FEA_RESULTS_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_RESULTS_INTEGER_CAPABILITIES),
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
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_field_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_field_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.plot",
    category = "fea",
    summary = "Create a RunMat figure for an FEA result field on its geometry mesh.",
    keywords = "fea,plot,visualize,mesh,von_mises,stress,field",
    descriptor(crate::builtins::fea::FEA_PLOT_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_plot_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_plot_from_args(args)
}

#[runtime_builtin(
    name = "fea.compare",
    category = "fea",
    summary = "Compare two persisted FEA runs by run id.",
    keywords = "fea,compare,run_id,quality",
    descriptor(crate::builtins::fea::FEA_COMPARE_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
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
    integer_capabilities(crate::builtins::fea::FEA_TRENDS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_trends_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_trends_object_from_args(args)
}
