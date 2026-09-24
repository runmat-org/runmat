use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, FormulationRequirement};
use runmat_analysis_fea::{
    run_linear_static_with_options, ComputeBackend, LinearStaticSolveOptions,
};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{
    analysis_complete_linear_static_run_op, analysis_prepare_linear_static_run_op,
    install_fea_solver_context, map_fea_run_error, AnalysisOperation, AnalysisRunOptions,
    AnalysisRunResult, StructuralLinearExecutor,
};

pub(super) fn analysis_run_linear_static_with_executor(
    model: &AnalysisModel,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    options: AnalysisRunOptions,
    context: OperationContext,
    executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let _solver_context = install_fea_solver_context();
    let prepared = analysis_prepare_linear_static_run_op(
        model,
        backend,
        formulation_requirement,
        options,
        context,
    )?;
    let run = if let Some(executor) = executor {
        let mesh = prepared.solver_mesh.as_ref().ok_or_else(|| {
            operation_error(
                AnalysisOperation::LinearStatic.name(),
                AnalysisOperation::LinearStatic.version(),
                &prepared.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_LINEAR_STATIC.PORTABLE_MESH_REQUIRED",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "portable structural execution requires a canonical solver mesh",
                BTreeMap::from([(
                    "analysis_model_id".to_string(),
                    prepared.workload.model.model_id.0.clone(),
                )]),
            )
        })?;
        executor
            .execute(&prepared.workload, mesh)
            .map_err(|message| {
                operation_error(
                    AnalysisOperation::LinearStatic.name(),
                    AnalysisOperation::LinearStatic.version(),
                    &prepared.context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_LINEAR_STATIC.PORTABLE_EXECUTION_FAILED",
                        error_type: OperationErrorType::Backend,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    message,
                    BTreeMap::from([(
                        "analysis_model_id".to_string(),
                        prepared.workload.model.model_id.0.clone(),
                    )]),
                )
            })?
            .run
    } else {
        run_linear_static_with_options(
            &prepared.workload.model,
            prepared.workload.backend,
            LinearStaticSolveOptions {
                formulation_requirement: prepared.workload.formulation_requirement,
                preconditioner_kind: prepared.workload.preconditioner,
                algebra_backend_kind: prepared.workload.algebra_backend,
                solver_mesh_artifact_path: prepared.options.solver_mesh_artifact_path.clone(),
                solver_mesh: prepared.solver_mesh.clone(),
                require_solver_mesh_for_solid: prepared.workload.require_solver_mesh_for_solid,
                initial_displacement: None,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            },
        )
        .map_err(|err| {
            map_fea_run_error(
                AnalysisOperation::LinearStatic,
                &prepared.workload.model,
                &prepared.context,
                err,
            )
        })?
    };
    analysis_complete_linear_static_run_op(prepared, run)
}
