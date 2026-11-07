use std::collections::BTreeMap;

use runmat_analysis_core::AnalysisModel;
use runmat_analysis_fea::{
    run_linear_static_with_options, ComputeBackend, LinearStaticSolveOptions,
};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{
    analysis_admit_linear_static_run_with_id, analysis_prepare_linear_static_run_op,
    install_fea_solver_context, map_fea_run_error, publish_standalone_run, AnalysisOperation,
    AnalysisRunOptions, AnalysisRunResult, StructuralLinearExecutor,
};

pub(super) fn analysis_run_linear_static_with_executor(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisRunOptions,
    context: OperationContext,
    executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    publish_standalone_run(
        execute_linear_static_with_executor(model, backend, options, context, executor)?,
        "RM.FEA.RUN_LINEAR_STATIC.ARTIFACT_STORE_FAILED",
    )
}

pub(super) fn execute_linear_static_with_executor(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisRunOptions,
    context: OperationContext,
    executor: Option<&dyn StructuralLinearExecutor>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    use futures::FutureExt;
    execute_linear_static_with_capacity(model, backend, options, context, executor, None)
        .now_or_never()
        .expect("the synchronous structural host does not await capacity")
}

pub(super) async fn execute_linear_static_with_capacity(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisRunOptions,
    context: OperationContext,
    executor: Option<&dyn StructuralLinearExecutor>,
    capacity: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let future = async {
        use runmat_analysis_fea::accelerator_admission::{
            admit_accelerator, complete_accelerator_admission, AcceleratorFamily,
        };
        // A portable executor receives an already admitted workload: placement is decided here,
        // with the one shared policy, before the task is prepared.
        let admission = match executor {
            Some(_) => Some(
                admit_accelerator(AcceleratorFamily::StructuralStatic, backend).map_err(
                    |error| {
                        map_fea_run_error(
                            AnalysisOperation::LinearStatic,
                            model,
                            &context,
                            error.into(),
                        )
                    },
                )?,
            ),
            None => None,
        };
        let mut prepared = analysis_prepare_linear_static_run_op(
            model,
            admission
                .as_ref()
                .map_or(backend, |admission| admission.effective),
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
            let run = executor
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
                .run;
            if let Some(admission) = admission {
                let mut run = run;
                complete_accelerator_admission(
                    AcceleratorFamily::StructuralStatic,
                    admission,
                    &mut run,
                )
                .map_err(|error| {
                    map_fea_run_error(
                        AnalysisOperation::LinearStatic,
                        &prepared.workload.model,
                        &prepared.context,
                        error.into(),
                    )
                })?;
                // The published result reports the requested placement next to the backend that ran.
                prepared.workload.backend = run.requested_backend;
                run
            } else {
                run
            }
        } else {
            let solve_options = LinearStaticSolveOptions {
                preconditioner_kind: prepared.workload.preconditioner,
                algebra_backend_kind: prepared.workload.algebra_backend,
                solver_mesh_artifact_path: prepared.options.solver_mesh_artifact_path.clone(),
                solver_mesh: prepared.solver_mesh.clone(),
                require_solver_mesh_for_solid: prepared.workload.require_solver_mesh_for_solid,
                initial_displacement: None,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            };
            let result = if let Some(capacity) = capacity {
                runmat_analysis_fea::run_linear_static_with_options_tet4_owned(
                    &prepared.workload.model,
                    prepared.workload.backend,
                    solve_options,
                    usize::MAX,
                    |_phase, work| async move {
                        capacity
                            .ensure_capacity(work)
                            .await
                            .map_err(|cause| runmat_analysis_fea::FeaRunError::Usage(cause))
                    },
                )
                .await
            } else {
                run_linear_static_with_options(
                    &prepared.workload.model,
                    prepared.workload.backend,
                    solve_options,
                )
            };
            result.map_err(|err| {
                map_fea_run_error(
                    AnalysisOperation::LinearStatic,
                    &prepared.workload.model,
                    &prepared.context,
                    err,
                )
            })?
        };
        analysis_admit_linear_static_run_with_id(prepared, run, None, Default::default())
    };
    let progress = {
        let _context = install_fea_solver_context();
        runmat_analysis_fea::progress::capture_fea_progress_context()
    };
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|cx| {
        let _context = runmat_analysis_fea::replace_fea_progress_context(progress.clone());
        std::future::Future::poll(future.as_mut(), cx)
    })
    .await
}
