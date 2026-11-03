//! Parent-owned study preparation/publication with an owned numerical-step boundary.
use super::*;
use runmat_analysis_core::ProcedureStep;
use std::{future::Future, pin::Pin};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisStudyStepRequest {
    pub model: AnalysisModel,
    pub procedure_step: ProcedureStep,
    pub execution: AnalysisExecutionStep,
    pub solver_mesh_artifact_path: Option<String>,
    pub solver_mesh: Option<SolverMeshArtifact>,
    pub require_accelerator: bool,
    pub context: OperationContext,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisStudyStepResult {
    pub run: OperationEnvelope<AnalysisRunResult>,
    pub resolved_solver_options: AnalysisSolverOptions,
}

/// An owned worker must return numerical values, never its process-local
/// device tokens. Actual GPU execution/provenance remains unchanged.
pub fn analysis_validate_study_step_transfer(
    result: &AnalysisStudyStepResult,
) -> Result<(), OperationErrorEnvelope> {
    let mut device_field = None;
    operations::visit_analysis_result_fields(&result.run.data, |field| {
        if matches!(
            &field.values,
            runmat_analysis_core::AnalysisFieldValues::DeviceRef(_)
        ) && device_field.is_none()
        {
            device_field = Some(field.field_id.clone());
        }
    });
    if let Some(field) = device_field {
        return Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION,
            &OperationContext::new(result.run.trace_id.clone(), result.run.request_id.clone()),
            OperationErrorSpec { error_code: "RM.FEA.WORKER.PROCESS_LOCAL_FIELD", error_type: OperationErrorType::Contract, retryable: false, severity: OperationErrorSeverity::Error },
            "An owned numerical worker must materialize device fields before transferring its result.",
            BTreeMap::from([("field_id".into(), field)]),
        ));
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub type AnalysisStudyStepFuture<'a> = Pin<
    Box<dyn Future<Output = Result<AnalysisStudyStepResult, OperationErrorEnvelope>> + Send + 'a>,
>;
#[cfg(target_arch = "wasm32")]
pub type AnalysisStudyStepFuture<'a> =
    Pin<Box<dyn Future<Output = Result<AnalysisStudyStepResult, OperationErrorEnvelope>> + 'a>>;

#[cfg(not(target_arch = "wasm32"))]
pub trait AnalysisStudyStepHost: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> AnalysisStudyStepHost for T {}
#[cfg(target_arch = "wasm32")]
pub trait AnalysisStudyStepHost {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> AnalysisStudyStepHost for T {}

/// The host computes one original step; it does not publish runs or study state.
pub trait AnalysisStudyStepExecutor: AnalysisStudyStepHost {
    fn allocated_cpu_millicores(&self) -> u64;
    fn execute_step<'a>(&'a self, request: AnalysisStudyStepRequest)
        -> AnalysisStudyStepFuture<'a>;
}

/// Execute the same numerical dispatcher used by synchronous study execution.
/// The owning host must establish its authenticated parent-supervised FEA scope;
/// a serialized request alone is not an authorization capability.
pub fn analysis_execute_study_step(
    request: AnalysisStudyStepRequest,
) -> Result<AnalysisStudyStepResult, OperationErrorEnvelope> {
    if fea_run_admission::FeaRunScope::current_allocation().is_none() {
        return Err(
            fea_run_admission::FeaRunDenied::SignInRequired.into_operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &request.context,
            ),
        );
    }
    if request.execution.step_id != request.procedure_step.step_id
        || !request
            .model
            .sequence
            .steps
            .contains(&request.procedure_step)
        || !request.procedure_step.state_inputs.is_empty()
    {
        return Err(study_sequence_contract_error(&request.context, "owned numerical step differs from its original authored procedure or requires upstream state"));
    }
    let AnalysisStudyStepRequest {
        model,
        procedure_step,
        execution,
        solver_mesh_artifact_path,
        solver_mesh,
        require_accelerator,
        context,
    } = request;
    let execute = || {
        study_execution::execute_study_step_with_policy(
            &model,
            &procedure_step,
            &execution,
            solver_mesh_artifact_path.as_deref(),
            &context,
            None,
            require_accelerator,
        )
        .map(|(run, resolved_solver_options)| AnalysisStudyStepResult {
            run,
            resolved_solver_options,
        })
    };
    match (solver_mesh_artifact_path.as_ref(), solver_mesh) {
        (Some(path), Some(mesh)) => {
            use runmat_meshing_core::CanonicalMeshingContract;
            mesh.validate_canonical()
                .map_err(|error| study_sequence_contract_error(&context, error))?;
            solver_mesh_artifact::with_scoped_solver_mesh(path.into(), mesh, execute)
        }
        (None, None) => execute(),
        _ => Err(study_sequence_contract_error(
            &context,
            "owned study mesh path and original input differ",
        )),
    }
}

pub async fn analysis_execute_study_with_step_executor(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    executor: &dyn AnalysisStudyStepExecutor,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let scope = admit(
        analysis_study_content_digest(spec),
        &context,
        executor,
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
    )?;
    let result = scope
        .run_checked_result_future(execute_admitted(spec, context.clone(), executor))
        .await;
    match result {
        Ok(result) => {
            let classification = outcome(&result);
            fea_run_admission::entrypoints::settle_result(
                scope,
                result,
                classification,
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &context,
            )
        }
        Err(error) => {
            let classification = error.outcome();
            fea_run_admission::entrypoints::settle_result(
                scope,
                Err(error.into_operation_error(
                    ANALYSIS_RUN_STUDY_OPERATION,
                    ANALYSIS_RUN_STUDY_OP_VERSION,
                    &context,
                )),
                classification,
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &context,
            )
        }
    }
}

pub async fn analysis_execute_study_sweep_with_step_executor(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
    executor: &dyn AnalysisStudyStepExecutor,
) -> Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope> {
    let identity = analysis_study_sweep_identity(spec, &context)?;
    let scope = admit(
        identity.digest(),
        &context,
        executor,
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
    )?;
    let result = scope
        .run_checked_result_future(async {
            let mut outcomes = Vec::with_capacity(spec.studies.len());
            for study in &spec.studies {
                match execute_admitted(study, context.clone(), executor).await {
                    Ok(run) => {
                        outcomes.push(AnalysisStudySweepRunOutcome::Succeeded(Box::new(run)))
                    }
                    Err(error) => {
                        outcomes.push(
                            if error.error_type == OperationErrorType::Cancelled
                                || crate::interrupt::is_cancelled()
                            {
                                AnalysisStudySweepRunOutcome::Cancelled(error)
                            } else {
                                AnalysisStudySweepRunOutcome::Failed(error)
                            },
                        );
                        if spec.fail_fast {
                            outcomes.resize_with(spec.studies.len(), || {
                                AnalysisStudySweepRunOutcome::NotRun
                            });
                            break;
                        }
                    }
                }
            }
            analysis_complete_study_sweep_op(spec, context.clone(), outcomes)
        })
        .await;
    match result {
        Ok(result) => {
            let classification = match &result {
                Ok(run) => fea_run_admission::entrypoints::sweep_outcome(&run.data),
                Err(_) => outcome(&result),
            };
            fea_run_admission::entrypoints::settle_result(
                scope,
                result,
                classification,
                ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                &context,
            )
        }
        Err(error) => {
            let classification = error.outcome();
            fea_run_admission::entrypoints::settle_result(
                scope,
                Err(error.into_operation_error(
                    ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                    ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                    &context,
                )),
                classification,
                ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                &context,
            )
        }
    }
}

fn admit(
    identity: runmat_execution::Digest,
    context: &OperationContext,
    executor: &dyn AnalysisStudyStepExecutor,
    operation: &str,
    version: &str,
) -> Result<fea_run_admission::FeaRunScope, OperationErrorEnvelope> {
    use fea_run_admission::{FeaRunKind, FeaRunRequest, FeaRunScope};
    let mut request = FeaRunRequest::local(
        FeaRunKind::MeshAndSolve,
        executor.allocated_cpu_millicores(),
        identity,
    )
    .map_err(|e| e.into_operation_error(operation, version, context))?;
    request.site = FeaRunScope::admission_site()
        .map_err(|e| e.into_operation_error(operation, version, context))?;
    FeaRunScope::admit(request).map_err(|e| e.into_operation_error(operation, version, context))
}

fn outcome<T>(result: &Result<T, OperationErrorEnvelope>) -> fea_run_admission::FeaRunOutcome {
    use fea_run_admission::FeaRunOutcome;
    match result {
        Ok(_) => FeaRunOutcome::Completed,
        Err(error)
            if error.error_type == OperationErrorType::Cancelled
                || crate::interrupt::is_cancelled() =>
        {
            FeaRunOutcome::Cancelled
        }
        Err(_) => FeaRunOutcome::Failed,
    }
}

async fn execute_admitted(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    executor: &dyn AnalysisStudyStepExecutor,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let mut prepared = prepare_study_semantics(spec, context)?;
    let mesh = meshing::resolve_study_mesh_async(spec)
        .await
        .map_err(|message| {
            operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &prepared.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.MESHING_FAILED",
                    error_type: OperationErrorType::Backend,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                BTreeMap::new(),
            )
        })?;
    prepared.solver_mesh_artifact_path = mesh.artifact_path;
    prepared.meshing_evidence_artifact_path = mesh.evidence_path;
    let mut completion = StudyStepCompletion::new(&prepared)?;
    for step in &prepared.steps {
        let path = step
            .authored_execution
            .solver_options
            .solver_mesh_artifact_path()
            .or(prepared.solver_mesh_artifact_path.as_deref())
            .map(str::to_owned);
        let solver_mesh = path
            .as_ref()
            .map(|path| solver_mesh_artifact::load_solver_mesh_artifact(std::path::Path::new(path)))
            .transpose()
            .map_err(|e| study_sequence_contract_error(&prepared.context, e))?;
        let result = executor
            .execute_step(AnalysisStudyStepRequest {
                model: prepared.model.clone(),
                procedure_step: step.procedure_step.clone(),
                execution: step.authored_execution.clone(),
                solver_mesh_artifact_path: path,
                solver_mesh,
                require_accelerator: crate::context::current_acceleration_policy()
                    .is_some_and(|policy| !policy.allow_host_fallback),
                context: prepared.context.clone(),
            })
            .await?;
        completion.push(&prepared, step, result.run, result.resolved_solver_options)?;
    }
    completion.finish(prepared)
}

/// Both synchronous and owned-step placement use this exact identity/state/publication loop.
pub(super) struct StudyStepCompletion {
    sequence_identity: SequenceIdentity,
    topology_digest: runmat_execution::Digest,
    completed: Vec<CompletedStudyStep>,
    states: Vec<CanonicalSemanticState>,
}
impl StudyStepCompletion {
    pub(super) fn new(prepared: &PreparedStudySequence) -> Result<Self, OperationErrorEnvelope> {
        if prepared
            .steps
            .iter()
            .any(|step| !step.procedure_step.state_inputs.is_empty())
        {
            return Err(operation_error(ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION, &prepared.context,
                OperationErrorSpec { error_code: "RM.FEA.RUN_STUDY.SEQUENCE_EXECUTOR_REQUIRED", error_type: OperationErrorType::Validation, retryable: false, severity: OperationErrorSeverity::Error },
                "studies that consume upstream state require the canonical structural sequence executor", BTreeMap::from([("study_id".to_string(), prepared.spec.study_id.clone())])));
        }
        StudyIdentity::from_authored_digest(analysis_study_content_digest(&prepared.spec), None)
            .map_err(|e| study_sequence_contract_error(&prepared.context, e))?;
        Ok(Self {
            sequence_identity: SequenceIdentity::from_solver_contract(&prepared.model.sequence)
                .map_err(|e| study_sequence_contract_error(&prepared.context, e))?,
            topology_digest: in_process_study_topology_digest(prepared)?,
            completed: Vec::with_capacity(prepared.steps.len()),
            states: Vec::new(),
        })
    }
    pub(super) fn push(
        &mut self,
        prepared: &PreparedStudySequence,
        step: &PreparedStudyStep,
        run: OperationEnvelope<AnalysisRunResult>,
        resolved_solver_options: AnalysisSolverOptions,
    ) -> Result<(), OperationErrorEnvelope> {
        let step_identity =
            StepIdentity::from_solver_contract(self.sequence_identity, &step.procedure_step)
                .map_err(|e| study_sequence_contract_error(&prepared.context, e))?;
        let result_bytes = serde_json::to_vec(&run.data.run)
            .map_err(|e| study_sequence_contract_error(&prepared.context, e))?;
        let solver_result_identity = SolverResultIdentity::from_step_result(
            step_identity,
            runmat_execution::Digest::sha256(result_bytes),
        )
        .map_err(|e| study_sequence_contract_error(&prepared.context, e))?;
        for declaration in &step.procedure_step.state_outputs {
            self.states.push(
                CanonicalSemanticState::new(CanonicalSemanticStateInput {
                    producer: step_identity,
                    declaration: declaration.clone(),
                    topology_digest: self.topology_digest,
                    source_result: solver_result_identity,
                    payload: semantic_state_payload(&run.data, declaration)
                        .map_err(|e| study_sequence_contract_error(&prepared.context, e))?,
                })
                .map_err(|e| study_sequence_contract_error(&prepared.context, e))?,
            );
        }
        self.completed.push(CompletedStudyStep {
            step_id: step.procedure_step.step_id.clone(),
            solver_result_identity,
            run: CompletedStudyStepRun::Standard(Box::new(run)),
            resolved_solver_options,
        });
        Ok(())
    }
    pub(super) fn finish(
        self,
        prepared: PreparedStudySequence,
    ) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
        complete_study_run(
            study_completion_from_prepared(prepared),
            self.completed,
            self.states,
            None,
        )?
        .publish_legacy()
    }
}
