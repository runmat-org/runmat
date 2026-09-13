use std::collections::BTreeMap;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::publication::{complete_study_run, CompletedStudyStep, PreparedStudyCompletion};
use super::{
    PreparedStructuralStudyIntent, PreparedStructuralStudyIntentStep,
    StructuralStudyCompletionIdentity,
};
use crate::analysis::{
    analysis_admit_linear_static_run_with_id, analysis_admit_transient_run_with_id,
    analysis_study_content_digest, meshing, prepare_linear_static_workload_template,
    prepare_study_semantics, study_fingerprint_from_digest, AnalysisSolverOptions,
    AnalysisStudyRunData, AnalysisStudySpec, PreparedStructuralLinearRun,
    StructuralStudyNumericalResult, StructuralStudySequenceNumericalResult,
    StructuralStudyTransientNumericalResult, ANALYSIS_RUN_STUDY_OPERATION,
    ANALYSIS_RUN_STUDY_OP_VERSION,
};

mod preparation;

pub use preparation::analysis_prepare_structural_study_intent_op;

/// Admits the remotely meshed and solved result of a path-free structural study intent.
///
/// Canonical mesh and evidence validation, boundary binding, numerical result admission, quality
/// classification, and persistence remain client-runtime responsibilities. The server allocation
/// identity deterministically fences local run publication. Retrying or concurrently attaching the
/// same completed server run therefore converges on one public run ID; collision checks in the
/// artifact store reject different content under that identity.
///
/// Mesh, run, and study-evidence artifacts are independent immutable publications because the
/// runtime does not yet have a transactional store spanning those domains. The mesh is published
/// first because its resolved path is part of normal run admission; a later failure can leave only
/// content-addressed, retry-safe objects, and retrying this operation converges on the same run ID.
pub fn analysis_complete_structural_study_intent_op(
    intent: PreparedStructuralStudyIntent,
    meshing_output: crate::meshing::MeshingExecutionOutput,
    result: StructuralStudyNumericalResult,
    completion_identity: StructuralStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    intent.validate().map_err(|message| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            message,
        )
    })?;
    meshing_output
        .evidence
        .validate(&meshing_output.artifact)
        .map_err(|error| {
            intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.INVALID_REMOTE_MESH_EVIDENCE",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                error.to_string(),
            )
        })?;
    let expected_source =
        runmat_meshing_core::StableDigest::from_bytes(*intent.geometry_source_digest.bytes());
    if meshing_output.artifact.geometry.source_digest != expected_source
        || meshing_output.artifact.geometry.geometry_revision
            != u64::from(intent.model.geometry_revision)
        || Some(&meshing_output.artifact.resolved_request) != intent.meshing_request.as_ref()
    {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.REMOTE_MESH_IDENTITY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "remote mesh source, geometry revision, or meshing request differs from the prepared study intent",
        ));
    }

    let aliases = runmat_analysis_core::BoundaryRegionAliases::new(
        meshing_output.boundary_region_ids.clone(),
    )
    .map_err(|error| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.BOUNDARY_RESOLUTION_FAILED",
            OperationErrorType::Validation,
            OperationErrorSeverity::Error,
            error.to_string(),
        )
    })?;
    let resolved_model =
        runmat_analysis_core::resolve_boundary_region_aliases(&intent.model, &aliases).map_err(
            |error| {
                intent_completion_error(
                    &intent,
                    "RM.FEA.RUN_STUDY.BOUNDARY_RESOLUTION_FAILED",
                    OperationErrorType::Validation,
                    OperationErrorSeverity::Error,
                    error.to_string(),
                )
            },
        )?;
    let persisted_mesh = meshing::persist_mesh_output(&meshing_output).map_err(|message| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.MESH_ARTIFACT_STORE_FAILED",
            OperationErrorType::Internal,
            OperationErrorSeverity::Error,
            message,
        )
    })?;
    match (intent.workloads.clone(), result) {
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence {
                workloads: authored_workloads,
            },
            StructuralStudyNumericalResult::LinearSequence(result),
        ) => complete_linear_sequence(
            intent,
            resolved_model,
            persisted_mesh,
            meshing_output.artifact,
            authored_workloads,
            result,
            completion_identity,
        ),
        (
            runmat_analysis_fea::execution::StructuralStudyWorkloads::DirectTransient {
                workload,
                ..
            },
            StructuralStudyNumericalResult::DirectTransient(result),
        ) => complete_direct_transient(
            intent,
            resolved_model,
            persisted_mesh,
            meshing_output.artifact,
            *workload,
            *result,
            completion_identity,
        ),
        _ => Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "remote structural-study result kind differs from its prepared workload kind",
        )),
    }
}

fn complete_linear_sequence(
    intent: PreparedStructuralStudyIntent,
    resolved_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: super::super::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralLinearWorkload,
    >,
    result: StructuralStudySequenceNumericalResult,
    completion_identity: StructuralStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let mut numerical_by_step = BTreeMap::new();
    for step in result.steps {
        if numerical_by_step
            .insert(step.step_id.clone(), step)
            .is_some()
        {
            return Err(intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "remote sequence contains a duplicate procedure-step result",
            ));
        }
    }
    let mut completion_steps = Vec::with_capacity(intent.steps.len());
    let mut authored_steps = Vec::with_capacity(intent.steps.len());
    for step in intent.steps.clone() {
        let numerical = numerical_by_step
            .remove(&step.procedure_step.step_id)
            .ok_or_else(|| {
                intent_completion_error(
                    &intent,
                    "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    OperationErrorType::Contract,
                    OperationErrorSeverity::Fatal,
                    "remote sequence omitted an authored procedure-step result",
                )
            })?;
        let mut workload = authored_workloads
            .remove(&step.procedure_step.step_id)
            .ok_or_else(|| {
                intent_completion_error(
                    &intent,
                    "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    OperationErrorType::Contract,
                    OperationErrorSeverity::Fatal,
                    "prepared linear workload is absent for an authored procedure step",
                )
            })?;
        workload.model = resolved_model.clone();
        let AnalysisSolverOptions::StructuralLinear(mut options) =
            step.authored_execution.solver_options.clone()
        else {
            return Err(intent_completion_error(
                &intent,
                "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                OperationErrorType::Contract,
                OperationErrorSeverity::Fatal,
                "linear sequence workload has non-linear authored solver options",
            ));
        };
        options.solver_mesh_artifact_path = persisted_mesh.artifact_path.clone();
        let prepared_linear = PreparedStructuralLinearRun {
            workload,
            solver_mesh: Some(solver_mesh.clone()),
            options: options.clone(),
            context: intent.context.clone(),
        };
        let assigned_run_id = completion_identity
            .local_run_id(intent.authored_study_digest, &step.procedure_step.step_id);
        let run = analysis_admit_linear_static_run_with_id(
            prepared_linear,
            numerical.result,
            Some(&assigned_run_id),
        )?;
        let mut authored_execution = step.authored_execution;
        authored_execution.solver_options =
            AnalysisSolverOptions::StructuralLinear(options.clone());
        authored_steps.push(super::PreparedStudyStep {
            authored_execution,
            procedure_step: step.procedure_step.clone(),
        });
        completion_steps.push(CompletedStudyStep {
            step_id: step.procedure_step.step_id,
            solver_result_identity: numerical.solver_result_identity,
            run,
            resolved_solver_options: AnalysisSolverOptions::StructuralLinear(options),
        });
    }
    if !numerical_by_step.is_empty() {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "remote sequence returned an unauthored procedure-step result",
        ));
    }
    if !authored_workloads.is_empty() {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "prepared linear workload inventory contains an unauthored step",
        ));
    }
    complete_study_run(
        PreparedStudyCompletion {
            context: intent.context,
            study_id: intent.study_id,
            model_template: intent.model_template,
            study_fingerprint: study_fingerprint_from_digest(intent.authored_study_digest),
            authored_study_digest: intent.authored_study_digest,
            model: resolved_model,
            steps: authored_steps,
            operation_sequence: intent.operation_sequence,
            solver_mesh_artifact_path: persisted_mesh.artifact_path,
            meshing_evidence_artifact_path: persisted_mesh.evidence_path,
        },
        completion_steps,
        result.states,
        result.frozen_project_digest,
    )
}

fn complete_direct_transient(
    intent: PreparedStructuralStudyIntent,
    resolved_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: super::super::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    mut workload: runmat_analysis_fea::execution::StructuralTransientWorkload,
    result: StructuralStudyTransientNumericalResult,
    completion_identity: StructuralStudyCompletionIdentity,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    if result.step_id != workload.step_id {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "remote direct-transient result names a different procedure step",
        ));
    }
    let [step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "direct-transient completion requires exactly one authored procedure step",
        ));
    };
    let AnalysisSolverOptions::StructuralTransient(mut options) =
        step.authored_execution.solver_options.clone()
    else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "direct-transient workload has non-transient authored solver options",
        ));
    };
    workload.model = resolved_model.clone();
    let recovered = runmat_analysis_fea::pipeline::transient::recover_completed_transient_run(
        runmat_analysis_fea::pipeline::transient::CompletedStructuralTransientRunInput {
            workload: &workload,
            solver_mesh: &solver_mesh,
            result: &result.result,
            numerical_receipt: &result.numerical_receipt,
        },
    )
    .map_err(|error| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.TRANSIENT_RESULT_ADMISSION_FAILED",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            error.to_string(),
        )
    })?;
    options.solver_mesh_artifact_path = persisted_mesh.artifact_path.clone();
    let assigned_run_id = completion_identity
        .local_run_id(intent.authored_study_digest, &step.procedure_step.step_id);
    let solver_result_identity = result.result.manifest().identity;
    let run = analysis_admit_transient_run_with_id(
        workload.backend,
        options.clone(),
        &intent.context,
        Some(solver_mesh.clone()),
        recovered,
        Some(&assigned_run_id),
    )?;
    let sequence_identity = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &resolved_model.sequence,
    )
    .map_err(|error| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            error.to_string(),
        )
    })?;
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence_identity,
        &step.procedure_step,
    )
    .map_err(|error| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            error.to_string(),
        )
    })?;
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let state_fields = super::super::collect_semantic_state_fields(&run.data);
    let states = step
        .procedure_step
        .state_outputs
        .iter()
        .map(|declaration| {
            runmat_analysis_fea::execution::CanonicalSemanticState::new(
                runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                    producer,
                    declaration: declaration.clone(),
                    topology_digest,
                    source_result: solver_result_identity,
                    payload:
                        runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                            fields: state_fields.clone(),
                        },
                },
            )
            .map_err(|error| {
                intent_completion_error(
                    &intent,
                    "RM.FEA.RUN_STUDY.STATE_INVALID",
                    OperationErrorType::Contract,
                    OperationErrorSeverity::Fatal,
                    error.to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let authored_execution = {
        let mut authored = step.authored_execution.clone();
        authored.solver_options = AnalysisSolverOptions::StructuralTransient(options.clone());
        authored
    };
    complete_study_run(
        PreparedStudyCompletion {
            context: intent.context,
            study_id: intent.study_id,
            model_template: intent.model_template,
            study_fingerprint: study_fingerprint_from_digest(intent.authored_study_digest),
            authored_study_digest: intent.authored_study_digest,
            model: resolved_model,
            steps: vec![super::PreparedStudyStep {
                authored_execution,
                procedure_step: step.procedure_step.clone(),
            }],
            operation_sequence: intent.operation_sequence,
            solver_mesh_artifact_path: persisted_mesh.artifact_path,
            meshing_evidence_artifact_path: persisted_mesh.evidence_path,
        },
        vec![CompletedStudyStep {
            step_id: result.step_id,
            solver_result_identity,
            run,
            resolved_solver_options: AnalysisSolverOptions::StructuralTransient(options),
        }],
        states,
        result.frozen_project_digest,
    )
}

/// Maps a placement-owned terminal failure into the structural-study operation contract.
pub fn structural_study_execution_failure(
    intent: &PreparedStructuralStudyIntent,
    failure: super::StructuralStudyExecutionFailure,
) -> OperationErrorEnvelope {
    let (error_code, error_type, severity, message) = match failure {
        super::StructuralStudyExecutionFailure::Failed(message) => (
            "RM.FEA.RUN_STUDY.REMOTE_EXECUTION_FAILED",
            OperationErrorType::Backend,
            OperationErrorSeverity::Error,
            message,
        ),
        super::StructuralStudyExecutionFailure::Cancelled => (
            "RM.FEA.RUN_STUDY.EXECUTION_CANCELLED",
            OperationErrorType::Cancelled,
            OperationErrorSeverity::Error,
            "structural study execution was cancelled".to_string(),
        ),
        super::StructuralStudyExecutionFailure::Indeterminate(message) => (
            "RM.FEA.RUN_STUDY.REMOTE_EXECUTION_INDETERMINATE",
            OperationErrorType::Internal,
            OperationErrorSeverity::Fatal,
            message,
        ),
    };
    intent_completion_error(intent, error_code, error_type, severity, message)
}

fn intent_completion_error(
    intent: &PreparedStructuralStudyIntent,
    error_code: &'static str,
    error_type: OperationErrorType,
    severity: OperationErrorSeverity,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &intent.context,
        OperationErrorSpec {
            error_code,
            error_type,
            retryable: error_type == OperationErrorType::Internal,
            severity,
        },
        message,
        BTreeMap::from([("study_id".to_string(), intent.study_id.clone())]),
    )
}
