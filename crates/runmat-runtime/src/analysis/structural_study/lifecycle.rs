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
    analysis_admit_linear_static_run_with_id, analysis_study_content_digest, meshing,
    prepare_linear_static_workload_template, prepare_study_semantics,
    study_fingerprint_from_digest, AnalysisSolverOptions, AnalysisStudyRunData, AnalysisStudySpec,
    PreparedStructuralLinearRun, StructuralStudySequenceNumericalResult,
    ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION,
};

/// Resolves and validates structural study semantics without running meshing or loading a solver
/// mesh artifact. The returned intent is path-free and can be checkpointed by a remote study
/// coordinator before geometry admission begins.
pub fn analysis_prepare_structural_study_intent_op(
    spec: &AnalysisStudySpec,
    context: OperationContext,
) -> Result<PreparedStructuralStudyIntent, OperationErrorEnvelope> {
    let study = prepare_study_semantics(spec, context)?;
    let mut steps = Vec::with_capacity(study.steps.len());
    for step in &study.steps {
        let AnalysisSolverOptions::StructuralLinear(mut options) =
            step.authored_execution.solver_options.clone()
        else {
            return Err(operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &study.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.PORTABLE_FAMILY_UNSUPPORTED",
                    error_type: OperationErrorType::Validation,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                "portable structural study preparation requires structural-linear procedure steps",
                BTreeMap::from([
                    ("study_id".to_string(), study.spec.study_id.clone()),
                    (
                        "step_id".to_string(),
                        step.procedure_step.step_id.as_str().to_string(),
                    ),
                ]),
            ));
        };
        options.solver_mesh_artifact_path = None;
        let workload = prepare_linear_static_workload_template(
            &study.model,
            &step.procedure_step.step_id,
            step.authored_execution.backend,
            step.authored_execution.requirement,
            &options,
        )?;
        let mut authored_execution = step.authored_execution.clone();
        authored_execution.solver_options =
            AnalysisSolverOptions::StructuralLinear(options.clone());
        steps.push(PreparedStructuralStudyIntentStep {
            authored_execution,
            procedure_step: step.procedure_step.clone(),
            workload,
            options,
        });
    }
    let geometry_source_digest =
        crate::meshing::parse_source_sha256(&study.spec.geometry.source.sha256)
            .map(runmat_execution::Digest::from_bytes)
            .map_err(|message| {
                operation_error(
                    ANALYSIS_RUN_STUDY_OPERATION,
                    ANALYSIS_RUN_STUDY_OP_VERSION,
                    &study.context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                        error_type: OperationErrorType::Input,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    message,
                    BTreeMap::from([("study_id".to_string(), study.spec.study_id.clone())]),
                )
            })?;
    let mut intent = PreparedStructuralStudyIntent {
        study_id: study.spec.study_id.clone(),
        model_template: study.spec.create_model_intent.template,
        geometry_source_digest,
        geometry_units: study.spec.geometry.units,
        meshing_settings: study.spec.meshing_settings.clone(),
        meshing_request: super::resolve_intent_meshing_request(
            study.spec.geometry.units,
            study.spec.meshing_settings.clone(),
        )
        .map_err(|message| {
            operation_error(
                ANALYSIS_RUN_STUDY_OPERATION,
                ANALYSIS_RUN_STUDY_OP_VERSION,
                &study.context,
                OperationErrorSpec {
                    error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                    error_type: OperationErrorType::Input,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                BTreeMap::from([("study_id".to_string(), study.spec.study_id.clone())]),
            )
        })?,
        meshing_contacts: meshing::meshing_contact_definitions_for_model(&study.model).collect(),
        context: study.context,
        authored_study_digest: analysis_study_content_digest(spec),
        intent_digest: runmat_execution::Digest::from_bytes([0_u8; 32]),
        model: study.model,
        steps,
        operation_sequence: study.operation_sequence,
    };
    intent.intent_digest = intent.semantic_digest().map_err(|message| {
        operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &intent.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            message,
            BTreeMap::from([("study_id".to_string(), intent.study_id.clone())]),
        )
    })?;
    intent.validate().map_err(|message| {
        operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &intent.context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            message,
            BTreeMap::from([("study_id".to_string(), intent.study_id.clone())]),
        )
    })?;
    Ok(intent)
}

/// Admits the remotely meshed and solved result of a path-free structural study intent.
///
/// Canonical mesh and evidence validation, boundary binding, numerical result admission, quality
/// classification, and persistence remain client-runtime responsibilities. The server allocation
/// identity deterministically fences local run publication. Retrying or concurrently attaching the
/// same completed server run therefore converges on one public run ID; collision checks in the
/// artifact store reject different content under that identity.
///
/// Numerical admission finishes before publication begins. Mesh, run, and study-evidence artifacts
/// are independent immutable publications because the runtime does not yet have a transactional
/// store spanning those domains; a failure between them is recoverable by retrying this operation.
pub fn analysis_complete_structural_study_intent_op(
    intent: PreparedStructuralStudyIntent,
    meshing_output: crate::meshing::MeshingExecutionOutput,
    result: StructuralStudySequenceNumericalResult,
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
        let mut workload = step.workload;
        workload.model = resolved_model.clone();
        let mut options = step.options;
        options.solver_mesh_artifact_path = persisted_mesh.artifact_path.clone();
        let prepared_linear = PreparedStructuralLinearRun {
            workload,
            solver_mesh: Some(meshing_output.artifact.clone()),
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
