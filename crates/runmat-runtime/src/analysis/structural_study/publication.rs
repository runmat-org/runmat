use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use runmat_analysis_core::{AnalysisModel, ProcedureStepId};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, SequenceIdentity, SolverResultIdentity, StepIdentity, StudyIdentity,
};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{PreparedStudySequence, PreparedStudyStep};
use crate::analysis::{
    current_fea_runtime_config, default_fea_artifact_root, persist_linear_static_run,
    run_operation_for_identity, storage, AnalysisExecutionStep, AnalysisModelTemplate,
    AnalysisRunResult, AnalysisSolverOptions, AnalysisStudyRunData,
    AnalysisStudyStatePublicationData, AnalysisStudyStepRunData, QualityGate, RunStatus,
    ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION,
};

const STUDY_EVIDENCE_SCHEMA_VERSION: &str = "analysis_study_evidence/2026-09";

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StudyEvidenceBinding {
    schema_version: String,
    study_fingerprint: String,
    stage: String,
    content_digest: String,
}

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct PersistedStudyEvidence {
    #[serde(rename = "$evidence")]
    binding: StudyEvidenceBinding,
    // Study-result readers consume these fields directly. Keep that artifact shape while reserving
    // a typed namespace for the immutable identity and content binding.
    #[serde(flatten)]
    payload: serde_json::Map<String, serde_json::Value>,
}

pub(in crate::analysis) fn study_evidence_root() -> PathBuf {
    let config = current_fea_runtime_config();
    config
        .study_artifact_root
        .or_else(|| {
            std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT")
                .or_else(|_| std::env::var("RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT"))
                .ok()
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| {
            config
                .artifact_root
                .unwrap_or_else(default_fea_artifact_root)
                .join("studies")
        })
}

pub(in crate::analysis) fn persist_immutable_study_evidence(
    study_fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
) -> Result<String, String> {
    persist_study_evidence(study_fingerprint, stage, payload, false)
}

pub(in crate::analysis) fn persist_content_addressed_study_evidence(
    study_fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
) -> Result<String, String> {
    persist_study_evidence(study_fingerprint, stage, payload, true)
}

fn persist_study_evidence(
    study_fingerprint: &str,
    stage: &str,
    payload: serde_json::Value,
    content_addressed: bool,
) -> Result<String, String> {
    let digest = study_fingerprint
        .strip_prefix("sha256:")
        .filter(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        })
        .ok_or_else(|| "study fingerprint must be a lowercase SHA-256 digest".to_string())?;
    if stage.is_empty()
        || stage.len() > 128
        || !stage.is_ascii()
        || !stage
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err("study evidence stage must be a safe 1-128 byte ASCII token".into());
    }
    let payload = payload
        .as_object()
        .cloned()
        .ok_or_else(|| "study evidence payload must be a JSON object".to_string())?;
    if payload.contains_key("$evidence") {
        return Err("study evidence payload may not define the reserved `$evidence` field".into());
    }

    let content_digest = payload_digest(&payload)?;
    let path_identity = if content_addressed {
        format!(
            "{stage}_sha256_{}",
            content_digest.trim_start_matches("sha256:")
        )
    } else {
        stage.to_string()
    };
    let evidence_root = study_evidence_root();
    storage::ensure_owned_directory(&evidence_root, "study evidence root")?;
    let root = evidence_root.join(format!("sha256_{digest}"));
    storage::ensure_owned_directory(&root, "study evidence fingerprint directory")?;
    let path = root.join(format!("{path_identity}.json"));
    let evidence = PersistedStudyEvidence {
        binding: StudyEvidenceBinding {
            schema_version: STUDY_EVIDENCE_SCHEMA_VERSION.to_string(),
            study_fingerprint: study_fingerprint.to_string(),
            stage: stage.to_string(),
            content_digest,
        },
        payload,
    };
    let bytes = serde_json::to_vec_pretty(&evidence)
        .map_err(|error| format!("failed to encode study evidence envelope: {error}"))?;
    match runmat_filesystem::atomic_create_new(&path, &bytes)
        .map_err(|error| format!("failed to publish immutable study evidence: {error}"))?
    {
        runmat_filesystem::AtomicCreateResult::Published { .. } => {}
        runmat_filesystem::AtomicCreateResult::Existing(existing_bytes) => {
            let existing = serde_json::from_slice::<PersistedStudyEvidence>(&existing_bytes)
                .map_err(|error| format!("failed to parse existing study evidence: {error}"))?;
            validate_evidence(&existing, study_fingerprint, stage)?;
            if existing != evidence {
                return Err(format!(
                    "study evidence identity already contains different content at {}",
                    path.display()
                ));
            }
        }
    }
    Ok(path.display().to_string())
}

fn validate_evidence(
    evidence: &PersistedStudyEvidence,
    expected_study_fingerprint: &str,
    expected_stage: &str,
) -> Result<(), String> {
    if evidence.binding.schema_version != STUDY_EVIDENCE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported study evidence schema `{}`",
            evidence.binding.schema_version
        ));
    }
    if evidence.binding.study_fingerprint != expected_study_fingerprint
        || evidence.binding.stage != expected_stage
    {
        return Err("study evidence path identity does not match its envelope".into());
    }
    if evidence.binding.content_digest != payload_digest(&evidence.payload)? {
        return Err("study evidence content digest does not match its payload".into());
    }
    Ok(())
}

fn payload_digest(payload: &serde_json::Map<String, serde_json::Value>) -> Result<String, String> {
    let payload_bytes = serde_json::to_vec(payload)
        .map_err(|error| format!("failed to encode study evidence payload: {error}"))?;
    Ok(runmat_execution::Digest::sha256(&payload_bytes).to_string())
}

pub(in crate::analysis) fn run_evidence_stage(run_id: &str) -> String {
    let digest = runmat_execution::Digest::sha256(run_id.as_bytes());
    format!("run-{}", digest.to_string().trim_start_matches("sha256:"))
}

pub(in crate::analysis) struct PreparedStudyCompletion {
    pub(super) context: OperationContext,
    pub(super) study_id: String,
    pub(super) model_template: AnalysisModelTemplate,
    pub(super) study_fingerprint: String,
    pub(super) authored_study_digest: runmat_execution::Digest,
    pub(super) model: AnalysisModel,
    pub(super) steps: Vec<PreparedStudyStep>,
    pub(super) operation_sequence: Vec<String>,
    pub(super) solver_mesh_artifact_path: Option<String>,
    pub(super) meshing_evidence_artifact_path: Option<String>,
}

pub(in crate::analysis) fn study_completion_from_prepared(
    prepared: PreparedStudySequence,
) -> PreparedStudyCompletion {
    let authored_study_digest = crate::analysis::analysis_study_content_digest(&prepared.spec);
    PreparedStudyCompletion {
        context: prepared.context,
        study_id: prepared.spec.study_id,
        model_template: prepared.spec.create_model_intent.template,
        study_fingerprint: prepared.study_fingerprint,
        authored_study_digest,
        model: prepared.model,
        steps: prepared.steps,
        operation_sequence: prepared.operation_sequence,
        solver_mesh_artifact_path: prepared.solver_mesh_artifact_path,
        meshing_evidence_artifact_path: prepared.meshing_evidence_artifact_path,
    }
}

pub(in crate::analysis) struct CompletedStudyStep {
    pub(in crate::analysis) step_id: ProcedureStepId,
    pub(in crate::analysis) solver_result_identity: SolverResultIdentity,
    pub(in crate::analysis) run: OperationEnvelope<AnalysisRunResult>,
    pub(in crate::analysis) resolved_solver_options: AnalysisSolverOptions,
}

pub(in crate::analysis) fn complete_study_run(
    prepared: PreparedStudyCompletion,
    completed_steps: Vec<CompletedStudyStep>,
    states: Vec<CanonicalSemanticState>,
    frozen_project_digest: Option<runmat_execution::Digest>,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let PreparedStudyCompletion {
        context,
        study_id,
        model_template,
        study_fingerprint,
        authored_study_digest,
        model,
        steps: authored_steps,
        operation_sequence,
        solver_mesh_artifact_path,
        meshing_evidence_artifact_path,
    } = prepared;
    if completed_steps.len() != authored_steps.len() {
        return Err(completion_error(
            &context,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "completed solver results do not exactly cover the authored sequence",
        ));
    }
    let mut completed_by_id = BTreeMap::new();
    for completed in completed_steps {
        if completed_by_id
            .insert(completed.step_id.clone(), completed)
            .is_some()
        {
            return Err(completion_error(
                &context,
                "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                "completed solver results contain a duplicate procedure step",
            ));
        }
    }
    let study_identity = StudyIdentity::new(authored_study_digest, frozen_project_digest)
        .map_err(|error| completion_error(&context, "RM.FEA.RUN_STUDY.IDENTITY_INVALID", error))?;
    let sequence_identity = SequenceIdentity::from_authored(study_identity, &model.sequence)
        .map_err(|error| completion_error(&context, "RM.FEA.RUN_STUDY.IDENTITY_INVALID", error))?;
    let mut step_results = Vec::with_capacity(authored_steps.len());
    let mut runs_to_persist = Vec::with_capacity(authored_steps.len());
    let mut solver_results = BTreeMap::new();
    for (ordinal, authored) in authored_steps.into_iter().enumerate() {
        let completed = completed_by_id
            .remove(&authored.procedure_step.step_id)
            .ok_or_else(|| {
                completion_error(
                    &context,
                    "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
                    "completed solver results are missing an authored procedure step",
                )
            })?;
        let run = completed.run;
        runs_to_persist.push(run.data.clone());
        let actual_identity = &run.data.run.identity;
        if actual_identity.family() != authored.procedure_step.family
            || actual_identity.procedure() != authored.procedure_step.procedure
            || actual_identity.solution_space() != authored.procedure_step.solution_space
        {
            return Err(completion_error(
                &context,
                "RM.FEA.RUN_STUDY.RESULT_IDENTITY_MISMATCH",
                "solver result identity does not match its authored procedure step",
            ));
        }
        let expected_operation = run_operation_for_identity(actual_identity).ok_or_else(|| {
            completion_error(
                &context,
                "RM.FEA.RUN_STUDY.RESULT_IDENTITY_UNSUPPORTED",
                "solver returned an admitted identity without a runtime operation",
            )
        })?;
        if run.operation != expected_operation.name()
            || run.op_version != expected_operation.version()
        {
            return Err(completion_error(
                &context,
                "RM.FEA.RUN_STUDY.OPERATION_IDENTITY_MISMATCH",
                "solver operation name or version does not match the result identity",
            ));
        }
        let resolved_execution = AnalysisExecutionStep {
            step_id: authored.authored_execution.step_id,
            backend: authored.authored_execution.backend,
            requirement: authored.authored_execution.requirement,
            solver_options: completed.resolved_solver_options,
        };
        let evidence_artifact_path = persist_immutable_study_evidence(
            &study_fingerprint,
            &run_evidence_stage(&run.data.run_id),
            serde_json::json!({
                "schema_version": "fea_study_sequence_step_artifact/v1",
                "study_id": study_id.clone(),
                "sequence_identity": sequence_identity,
                "ordinal": ordinal,
                "procedure_step": authored.procedure_step,
                "execution": resolved_execution,
                "solver_result_identity": completed.solver_result_identity,
                "identity": run.data.run.identity,
                "run_operation": run.operation,
                "run_op_version": run.op_version,
                "run_id": run.data.run_id,
                "run_status": run.data.run_status,
                "numerically_acceptable": run.data.numerically_acceptable,
                "production_publishable": run.data.production_publishable,
                "solver_convergence": run.data.solver_convergence,
                "result_quality": run.data.result_quality,
                "quality_reasons": run.data.quality_reasons,
                "production_publication_reasons": run.data.production_publication_reasons,
                "provenance": run.data.provenance,
            }),
        )
        .map_err(|error| {
            completion_storage_error(&context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
        })?;
        solver_results.insert(
            authored.procedure_step.step_id.clone(),
            completed.solver_result_identity,
        );
        step_results.push(AnalysisStudyStepRunData {
            ordinal,
            procedure_step: authored.procedure_step,
            execution: resolved_execution,
            identity: run.data.run.identity,
            run_operation: run.operation,
            run_op_version: run.op_version,
            run_id: run.data.run_id,
            run_status: run.data.run_status,
            numerically_acceptable: run.data.numerically_acceptable,
            production_publishable: run.data.production_publishable,
            solver_convergence: run.data.solver_convergence,
            result_quality: run.data.result_quality,
            quality_reasons: run.data.quality_reasons,
            production_publication_reasons: run.data.production_publication_reasons,
            provenance: run.data.provenance,
            evidence_artifact_path,
        });
    }
    if !completed_by_id.is_empty() {
        return Err(completion_error(
            &context,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "completed solver results contain unauthored procedure steps",
        ));
    }
    let state_results = validate_and_persist_states(
        &context,
        &study_fingerprint,
        sequence_identity,
        &model,
        &solver_results,
        states,
    )?;
    for run in &runs_to_persist {
        persist_linear_static_run(&context, run)?;
    }
    let run_status = step_results
        .iter()
        .fold(RunStatus::Completed, |status, step| {
            worst_status(status, step.run_status)
        });
    let solver_convergence = step_results.iter().fold(QualityGate::Pass, |gate, step| {
        worst_gate(gate, step.solver_convergence)
    });
    let result_quality = step_results.iter().fold(QualityGate::Pass, |gate, step| {
        worst_gate(gate, step.result_quality)
    });
    let numerically_acceptable = step_results.iter().all(|step| step.numerically_acceptable);
    let production_publishable = step_results.iter().all(|step| step.production_publishable);
    let aggregate_stage_digest = runmat_execution::Digest::sha256(
        &serde_json::to_vec(&(&sequence_identity, &step_results, &state_results)).map_err(
            |error| completion_error(&context, "RM.FEA.RUN_STUDY.IDENTITY_INVALID", error),
        )?,
    );
    let aggregate_stage = format!(
        "sequence-run-{}",
        aggregate_stage_digest
            .to_string()
            .trim_start_matches("sha256:")
    );
    let evidence_artifact_path = persist_immutable_study_evidence(
        &study_fingerprint,
        &aggregate_stage,
        serde_json::json!({
            "schema_version": "fea_study_sequence_run_artifact/v1",
            "study_id": study_id.clone(),
            "model_id": model.model_id.0.clone(),
            "model_template": model_template,
            "sequence_identity": sequence_identity,
            "steps": step_results.clone(),
            "states": state_results.clone(),
            "solver_mesh_artifact_path": solver_mesh_artifact_path.clone(),
            "meshing_evidence_artifact_path": meshing_evidence_artifact_path.clone(),
            "study_fingerprint": study_fingerprint.clone(),
            "operation_sequence": operation_sequence.clone(),
            "run_status": run_status,
            "numerically_acceptable": numerically_acceptable,
            "production_publishable": production_publishable,
            "solver_convergence": solver_convergence,
            "result_quality": result_quality,
        }),
    )
    .map_err(|error| {
        completion_storage_error(&context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &context,
        AnalysisStudyRunData {
            study_id,
            model_id: model.model_id.0.clone(),
            model_template,
            sequence_identity,
            steps: step_results,
            states: state_results,
            solver_mesh_artifact_path,
            meshing_evidence_artifact_path,
            study_fingerprint,
            operation_sequence,
            run_status,
            numerically_acceptable,
            production_publishable,
            solver_convergence,
            result_quality,
            evidence_artifact_path,
        },
    ))
}

fn validate_and_persist_states(
    context: &OperationContext,
    study_fingerprint: &str,
    sequence_identity: SequenceIdentity,
    model: &AnalysisModel,
    solver_results: &BTreeMap<ProcedureStepId, SolverResultIdentity>,
    states: Vec<CanonicalSemanticState>,
) -> Result<Vec<AnalysisStudyStatePublicationData>, OperationErrorEnvelope> {
    let expected = model
        .sequence
        .steps
        .iter()
        .flat_map(|step| {
            step.state_outputs
                .iter()
                .map(move |declaration| (step, declaration))
        })
        .collect::<Vec<_>>();
    if states.len() != expected.len() {
        return Err(completion_error(
            context,
            "RM.FEA.RUN_STUDY.STATE_INVENTORY_MISMATCH",
            "published states do not exactly cover the authored declarations",
        ));
    }
    let mut seen = BTreeSet::new();
    for (state, (producer, declaration)) in states.iter().zip(&expected) {
        state
            .validate()
            .map_err(|error| completion_error(context, "RM.FEA.RUN_STUDY.STATE_INVALID", error))?;
        let producer_identity = StepIdentity::from_authored(sequence_identity, producer)
            .map_err(|error| completion_error(context, "RM.FEA.RUN_STUDY.STATE_INVALID", error))?;
        if state.producer != producer_identity
            || state.state_id != declaration.state_id
            || state.kind != declaration.kind
            || state.completeness != declaration.completeness
            || state.persistence != declaration.persistence
            || solver_results.get(&producer.step_id) != Some(&state.source_result)
            || !seen.insert((producer.step_id.clone(), declaration.state_id.clone()))
        {
            return Err(completion_error(
                context,
                "RM.FEA.RUN_STUDY.STATE_INVENTORY_MISMATCH",
                "published state identity, order, declaration, or solver lineage changed",
            ));
        }
    }
    let mut results = Vec::with_capacity(states.len());
    for (state, (producer, _)) in states.into_iter().zip(expected) {
        let artifact_path = persist_semantic_state(study_fingerprint, &state).map_err(|error| {
            completion_storage_error(
                context,
                "RM.FEA.RUN_STUDY.STATE_ARTIFACT_STORE_FAILED",
                error,
            )
        })?;
        results.push(AnalysisStudyStatePublicationData {
            producer_step: producer.step_id.clone(),
            state_id: state.state_id,
            kind: state.kind,
            completeness: state.completeness,
            persistence: state.persistence,
            identity: state.identity,
            content_digest: state.content_digest,
            topology_digest: state.topology_digest,
            source_result: state.source_result,
            artifact_path,
        });
    }
    Ok(results)
}

fn persist_semantic_state(
    study_fingerprint: &str,
    state: &CanonicalSemanticState,
) -> Result<String, String> {
    let fingerprint = study_fingerprint
        .strip_prefix("sha256:")
        .ok_or_else(|| "study fingerprint is not a SHA-256 identity".to_string())?;
    let state_digest = state
        .identity
        .digest()
        .to_string()
        .trim_start_matches("sha256:")
        .to_string();
    let directory = study_evidence_root()
        .join(format!("sha256_{fingerprint}"))
        .join("states");
    storage::ensure_owned_directory(&directory, "study semantic state directory")?;
    let path = directory.join(format!("sha256_{state_digest}.state"));
    let bytes = state.canonical_bytes().map_err(|error| error.to_string())?;
    match runmat_filesystem::atomic_create_new(&path, &bytes)
        .map_err(|error| format!("failed to publish immutable semantic state: {error}"))?
    {
        runmat_filesystem::AtomicCreateResult::Published { .. } => {}
        runmat_filesystem::AtomicCreateResult::Existing(existing) if existing == bytes => {}
        runmat_filesystem::AtomicCreateResult::Existing(_) => {
            return Err("semantic state identity already contains different content".into());
        }
    }
    Ok(path.display().to_string())
}

fn worst_gate(left: QualityGate, right: QualityGate) -> QualityGate {
    match (left, right) {
        (QualityGate::Fail, _) | (_, QualityGate::Fail) => QualityGate::Fail,
        (QualityGate::Warn, _) | (_, QualityGate::Warn) => QualityGate::Warn,
        _ => QualityGate::Pass,
    }
}

fn worst_status(left: RunStatus, right: RunStatus) -> RunStatus {
    match (left, right) {
        (RunStatus::Rejected, _) | (_, RunStatus::Rejected) => RunStatus::Rejected,
        (RunStatus::Degraded, _) | (_, RunStatus::Degraded) => RunStatus::Degraded,
        _ => RunStatus::Completed,
    }
}

fn completion_error(
    context: &OperationContext,
    code: &'static str,
    message: impl std::fmt::Display,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: code,
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message.to_string(),
        BTreeMap::new(),
    )
}

fn completion_storage_error(
    context: &OperationContext,
    code: &'static str,
    message: impl std::fmt::Display,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: code,
            error_type: OperationErrorType::Internal,
            retryable: true,
            severity: OperationErrorSeverity::Error,
        },
        message.to_string(),
        BTreeMap::new(),
    )
}
