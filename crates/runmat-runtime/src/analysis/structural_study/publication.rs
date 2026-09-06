use std::collections::BTreeMap;
use std::path::PathBuf;

use runmat_analysis_core::{AnalysisModel, ProcedureStep};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::PreparedStudyRun;
use crate::analysis::{
    current_fea_runtime_config, default_fea_artifact_root, run_operation_for_identity, storage,
    AnalysisExecutionStep, AnalysisModelTemplate, AnalysisRunResult, AnalysisSolverOptions,
    AnalysisStudyRunData, ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION,
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

    let evidence_root = study_evidence_root();
    storage::ensure_owned_directory(&evidence_root, "study evidence root")?;
    let root = evidence_root.join(format!("sha256_{digest}"));
    storage::ensure_owned_directory(&root, "study evidence fingerprint directory")?;
    let path = root.join(format!("{stage}.json"));
    let content_digest = payload_digest(&payload)?;
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
    pub(super) model: AnalysisModel,
    pub(super) authored_execution: AnalysisExecutionStep,
    pub(super) procedure_step: ProcedureStep,
    pub(super) operation_sequence: Vec<String>,
    pub(super) solver_mesh_artifact_path: Option<String>,
    pub(super) meshing_evidence_artifact_path: Option<String>,
}

pub(in crate::analysis) fn study_completion_from_prepared(
    prepared: PreparedStudyRun,
) -> PreparedStudyCompletion {
    PreparedStudyCompletion {
        context: prepared.context,
        study_id: prepared.spec.study_id,
        model_template: prepared.spec.create_model_intent.template,
        study_fingerprint: prepared.study_fingerprint,
        model: prepared.model,
        authored_execution: prepared.authored_execution,
        procedure_step: prepared.procedure_step,
        operation_sequence: prepared.operation_sequence,
        solver_mesh_artifact_path: prepared.solver_mesh_artifact_path,
        meshing_evidence_artifact_path: prepared.meshing_evidence_artifact_path,
    }
}

pub(in crate::analysis) fn complete_study_run(
    prepared: PreparedStudyCompletion,
    run_envelope: OperationEnvelope<AnalysisRunResult>,
    resolved_solver_options: AnalysisSolverOptions,
) -> Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope> {
    let PreparedStudyCompletion {
        context,
        study_id,
        model_template,
        study_fingerprint,
        model,
        authored_execution,
        procedure_step,
        operation_sequence,
        solver_mesh_artifact_path,
        meshing_evidence_artifact_path,
    } = prepared;
    let actual_identity = &run_envelope.data.run.identity;
    if actual_identity.family() != procedure_step.family
        || actual_identity.procedure() != procedure_step.procedure
        || actual_identity.solution_space() != procedure_step.solution_space
    {
        return Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.RESULT_IDENTITY_MISMATCH",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "solver result identity does not match the requested procedure step",
            BTreeMap::from([
                (
                    "requested_identity".to_string(),
                    format!("{procedure_step:?}"),
                ),
                (
                    "actual_identity".to_string(),
                    format!("{actual_identity:?}"),
                ),
            ]),
        ));
    }
    let run_operation = run_envelope.operation.clone();
    let run_op_version = run_envelope.op_version.clone();
    let expected_operation = run_operation_for_identity(actual_identity).ok_or_else(|| {
        operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.RESULT_IDENTITY_UNSUPPORTED",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "solver returned an admitted identity without a runtime operation",
            BTreeMap::from([(
                "actual_identity".to_string(),
                format!("{actual_identity:?}"),
            )]),
        )
    })?;
    if run_operation != expected_operation.name() || run_op_version != expected_operation.version()
    {
        return Err(operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.OPERATION_IDENTITY_MISMATCH",
                error_type: OperationErrorType::Contract,
                retryable: false,
                severity: OperationErrorSeverity::Fatal,
            },
            "solver operation name or version does not match the result identity",
            BTreeMap::from([
                ("run_operation".to_string(), run_operation.clone()),
                ("run_op_version".to_string(), run_op_version.clone()),
                (
                    "identity_operation".to_string(),
                    expected_operation.name().to_string(),
                ),
                (
                    "identity_op_version".to_string(),
                    expected_operation.version().to_string(),
                ),
            ]),
        ));
    }
    let resolved_execution = AnalysisExecutionStep {
        step_id: authored_execution.step_id.clone(),
        backend: authored_execution.backend,
        requirement: authored_execution.requirement,
        solver_options: resolved_solver_options,
    };
    let evidence_stage = run_evidence_stage(&run_envelope.data.run_id);
    let evidence_artifact_path = persist_immutable_study_evidence(
        &study_fingerprint,
        &evidence_stage,
        serde_json::json!({
            "schema_version": "fea_study_run_artifact/v1",
            "study_id": study_id.clone(),
            "model_id": model.model_id.0.clone(),
            "model_template": model_template,
            "procedure_step": procedure_step,
            "execution": resolved_execution,
            "identity": run_envelope.data.run.identity.clone(),
            "solver_mesh_artifact_path": solver_mesh_artifact_path.clone(),
            "meshing_evidence_artifact_path": meshing_evidence_artifact_path.clone(),
            "study_fingerprint": study_fingerprint.clone(),
            "operation_sequence": operation_sequence.clone(),
            "run_operation": run_operation.clone(),
            "run_op_version": run_op_version.clone(),
            "run_id": run_envelope.data.run_id.clone(),
            "run_status": run_envelope.data.run_status,
            "numerically_acceptable": run_envelope.data.numerically_acceptable,
            "production_publishable": run_envelope.data.production_publishable,
            "solver_convergence": run_envelope.data.solver_convergence,
            "result_quality": run_envelope.data.result_quality,
            "quality_reasons": run_envelope.data.quality_reasons.clone(),
            "production_publication_reasons": run_envelope.data.production_publication_reasons.clone(),
            "provenance": run_envelope.data.provenance.clone(),
        }),
    )
    .map_err(|err| {
        operation_error(
            ANALYSIS_RUN_STUDY_OPERATION,
            ANALYSIS_RUN_STUDY_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to persist study run evidence artifact: {err}"),
            BTreeMap::from([
                ("study_id".to_string(), study_id.clone()),
                ("run_id".to_string(), run_envelope.data.run_id.clone()),
            ]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &context,
        AnalysisStudyRunData {
            study_id,
            model_id: model.model_id.0.clone(),
            model_template,
            procedure_step,
            execution: resolved_execution,
            identity: run_envelope.data.run.identity.clone(),
            solver_mesh_artifact_path,
            meshing_evidence_artifact_path,
            study_fingerprint,
            operation_sequence,
            run_operation,
            run_op_version,
            run_id: run_envelope.data.run_id,
            run_status: run_envelope.data.run_status,
            numerically_acceptable: run_envelope.data.numerically_acceptable,
            production_publishable: run_envelope.data.production_publishable,
            solver_convergence: run_envelope.data.solver_convergence,
            result_quality: run_envelope.data.result_quality,
            quality_reasons: run_envelope.data.quality_reasons,
            production_publication_reasons: run_envelope.data.production_publication_reasons,
            provenance: run_envelope.data.provenance,
            evidence_artifact_path,
        },
    ))
}
