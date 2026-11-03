use std::collections::BTreeMap;

use runmat_analysis_fea::execution::{
    StudyIdentity, SweepCompositionPolicy, SweepDefinition, SweepIdentity,
};
use runmat_execution::Digest;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{
    sanitize_study_sweep_id, study_evidence_root, AnalysisStudyRunData, AnalysisStudySweepData,
    AnalysisStudySweepFailureEntry, AnalysisStudySweepFailureKind, AnalysisStudySweepRunEntry,
    AnalysisStudySweepSpec, ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
    ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
};

/// One authored sweep entry after semantic preparation and numerical placement.
///
/// Placement adapters return this boundary to the runtime; only the runtime may apply sweep
/// failure policy or publish the aggregate evidence artifact.
#[derive(Debug)]
pub enum AnalysisStudySweepRunOutcome {
    Succeeded(Box<OperationEnvelope<AnalysisStudyRunData>>),
    Failed(OperationErrorEnvelope),
    Cancelled(OperationErrorEnvelope),
    Indeterminate(OperationErrorEnvelope),
    NotRun,
}

/// Composes and publishes a complete sweep in authored order.
///
/// The caller may complete entries in any order, but it must restore authored order before
/// crossing this boundary. The runtime revalidates cardinality and successful study identity so a
/// placement adapter cannot substitute or omit a result.
pub fn analysis_complete_study_sweep_op(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
    outcomes: Vec<AnalysisStudySweepRunOutcome>,
) -> Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope> {
    let identity = analysis_study_sweep_identity(spec, &context)?;
    validate_outcome_cardinality(spec, outcomes.len(), &context)?;

    if spec.fail_fast {
        if let Some((index, kind, error)) =
            outcomes
                .iter()
                .enumerate()
                .find_map(|(index, outcome)| match outcome {
                    AnalysisStudySweepRunOutcome::Failed(error) => {
                        Some((index, AnalysisStudySweepFailureKind::Failed, error))
                    }
                    AnalysisStudySweepRunOutcome::Cancelled(error) => {
                        Some((index, AnalysisStudySweepFailureKind::Cancelled, error))
                    }
                    AnalysisStudySweepRunOutcome::Indeterminate(error) => {
                        Some((index, AnalysisStudySweepFailureKind::Indeterminate, error))
                    }
                    AnalysisStudySweepRunOutcome::Succeeded(_)
                    | AnalysisStudySweepRunOutcome::NotRun => None,
                })
        {
            return Err(study_failure(
                spec,
                &spec.studies[index].study_id,
                index,
                kind,
                error,
                &context,
            ));
        }
    }

    let mut run_entries = Vec::with_capacity(spec.studies.len());
    let mut failure_entries = Vec::new();
    for (index, (study, outcome)) in spec.studies.iter().zip(outcomes).enumerate() {
        match outcome {
            AnalysisStudySweepRunOutcome::Succeeded(run) => {
                if run.data.study_id != study.study_id {
                    return Err(operation_error(
                        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                        &context,
                        OperationErrorSpec {
                            error_code: "RM.FEA.RUN_STUDY_SWEEP.RESULT_SUBSTITUTED",
                            error_type: OperationErrorType::Validation,
                            retryable: false,
                            severity: OperationErrorSeverity::Error,
                        },
                        "study sweep result differs from its authored entry",
                        BTreeMap::from([
                            ("sweep_id".to_string(), spec.sweep_id.clone()),
                            ("study_id".to_string(), study.study_id.clone()),
                            ("study_index".to_string(), index.to_string()),
                            ("result_study_id".to_string(), run.data.study_id),
                        ]),
                    ));
                }
                run_entries.push(run_entry(run.data));
            }
            AnalysisStudySweepRunOutcome::Failed(error) => {
                if spec.fail_fast {
                    return Err(study_failure(
                        spec,
                        study.study_id.as_str(),
                        index,
                        AnalysisStudySweepFailureKind::Failed,
                        &error,
                        &context,
                    ));
                }
                failure_entries.push(AnalysisStudySweepFailureEntry {
                    study_id: study.study_id.clone(),
                    study_index: index,
                    kind: AnalysisStudySweepFailureKind::Failed,
                    error_code: error.error_code,
                    message: error.message,
                });
            }
            AnalysisStudySweepRunOutcome::Cancelled(error) => {
                if spec.fail_fast {
                    return Err(study_failure(
                        spec,
                        study.study_id.as_str(),
                        index,
                        AnalysisStudySweepFailureKind::Cancelled,
                        &error,
                        &context,
                    ));
                }
                failure_entries.push(AnalysisStudySweepFailureEntry {
                    study_id: study.study_id.clone(),
                    study_index: index,
                    kind: AnalysisStudySweepFailureKind::Cancelled,
                    error_code: error.error_code,
                    message: error.message,
                });
            }
            AnalysisStudySweepRunOutcome::Indeterminate(error) => {
                if spec.fail_fast {
                    return Err(study_failure(
                        spec,
                        study.study_id.as_str(),
                        index,
                        AnalysisStudySweepFailureKind::Indeterminate,
                        &error,
                        &context,
                    ));
                }
                failure_entries.push(AnalysisStudySweepFailureEntry {
                    study_id: study.study_id.clone(),
                    study_index: index,
                    kind: AnalysisStudySweepFailureKind::Indeterminate,
                    error_code: error.error_code,
                    message: error.message,
                });
            }
            AnalysisStudySweepRunOutcome::NotRun => {
                return Err(operation_error(
                    ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                    ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                    &context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_STUDY_SWEEP.OUTCOME_MISSING",
                        error_type: OperationErrorType::Contract,
                        retryable: false,
                        severity: OperationErrorSeverity::Fatal,
                    },
                    "study sweep contains an entry that was not executed",
                    BTreeMap::from([
                        ("sweep_id".to_string(), spec.sweep_id.clone()),
                        ("study_id".to_string(), study.study_id.clone()),
                        ("study_index".to_string(), index.to_string()),
                    ]),
                ));
            }
        }
    }

    publish_sweep(spec, identity, context, run_entries, failure_entries)
}

/// Validates an authored sweep and derives its stable logical identity before any work starts.
pub fn analysis_study_sweep_identity(
    spec: &AnalysisStudySweepSpec,
    context: &OperationContext,
) -> Result<SweepIdentity, OperationErrorEnvelope> {
    analysis_study_sweep_definition(spec, context)?
        .identity()
        .map_err(|error| invalid_sweep_error(context, error.to_string(), None))
}

pub fn analysis_study_sweep_definition(
    spec: &AnalysisStudySweepSpec,
    context: &OperationContext,
) -> Result<SweepDefinition, OperationErrorEnvelope> {
    let mut issue_codes = Vec::new();
    if spec.sweep_id.trim().is_empty() {
        issue_codes.push("RM.FEA.STUDY_SWEEP.ID_EMPTY");
    }
    if spec.studies.is_empty() {
        issue_codes.push("RM.FEA.STUDY_SWEEP.STUDIES_EMPTY");
    }
    if issue_codes.is_empty() {
        match sweep_definition(spec) {
            Ok(definition) => return Ok(definition),
            Err(_) => {
                issue_codes.push("RM.FEA.STUDY_SWEEP.IDENTITY_INVALID");
            }
        }
    }
    Err(invalid_sweep_error(
        context,
        "study sweep spec or outcome set is invalid",
        Some(issue_codes),
    ))
}

fn invalid_sweep_error(
    context: &OperationContext,
    message: impl Into<String>,
    issue_codes: Option<Vec<&str>>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY_SWEEP.INVALID_SPEC",
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        message,
        issue_codes.map_or_else(BTreeMap::new, |codes| {
            BTreeMap::from([("issue_codes".to_string(), codes.join(","))])
        }),
    )
}

fn validate_outcome_cardinality(
    spec: &AnalysisStudySweepSpec,
    outcome_count: usize,
    context: &OperationContext,
) -> Result<(), OperationErrorEnvelope> {
    if outcome_count == spec.studies.len() {
        return Ok(());
    }
    Err(operation_error(
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY_SWEEP.OUTCOME_CARDINALITY",
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        "study sweep outcome count differs from its authored study count",
        BTreeMap::from([
            ("sweep_id".to_string(), spec.sweep_id.clone()),
            ("study_count".to_string(), spec.studies.len().to_string()),
            ("outcome_count".to_string(), outcome_count.to_string()),
        ]),
    ))
}

fn sweep_definition(spec: &AnalysisStudySweepSpec) -> Result<SweepDefinition, String> {
    let studies = spec
        .studies
        .iter()
        .map(|study| {
            StudyIdentity::from_authored_digest(super::analysis_study_content_digest(study), None)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let sweep_contract_digest = Digest::sha256(
        [
            b"runmat-authored-analysis-sweep/v1\0".as_slice(),
            spec.sweep_id.as_bytes(),
        ]
        .concat(),
    );
    SweepDefinition::new(
        sweep_contract_digest,
        studies,
        SweepCompositionPolicy::from_fail_fast(spec.fail_fast),
    )
    .map_err(|error| error.to_string())
}

fn study_failure(
    spec: &AnalysisStudySweepSpec,
    study_id: &str,
    index: usize,
    kind: AnalysisStudySweepFailureKind,
    error: &OperationErrorEnvelope,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    let (error_code, error_type, retryable, severity, terminal_label) = match kind {
        AnalysisStudySweepFailureKind::Failed => (
            "RM.FEA.RUN_STUDY_SWEEP.STUDY_FAILED",
            OperationErrorType::Validation,
            false,
            OperationErrorSeverity::Error,
            "failed",
        ),
        AnalysisStudySweepFailureKind::Cancelled => (
            "RM.FEA.RUN_STUDY_SWEEP.STUDY_CANCELLED",
            OperationErrorType::Cancelled,
            false,
            OperationErrorSeverity::Error,
            "cancelled",
        ),
        AnalysisStudySweepFailureKind::Indeterminate => (
            "RM.FEA.RUN_STUDY_SWEEP.STUDY_INDETERMINATE",
            error.error_type,
            error.retryable,
            error.severity,
            "became indeterminate",
        ),
    };
    operation_error(
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code,
            error_type,
            retryable,
            severity,
        },
        format!(
            "study sweep {terminal_label} at index {index} for study_id {study_id}: {}",
            error.error_code
        ),
        BTreeMap::from([
            ("sweep_id".to_string(), spec.sweep_id.clone()),
            ("study_id".to_string(), study_id.to_string()),
            ("study_index".to_string(), index.to_string()),
            ("failure_kind".to_string(), kind.as_str().to_string()),
            ("cause_error_code".to_string(), error.error_code.clone()),
        ]),
    )
}

fn run_entry(run: AnalysisStudyRunData) -> AnalysisStudySweepRunEntry {
    AnalysisStudySweepRunEntry {
        study_id: run.study_id.clone(),
        result: Box::new(run),
    }
}

fn publish_sweep(
    spec: &AnalysisStudySweepSpec,
    identity: SweepIdentity,
    context: OperationContext,
    run_entries: Vec<AnalysisStudySweepRunEntry>,
    failure_entries: Vec<AnalysisStudySweepFailureEntry>,
) -> Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope> {
    let evidence_root = study_evidence_root()
        .join("sweeps")
        .join(sanitize_study_sweep_id(&spec.sweep_id))
        .join("run.json");
    let payload = serde_json::json!({
        "schema_version": "fea_study_sweep_run_artifact/v1",
        "sweep_id": spec.sweep_id.clone(),
        "identity": identity,
        "fail_fast": spec.fail_fast,
        "study_count": spec.studies.len(),
        "success_count": run_entries.len(),
        "failed_count": failure_entries.len(),
        "failure_entries": failure_entries,
        "run_entries": run_entries,
    });
    let payload_bytes = serde_json::to_vec_pretty(&payload).map_err(|error| {
        artifact_error(
            spec,
            &context,
            format!("failed to encode study sweep evidence payload: {error}"),
        )
    })?;
    super::evidence_filesystem::write_captured_evidence(&evidence_root, payload_bytes)
        .map_err(|error| sweep_publication_error(spec, &context, error))?;

    Ok(OperationEnvelope::new(
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        &context,
        AnalysisStudySweepData {
            sweep_id: spec.sweep_id.clone(),
            identity,
            study_count: spec.studies.len(),
            success_count: run_entries.len(),
            failed_count: failure_entries.len(),
            failure_entries,
            run_entries,
            evidence_artifact_path: evidence_root.display().to_string(),
        },
    ))
}

fn artifact_error(
    spec: &AnalysisStudySweepSpec,
    context: &OperationContext,
    message: String,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY_SWEEP.ARTIFACT_STORE_FAILED",
            error_type: OperationErrorType::Internal,
            retryable: true,
            severity: OperationErrorSeverity::Error,
        },
        message,
        BTreeMap::from([("sweep_id".to_string(), spec.sweep_id.clone())]),
    )
}

pub(super) fn sweep_publication_error(
    spec: &AnalysisStudySweepSpec,
    context: &OperationContext,
    error: super::storage::AnalysisPublicationError,
) -> OperationErrorEnvelope {
    super::prepared_study_publication::publication_failure_at(
        error,
        ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
        ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
        "RM.FEA.RUN_STUDY_SWEEP.PUBLICATION_RESOURCE_LIMIT",
        "RM.FEA.RUN_STUDY_SWEEP.PUBLICATION_DURABILITY_INDETERMINATE",
        context,
        |message| {
            if crate::interrupt::is_cancelled() {
                operation_error(
                    ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                    ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                    context,
                    OperationErrorSpec {
                        error_code: "RM.FEA.RUN_STUDY_SWEEP.CANCELLED",
                        error_type: OperationErrorType::Cancelled,
                        retryable: false,
                        severity: OperationErrorSeverity::Error,
                    },
                    message,
                    BTreeMap::from([("sweep_id".into(), spec.sweep_id.clone())]),
                )
            } else {
                artifact_error(spec, context, message)
            }
        },
    )
}
