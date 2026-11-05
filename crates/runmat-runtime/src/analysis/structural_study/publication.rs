use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{AnalysisModel, ProcedureStepId};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, SequenceIdentity, SolverResultIdentity, StepIdentity, StudyIdentity,
};

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{PreparedStudySequence, PreparedStudyStep};
use crate::analysis::frequency_result::CompletedStructuralFrequencyStep;
use crate::analysis::{
    run_operation_for_identity, storage, AnalysisExecutionStep, AnalysisRunResult,
    AnalysisSolverOptions, AnalysisStudyRunData, AnalysisStudyStatePublicationData,
    AnalysisStudyStepRunData, QualityGate, RunStatus, ANALYSIS_RUN_STUDY_OPERATION,
    ANALYSIS_RUN_STUDY_OP_VERSION,
};

pub(in crate::analysis) mod evidence;
#[cfg(test)]
pub(in crate::analysis) use evidence::persist_immutable_study_evidence;
pub(in crate::analysis) use evidence::{
    persist_content_addressed_study_evidence, run_evidence_stage, study_evidence_root,
    StudyEvidenceReadError,
};

mod completion_contract;
pub(in crate::analysis) use completion_contract::{
    standalone_completion, standalone_thermal_completion, study_completion_from_prepared,
    CompletedStudyStep, CompletedStudyStepRun, PreparedStudyCompletion, ThermalPrefixCompletion,
};

pub(in crate::analysis) struct CompletedStructuralResponseSpectrumStep {
    pub identity: runmat_analysis_fea::AnalysisRunIdentity,
    pub run_id: String,
    pub numerically_acceptable: bool,
    pub run_status: RunStatus,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub quality_reasons: Vec<crate::analysis::QualityReason>,
    pub provenance: crate::analysis::RunProvenance,
    pub numerical_result: crate::analysis::StructuralResponseSpectrumResultsData,
    pub persistence: AnalysisRunResult,
}

pub(in crate::analysis) struct CompletedStructuralRandomResponseStep {
    pub identity: runmat_analysis_fea::AnalysisRunIdentity,
    pub run_id: String,
    pub numerically_acceptable: bool,
    pub run_status: RunStatus,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub quality_reasons: Vec<crate::analysis::QualityReason>,
    pub provenance: crate::analysis::RunProvenance,
    pub numerical_result: crate::analysis::StructuralRandomResponseResultsData,
    pub persistence: AnalysisRunResult,
}

/// A step's numerical result before publication. Acoustic sweeps keep their complete
/// in-memory data until entry states are validated and persisted.
enum CompletedNumericalResult {
    Published(crate::analysis::AnalysisStudyStepNumericalResultData),
    AcousticFrequency(Box<crate::analysis::study_publication::AcousticFrequencyCompletionResults>),
}

struct CompletedRunFacts {
    identity: runmat_analysis_fea::AnalysisRunIdentity,
    operation: String,
    op_version: String,
    run_id: String,
    run_status: RunStatus,
    numerically_acceptable: bool,
    solver_convergence: QualityGate,
    result_quality: QualityGate,
    quality_reasons: Vec<crate::analysis::QualityReason>,
    provenance: crate::analysis::RunProvenance,
    numerical_result: CompletedNumericalResult,
    standard_persistence: Option<AnalysisRunResult>,
    frequency_persistence:
        Option<crate::analysis::frequency_result::StructuralFrequencyPersistence>,
    acoustic_frequency_persistence:
        Option<crate::analysis::study_publication::AcousticFrequencyPersistence>,
}

mod run_facts;
mod state_publication;
use state_publication::{validate_and_persist_states, validate_entry_scoped_acoustic_states};
mod deferred;
pub(in crate::analysis) use deferred::DeferredStudyRun;
pub(in crate::analysis) mod original_fields;
mod preparation;
mod run_persistence;
pub(in crate::analysis) use preparation::complete_study_run;
use run_facts::completed_run_facts;

pub(in crate::analysis) fn complete_study_run_prepared(
    mut prepared: PreparedStudyCompletion,
    mut completed_steps: Vec<CompletedStudyStep>,
    mut states: Vec<CanonicalSemanticState>,
    frozen_project_digest: Option<runmat_execution::Digest>,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let mut thermal_observations = None;
    let mut thermal_original_fields = None;
    if let Some(prefix) = prepared.thermal_prefix.take() {
        thermal_original_fields = Some(original_fields::thermal(&prefix, &prepared.context)?);
        if prefix.thermal.observations.is_some() {
            let CompletedStudyStepRun::Standard(run) = &prefix.completed.run else {
                return Err(completion_error(
                    &prepared.context,
                    "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
                    "thermal observations require a standard thermal run",
                ));
            };
            thermal_observations = Some((
                run.data.run_id.clone(),
                prefix.thermal,
                prefix.observation_source,
            ));
        }
        prepared.steps.insert(0, prefix.authored);
        completed_steps.insert(0, prefix.completed);
        states.splice(0..0, prefix.states);
    }
    let PreparedStudyCompletion {
        thermal_prefix: _,
        context,
        study_id,
        study_fingerprint,
        authored_study_digest,
        model,
        steps: authored_steps,
        operation_sequence,
        solver_mesh_artifact_path,
        meshing_evidence_artifact_path,
    } = prepared;
    let crate::analysis::study_publication::preparation::PublicationPreparation {
        staging,
        store,
        ..
    } = preparation;
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
    let _study_identity =
        StudyIdentity::from_authored_digest(authored_study_digest, frozen_project_digest).map_err(
            |error| completion_error(&context, "RM.FEA.RUN_STUDY.IDENTITY_INVALID", error),
        )?;
    let sequence_identity = SequenceIdentity::from_solver_contract(&model.sequence)
        .map_err(|error| completion_error(&context, "RM.FEA.RUN_STUDY.IDENTITY_INVALID", error))?;
    let mut step_results = Vec::with_capacity(authored_steps.len());
    let mut runs_to_persist = Vec::with_capacity(authored_steps.len());
    let mut frequency_runs_to_persist = Vec::new();
    let mut acoustic_frequency_runs_to_persist = Vec::new();
    let mut entry_scoped_state_steps = BTreeSet::new();
    let mut solver_results = BTreeMap::new();
    let mut original_sources = BTreeMap::new();
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
        let (completed_run, source) = original_fields::take(completed.run);
        let run = completed_run_facts(completed_run);
        if let Some(source) = source {
            original_fields::validate_source(
                &context,
                sequence_identity,
                &authored.procedure_step,
                completed.solver_result_identity,
                &source,
            )?;
            original_sources.insert(run.run_id.clone(), source);
        }
        let numerical_result = match run.numerical_result {
            CompletedNumericalResult::Published(result) => result,
            CompletedNumericalResult::AcousticFrequency(completion) => {
                validate_entry_scoped_acoustic_states(
                    &context,
                    &authored.procedure_step,
                    &completion,
                )?;
                entry_scoped_state_steps.insert(authored.procedure_step.step_id.clone());
                let published =
                    completion
                        .publish(&study_fingerprint, &staging)
                        .map_err(|error| {
                            completion_storage_error(
                                &context,
                                "RM.FEA.RUN_STUDY.STATE_ARTIFACT_STORE_FAILED",
                                error,
                            )
                        })?;
                crate::analysis::AnalysisStudyStepNumericalResultData::AcousticFrequency(Box::new(
                    published,
                ))
            }
        };
        if let Some(standard) = run.standard_persistence.clone() {
            runs_to_persist.push(standard);
        }
        let actual_identity = &run.identity;
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
            accelerator_admission: authored.authored_execution.accelerator_admission,
            solver_options: completed.resolved_solver_options,
        };
        let evidence_artifact_path = staging
            .persist_evidence(
                &study_fingerprint,
                &run_evidence_stage(&run.run_id),
                serde_json::json!({
                    "schema_version": "fea_study_sequence_step_artifact/v1",
                    "study_id": study_id.clone(),
                    "sequence_identity": sequence_identity,
                    "ordinal": ordinal,
                    "procedure_step": authored.procedure_step,
                    "execution": resolved_execution,
                    "solver_result_identity": completed.solver_result_identity,
                    "identity": run.identity,
                    "run_operation": run.operation,
                    "run_op_version": run.op_version,
                    "run_id": run.run_id,
                    "run_status": run.run_status,
                    "numerically_acceptable": run.numerically_acceptable,
                    "solver_convergence": run.solver_convergence,
                    "result_quality": run.result_quality,
                    "quality_reasons": run.quality_reasons,
                    "provenance": run.provenance,
                    "numerical_result": numerical_result,
                }),
            )
            .map_err(|error| {
                completion_storage_error(&context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
            })?;
        solver_results.insert(
            authored.procedure_step.step_id.clone(),
            completed.solver_result_identity,
        );
        let step_result = AnalysisStudyStepRunData {
            ordinal,
            procedure_step: authored.procedure_step,
            execution: resolved_execution,
            identity: run.identity,
            run_operation: run.operation,
            run_op_version: run.op_version,
            run_id: run.run_id,
            run_status: run.run_status,
            numerically_acceptable: run.numerically_acceptable,
            solver_convergence: run.solver_convergence,
            result_quality: run.result_quality,
            quality_reasons: run.quality_reasons,
            provenance: run.provenance,
            numerical_result,
            evidence_artifact_path,
        };
        if let Some(persistence) = run.frequency_persistence {
            frequency_runs_to_persist.push((step_result.clone(), persistence));
        }
        if let Some(persistence) = run.acoustic_frequency_persistence {
            acoustic_frequency_runs_to_persist.push((step_result.clone(), persistence));
        }
        step_results.push(step_result);
    }
    if !completed_by_id.is_empty() {
        return Err(completion_error(
            &context,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "completed solver results contain unauthored procedure steps",
        ));
    }
    for run in &runs_to_persist {
        run_persistence::stage(&context, &staging, run)?;
        original_fields::stage(
            &context,
            &staging,
            &run.run_id,
            &thermal_original_fields,
            &original_sources,
        )?;
        // Attach source-backed evidence with its owning run before retaining
        // independent suffix outputs. All entries remain hidden until commit.
        if thermal_observations
            .as_ref()
            .is_some_and(|(run_id, _, _)| run_id == &run.run_id)
        {
            let (run_id, completed, source) = thermal_observations.take().unwrap();
            staging
                .stage_thermal_observations(&run_id, &completed, &source)
                .map_err(|error| {
                    completion_storage_error(
                        &context,
                        "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                        error,
                    )
                })?;
        }
    }
    if thermal_observations.is_some() {
        return Err(completion_error(
            &context,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            "thermal observations have no staged thermal run",
        ));
    }
    let state_results = validate_and_persist_states(
        &context,
        &study_fingerprint,
        sequence_identity,
        &model,
        &solver_results,
        states,
        &entry_scoped_state_steps,
        &staging,
    )?;
    // A frequency sweep saves the render tiles of the mesh it was solved on, as every
    // standard run does, so its figures replay from the saved record alone.
    let frequency_render_topology = if frequency_runs_to_persist.is_empty() {
        None
    } else {
        solver_mesh_artifact_path
            .as_deref()
            .map(std::path::Path::new)
            .map(crate::analysis::solver_mesh_artifact::load_solver_mesh_artifact)
            .transpose()
            .map_err(|error| {
                completion_error(&context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
            })?
            .and_then(|mesh| crate::analysis::render_topology_from_solver_mesh(Some(&mesh)))
    };
    for (step, persistence) in &frequency_runs_to_persist {
        staging
            .stage_structural_frequency(
                step,
                &persistence.object_catalog,
                &persistence.artifact_roots,
                frequency_render_topology.as_ref(),
                persistence.source.as_ref(),
            )
            .map_err(|error| {
                completion_storage_error(&context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
            })?;
    }
    for (step, persistence) in &acoustic_frequency_runs_to_persist {
        staging
            .stage_acoustic_frequency(
                step,
                &persistence.publication_catalog,
                &persistence.publication_roots,
                &persistence.source,
            )
            .map_err(|error| {
                completion_storage_error(&context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
            })?;
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
    let evidence_artifact_path = staging
        .persist_evidence(
            &study_fingerprint,
            "sequence-run",
            serde_json::json!({
                "schema_version": "fea_study_sequence_run_artifact/v1",
                "study_id": study_id.clone(),
                "model_id": model.model_id.0.clone(),
                "sequence_identity": sequence_identity,
                "steps": step_results.clone(),
                "states": state_results.clone(),
                "solver_mesh_artifact_path": solver_mesh_artifact_path.clone(),
                "meshing_evidence_artifact_path": meshing_evidence_artifact_path.clone(),
                "study_fingerprint": study_fingerprint.clone(),
                "operation_sequence": operation_sequence.clone(),
                "run_status": run_status,
                "numerically_acceptable": numerically_acceptable,
                "solver_convergence": solver_convergence,
                "result_quality": result_quality,
            }),
        )
        .map_err(|error| {
            completion_storage_error(&context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
        })?;

    let published = OperationEnvelope::new(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &context,
        AnalysisStudyRunData {
            assessments: Vec::new(),
            study_id,
            model_id: model.model_id.0.clone(),
            sequence_identity,
            steps: step_results,
            states: state_results,
            solver_mesh_artifact_path,
            meshing_evidence_artifact_path,
            study_fingerprint,
            operation_sequence,
            run_status,
            numerically_acceptable,
            solver_convergence,
            result_quality,
            evidence_artifact_path,
            result_manifest_artifact_path: None,
            diagnostics: Vec::new(),
        },
    );
    Ok(DeferredStudyRun {
        published,
        staging,
        store,
        definition: authored_study_digest,
    })
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

pub(in crate::analysis) fn completion_storage_error(
    context: &OperationContext,
    code: &'static str,
    message: impl Into<storage::AnalysisPublicationError>,
) -> OperationErrorEnvelope {
    crate::analysis::prepared_study_publication::publication_failure(
        message.into(),
        context,
        |message| {
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
        },
    )
}
