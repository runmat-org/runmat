//! Publish complete frequency metadata over original immutable field partitions.
use crate::analysis::*;
use runmat_analysis_fea::{
    contracts::{ExecutionBackend, SolverMethod},
    planning::resource::FeaResourcePhase,
    solve::backend::kind::LinearAlgebraBackendKind,
    ComputeBackend,
};

#[derive(Clone)]
pub(in crate::analysis) struct AcousticFrequencyPersistence {
    pub publication_catalog: runmat_execution::value::PublishedObjectCatalog,
    pub publication_roots: FrequencySampleArtifactRootInventory,
    pub source: AcousticStudyObjectSource,
}

pub(in crate::analysis) struct CompletedAcousticFrequencyStep {
    pub identity: runmat_analysis_fea::AnalysisRunIdentity,
    pub run_id: String,
    pub run_status: RunStatus,
    pub numerically_acceptable: bool,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub quality_reasons: Vec<QualityReason>,
    pub provenance: RunProvenance,
    pub completion: Box<AcousticFrequencyCompletionResults>,
    pub persistence: AcousticFrequencyPersistence,
}

/// Complete in-memory acoustic sweep, admitted and validated before publication. Only its
/// slim projection (`AcousticFrequencyResultsData`) is persisted.
pub(in crate::analysis) struct AcousticFrequencyCompletionResults {
    pub workload: runmat_analysis_fea::execution::AcousticFrequencyWorkload,
    pub result: runmat_analysis_fea::execution::AcousticFrequencyResult,
    pub publication: runmat_analysis_fea::execution::FrequencySamplePublicationManifest,
    pub entries: Vec<AcousticStudyEntryCompletion>,
    pub history_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub series_reductions: Vec<runmat_analysis_fea::results::observation::SeriesReductionResult>,
    pub resource_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
}

impl AcousticFrequencyCompletionResults {
    /// Persist entry states once as canonical binary and keep scalar entry summaries.
    pub(in crate::analysis) fn publish(
        &self,
        study_fingerprint: &str,
        staging: &crate::analysis::storage::AnalysisPublicationBuilder,
    ) -> Result<AcousticFrequencyResultsData, crate::analysis::storage::AnalysisPublicationError>
    {
        let entries = self
            .entries
            .iter()
            .map(|entry| {
                Ok(AcousticFrequencyEntryEvidence {
                    entry: entry.entry,
                    host_workload_digest: entry.host_workload_digest,
                    result: entry.result.summary().map_err(|error| error.to_string())?,
                    numerical_receipt: entry.numerical_receipt.clone(),
                    observations: entry.observation_results.clone(),
                    observation_catalog: entry.observation_catalog.clone(),
                    accuracy: entry.accuracy.clone(),
                    states: entry
                        .states
                        .iter()
                        .map(|state| {
                            crate::analysis::study_result_manifest::PersistedStateEvidence::persist_in(
                                study_fingerprint,
                                state,
                                staging,
                            )
                        })
                        .collect::<Result<_, _>>()?,
                    baseline_usage: entry.baseline_usage.clone(),
                    terminal_usage: entry.terminal_usage.clone(),
                    publication: entry.publication.clone(),
                })
            })
            .collect::<Result<Vec<_>, crate::analysis::storage::AnalysisPublicationError>>()?;
        Ok(AcousticFrequencyResultsData {
            workload: self.workload.clone(),
            publication: self.publication.clone(),
            entries,
            history_catalog: self.history_catalog.clone(),
            series_reductions: self.series_reductions.clone(),
            resource_usage: self.resource_usage.clone(),
        })
    }
}

pub(super) fn admit(
    completed: &AcousticStudyCompletion,
    source: &AcousticStudyObjectSource,
    options: &AnalysisAcousticRunOptions,
    run_id: &str,
) -> Result<CompletedAcousticFrequencyStep, String> {
    let [common, ..] = completed.entries.as_slice() else {
        return Err("acoustic frequency completion is empty".into());
    };
    let identity = common.result.formulation().clone();
    if completed.entries.iter().any(|entry| {
        entry.result.formulation() != &identity || entry.workload.solver != common.workload.solver
    }) {
        return Err("acoustic entries differ in common formulation or numerical controls".into());
    }
    let mut quality_reasons = Vec::new();
    let mut solver_convergence = QualityGate::Pass;
    let mut result_quality = QualityGate::Pass;
    for entry in &completed.entries {
        crate::analysis::structural_study::lifecycle::accuracy_completion::validate_publication_chain(
            entry.host_workload_digest,
            &FeaResourcePhase::AcousticHarmonic(entry.workload.identity().map_err(|e| e.to_string())?),
            &entry.accuracy, &entry.baseline_usage, &entry.terminal_usage, &entry.publication,
        )?;
        let accuracy =
            crate::analysis::structural_study::lifecycle::accuracy_completion::classify_accuracy(
                &entry.accuracy,
            );
        solver_convergence = worst_gate(solver_convergence, accuracy.solver_convergence);
        result_quality = worst_gate(result_quality, accuracy.result_quality);
        quality_reasons.extend(accuracy.quality_reasons);
    }
    let numerically_acceptable =
        solver_convergence == QualityGate::Pass && result_quality == QualityGate::Pass;
    let run_status =
        crate::analysis::result_policy::classify_run_status(result_quality, numerically_acceptable);
    Ok(CompletedAcousticFrequencyStep {
        identity,
        run_id: run_id.into(),
        run_status,
        numerically_acceptable,
        solver_convergence,
        result_quality,
        quality_reasons,
        provenance: RunProvenance {
            requested_backend: ComputeBackend::Cpu,
            execution_backend: ExecutionBackend::Cpu,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            solver_device_apply_k_ratio: 0.0,
            solver_host_sync_count: 0,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method: SolverMethod::AcousticCanonicalHelmholtzCertifiedLu,
            preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events: Vec::new(),
            effective_memory_ceiling_bytes: None,
            static_solve_path: None,
        },
        completion: Box::new(AcousticFrequencyCompletionResults {
            workload: completed.workload.clone(),
            result: completed.result.clone(),
            publication: completed.publication.clone(),
            entries: completed.entries.clone(),
            history_catalog: completed.history_catalog.clone(),
            series_reductions: completed.series_reductions.clone(),
            resource_usage: completed.resource_usage.clone(),
        }),
        persistence: AcousticFrequencyPersistence {
            publication_catalog: completed.publication_catalog.clone(),
            publication_roots: completed.publication_roots.clone(),
            source: source.clone(),
        },
    })
}

fn worst_gate(left: QualityGate, right: QualityGate) -> QualityGate {
    match (left, right) {
        (QualityGate::Fail, _) | (_, QualityGate::Fail) => QualityGate::Fail,
        (QualityGate::Warn, _) | (_, QualityGate::Warn) => QualityGate::Warn,
        _ => QualityGate::Pass,
    }
}
