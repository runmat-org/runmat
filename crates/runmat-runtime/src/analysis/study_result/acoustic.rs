//! Placement-neutral accepted acoustic evidence with authenticated object-source authority.
use runmat_analysis_fea::execution::{
    AcousticFrequencyResult, AcousticFrequencyWorkload, AcousticHarmonicSolverResult,
    AcousticHarmonicWorkload, AcousticNumericalReceiptArtifact, CanonicalSemanticState,
    FrequencySamplePublicationManifest,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, PublishedObjectSource, ValueRef},
    Digest,
};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct AcousticStudyCompletion {
    pub compiled: runmat_analysis_fea::study::CompiledAnalysisStudy,
    pub workload: AcousticFrequencyWorkload,
    pub result: AcousticFrequencyResult,
    pub entries: Vec<AcousticStudyEntryCompletion>,
    /// Histories over the sweep's frequencies, finalized after the last entry; the entry
    /// catalogs publish only the outputs their frequency schedules.
    pub history_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    /// The series reductions of `history_catalog`'s histories, in authored order.
    pub series_reductions: Vec<runmat_analysis_fea::results::observation::SeriesReductionResult>,
    pub publication: FrequencySamplePublicationManifest,
    pub resource_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub publication_catalog: PublishedObjectCatalog,
    pub publication_roots: super::super::FrequencySampleArtifactRootInventory,
    pub frozen_project_digest: Option<Digest>,
    pub root: ValueRef,
    pub catalog: PublishedObjectCatalog,
}

/// Original numerical and lifecycle evidence for one exact authored frequency entry.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticStudyEntryCompletion {
    pub entry: runmat_analysis_fea::frequency::FrequencyResponsePlanEntry,
    pub workload: AcousticHarmonicWorkload,
    pub host_workload_digest: Digest,
    pub result: AcousticHarmonicSolverResult,
    pub numerical_receipt: AcousticNumericalReceiptArtifact,
    /// The authored outputs this frequency schedules, with their publication catalog.
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub observation_results: Vec<runmat_analysis_fea::results::observation::ObservationResult>,
    pub accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub states: Vec<CanonicalSemanticState>,
    pub baseline_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub terminal_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
}

/// Placement owns replay authentication; runtime owns public artifact projection only.
pub trait AcousticStudySource: PublishedObjectSource {
    fn authenticate(&self, completed: &AcousticStudyCompletion) -> Result<(), String>;
}

#[derive(Clone)]
pub struct AcousticStudyObjectSource(Arc<dyn AcousticStudySource>);
impl std::fmt::Debug for AcousticStudyObjectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcousticStudyObjectSource")
            .finish_non_exhaustive()
    }
}
impl AcousticStudyObjectSource {
    pub fn new(source: impl AcousticStudySource + 'static) -> Self {
        Self(Arc::new(source))
    }
    pub(in crate::analysis) fn authenticate(
        &self,
        completed: &AcousticStudyCompletion,
    ) -> Result<(), String> {
        self.0.authenticate(completed)
    }
    pub(in crate::analysis) fn persist(
        &self,
        completed: &AcousticStudyCompletion,
        definition: Digest,
        staging: &crate::analysis::storage::AnalysisPublicationBuilder,
    ) -> Result<Vec<String>, crate::analysis::storage::AnalysisPublicationError> {
        {
            let _source_phase = staging.begin_source_phase()?;
            staging.require_source_objects(&completed.catalog.objects)?;
            self.authenticate(completed)?;
        }
        super::super::study_publication::catalog::persist_objects(
            self,
            &completed.catalog,
            definition,
            staging,
        )
    }
}
impl PublishedObjectSource for AcousticStudyObjectSource {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.0.read_verified(reference)
    }
}
