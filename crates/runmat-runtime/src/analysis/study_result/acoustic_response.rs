//! Placement-neutral accepted static acoustic response evidence with authenticated object-source authority.
use runmat_analysis_fea::execution::{
    AcousticResponseNumericalReceiptArtifact, AcousticResponseSolverResult,
    AcousticResponseWorkload, CanonicalSemanticState,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, PublishedObjectSource, ValueRef},
    Digest,
};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct AcousticResponseStudyCompletion {
    pub compiled: runmat_analysis_fea::study::CompiledAnalysisStudy,
    pub workload: AcousticResponseWorkload,
    pub host_workload_digest: Digest,
    pub result: AcousticResponseSolverResult,
    pub numerical_receipt: AcousticResponseNumericalReceiptArtifact,
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub observation_results: Vec<runmat_analysis_fea::results::observation::ObservationResult>,
    pub observation_histories: Vec<runmat_analysis_fea::results::observation::ObservationHistory>,
    pub series_reductions: Vec<runmat_analysis_fea::results::observation::SeriesReductionResult>,
    pub accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub states: Vec<CanonicalSemanticState>,
    pub baseline_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub terminal_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    pub frozen_project_digest: Option<Digest>,
    pub root: ValueRef,
    pub catalog: PublishedObjectCatalog,
}

/// Placement owns replay authentication; runtime owns public artifact projection only.
pub trait AcousticResponseStudySource: PublishedObjectSource {
    fn authenticate(&self, completed: &AcousticResponseStudyCompletion) -> Result<(), String>;
}

#[derive(Clone)]
pub struct AcousticResponseStudyObjectSource(Arc<dyn AcousticResponseStudySource>);
impl std::fmt::Debug for AcousticResponseStudyObjectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcousticResponseStudyObjectSource")
            .finish_non_exhaustive()
    }
}
impl AcousticResponseStudyObjectSource {
    pub fn new(source: impl AcousticResponseStudySource + 'static) -> Self {
        Self(Arc::new(source))
    }
    pub(in crate::analysis) fn authenticate(
        &self,
        completed: &AcousticResponseStudyCompletion,
    ) -> Result<(), String> {
        self.0.authenticate(completed)
    }
    pub(in crate::analysis) fn persist(
        &self,
        completed: &AcousticResponseStudyCompletion,
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
impl PublishedObjectSource for AcousticResponseStudyObjectSource {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.0.read_verified(reference)
    }
}
