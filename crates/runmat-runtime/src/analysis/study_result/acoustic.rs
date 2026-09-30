//! Placement-neutral accepted acoustic evidence with authenticated object-source authority.
use runmat_analysis_fea::execution::{
    AcousticHarmonicSolverResult, AcousticHarmonicWorkload, AcousticNumericalReceiptArtifact,
    CanonicalSemanticState,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, PublishedObjectSource, ValueRef},
    Digest,
};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct AcousticStudyCompletion {
    pub compiled: runmat_analysis_fea::study::CompiledAnalysisStudy,
    pub workload: AcousticHarmonicWorkload,
    pub host_workload_digest: Digest,
    pub result: AcousticHarmonicSolverResult,
    pub numerical_receipt: AcousticNumericalReceiptArtifact,
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
    ) -> Result<Vec<String>, String> {
        self.authenticate(completed)?;
        super::super::study_result_manifest::persist_catalog_objects(
            self,
            &completed.catalog,
            definition,
            "acoustic-object",
        )
    }
}
impl PublishedObjectSource for AcousticStudyObjectSource {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.0.read_verified(reference)
    }
}
