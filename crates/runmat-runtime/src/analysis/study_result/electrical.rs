//! Placement-neutral accepted static electrical evidence with authenticated object-source authority.
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, ElectricalNumericalReceiptArtifact, ElectricalStaticSolverResult,
    ElectricalStaticWorkload,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, PublishedObjectSource, ValueRef},
    Digest,
};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ElectricalStudyCompletion {
    pub compiled: runmat_analysis_fea::study::CompiledAnalysisStudy,
    pub workload: ElectricalStaticWorkload,
    pub host_workload_digest: Digest,
    pub result: ElectricalStaticSolverResult,
    pub numerical_receipt: ElectricalNumericalReceiptArtifact,
    pub fields: runmat_analysis_fea::results::field::RecoveredFieldArtifactManifest,
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub observation_results: Vec<runmat_analysis_fea::results::observation::ObservationResult>,
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
pub trait ElectricalStudySource: PublishedObjectSource {
    fn authenticate(&self, completed: &ElectricalStudyCompletion) -> Result<(), String>;
}

#[derive(Clone)]
pub struct ElectricalStudyObjectSource(Arc<dyn ElectricalStudySource>);
impl std::fmt::Debug for ElectricalStudyObjectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElectricalStudyObjectSource")
            .finish_non_exhaustive()
    }
}
impl ElectricalStudyObjectSource {
    pub fn new(source: impl ElectricalStudySource + 'static) -> Self {
        Self(Arc::new(source))
    }
    pub(in crate::analysis) fn authenticate(
        &self,
        completed: &ElectricalStudyCompletion,
    ) -> Result<(), String> {
        self.0.authenticate(completed)
    }
    pub(in crate::analysis) fn persist(
        &self,
        completed: &ElectricalStudyCompletion,
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
impl PublishedObjectSource for ElectricalStudyObjectSource {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.0.read_verified(reference)
    }
}
