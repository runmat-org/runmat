//! Original J2 closure, independent from projected element fields and host storage.
use crate::analysis::storage::AnalysisPublicationError;
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, NonlinearSolidBlockArtifact, NonlinearSolidNumericalReceiptArtifact,
    NonlinearSolidWorkload,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, PublishedObjectSource, ValueRef},
    Digest,
};
use std::sync::Arc;
mod retention;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct NonlinearSolidStudyCompletion {
    pub compiled: runmat_analysis_fea::study::CompiledAnalysisStudy,
    pub workload: NonlinearSolidWorkload,
    pub terminal_block: NonlinearSolidBlockArtifact,
    pub numerical_receipt: NonlinearSolidNumericalReceiptArtifact,
    pub host_workload_digest: Digest,
    pub block_count: u64,
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub accuracy: runmat_analysis_fea::results::accuracy::AccuracyArtifact,
    pub states: Vec<CanonicalSemanticState>,
    pub baseline_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub terminal_usage: runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    pub publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    pub frozen_project_digest: Option<Digest>,
    pub root: ValueRef,
    pub catalog: PublishedObjectCatalog,
}

pub trait NonlinearSolidStudySource: PublishedObjectSource {
    /// Replay the original root within the captured host phase allowance as well
    /// as the authored ceiling. The allowance includes the retained completion;
    /// publication uses its captured preparation token to transfer that existing
    /// charge. The source still profiles its own catalog/access and cannot enlarge
    /// the supplied allowance.
    fn authenticate(
        &self,
        completed: &NonlinearSolidStudyCompletion,
        maximum_workspace_bytes: u64,
    ) -> Result<(), AnalysisPublicationError>;
    /// Shared FEA projection of one admitted original block; no equilibrium re-solve.
    /// Earlier public history stays charged during authentication, import and recovery.
    fn materialize_block(
        &self,
        completed: &NonlinearSolidStudyCompletion,
        ordinal: u64,
        additional_retained_bytes: u64,
        maximum_workspace_bytes: u64,
    ) -> Result<runmat_analysis_fea::FeaNonlinearRunResult, AnalysisPublicationError>;
}
#[derive(Clone)]
pub struct NonlinearSolidStudyObjectSource(Arc<dyn NonlinearSolidStudySource>);
impl std::fmt::Debug for NonlinearSolidStudyObjectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NonlinearSolidStudyObjectSource")
            .finish_non_exhaustive()
    }
}
impl NonlinearSolidStudyObjectSource {
    pub fn new(source: impl NonlinearSolidStudySource + 'static) -> Self {
        Self(Arc::new(source))
    }
    pub fn authenticate(
        &self,
        completed: &NonlinearSolidStudyCompletion,
        maximum_workspace_bytes: u64,
    ) -> Result<(), AnalysisPublicationError> {
        self.0.authenticate(completed, maximum_workspace_bytes)
    }
    pub fn materialize_block(
        &self,
        completed: &NonlinearSolidStudyCompletion,
        ordinal: u64,
        additional_retained_bytes: u64,
        maximum_workspace_bytes: u64,
    ) -> Result<runmat_analysis_fea::FeaNonlinearRunResult, AnalysisPublicationError> {
        if ordinal >= completed.block_count {
            return Err("nonlinear original block ordinal absent".into());
        }
        self.0.materialize_block(
            completed,
            ordinal,
            additional_retained_bytes,
            maximum_workspace_bytes,
        )
    }
    pub(in crate::analysis) fn persist(
        &self,
        completed: &NonlinearSolidStudyCompletion,
        definition: Digest,
        staging: &crate::analysis::storage::AnalysisPublicationBuilder,
        retained_input: &crate::analysis::storage::publication::PreparationReservation,
    ) -> Result<Vec<String>, crate::analysis::storage::AnalysisPublicationError> {
        {
            let _source_phase = staging.begin_source_phase()?;
            staging.require_source_objects(&completed.catalog.objects)?;
            self.authenticate(
                completed,
                staging.source_workspace_with_preparation(retained_input, completed)?,
            )?;
        }
        super::super::study_publication::catalog::persist_objects(
            self,
            &completed.catalog,
            definition,
            staging,
        )
    }
}
impl PublishedObjectSource for NonlinearSolidStudyObjectSource {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.0.read_verified(reference)
    }
}
