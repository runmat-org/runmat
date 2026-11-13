//! Original finite-volume evidence with placement-owned immutable-root authentication.
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, FluidNumericalReceiptArtifact, FluidSteadySolverResult,
    FluidSteadyWorkload,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, PublishedObjectSource, ValueRef},
    Digest,
};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FluidStudyCompletion {
    /// The compiled study and its workload, planned once by the coordinator on the study's
    /// mesh (sealed: holding it is the proof of that planning).
    pub product: runmat_analysis_fea::study::FluidStudyProduct,
    pub host_workload_digest: Digest,
    pub result: FluidSteadySolverResult,
    pub numerical_receipt: FluidNumericalReceiptArtifact,
    pub fields: runmat_analysis_fea::results::field::RecoveredFieldArtifactManifest,
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub observation_results: Vec<runmat_analysis_fea::results::observation::ObservationResult>,
    /// Authored histories over a transient's retained samples, with their series reductions.
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

impl FluidStudyCompletion {
    pub fn compiled(&self) -> &runmat_analysis_fea::study::CompiledAnalysisStudy {
        self.product.compiled()
    }

    pub fn workload(&self) -> &FluidSteadyWorkload {
        self.product.workload()
    }
}

/// Authenticate every DTO member against a cold replay of the original immutable root.
pub trait FluidStudySource: PublishedObjectSource {
    fn authenticate(&self, completed: &FluidStudyCompletion) -> Result<(), String>;
    /// Decode only the original governed recovery artifact, after authenticating
    /// its source closure and original recovery-plan output/heap admission.
    /// Implementations must not recover fresh arrays from solver values.
    fn materialize_fields(
        &self,
        completed: &FluidStudyCompletion,
    ) -> Result<runmat_analysis_fea::results::field::RecoveredFieldArtifact, String>;
}
#[derive(Clone)]
pub struct FluidStudyObjectSource(Arc<dyn FluidStudySource>);
impl std::fmt::Debug for FluidStudyObjectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FluidStudyObjectSource")
            .finish_non_exhaustive()
    }
}
impl FluidStudyObjectSource {
    pub fn new(source: impl FluidStudySource + 'static) -> Self {
        Self(Arc::new(source))
    }
    pub(in crate::analysis) fn authenticate(
        &self,
        completed: &FluidStudyCompletion,
    ) -> Result<(), String> {
        self.0.authenticate(completed)
    }
    pub(in crate::analysis) fn materialize_fields(
        &self,
        completed: &FluidStudyCompletion,
    ) -> Result<runmat_analysis_fea::results::field::RecoveredFieldArtifact, String> {
        self.authenticate(completed)?;
        let fields = self.0.materialize_fields(completed)?;
        if fields.manifest().map_err(|error| error.to_string())? != completed.fields {
            return Err("fluid source materialized a different original field artifact".into());
        }
        Ok(fields)
    }
    pub(in crate::analysis) fn persist(
        &self,
        completed: &FluidStudyCompletion,
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
impl PublishedObjectSource for FluidStudyObjectSource {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.0.read_verified(reference)
    }
}
