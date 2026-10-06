//! Placement-neutral replay authentication and complete durable thermal evidence.
use super::ThermalStudyCompletion;
use runmat_analysis_core::ObservationId;
use runmat_analysis_fea::results::{
    field::RecoveredFieldSample,
    observation::{
        ObservationArtifactIdentity, ObservationHistoryIdentity, ObservationHistorySample,
        ObservationPublicationCatalogIdentity, ObservationResult, SeriesReductionResult,
        SeriesReductionResultIdentity,
    },
};
use runmat_execution::{value::PublishedObjectSource, Digest};
use std::sync::Arc;
pub trait ThermalStudySource: PublishedObjectSource {
    fn authenticate(&self, completed: &ThermalStudyCompletion) -> Result<(), String>;
    /// Source-owned replay/read/decode peak, excluding the caller's already retained DTO.
    /// Profiling must enforce the allowance before any payload allocation or typed decode.
    fn publication_workspace(
        &self,
        completed: &ThermalStudyCompletion,
        maximum_workspace_bytes: u64,
    ) -> Result<u64, crate::analysis::storage::AnalysisPublicationError>;
    fn authenticate_admitted(
        &self,
        completed: &ThermalStudyCompletion,
        maximum_workspace_bytes: u64,
    ) -> Result<(), crate::analysis::storage::AnalysisPublicationError>;
    fn read_observation(
        &self,
        catalog: ObservationPublicationCatalogIdentity,
        artifact: ObservationArtifactIdentity,
        observation: &ObservationId,
    ) -> Result<ObservationResult, String>;
    fn read_history_sample(
        &self,
        catalog: ObservationPublicationCatalogIdentity,
        history: ObservationHistoryIdentity,
        coordinate: RecoveredFieldSample,
    ) -> Result<ObservationHistorySample, String>;
    fn read_series_reduction(
        &self,
        catalog: ObservationPublicationCatalogIdentity,
        series: SeriesReductionResultIdentity,
    ) -> Result<SeriesReductionResult, String>;
}
#[derive(Clone)]
pub struct ThermalStudyObjectSource(Arc<dyn ThermalStudySource>);
impl std::fmt::Debug for ThermalStudyObjectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThermalStudyObjectSource")
            .finish_non_exhaustive()
    }
}
impl ThermalStudyObjectSource {
    pub fn new(source: impl ThermalStudySource + 'static) -> Self {
        Self(Arc::new(source))
    }
    pub fn read_observation(
        &self,
        catalog: ObservationPublicationCatalogIdentity,
        artifact: ObservationArtifactIdentity,
        observation: &ObservationId,
    ) -> Result<ObservationResult, String> {
        self.0.read_observation(catalog, artifact, observation)
    }
    pub fn read_history_sample(
        &self,
        catalog: ObservationPublicationCatalogIdentity,
        history: ObservationHistoryIdentity,
        coordinate: RecoveredFieldSample,
    ) -> Result<ObservationHistorySample, String> {
        self.0.read_history_sample(catalog, history, coordinate)
    }
    pub fn read_series_reduction(
        &self,
        catalog: ObservationPublicationCatalogIdentity,
        series: SeriesReductionResultIdentity,
    ) -> Result<SeriesReductionResult, String> {
        self.0.read_series_reduction(catalog, series)
    }
    pub(in crate::analysis) fn authenticate(
        &self,
        completed: &ThermalStudyCompletion,
    ) -> Result<(), String> {
        self.0.authenticate(completed)
    }
    pub(in crate::analysis) fn authenticate_admitted(
        &self,
        completed: &ThermalStudyCompletion,
        maximum: u64,
    ) -> Result<(), crate::analysis::storage::AnalysisPublicationError> {
        self.0.authenticate_admitted(completed, maximum)
    }
    pub(in crate::analysis) fn admit_publication(
        &self,
        completed: &ThermalStudyCompletion,
        staging: &mut crate::analysis::storage::AnalysisPublicationBuilder,
    ) -> Result<(), crate::analysis::storage::AnalysisPublicationError> {
        let maximum = staging.source_workspace_limit()?;
        let required = {
            let _phase = staging.begin_source_profile(maximum)?;
            self.0.publication_workspace(completed, maximum)?
        };
        staging.admit_profiled_source_objects(&completed.catalog.objects, required)
    }
    pub(in crate::analysis) fn persist(
        &self,
        completed: &ThermalStudyCompletion,
        study_digest: Digest,
        staging: &crate::analysis::storage::AnalysisPublicationBuilder,
    ) -> Result<Vec<String>, crate::analysis::storage::AnalysisPublicationError> {
        {
            let _source_phase = staging.begin_source_phase()?;
            staging.require_source_objects(&completed.catalog.objects)?;
            self.authenticate_admitted(
                completed,
                staging.source_replay_allowance(&completed.catalog.objects)?,
            )?;
        }
        super::super::study_publication::catalog::persist_objects(
            self,
            &completed.catalog,
            study_digest,
            staging,
        )
    }
}

impl PublishedObjectSource for ThermalStudyObjectSource {
    fn read_verified(
        &self,
        reference: &runmat_execution::value::ValueRef,
    ) -> Result<Vec<u8>, String> {
        self.0.read_verified(reference)
    }
}
