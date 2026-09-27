//! Measurements at the actual cache owner, not aggregate metric differences.
use runmat_accelerate_api::observation::{
    current_provider_observation, ProviderObservationStage, ProviderOperationRecorder,
};
use runmat_execution::CandidatePreparationState;
use runmat_time::Instant;

pub(super) fn poisoned_cache() {
    log::error!("WGPU pipeline cache is poisoned; executing uncached without preparation evidence");
    if let Some(recorder) = current_provider_observation() {
        recorder
            .fail(runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed);
    }
}

pub(super) struct PipelineObservation(Option<(ProviderOperationRecorder, Instant)>);
impl PipelineObservation {
    pub(super) fn begin() -> Self {
        Self(current_provider_observation().map(|recorder| (recorder, Instant::now())))
    }
    pub(super) fn finish(self, preparation: CandidatePreparationState) {
        if let Some((recorder, start)) = self.0 {
            recorder.record_pipeline(preparation);
            recorder.record_stage(ProviderObservationStage::Preparation, start.elapsed());
        }
    }
}
