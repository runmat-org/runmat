mod attribution;
mod builder;
mod contract;
mod schedule;
mod series_contract;
mod series_reduce;

pub use attribution::{
    AccuracyEventAttribution, ObservationEventAttribution, SeriesEventAttribution,
    SolverEventAttribution, SolverEventSource,
};
pub use builder::BoundedObservationHistory;
pub use contract::{
    ObservationHistory, ObservationHistoryEntry, ObservationHistoryIdentity,
    ObservationHistoryManifest, ObservationHistorySample, ObservationHistorySampleIdentity,
};
pub(crate) use schedule::{
    exact_scheduled_coordinates, sample_matches_history_policy, sample_matches_policy,
};
pub use series_contract::{
    SeriesExtremumLocation, SeriesReductionContribution, SeriesReductionContributionIdentity,
    SeriesReductionEvidence, SeriesReductionPayload, SeriesReductionResult,
    SeriesReductionResultIdentity, SeriesReductionSource,
};
pub use series_reduce::reduce_observation_history;

pub(super) use schedule::{resolve_history_sample_bounds, sample_matches_domain};
