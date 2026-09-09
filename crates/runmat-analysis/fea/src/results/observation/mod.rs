//! Resolution and evaluation of authored observations over recovered FEA fields.
//!
//! Field recovery remains owned by each physics pipeline. This module binds authored requests to
//! typed field declarations and admitted topology, evaluates the resulting immutable plan, and
//! publishes canonical observation objects.

mod contract;
mod evaluate;
mod history;
mod publish;
mod resolve;

pub use contract::{
    ObservationArtifact, ObservationArtifactEntry, ObservationArtifactIdentity,
    ObservationArtifactInput, ObservationArtifactManifest, ObservationEntity, ObservationError,
    ObservationLimits, ObservationResult, ObservationResultIdentity, ObservationSampleBounds,
    ResolvedComponentProjection, ResolvedObservation, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity, ResolvedObservationTarget,
};
pub use evaluate::{evaluate_observation_sample, evaluate_observations};
pub use history::{
    BoundedObservationHistory, ObservationHistory, ObservationHistoryEntry,
    ObservationHistoryIdentity, ObservationHistoryManifest, ObservationHistorySample,
    ObservationHistorySampleIdentity,
};
pub use publish::publish_observation;
pub use resolve::resolve_observation_plan;

#[cfg(test)]
mod tests;
