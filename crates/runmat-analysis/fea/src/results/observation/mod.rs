//! Resolution and evaluation of authored observations over recovered FEA fields.
//!
//! Field recovery remains owned by each physics pipeline. This module binds authored requests to
//! typed field declarations and admitted topology, evaluates the resulting immutable plan, and
//! publishes canonical observation objects.

mod contract;
mod distributed;
mod distributed_numeric;
mod evaluate;
mod history;
mod partition;
mod publish;
mod resolve;
mod topology;

pub use contract::{
    ObservationArtifact, ObservationArtifactEntry, ObservationArtifactIdentity,
    ObservationArtifactInput, ObservationArtifactManifest, ObservationEntity, ObservationError,
    ObservationLimits, ObservationResult, ObservationResultIdentity, ObservationSampleBounds,
    ResolvedComponentProjection, ResolvedObservation, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity, ResolvedObservationTarget,
};
pub use distributed::{
    evaluate_solid_observation_partition, evaluate_structural_observation_partition,
    prepare_solid_observation_field_partition, prepare_structural_observation_field_partition,
    reduce_observation_partitions, structural_observation_partition_source_indices,
    ObservationPartitionCombination, ObservationPartitionContribution,
    ObservationPartitionContributionIdentity, ObservationPartitionEvidence,
    ObservationPartitionEvidenceIdentity, ObservationPartitionPayload,
};
pub use evaluate::{
    evaluate_observation_sample, evaluate_observations, evaluate_structural_observation_sample,
    evaluate_structural_observations,
};
pub use history::{
    BoundedObservationHistory, ObservationHistory, ObservationHistoryEntry,
    ObservationHistoryIdentity, ObservationHistoryManifest, ObservationHistorySample,
    ObservationHistorySampleIdentity,
};
pub use partition::{
    ObservationExecutionPolicy, ObservationPartition, ObservationPartitionIdentity,
    ObservationPartitionPlan, ObservationPartitionPlanIdentity, ObservationPartitionTarget,
    DEFAULT_MAXIMUM_OBSERVATION_PARTITIONS,
};
pub use publish::publish_observation;
pub use resolve::{resolve_observation_plan, resolve_structural_observation_plan};

#[cfg(test)]
mod partition_tests;
#[cfg(test)]
mod tests;
