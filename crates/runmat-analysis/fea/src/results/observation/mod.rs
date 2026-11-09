//! Resolution and evaluation of authored observations over recovered FEA fields.
//!
//! Field recovery remains owned by each physics pipeline. This module binds authored requests to
//! typed field declarations and admitted topology, evaluates the resulting immutable plan, and
//! publishes canonical observation objects.

mod acceleration;
mod acceleration_evidence;
mod catalog;
mod contract;
mod distributed;
mod distributed_numeric;
mod entity;
mod evaluate;
mod extrema;
mod history;
mod partition;
mod publication;
mod publish;
pub use publication::{
    ObservationHistoryPublication, ObservationPublicationCatalog,
    ObservationPublicationCatalogIdentity, ObservationPublicationEntry,
    ObservationPublicationScope, ObservationSamplePublication, SeriesReductionPublication,
    SeriesReductionPublicationContribution,
};
mod resolve;
mod sampled;
pub(crate) use sampled::validate_sampled_requests;
mod source;
pub use sampled::{
    evaluate_sampled_observation_partition, evaluate_sampled_observations,
    prepare_sampled_observation_field_partition, resolve_sampled_observation_plan,
    sampled_observation_partition_source_indices,
};
mod topology;
pub use topology::resources::{
    solid_observation_topology_resident_bytes, solid_observation_topology_resident_bytes_for_mesh,
};
pub use topology::structural_observation_derivation_work_upper_bound;

pub use source::{
    thermal_source_validation_resident_bytes, ObservationFieldSelection,
    ObservationFieldSelectionIdentity, ObservationFieldSource,
};

pub(crate) use acceleration::governed_extrema_device_resident_bytes;
pub use catalog::{
    ObservationFieldCatalog, ObservationFieldCatalogSource, ObservationSelectionCoordinate,
    VolumeMeshObservationAuthority,
};
pub(crate) use contract::OBSERVATION_LIMITS;

pub use contract::{
    ObservationArtifact, ObservationArtifactEntry, ObservationArtifactIdentity,
    ObservationArtifactInput, ObservationArtifactManifest, ObservationEntity, ObservationError,
    ObservationLimits, ObservationResult, ObservationResultIdentity, ObservationSampleBounds,
    ResolvedComponentProjection, ResolvedObservation, ResolvedObservationPlan,
    ResolvedObservationPlanContents, ResolvedObservationPlanIdentity, ResolvedObservationPoint,
    ResolvedObservationTarget,
};
pub use distributed::{
    evaluate_solid_observation_partition, evaluate_structural_observation_partition,
    prepare_solid_observation_field_partition, prepare_structural_observation_field_partition,
    reduce_observation_partitions, solid_observation_partition_source_indices,
    structural_observation_partition_source_indices, ObservationPartitionCombination,
    ObservationPartitionContribution, ObservationPartitionContributionIdentity,
    ObservationPartitionEvidence, ObservationPartitionEvidenceIdentity,
    ObservationPartitionPayload, ObservationPartitionPayloadCapacity,
    SolidObservationSourceSelection,
};
pub use evaluate::{
    authored_histories_sample, authored_observations_sample, evaluate_observation_event_sample,
    evaluate_observation_sample, evaluate_observations,
    evaluate_structural_observation_event_sample, evaluate_structural_observation_sample,
    evaluate_structural_observations, resolved_observation_accepts_sample,
};
pub(crate) use history::{
    exact_scheduled_coordinate_at, exact_scheduled_sample_count, sample_matches_policy,
};
pub use history::{
    reduce_observation_history, AccuracyEventAttribution, BoundedObservationHistory,
    ObservationEventAttribution, ObservationHistory, ObservationHistoryCheckpoint,
    ObservationHistoryCheckpointIdentity, ObservationHistoryCheckpointManifest,
    ObservationHistoryEntry, ObservationHistoryIdentity, ObservationHistoryManifest,
    ObservationHistorySample, ObservationHistorySampleIdentity, ObservationHistoryStream,
    ObservationHistoryStreamIdentity, SeriesEventAttribution, SeriesExtremumLocation,
    SeriesReductionContribution, SeriesReductionContributionIdentity, SeriesReductionEvidence,
    SeriesReductionPayload, SeriesReductionResult, SeriesReductionResultIdentity,
    SeriesReductionSource, SolverEventAttribution, SolverEventSource,
};
pub use partition::{
    ObservationExecutionPolicy, ObservationPartition, ObservationPartitionIdentity,
    ObservationPartitionPlan, ObservationPartitionPlanContents, ObservationPartitionPlanIdentity,
    ObservationPartitionTarget, DEFAULT_MAXIMUM_OBSERVATION_PARTITIONS,
};
pub use publish::publish_observation;
pub use resolve::{
    resolve_observation_plan, resolve_structural_observation_plan,
    resolve_structural_observation_plan_with_connectors,
};
pub(crate) use topology::ObservationTopology;

#[cfg(test)]
mod acceleration_tests;
#[cfg(test)]
mod partition_tests;
#[cfg(test)]
mod partition_payload_tests;
#[cfg(test)]
mod tests;
pub use acceleration::{
    classify_observation_partition_acceleration,
    evaluate_structural_observation_partition_accelerated, ObservationAccelerationEligibility,
    StructuralObservationAccelerationInput,
};
pub use acceleration_evidence::{
    ObservationAccelerationEvidence, ObservationAccelerationEvidenceIdentity,
    ObservationAccelerationOperationSummary, ObservationAccelerationSummary,
    ObservationAccelerationSummaryIdentity, OBSERVATION_ACCELERATION_EVIDENCE_MAX_ENCODED_BYTES,
};
