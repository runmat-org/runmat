//! Physics-owned recovered fields exposed to observation evaluation.

mod catalog;
mod contract;
mod entity;
mod finite_volume;
mod partition;
mod partition_set;
mod physical_storage;
pub use physical_storage::RecoveredQuantityLayout;
mod sampling;
mod source_plan;
pub use source_plan::{OriginalFieldSourceEntry, OriginalFieldSourcePlan};
mod topology;
mod view;

pub use catalog::RecoveredFieldCatalog;
pub use contract::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldArtifactEntry,
    RecoveredFieldArtifactIdentity, RecoveredFieldArtifactManifest, RecoveredFieldDefinition,
    RecoveredFieldDefinitionIdentity, RecoveredFieldDerivation, RecoveredFieldError,
    RecoveredFieldIdentity, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    RecoveredFieldSelection, RecoveredFieldSelectionIdentity,
};
pub use entity::{
    RecoveredFieldEntityCounts, RecoveredFieldEntityDomain, RecoveredFieldSupport, SupportTopology,
};
pub use finite_volume::{FiniteVolumeFieldTopology, NativeOrientedFace};
pub use partition::{RecoveredFieldPartition, RecoveredFieldPartitionIdentity};
pub use partition_set::{
    canonical_field_partition_descriptors, canonical_field_partitions, reconstruct_recovered_field,
    select_canonical_field_partitions, validate_canonical_field_partition_descriptors,
    RecoveredFieldPartitionDescriptor, RecoveredFieldPartitionSelection,
    RecoveredFieldPartitionSet, RecoveredFieldPartitionSetIdentity,
    RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};
pub(crate) use sampling::{RepeatedShape, SampledTopologyShape};
pub use sampling::{
    SampledOrientedEdge, SampledVolumeFieldTopology, VolumeFieldIntegrationSample,
    VolumeFieldSampleGeometry, VolumeFieldSamplingRule,
};
pub use topology::StructuralRecoveredFieldTopology;
pub use view::{CanonicalFieldPartitions, RecoveredFieldView};

#[cfg(test)]
mod tests;
