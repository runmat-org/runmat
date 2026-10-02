//! Physics-owned recovered fields exposed to observation evaluation.

mod catalog;
mod contract;
mod entity;
mod finite_volume;
mod partition;
mod partition_set;
mod sampling;
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
pub use entity::{RecoveredFieldEntityCounts, RecoveredFieldEntityDomain};
pub use finite_volume::{FiniteVolumeFieldTopology, NativeOrientedFace};
pub use partition::{RecoveredFieldPartition, RecoveredFieldPartitionIdentity};
pub use partition_set::{
    canonical_field_partition_descriptors, canonical_field_partitions, reconstruct_recovered_field,
    select_canonical_field_partitions, validate_canonical_field_partition_descriptors,
    RecoveredFieldPartitionDescriptor, RecoveredFieldPartitionSelection,
    RecoveredFieldPartitionSet, RecoveredFieldPartitionSetIdentity,
    RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};
pub use sampling::{
    SampledOrientedEdge, SampledVolumeFieldTopology, VolumeFieldIntegrationSample,
    VolumeFieldSamplingRule,
};
pub use topology::StructuralRecoveredFieldTopology;
pub use view::{CanonicalFieldPartitions, RecoveredFieldView};

#[cfg(test)]
mod tests;
