//! Retained sampled-field catalog and its bounded validation scratch; not full manifest metadata.
use super::{ByteCount, ResourcePlanningError};
use crate::{
    execution::ThermalSolveSchedule,
    physics::thermal::history_fields,
    results::field::{
        RecoveredFieldArtifactEntry, RecoveredFieldIdentity, RecoveredFieldPartitionIdentity,
        RecoveredFieldSample, RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
    },
};
use runmat_canonical_codec::SerializedCanonicalProfile;
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalManifestCatalogResourceFacts {
    pub catalog: SerializedCanonicalProfile,
    pub validation_scratch_bytes: ByteCount,
    pub field_identity_bytes: ByteCount,
}

impl ThermalManifestCatalogResourceFacts {
    pub fn derive(
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
        interface_nodes: usize,
    ) -> Result<Self, ResourcePlanningError> {
        let samples = schedule.sample_count();
        let ordinal = samples.checked_sub(1).ok_or_else(|| {
            ResourcePlanningError::Invalid("thermal catalog requires samples".into())
        })?;
        let entities = history_fields::snapshot_entities(
            mesh.topology.nodes.len(),
            mesh.topology.volume_elements.len(),
            interface_nodes,
        );
        let definitions = entities
            .iter()
            .map(|&(index, _)| history_fields::snapshot_definition(index, ordinal))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let counts = entities.iter().map(|&(_, count)| count).collect::<Vec<_>>();
        let catalog = RecoveredFieldArtifactEntry::profile_sampled_catalog_shape(
            &definitions,
            &counts,
            RecoveredFieldSample::TimeSeconds(0.0),
            samples,
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let partitions = counts
            .iter()
            .try_fold(0usize, |sum, count| {
                sum.checked_add(count.div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT))
            })
            .and_then(|count| count.checked_mul(samples))
            .ok_or(ResourcePlanningError::Overflow(
                "thermal catalog partition count",
            ))?;
        let validation_scratch_bytes = vector_bytes::<RecoveredFieldPartitionIdentity>(partitions)?;
        let field_identity_bytes = vector_bytes::<Vec<RecoveredFieldIdentity>>(samples)?
            .0
            .checked_add(
                vector_bytes::<RecoveredFieldIdentity>(counts.len())?
                    .0
                    .checked_mul(samples as u64)
                    .ok_or(ResourcePlanningError::Overflow(
                        "thermal catalog identities",
                    ))?,
            )
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal catalog identities",
            ))?;
        Ok(Self {
            catalog,
            validation_scratch_bytes,
            field_identity_bytes,
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        // Conservative sum: retained catalog + canonical tree/buffer estimate + bounded scratch.
        self.catalog
            .materialization_resident_bytes_estimate
            .checked_add(self.validation_scratch_bytes.0)
            .and_then(|bytes| bytes.checked_add(self.field_identity_bytes.0))
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow("thermal manifest catalog"))
    }
}

fn vector_bytes<T>(count: usize) -> Result<ByteCount, ResourcePlanningError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<T>>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .map(ByteCount)
        .ok_or(ResourcePlanningError::Overflow("thermal catalog scratch"))
}
