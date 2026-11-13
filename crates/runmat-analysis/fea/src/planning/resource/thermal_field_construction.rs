//! Sequential field identity and partition construction scratch, excluding retained transport.
use super::{ByteCount, ResourcePlanningError};
use crate::{
    execution::ThermalSolveSchedule,
    physics::thermal::history_fields,
    results::field::{
        RecoveredField, RecoveredFieldPartition, RecoveredFieldSample,
        RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
    },
};
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalFieldConstructionResourceFacts {
    pub largest_identity_materialization_bytes: ByteCount,
    pub largest_partition_construction_bytes: ByteCount,
}

impl ThermalFieldConstructionResourceFacts {
    pub fn derive(
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
        interface_nodes: usize,
    ) -> Result<Self, ResourcePlanningError> {
        let ordinal = schedule.sample_count().checked_sub(1).ok_or_else(|| {
            ResourcePlanningError::Invalid("thermal field construction requires samples".into())
        })?;
        let mut facts = Self {
            largest_identity_materialization_bytes: ByteCount(0),
            largest_partition_construction_bytes: ByteCount(0),
        };
        for (index, entities) in history_fields::snapshot_entities(
            mesh.topology.nodes.len(),
            mesh.topology.volume_elements.len(),
            interface_nodes,
        ) {
            // Last ordinal has the longest decimal suffix. Float profile width is value-independent.
            let definition =
                history_fields::snapshot_definition(index, ordinal).map_err(invalid)?;
            let sample = RecoveredFieldSample::TimeSeconds(0.0);
            // `identity_for` encodes with the direct encoder; its workspace is that profile.
            let identity =
                RecoveredField::identity_encoding_workspace_bytes(&definition, sample, entities)
                    .map_err(invalid)?;
            facts.largest_identity_materialization_bytes.0 =
                facts.largest_identity_materialization_bytes.0.max(identity);
            let partition_entities = entities.min(RECOVERED_FIELD_PARTITION_ENTITY_COUNT);
            let profile =
                RecoveredFieldPartition::profile_shape(&definition, sample, partition_entities)
                    .map_err(invalid)?;
            let components = definition.storage_component_count().map_err(invalid)?;
            let extraction = partition_entities
                .checked_mul(components)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
                .and_then(|n| {
                    partition_entities
                        .checked_mul(std::mem::size_of::<usize>())
                        .and_then(|indices| n.checked_add(indices))
                })
                .and_then(|n| n.checked_add(std::mem::size_of::<RecoveredFieldPartition>()))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal partition extraction",
                ))?;
            let bytes = profile
                .materialization_resident_bytes_estimate
                .checked_add(extraction)
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal partition construction",
                ))?;
            facts.largest_partition_construction_bytes.0 =
                facts.largest_partition_construction_bytes.0.max(bytes);
        }
        Ok(facts)
    }

    /// Identity encoding ends before partition extraction starts; retained histories are separate.
    pub fn reservation_bytes(&self) -> ByteCount {
        ByteCount(
            self.largest_identity_materialization_bytes
                .0
                .max(self.largest_partition_construction_bytes.0),
        )
    }
}

fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}
