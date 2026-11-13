//! Full canonical header scratch, separate from retained producer evidence and transport chunks.
use super::{
    ByteCount, ResourcePlanningError, ThermalManifestCatalogResourceFacts,
    ThermalMetadataResourceFacts, ThermalSourceEvidenceResourceFacts,
};
use crate::{
    admission::AnalysisRunIdentity,
    execution::{ThermalSolveSchedule, ThermalSolverResultManifest},
};
use runmat_canonical_codec::SerializedCanonicalProfile;
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalManifestMetadataResourceFacts {
    pub wrapper: SerializedCanonicalProfile,
    pub scalar_inventories: SerializedCanonicalProfile,
    pub complete_header: SerializedCanonicalProfile,
    pub numerical_identity: SerializedCanonicalProfile,
}

impl ThermalManifestMetadataResourceFacts {
    pub fn derive(
        identity: &AnalysisRunIdentity,
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
        metadata: &ThermalMetadataResourceFacts,
        sources: &ThermalSourceEvidenceResourceFacts,
        catalog: &ThermalManifestCatalogResourceFacts,
        interface_nodes: usize,
    ) -> Result<Self, ResourcePlanningError> {
        let wrapper = ThermalSolverResultManifest::wrapper_profile(identity)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let scalar_inventories = ThermalSolverResultManifest::scalar_inventory_profile(
            mesh.topology.nodes.len(),
            mesh.topology.volume_elements.len(),
            schedule.sample_count(),
            schedule.solve_count(),
            interface_nodes,
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let mut complete_header = wrapper;
        // The shared codec estimate is a positive linear sum of nodes, map entries, strings,
        // encoded bytes and fixed overhead. Adding standalone component profiles double-counts
        // wrappers/envelopes, hence bounds insertion into the skeleton rather than omitting them.
        for profile in [
            scalar_inventories,
            metadata.initial_state,
            metadata.physical_diagnostics,
            metadata.assignment_diagnostics,
            metadata.solver_diagnostics,
            catalog.catalog,
        ] {
            add(&mut complete_header, profile, 1)?;
        }
        if let Some(profile) = sources.serialization {
            add(&mut complete_header, profile, 1)?;
        }
        add(
            &mut complete_header,
            metadata.backend_evidence.record,
            u64::try_from(metadata.backend_evidence.maximum_records)
                .map_err(|_| ResourcePlanningError::Overflow("thermal manifest backend count"))?,
        )?;
        let mut numerical_identity = ThermalSolverResultManifest::numerical_identity_profile(
            identity,
            mesh.topology.nodes.len(),
            mesh.topology.volume_elements.len(),
            schedule.sample_count(),
            schedule.solve_count(),
            interface_nodes,
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        add(&mut numerical_identity, metadata.initial_state, 1)?;
        Ok(Self {
            wrapper,
            scalar_inventories,
            complete_header,
            numerical_identity,
        })
    }

    pub fn reservation_bytes(&self) -> ByteCount {
        // Identity validation completes before canonical header encoding starts.
        ByteCount(
            self.complete_header
                .materialization_resident_bytes_estimate
                .max(
                    self.numerical_identity
                        .materialization_resident_bytes_estimate,
                ),
        )
    }
}

fn add(
    total: &mut SerializedCanonicalProfile,
    value: SerializedCanonicalProfile,
    count: u64,
) -> Result<(), ResourcePlanningError> {
    let encoded = value
        .encoded_bytes_upper_bound
        .checked_mul(count)
        .and_then(|bytes| total.encoded_bytes_upper_bound.checked_add(bytes))
        .ok_or(ResourcePlanningError::Overflow(
            "thermal manifest encoded profile",
        ))?;
    let resident = value
        .materialization_resident_bytes_estimate
        .checked_mul(count)
        .and_then(|bytes| {
            total
                .materialization_resident_bytes_estimate
                .checked_add(bytes)
        })
        .ok_or(ResourcePlanningError::Overflow(
            "thermal manifest materialization profile",
        ))?;
    total.encoded_bytes_upper_bound = encoded;
    total.materialization_resident_bytes_estimate = resident;
    Ok(())
}

#[cfg(test)]
mod tests;
