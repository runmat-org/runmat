//! Compiler-owned identities bound to one immutable explicit structural model.
use runmat_analysis_core::StructuralModel;
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;

use crate::physics::structural::topology::{
    explicit_element_identity, explicit_identity_resource_bytes, explicit_topology_digest,
};

use super::{add, require_fits, scale, PropertyCompilationError};

pub(crate) struct PreparedExplicitTopology<'a> {
    structural: &'a StructuralModel,
    topology_digest: Digest,
    element_identities: Vec<StableDigest>,
    pub(crate) estimated_resident_bytes: u64,
}

impl<'a> PreparedExplicitTopology<'a> {
    pub(crate) fn prepare(
        structural: &'a StructuralModel,
        maximum: u64,
    ) -> Result<Self, PropertyCompilationError> {
        let identity_bytes = explicit_identity_resource_bytes(structural).map_err(|error| {
            PropertyCompilationError::InvalidExplicitTopology(error.to_string())
        })?;
        let cache_bytes = scale(
            structural.elements.len(),
            std::mem::size_of::<StableDigest>() as u64,
        )?;
        let estimated_resident_bytes = add(identity_bytes, cache_bytes)?;
        // No canonical encoding or identity-cache allocation precedes this admission.
        require_fits(estimated_resident_bytes, maximum)?;
        let topology_digest = explicit_topology_digest(structural).map_err(|error| {
            PropertyCompilationError::InvalidExplicitTopology(error.to_string())
        })?;
        let mut element_identities = Vec::new();
        element_identities
            .try_reserve_exact(structural.elements.len())
            .map_err(|error| PropertyCompilationError::Resource(error.to_string()))?;
        for element in &structural.elements {
            element_identities.push(
                explicit_element_identity(
                    topology_digest,
                    &element.element_id,
                    &element.region_id,
                    &element.kind,
                )
                .map_err(|error| {
                    PropertyCompilationError::InvalidExplicitTopology(error.to_string())
                })?,
            );
        }
        Ok(Self {
            structural,
            topology_digest,
            element_identities,
            estimated_resident_bytes,
        })
    }

    pub(crate) fn structural(&self) -> &'a StructuralModel {
        self.structural
    }

    pub(crate) fn topology_digest(&self) -> Digest {
        self.topology_digest
    }

    pub(crate) fn element_identities(&self) -> &[StableDigest] {
        &self.element_identities
    }
}
