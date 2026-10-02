//! Explicit, mesh-bound integration inventory for canonical solid history.
//!
//! A rule identifies point ownership; it does not certify integration convergence.
//! In particular, the positive Duffy alternatives remain experimental envelopes.

use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{
    CanonicalMeshingContract, ElementOrder, SolverMeshArtifact, StableDigest,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::quadrature::{
    tensor_tetrahedron_rule, GAUSS_LEGENDRE_EIGHT_UNIT, GAUSS_LEGENDRE_FIVE_UNIT,
    GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT, KEAST_FIFTEEN,
};

const SCHEMA_VERSION: u16 = 1;
mod resources;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalSolidIntegrationRule {
    Keast15,
    Duffy5,
    Duffy8,
    Duffy12,
    Duffy16,
}

impl CanonicalSolidIntegrationRule {
    pub fn point_count(self) -> usize {
        match self {
            Self::Keast15 => 15,
            Self::Duffy5 => 125,
            Self::Duffy8 => 512,
            Self::Duffy12 => 1728,
            Self::Duffy16 => 4096,
        }
    }

    /// Exact deterministic ordering from the shared reference-rule authority.
    /// This iterator does not allocate or construct physical quadrature.
    pub fn reference_points(self) -> impl Iterator<Item = ([f64; 4], f64)> {
        let tensor_rule: &'static [(f64, f64)] = match self {
            Self::Keast15 => &[],
            Self::Duffy5 => &GAUSS_LEGENDRE_FIVE_UNIT,
            Self::Duffy8 => &GAUSS_LEGENDRE_EIGHT_UNIT,
            Self::Duffy12 => &GAUSS_LEGENDRE_TWELVE_UNIT,
            Self::Duffy16 => &GAUSS_LEGENDRE_SIXTEEN_UNIT,
        };
        KEAST_FIFTEEN
            .into_iter()
            .take(if self == Self::Keast15 { 15 } else { 0 })
            .chain(tensor_tetrahedron_rule(tensor_rule))
    }

    pub fn reference_identity(self) -> Result<Digest, SolidIntegrationPlanError> {
        // Float bits bind signed zero and exact ordered reference data, not rounded text.
        let points = self
            .reference_points()
            .map(|(position, weight)| (position.map(f64::to_bits), weight.to_bits()))
            .collect::<Vec<_>>();
        encode_contract(
            b"runmat-canonical-solid-reference-rule/v1\0",
            "canonical-solid-reference-rule",
            &(SCHEMA_VERSION, self, points),
            CanonicalLimits::new(1_048_576, 4096, 256, 16),
        )
        .map(Digest::sha256)
        .map_err(|error| SolidIntegrationPlanError::Identity(error.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSolidIntegrationElement {
    pub element_id: u64,
    pub stable_identity: StableDigest,
    pub order: ElementOrder,
    pub node_count: usize,
    pub point_count: usize,
    pub point_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSolidIntegrationPlan {
    schema_version: u16,
    mesh_digest: StableDigest,
    rule: CanonicalSolidIntegrationRule,
    reference_identity: Digest,
    elements: Vec<CanonicalSolidIntegrationElement>,
    total_point_count: usize,
    total_point_node_count: usize,
    identity: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SolidIntegrationPlanError {
    #[error("invalid canonical solid mesh: {0}")]
    Mesh(String),
    #[error("solid integration count exceeds the addressable inventory")]
    CountOverflow,
    #[error("solid integration identity encoding failed: {0}")]
    Identity(String),
    #[error("solid integration plan does not match the exact mesh and requested rule")]
    AuthorityMismatch,
}

impl CanonicalSolidIntegrationPlan {
    /// Requested storage for construction, excluding the borrowed mesh, allocator
    /// overhead and RSS. This is not whole nonlinear-task resource admission.
    /// Raw malformed connectivity is profiled without physical projection.
    pub fn derivation_resident_bytes_estimate(
        mesh: &SolverMeshArtifact,
        rule: CanonicalSolidIntegrationRule,
    ) -> Result<u64, SolidIntegrationPlanError> {
        resources::derivation_resident_bytes_estimate(mesh, rule)
    }

    /// Retained requested storage, including actual ordered inventory capacity.
    pub fn retained_resident_bytes_estimate(&self) -> Result<u64, SolidIntegrationPlanError> {
        resources::retained_resident_bytes_estimate(self)
    }

    pub fn derive(
        mesh: &SolverMeshArtifact,
        rule: CanonicalSolidIntegrationRule,
    ) -> Result<Self, SolidIntegrationPlanError> {
        mesh.validate_canonical()
            .map_err(|error| SolidIntegrationPlanError::Mesh(error.to_string()))?;
        let mut elements = Vec::with_capacity(mesh.topology.volume_elements.len());
        let mut total_point_count = 0;
        let mut total_point_node_count = 0;
        for element in &mesh.topology.volume_elements {
            let node_count = match element.order {
                ElementOrder::Tet4 => 4,
                ElementOrder::Tet10 => 10,
            };
            if element.node_ids.len() != node_count {
                return Err(SolidIntegrationPlanError::Mesh(
                    "solid integration node inventory differs from its order".into(),
                ));
            }
            let point_count = rule.point_count();
            elements.push(CanonicalSolidIntegrationElement {
                element_id: element.element_id,
                stable_identity: element.stable_identity,
                order: element.order,
                node_count,
                point_count,
                point_offset: total_point_count,
            });
            accumulate_counts(
                &mut total_point_count,
                &mut total_point_node_count,
                point_count,
                node_count,
            )?;
        }
        let reference_identity = rule.reference_identity()?;
        let maximum_encoded_bytes = elements
            .len()
            .checked_mul(512)
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or(SolidIntegrationPlanError::CountOverflow)?;
        let identity = encode_contract(
            b"runmat-canonical-solid-integration-plan/v1\0",
            "canonical-solid-integration-plan",
            &(
                SCHEMA_VERSION,
                mesh.canonical_digest,
                rule,
                reference_identity,
                &elements,
                total_point_count,
                total_point_node_count,
            ),
            CanonicalLimits::new(maximum_encoded_bytes, elements.len().max(16), 256, 16),
        )
        .map(Digest::sha256)
        .map_err(|error| SolidIntegrationPlanError::Identity(error.to_string()))?;
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            mesh_digest: mesh.canonical_digest,
            rule,
            reference_identity,
            elements,
            total_point_count,
            total_point_node_count,
            identity,
        })
    }

    /// Untrusted decoded plans must validate before allocating or interpreting history.
    pub fn validate(
        &self,
        mesh: &SolverMeshArtifact,
        expected_rule: CanonicalSolidIntegrationRule,
    ) -> Result<(), SolidIntegrationPlanError> {
        if self != &Self::derive(mesh, expected_rule)? {
            return Err(SolidIntegrationPlanError::AuthorityMismatch);
        }
        Ok(())
    }

    pub fn identity(&self) -> Digest {
        self.identity
    }

    pub fn mesh_digest(&self) -> StableDigest {
        self.mesh_digest
    }

    pub fn rule(&self) -> CanonicalSolidIntegrationRule {
        self.rule
    }

    pub fn reference_identity(&self) -> Digest {
        self.reference_identity
    }

    pub fn elements(&self) -> &[CanonicalSolidIntegrationElement] {
        &self.elements
    }

    pub fn total_point_count(&self) -> usize {
        self.total_point_count
    }

    pub fn total_point_node_count(&self) -> usize {
        self.total_point_node_count
    }
}

fn accumulate_counts(
    total_points: &mut usize,
    total_point_nodes: &mut usize,
    points: usize,
    nodes: usize,
) -> Result<(), SolidIntegrationPlanError> {
    let next_points = total_points
        .checked_add(points)
        .ok_or(SolidIntegrationPlanError::CountOverflow)?;
    let next_point_nodes = points
        .checked_mul(nodes)
        .and_then(|value| total_point_nodes.checked_add(value))
        .ok_or(SolidIntegrationPlanError::CountOverflow)?;
    *total_points = next_points;
    *total_point_nodes = next_point_nodes;
    Ok(())
}

#[cfg(test)]
mod tests;
