//! Explicit, mesh-bound integration inventory for canonical solid history.
//!
//! A rule identifies point ownership; it does not certify integration convergence.
//! In particular, the positive Duffy alternatives remain unadmitted envelopes.

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

const SCHEMA_VERSION: u16 = 2;
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
    /// The element's own rule: the requested rule for straight-sided elements, and for a curved
    /// Tet10 the converged rule of the shared Tet10 element-validity rule.
    pub rule: CanonicalSolidIntegrationRule,
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
    #[error("solid integration numerical accounting failed: {0}")]
    Accounting(runmat_simulation_usage::LedgerError),
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
            let element_rule = element_rule(mesh, element, rule)?;
            let point_count = element_rule.point_count();
            elements.push(CanonicalSolidIntegrationElement {
                element_id: element.element_id,
                stable_identity: element.stable_identity,
                order: element.order,
                rule: element_rule,
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

/// The rule that integrates one element. A curved Tet10 integrates with the quadrature the
/// shared Tet10 element-validity rule converged on (`runmat_meshing_core::
/// tetrahedron10_stiffness_moments`), never with a fixed rule; straight-sided elements use the
/// requested rule. An element the shared rule rejects has no admissible quadrature.
fn element_rule(
    mesh: &SolverMeshArtifact,
    element: &runmat_meshing_core::SolverVolumeElement,
    requested: CanonicalSolidIntegrationRule,
) -> Result<CanonicalSolidIntegrationRule, SolidIntegrationPlanError> {
    let Some(coordinates) = tet10_coordinates(mesh, element)? else {
        return Ok(requested);
    };
    if runmat_meshing_core::tetrahedron10_is_affine(&coordinates) {
        return Ok(requested);
    }
    match runmat_meshing_core::tetrahedron10_stiffness_moments_for_category(
        &coordinates,
        runmat_simulation_usage::UsageCategory::Assembly,
    ) {
        Ok(moments) => Ok(curved_rule(moments.rule).unwrap_or(requested)),
        Err(runmat_meshing_core::Tet10ElementInvalidity::Accounting(error)) => {
            Err(SolidIntegrationPlanError::Accounting(error))
        }
        Err(invalidity) => Err(SolidIntegrationPlanError::Mesh(format!(
            "element {} fails the shared Tet10 element-validity rule: {invalidity:?}",
            element.element_id
        ))),
    }
}

/// The plan rule carrying the shared rule's converged tensor quadrature.
pub(crate) fn curved_rule(
    rule: runmat_meshing_core::Tet10StiffnessRule,
) -> Option<CanonicalSolidIntegrationRule> {
    match rule {
        runmat_meshing_core::Tet10StiffnessRule::Affine => None,
        runmat_meshing_core::Tet10StiffnessRule::GaussTwelve => {
            Some(CanonicalSolidIntegrationRule::Duffy12)
        }
        runmat_meshing_core::Tet10StiffnessRule::GaussSixteen => {
            Some(CanonicalSolidIntegrationRule::Duffy16)
        }
    }
}

/// A Tet10 element's node coordinates in canonical order, or `None` for other orders.
pub(super) fn tet10_coordinates(
    mesh: &SolverMeshArtifact,
    element: &runmat_meshing_core::SolverVolumeElement,
) -> Result<Option<[[f64; 3]; 10]>, SolidIntegrationPlanError> {
    if element.order != ElementOrder::Tet10 {
        return Ok(None);
    }
    let mut coordinates = [[0.0; 3]; 10];
    for (slot, node_id) in coordinates.iter_mut().zip(&element.node_ids) {
        let index = mesh
            .topology
            .nodes
            .binary_search_by_key(node_id, |node| node.node_id)
            .map_err(|_| {
                SolidIntegrationPlanError::Mesh("solid integration node is absent".into())
            })?;
        *slot = mesh.topology.nodes[index].coordinates_m.canonical_value();
    }
    Ok(Some(coordinates))
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
