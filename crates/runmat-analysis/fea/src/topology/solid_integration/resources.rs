//! Count-only requested storage for the integration authority constructor.
//! Not allocator bookkeeping, measured RSS, or a complete nonlinear-task admission.

use std::mem::size_of;

use runmat_canonical_codec::{serialized_materialization_resident_bytes_estimate, CanonicalLimits};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};
use serde::{Serialize, Serializer};

use super::{
    CanonicalSolidIntegrationElement, CanonicalSolidIntegrationPlan, CanonicalSolidIntegrationRule,
    Digest, SolidIntegrationPlanError, SCHEMA_VERSION,
};

const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

pub(super) fn retained_resident_bytes_estimate(
    plan: &CanonicalSolidIntegrationPlan,
) -> Result<u64, SolidIntegrationPlanError> {
    let mut total = size_of::<CanonicalSolidIntegrationPlan>() as u64;
    add(
        &mut total,
        slots::<CanonicalSolidIntegrationElement>(plan.elements.capacity())?,
    )?;
    Ok(total)
}

/// Sum sequential constructor phases conservatively, excluding the borrowed mesh.
/// Does not validate the mesh, encode a contract, or construct physical quadrature.
/// Shared mesh validation profiling counts raw malformed connectivity as supplied.
pub(super) fn derivation_resident_bytes_estimate(
    mesh: &SolverMeshArtifact,
    rule: CanonicalSolidIntegrationRule,
) -> Result<u64, SolidIntegrationPlanError> {
    let element_count = mesh.topology.volume_elements.len();
    let total_points = element_count
        .checked_mul(rule.point_count())
        .ok_or(SolidIntegrationPlanError::CountOverflow)?;
    let mut total_point_nodes = 0_usize;
    for element in &mesh.topology.volume_elements {
        let nodes = node_count(element.order);
        total_point_nodes = rule
            .point_count()
            .checked_mul(nodes)
            .and_then(|count| total_point_nodes.checked_add(count))
            .ok_or(SolidIntegrationPlanError::CountOverflow)?;
    }
    let mut total = mesh
        .validation_resident_bytes_estimate()
        .map_err(|error| SolidIntegrationPlanError::Mesh(error.to_string()))?;
    add(
        &mut total,
        size_of::<CanonicalSolidIntegrationPlan>() as u64,
    )?;
    add(
        &mut total,
        slots::<CanonicalSolidIntegrationElement>(element_count)?,
    )?;
    // reference_identity collects exact bit tuples before shared canonical encoding.
    // Its iterator lacks an exact-size promise; include conservative Vec growth.
    let reference_slots = rule
        .point_count()
        .checked_mul(2)
        .and_then(|count| count.checked_add(4))
        .ok_or(SolidIntegrationPlanError::CountOverflow)?;
    add(&mut total, slots::<([u64; 4], u64)>(reference_slots)?)?;
    add(
        &mut total,
        profile(&(SCHEMA_VERSION, rule, ReferenceBits(rule)))?,
    )?;
    // Digests serialize to fixed-width values. A placeholder preserves the exact
    // identity contract shape without computing reference or plan encodings.
    let reference_identity = Digest::sha256(b"solid-integration-resource-profile");
    add(
        &mut total,
        profile(&(
            SCHEMA_VERSION,
            mesh.canonical_digest,
            rule,
            reference_identity,
            ElementInventory { mesh, rule },
            total_points,
            total_point_nodes,
        ))?,
    )?;
    Ok(total)
}

struct ReferenceBits(CanonicalSolidIntegrationRule);
impl Serialize for ReferenceBits {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.0
                .reference_points()
                .map(|(position, weight)| (position.map(f64::to_bits), weight.to_bits())),
        )
    }
}

struct ElementInventory<'a> {
    mesh: &'a SolverMeshArtifact,
    rule: CanonicalSolidIntegrationRule,
}
impl Serialize for ElementInventory<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // The entire count was checked before this borrowed traversal.
        serializer.collect_seq(self.mesh.topology.volume_elements.iter().enumerate().map(
            |(index, element)| CanonicalSolidIntegrationElement {
                element_id: element.element_id,
                stable_identity: element.stable_identity,
                order: element.order,
                node_count: node_count(element.order),
                point_count: self.rule.point_count(),
                point_offset: index * self.rule.point_count(),
            },
        ))
    }
}

fn node_count(order: ElementOrder) -> usize {
    match order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    }
}

fn profile<T: Serialize>(value: &T) -> Result<u64, SolidIntegrationPlanError> {
    serialized_materialization_resident_bytes_estimate(value, PROFILE_LIMITS)
        .map_err(|error| SolidIntegrationPlanError::Identity(error.to_string()))
}

fn slots<T>(count: usize) -> Result<u64, SolidIntegrationPlanError> {
    u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<T>() as u64))
        .ok_or(SolidIntegrationPlanError::CountOverflow)
}

fn add(total: &mut u64, bytes: u64) -> Result<(), SolidIntegrationPlanError> {
    *total = total
        .checked_add(bytes)
        .ok_or(SolidIntegrationPlanError::CountOverflow)?;
    Ok(())
}

#[cfg(test)]
mod tests;
