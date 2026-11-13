//! Count-only requested storage for the integration authority constructor.
//! Not allocator bookkeeping, measured RSS, or a complete nonlinear-task admission.

use std::mem::size_of;

use runmat_canonical_codec::{profile_serialized_shape, CanonicalLimits};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};
use serde::{ser::SerializeSeq, Serialize, Serializer};

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
    let mut total_points = 0_usize;
    let mut total_point_nodes = 0_usize;
    for element in &mesh.topology.volume_elements {
        let nodes = node_count(element.order);
        let points = inventory_rule(mesh, element, rule).point_count();
        total_points = total_points
            .checked_add(points)
            .ok_or(SolidIntegrationPlanError::CountOverflow)?;
        total_point_nodes = points
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
        // The chained reference iterator has no exact size hint; declare the rule's point count
        // so the projected header is the owned record's exact-length header.
        let mut sequence = serializer.serialize_seq(Some(self.0.point_count()))?;
        for (position, weight) in self.0.reference_points() {
            sequence.serialize_element(&(position.map(f64::to_bits), weight.to_bits()))?;
        }
        sequence.end()
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
            |(index, element)| {
                let rule = inventory_rule(self.mesh, element, self.rule);
                CanonicalSolidIntegrationElement {
                    element_id: element.element_id,
                    stable_identity: element.stable_identity,
                    order: element.order,
                    rule,
                    node_count: node_count(element.order),
                    point_count: rule.point_count(),
                    // Integer widths are value independent in the shape profile.
                    point_offset: index * rule.point_count(),
                }
            },
        ))
    }
}

/// The rule bound for one element without running the shared quadrature: the requested rule for
/// a straight-sided element, otherwise the sixteen-point tensor rule, the largest the shared
/// Tet10 rule converges on (its encoding has the same width as the twelve-point rule).
fn inventory_rule(
    mesh: &SolverMeshArtifact,
    element: &runmat_meshing_core::SolverVolumeElement,
    requested: CanonicalSolidIntegrationRule,
) -> CanonicalSolidIntegrationRule {
    match super::tet10_coordinates(mesh, element) {
        Ok(None) => requested,
        Ok(Some(coordinates)) if runmat_meshing_core::tetrahedron10_is_affine(&coordinates) => {
            requested
        }
        _ if requested.point_count() > CanonicalSolidIntegrationRule::Duffy16.point_count() => {
            requested
        }
        _ => CanonicalSolidIntegrationRule::Duffy16,
    }
}

fn node_count(order: ElementOrder) -> usize {
    match order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    }
}

fn profile<T: Serialize>(value: &T) -> Result<u64, SolidIntegrationPlanError> {
    profile_serialized_shape(value, PROFILE_LIMITS)
        .map(|profile| profile.materialization_resident_bytes_estimate)
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
