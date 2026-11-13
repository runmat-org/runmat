//! Deterministic node and surface projection for tie and periodic relations.

use runmat_analysis_core::{PairingPolicy, PeriodicTransform};
use runmat_meshing_core::{BoundaryTriangleOrder, SolverMeshArtifact, StableDigest};

use super::contract::{
    CanonicalStructuralConstraintRow, CanonicalStructuralConstraintTerm, CanonicalStructuralDof,
    CanonicalStructuralDofOwner, ConstraintPairingEvidence, ConstraintPairingWeight,
    ConstraintPlanError, ResolvedConstraintTarget,
};
use super::row;
use super::vector::{magnitude, subtract};

#[cfg(test)]
mod tests;

mod geometry;
use geometry::{closest_triangle_point, identity, rotation_matrix, transform_position};
pub(in crate::constraints) mod profile;
use crate::topology::storage::{solver_node_identities, CanonicalNodeValues as NodeValues};
type NodeCoordinates = NodeValues<[f64; 3]>;
use super::workspace::{check_cancelled, multiply, reserve};
use profile::row_term_count;
pub use profile::{constraint_pairing_workspace_profile, ConstraintPairingWorkspaceProfile};

pub(crate) fn pair_targets(
    mesh: &SolverMeshArtifact,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    policy: &PairingPolicy,
    transform: &PeriodicTransform,
    constraint_id: &str,
) -> Result<Vec<ConstraintPairingEvidence>, ConstraintPlanError> {
    check_cancelled()?;
    validate_inputs(policy, transform, constraint_id)?;
    require_nonempty_targets(primary, secondary, constraint_id)?;
    let profile = constraint_pairing_workspace_profile(mesh, primary, secondary, policy)?;
    let nodes = NodeCoordinates::derive(
        mesh.topology
            .nodes
            .iter()
            .map(|node| (node.stable_identity, node.coordinates_m.canonical_value())),
        profile.coordinate_index_slots,
    )?;
    pair_targets_with_nodes(
        mesh,
        &nodes,
        primary,
        secondary,
        policy,
        transform,
        constraint_id,
    )
}

pub(super) fn pair_mixed_targets(
    mesh: &SolverMeshArtifact,
    topology: &crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    policy: &PairingPolicy,
    transform: &PeriodicTransform,
    constraint_id: &str,
) -> Result<Vec<ConstraintPairingEvidence>, ConstraintPlanError> {
    check_cancelled()?;
    validate_inputs(policy, transform, constraint_id)?;
    require_nonempty_targets(primary, secondary, constraint_id)?;
    let profile = profile::derive(mesh, topology.nodes.len(), primary, secondary, policy)?;
    let nodes = NodeCoordinates::derive(
        topology
            .nodes
            .iter()
            .map(|node| (node.identity, node.coordinates_m.canonical_value())),
        profile.coordinate_index_slots,
    )?;
    pair_targets_with_nodes(
        mesh,
        &nodes,
        primary,
        secondary,
        policy,
        transform,
        constraint_id,
    )
}

fn pair_targets_with_nodes(
    mesh: &SolverMeshArtifact,
    nodes: &NodeCoordinates,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    policy: &PairingPolicy,
    transform: &PeriodicTransform,
    constraint_id: &str,
) -> Result<Vec<ConstraintPairingEvidence>, ConstraintPlanError> {
    check_cancelled()?;
    let mut pairings = match policy {
        PairingPolicy::Matching => {
            pair_matching(mesh, nodes, primary, secondary, transform, constraint_id)?
        }
        PairingPolicy::GeometricTolerance(tolerance) => pair_geometrically(
            mesh,
            nodes,
            primary,
            secondary,
            transform,
            tolerance.canonical_value(),
            constraint_id,
        )?,
    };
    pairings.sort_unstable_by_key(|pairing| pairing.secondary_node_identity);
    Ok(pairings)
}

fn require_nonempty_targets(
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    constraint_id: &str,
) -> Result<(), ConstraintPlanError> {
    for target in [primary, secondary] {
        if target.matched_node_identities.is_empty() {
            return Err(ConstraintPlanError::EmptyTarget {
                constraint_id: constraint_id.into(),
                region_id: target.authored.region_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_inputs(
    policy: &PairingPolicy,
    transform: &PeriodicTransform,
    constraint_id: &str,
) -> Result<(), ConstraintPlanError> {
    let valid_policy = match policy {
        PairingPolicy::Matching => true,
        PairingPolicy::GeometricTolerance(tolerance) => {
            tolerance.canonical_value().is_finite() && tolerance.canonical_value() >= 0.0
        }
    };
    let valid_transform = match transform {
        PeriodicTransform::Identity => true,
        PeriodicTransform::Translation(value) => value.is_finite(),
        PeriodicTransform::Rotation {
            origin,
            axis,
            angle,
        } => {
            origin.is_finite()
                && angle.is_finite()
                && axis.iter().all(|value| value.is_finite())
                && magnitude(*axis) > 1.0e-12
        }
    };
    if valid_policy && valid_transform {
        Ok(())
    } else {
        Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` has an invalid pairing policy or transform"
        )))
    }
}

pub(super) fn displacement_transform(transform: &PeriodicTransform) -> [[f64; 3]; 3] {
    match transform {
        PeriodicTransform::Identity | PeriodicTransform::Translation(_) => identity(),
        PeriodicTransform::Rotation { axis, angle, .. } => {
            rotation_matrix(*axis, angle.canonical_value())
        }
    }
}

fn pair_matching(
    mesh: &SolverMeshArtifact,
    nodes: &NodeCoordinates,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    transform: &PeriodicTransform,
    constraint_id: &str,
) -> Result<Vec<ConstraintPairingEvidence>, ConstraintPlanError> {
    if primary.matched_node_identities.len() != secondary.matched_node_identities.len() {
        return Err(ConstraintPlanError::NonBijectivePairing {
            constraint_id: constraint_id.into(),
        });
    }
    let tolerance = matching_tolerance(mesh);
    let mut used = reserve::<StableDigest>(primary.matched_node_identities.len())?;
    let mut pairings = reserve(secondary.matched_node_identities.len())?;
    for secondary_identity in &secondary.matched_node_identities {
        check_cancelled()?;
        let secondary_point = node_point(nodes, *secondary_identity, constraint_id)?;
        let nearest = primary
            .matched_node_identities
            .iter()
            .enumerate()
            .map(|(ordinal, identity)| {
                if ordinal % 1024 == 0 {
                    check_cancelled()?;
                }
                let point = node_point(nodes, *identity, constraint_id)?;
                let distance = magnitude(subtract(
                    secondary_point,
                    transform_position(transform, point),
                ));
                Ok::<_, ConstraintPlanError>((*identity, distance))
            })
            .try_fold(None, |nearest: Option<(StableDigest, f64)>, candidate| {
                let candidate = candidate?;
                Ok::<_, ConstraintPlanError>(Some(match nearest {
                    Some(previous)
                        if previous
                            .1
                            .total_cmp(&candidate.1)
                            .then_with(|| previous.0.cmp(&candidate.0))
                            .is_le() =>
                    {
                        previous
                    }
                    _ => candidate,
                }))
            })?
            .filter(|(_, distance)| *distance <= tolerance)
            .ok_or(ConstraintPlanError::PairingUnavailable {
                constraint_id: constraint_id.into(),
                secondary_node_identity: *secondary_identity,
                reason: "no transformed primary node matches exactly",
            })?;
        let Err(position) = used.binary_search(&nearest.0) else {
            return Err(ConstraintPlanError::NonBijectivePairing {
                constraint_id: constraint_id.into(),
            });
        };
        used.insert(position, nearest.0);
        pairings.push(ConstraintPairingEvidence {
            secondary_node_identity: *secondary_identity,
            primary_weights: vec![ConstraintPairingWeight {
                primary_node_identity: nearest.0,
                weight: 1.0,
            }],
            geometric_error_meters: nearest.1,
        });
    }
    Ok(pairings)
}

fn matching_tolerance(mesh: &SolverMeshArtifact) -> f64 {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    let mut coordinate_scale = 1.0_f64;
    for node in &mesh.topology.nodes {
        for component in 0..3 {
            minimum[component] =
                minimum[component].min(node.coordinates_m.canonical_value()[component]);
            maximum[component] =
                maximum[component].max(node.coordinates_m.canonical_value()[component]);
            coordinate_scale =
                coordinate_scale.max(node.coordinates_m.canonical_value()[component].abs());
        }
    }
    let model_scale = magnitude(subtract(maximum, minimum));
    let tolerance = mesh.resolved_request.tolerance;
    tolerance
        .source_tolerance_m
        .canonical_value()
        .max(tolerance.absolute_floor_m.canonical_value())
        .max(tolerance.model_relative_term * model_scale)
        .max(64.0 * f64::EPSILON * coordinate_scale)
}

fn pair_geometrically(
    mesh: &SolverMeshArtifact,
    nodes: &NodeCoordinates,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    transform: &PeriodicTransform,
    tolerance: f64,
    constraint_id: &str,
) -> Result<Vec<ConstraintPairingEvidence>, ConstraintPlanError> {
    let identities_by_id = solver_node_identities(mesh)?;
    let faces = mesh.topology.boundary_faces.iter().filter(|face| {
        face.provenance
            .iter()
            .any(|entity| primary.matched_entities.contains(entity))
    });
    if faces.clone().next().is_none() {
        return Err(ConstraintPlanError::PairingUnavailable {
            constraint_id: constraint_id.into(),
            secondary_node_identity: secondary.matched_node_identities[0],
            reason: "primary target contains no boundary facets",
        });
    }
    let mut pairings = reserve(secondary.matched_node_identities.len())?;
    for secondary_identity in &secondary.matched_node_identities {
        check_cancelled()?;
        let point = node_point(nodes, *secondary_identity, constraint_id)?;
        let mut best: Option<(f64, StableDigest, Vec<ConstraintPairingWeight>)> = None;
        for face in faces.clone() {
            check_cancelled()?;
            let count = profile::facet_node_count(face.order);
            if face.node_ids.len() != count {
                return Err(ConstraintPlanError::Invalid(
                    "pairing facet node count is malformed".into(),
                ));
            }
            let mut face_nodes = [(StableDigest::ZERO, [0.0; 3]); 6];
            for (slot, node_id) in face_nodes.iter_mut().zip(&face.node_ids) {
                let identity = identities_by_id.get(node_id).copied().ok_or(
                    ConstraintPlanError::PairingUnavailable {
                        constraint_id: constraint_id.into(),
                        secondary_node_identity: *secondary_identity,
                        reason: "primary facet references an unavailable node",
                    },
                )?;
                *slot = (
                    identity,
                    transform_position(transform, node_point(nodes, identity, constraint_id)?),
                );
            }
            let (closest, barycentric) =
                closest_triangle_point(point, face_nodes[0].1, face_nodes[1].1, face_nodes[2].1);
            let distance = magnitude(subtract(point, closest));
            // Allocate weights only for a candidate that replaces the retained best facet.
            // Distance and stable identity use the same deterministic tie break as before.
            if best.as_ref().is_none_or(|current| {
                distance
                    .total_cmp(&current.0)
                    .then_with(|| face.stable_identity.cmp(&current.1))
                    .is_lt()
            }) {
                let weights = interpolation_weights(face.order, &face_nodes[..count], barycentric)?;
                best = Some((distance, face.stable_identity, weights));
            }
        }
        let Some((distance, _, primary_weights)) = best.filter(|best| best.0 <= tolerance) else {
            return Err(ConstraintPlanError::PairingUnavailable {
                constraint_id: constraint_id.into(),
                secondary_node_identity: *secondary_identity,
                reason: "closest primary surface point exceeds geometric tolerance",
            });
        };
        pairings.push(ConstraintPairingEvidence {
            secondary_node_identity: *secondary_identity,
            primary_weights,
            geometric_error_meters: distance,
        });
    }
    Ok(pairings)
}

fn interpolation_weights(
    order: BoundaryTriangleOrder,
    nodes: &[(StableDigest, [f64; 3])],
    barycentric: [f64; 3],
) -> Result<Vec<ConstraintPairingWeight>, ConstraintPlanError> {
    let [a, b, c] = barycentric;
    let values = match order {
        BoundaryTriangleOrder::Tri3 => [a, b, c, 0.0, 0.0, 0.0],
        BoundaryTriangleOrder::Tri6 => [
            a * (2.0 * a - 1.0),
            b * (2.0 * b - 1.0),
            c * (2.0 * c - 1.0),
            4.0 * a * b,
            4.0 * b * c,
            4.0 * c * a,
        ],
    };
    let mut weights = reserve(nodes.len())?;
    weights.extend(
        nodes
            .iter()
            .zip(values)
            .filter(|(_, weight)| weight.abs() > 1.0e-14)
            .map(|((identity, _), weight)| ConstraintPairingWeight {
                primary_node_identity: *identity,
                weight,
            }),
    );
    weights.sort_unstable_by_key(|weight| weight.primary_node_identity);
    Ok(weights)
}

fn node_point(
    nodes: &NodeCoordinates,
    identity: StableDigest,
    constraint_id: &str,
) -> Result<[f64; 3], ConstraintPlanError> {
    nodes
        .get(&identity)
        .copied()
        .ok_or_else(|| ConstraintPlanError::PairingUnavailable {
            constraint_id: constraint_id.into(),
            secondary_node_identity: identity,
            reason: "resolved target node is absent from solver topology",
        })
}

pub(super) fn pairing_rows(
    constraint_id: &str,
    pairings: &[ConstraintPairingEvidence],
    transform: &PeriodicTransform,
    transform_displacement: bool,
) -> Result<Vec<CanonicalStructuralConstraintRow>, ConstraintPlanError> {
    let basis = if transform_displacement {
        displacement_transform(transform)
    } else {
        identity()
    };
    let row_count = multiply(pairings.len(), 3)?;
    let mut term_slots = 0;
    let mut maximum_terms_per_row = 0;
    for pairing in pairings {
        check_cancelled()?;
        for direction in basis {
            let count = row_term_count(pairing, direction)?;
            term_slots = super::workspace::count_add(term_slots, count)?;
            maximum_terms_per_row = maximum_terms_per_row.max(count);
        }
    }
    let mut rows = super::lowering::variable_rows(
        row_count,
        term_slots,
        maximum_terms_per_row,
        constraint_id,
    )?;
    for pairing in pairings {
        check_cancelled()?;
        for (secondary_dof, transformed_direction) in
            CanonicalStructuralDof::TRANSLATIONS.into_iter().zip(basis)
        {
            let mut terms = reserve(row_term_count(pairing, transformed_direction)?)?;
            terms.push(CanonicalStructuralConstraintTerm {
                owner: CanonicalStructuralDofOwner::StructuralNode(pairing.secondary_node_identity),
                dof: secondary_dof,
                coefficient: 1.0,
            });
            for weight in &pairing.primary_weights {
                terms.extend(
                    CanonicalStructuralDof::TRANSLATIONS
                        .into_iter()
                        .zip(transformed_direction)
                        .filter(|(_, coefficient)| *coefficient != 0.0)
                        .map(|(dof, coefficient)| CanonicalStructuralConstraintTerm {
                            owner: CanonicalStructuralDofOwner::StructuralNode(
                                weight.primary_node_identity,
                            ),
                            dof,
                            coefficient: -weight.weight * coefficient,
                        }),
                );
            }
            rows.push(row::new_sparse(constraint_id, terms, 0.0)?);
        }
    }
    Ok(rows)
}
