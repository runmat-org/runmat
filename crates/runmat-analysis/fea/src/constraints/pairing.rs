//! Deterministic node and surface projection for tie and periodic relations.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{PairingPolicy, PeriodicTransform};
use runmat_meshing_core::{BoundaryTriangleOrder, SolverMeshArtifact, StableDigest};

use super::contract::{
    CanonicalStructuralConstraintRow, CanonicalStructuralConstraintTerm, CanonicalStructuralDof,
    CanonicalStructuralDofOwner, ConstraintPairingEvidence, ConstraintPairingWeight,
    ConstraintPlanError, ResolvedConstraintTarget,
};
use super::row;
use super::vector::{add, cross, dot, magnitude, scale, subtract};

pub(super) fn pair_targets(
    mesh: &SolverMeshArtifact,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    policy: &PairingPolicy,
    transform: &PeriodicTransform,
    constraint_id: &str,
) -> Result<Vec<ConstraintPairingEvidence>, ConstraintPlanError> {
    validate_inputs(policy, transform, constraint_id)?;
    let nodes = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.stable_identity, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
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
    let nodes = topology
        .nodes
        .iter()
        .map(|node| (node.identity, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
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
    nodes: &BTreeMap<StableDigest, [f64; 3]>,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    policy: &PairingPolicy,
    transform: &PeriodicTransform,
    constraint_id: &str,
) -> Result<Vec<ConstraintPairingEvidence>, ConstraintPlanError> {
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
            tolerance.meters,
            constraint_id,
        )?,
    };
    pairings.sort_by_key(|pairing| pairing.secondary_node_identity);
    Ok(pairings)
}

fn validate_inputs(
    policy: &PairingPolicy,
    transform: &PeriodicTransform,
    constraint_id: &str,
) -> Result<(), ConstraintPlanError> {
    let valid_policy = match policy {
        PairingPolicy::Matching => true,
        PairingPolicy::GeometricTolerance(tolerance) => {
            tolerance.meters.is_finite() && tolerance.meters >= 0.0
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
        PeriodicTransform::Rotation { axis, angle, .. } => rotation_matrix(*axis, angle.radians),
    }
}

fn pair_matching(
    mesh: &SolverMeshArtifact,
    nodes: &BTreeMap<StableDigest, [f64; 3]>,
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
    let mut used = BTreeSet::new();
    let mut pairings = Vec::with_capacity(secondary.matched_node_identities.len());
    for secondary_identity in &secondary.matched_node_identities {
        let secondary_point = node_point(nodes, *secondary_identity, constraint_id)?;
        let nearest = primary
            .matched_node_identities
            .iter()
            .map(|identity| {
                let point = node_point(nodes, *identity, constraint_id)?;
                let distance = magnitude(subtract(
                    secondary_point,
                    transform_position(transform, point),
                ));
                Ok((*identity, distance))
            })
            .collect::<Result<Vec<_>, ConstraintPlanError>>()?
            .into_iter()
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            })
            .filter(|(_, distance)| *distance <= tolerance)
            .ok_or(ConstraintPlanError::PairingUnavailable {
                constraint_id: constraint_id.into(),
                secondary_node_identity: *secondary_identity,
                reason: "no transformed primary node matches exactly",
            })?;
        if !used.insert(nearest.0) {
            return Err(ConstraintPlanError::NonBijectivePairing {
                constraint_id: constraint_id.into(),
            });
        }
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
            minimum[component] = minimum[component].min(node.coordinates_m[component]);
            maximum[component] = maximum[component].max(node.coordinates_m[component]);
            coordinate_scale = coordinate_scale.max(node.coordinates_m[component].abs());
        }
    }
    let model_scale = magnitude(subtract(maximum, minimum));
    let tolerance = mesh.resolved_request.tolerance;
    tolerance
        .source_tolerance_m
        .max(tolerance.absolute_floor_m)
        .max(tolerance.model_relative_term * model_scale)
        .max(64.0 * f64::EPSILON * coordinate_scale)
}

fn pair_geometrically(
    mesh: &SolverMeshArtifact,
    nodes: &BTreeMap<StableDigest, [f64; 3]>,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    transform: &PeriodicTransform,
    tolerance: f64,
    constraint_id: &str,
) -> Result<Vec<ConstraintPairingEvidence>, ConstraintPlanError> {
    let identities_by_id = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.stable_identity))
        .collect::<BTreeMap<_, _>>();
    let faces = mesh
        .topology
        .boundary_faces
        .iter()
        .filter(|face| {
            face.provenance
                .iter()
                .any(|entity| primary.matched_entities.contains(entity))
        })
        .collect::<Vec<_>>();
    if faces.is_empty() {
        return Err(ConstraintPlanError::PairingUnavailable {
            constraint_id: constraint_id.into(),
            secondary_node_identity: secondary.matched_node_identities[0],
            reason: "primary target contains no boundary facets",
        });
    }
    let mut pairings = Vec::with_capacity(secondary.matched_node_identities.len());
    for secondary_identity in &secondary.matched_node_identities {
        let point = node_point(nodes, *secondary_identity, constraint_id)?;
        let mut best: Option<(f64, StableDigest, Vec<ConstraintPairingWeight>)> = None;
        for face in &faces {
            let face_nodes = face
                .node_ids
                .iter()
                .map(|node_id| {
                    let identity = identities_by_id.get(node_id).copied().ok_or(
                        ConstraintPlanError::PairingUnavailable {
                            constraint_id: constraint_id.into(),
                            secondary_node_identity: *secondary_identity,
                            reason: "primary facet references an unavailable node",
                        },
                    )?;
                    Ok((
                        identity,
                        transform_position(transform, node_point(nodes, identity, constraint_id)?),
                    ))
                })
                .collect::<Result<Vec<_>, ConstraintPlanError>>()?;
            let (closest, barycentric) =
                closest_triangle_point(point, face_nodes[0].1, face_nodes[1].1, face_nodes[2].1);
            let distance = magnitude(subtract(point, closest));
            let weights = interpolation_weights(face.order, &face_nodes, barycentric);
            let candidate = (distance, face.stable_identity, weights);
            if best.as_ref().is_none_or(|current| {
                candidate
                    .0
                    .total_cmp(&current.0)
                    .then_with(|| candidate.1.cmp(&current.1))
                    .is_lt()
            }) {
                best = Some(candidate);
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
) -> Vec<ConstraintPairingWeight> {
    let [a, b, c] = barycentric;
    let values = match order {
        BoundaryTriangleOrder::Tri3 => vec![a, b, c],
        BoundaryTriangleOrder::Tri6 => vec![
            a * (2.0 * a - 1.0),
            b * (2.0 * b - 1.0),
            c * (2.0 * c - 1.0),
            4.0 * a * b,
            4.0 * b * c,
            4.0 * c * a,
        ],
    };
    let mut weights = nodes
        .iter()
        .zip(values)
        .filter(|(_, weight)| weight.abs() > 1.0e-14)
        .map(|((identity, _), weight)| ConstraintPairingWeight {
            primary_node_identity: *identity,
            weight,
        })
        .collect::<Vec<_>>();
    weights.sort_by_key(|weight| weight.primary_node_identity);
    weights
}

fn node_point(
    nodes: &BTreeMap<StableDigest, [f64; 3]>,
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
    let mut rows = Vec::with_capacity(pairings.len() * 3);
    for pairing in pairings {
        for (secondary_dof, transformed_direction) in
            CanonicalStructuralDof::TRANSLATIONS.into_iter().zip(basis)
        {
            let mut terms = vec![CanonicalStructuralConstraintTerm {
                owner: CanonicalStructuralDofOwner::StructuralNode(pairing.secondary_node_identity),
                dof: secondary_dof,
                coefficient: 1.0,
            }];
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

fn transform_position(transform: &PeriodicTransform, point: [f64; 3]) -> [f64; 3] {
    match transform {
        PeriodicTransform::Identity => point,
        PeriodicTransform::Translation(value) => add(point, value.meters),
        PeriodicTransform::Rotation {
            origin,
            axis,
            angle,
        } => add(
            origin.meters,
            matrix_vector(
                rotation_matrix(*axis, angle.radians),
                subtract(point, origin.meters),
            ),
        ),
    }
}

fn rotation_matrix(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let inverse_length = magnitude(axis).recip();
    let [x, y, z] = scale(axis, inverse_length);
    let cosine = angle.cos();
    let sine = angle.sin();
    let one_minus_cosine = 1.0 - cosine;
    [
        [
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ],
        [
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ],
        [
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ],
    ]
}

fn matrix_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| dot(row, vector))
}

fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

// Ericson's closest-point regions, returning a point and barycentric coordinates.
fn closest_triangle_point(
    point: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let ab = subtract(b, a);
    let ac = subtract(c, a);
    if magnitude(cross(ab, ac)) <= 1.0e-18 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let ap = subtract(point, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = subtract(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add(a, scale(ab, v)), [1.0 - v, v, 0.0]);
    }
    let cp = subtract(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add(a, scale(ac, w)), [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(b, scale(subtract(c, b), w)), [0.0, 1.0 - w, w]);
    }
    let denominator = (va + vb + vc).recip();
    let v = vb * denominator;
    let w = vc * denominator;
    (add(a, add(scale(ab, v), scale(ac, w))), [1.0 - v - w, v, w])
}
