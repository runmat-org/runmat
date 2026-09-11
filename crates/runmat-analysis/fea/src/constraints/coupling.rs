//! Reference-point coupling equations and conservative target weights.

use std::collections::BTreeMap;

use runmat_analysis_core::{ReferencePoint, StructuralRotationDof};
use runmat_execution::Digest;
use runmat_meshing_core::{BoundaryTriangleOrder, SolverMeshArtifact, StableDigest};

use super::contract::{
    CanonicalStructuralConstraintRow, CanonicalStructuralConstraintTerm, CanonicalStructuralDof,
    CanonicalStructuralDofOwner, ConstraintCouplingEvidence, ConstraintCouplingKind,
    ConstraintCouplingWeight, ConstraintPlanError, ResolvedConstraintTarget,
};
use super::row;
use super::vector::{cross, magnitude, scale, subtract};

pub(super) fn compile_reference_coupling(
    mesh: &SolverMeshArtifact,
    target: &ResolvedConstraintTarget,
    reference_point: &ReferencePoint,
    reference_point_identity: Digest,
    kind: ConstraintCouplingKind,
    constraint_id: &str,
) -> Result<
    (
        Vec<CanonicalStructuralConstraintRow>,
        ConstraintCouplingEvidence,
    ),
    ConstraintPlanError,
> {
    let weights = target_weights(mesh, target, constraint_id)?;
    let coordinates = coordinates_by_identity(mesh);
    let characteristic_length_meters = weights
        .iter()
        .map(|weight| {
            let position = coordinates
                .get(&weight.node_identity)
                .copied()
                .ok_or_else(|| {
                    ConstraintPlanError::Invalid(format!(
                        "constraint `{constraint_id}` coupling node is absent from solver topology"
                    ))
                })?;
            Ok(weight.weight
                * magnitude(subtract(position, reference_point.position.meters)).powi(2))
        })
        .collect::<Result<Vec<_>, ConstraintPlanError>>()?
        .into_iter()
        .sum::<f64>()
        .sqrt();
    validate_rotational_geometry(
        &weights,
        &coordinates,
        characteristic_length_meters,
        constraint_id,
    )?;
    let rows = match kind {
        ConstraintCouplingKind::Rigid => rigid_rows(
            mesh,
            &weights,
            reference_point,
            reference_point_identity,
            constraint_id,
        )?,
        ConstraintCouplingKind::Distributing => distributing_rows(
            mesh,
            &weights,
            reference_point,
            reference_point_identity,
            constraint_id,
        )?,
    };
    Ok((
        rows,
        ConstraintCouplingEvidence {
            kind,
            reference_point_identity,
            target_weights: weights,
            characteristic_length_meters,
        },
    ))
}

fn validate_rotational_geometry(
    weights: &[ConstraintCouplingWeight],
    coordinates: &BTreeMap<StableDigest, [f64; 3]>,
    characteristic_length: f64,
    constraint_id: &str,
) -> Result<(), ConstraintPlanError> {
    if !characteristic_length.is_finite() || characteristic_length <= 1.0e-12 {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` coupling has no rotational lever arm"
        )));
    }
    let centroid = weights.iter().try_fold([0.0; 3], |sum, weight| {
        coordinates
            .get(&weight.node_identity)
            .copied()
            .map(|position| super::vector::add(sum, scale(position, weight.weight)))
            .ok_or_else(|| {
                ConstraintPlanError::Invalid(format!(
                    "constraint `{constraint_id}` coupling node is absent from solver topology"
                ))
            })
    })?;
    let mut inertia = [[0.0; 3]; 3];
    for weight in weights {
        let arm = subtract(coordinates[&weight.node_identity], centroid);
        let squared_radius = arm.iter().map(|value| value * value).sum::<f64>();
        for row in 0..3 {
            for column in 0..3 {
                inertia[row][column] += weight.weight
                    * if row == column {
                        squared_radius - arm[row] * arm[column]
                    } else {
                        -arm[row] * arm[column]
                    };
            }
        }
    }
    let scale = inertia
        .iter()
        .flatten()
        .map(|value| value.abs())
        .sum::<f64>();
    let tolerance = scale.max(f64::MIN_POSITIVE) * 1.0e-12;
    let first = inertia[0][0];
    if !first.is_finite() || first <= tolerance {
        return Err(degenerate_coupling(constraint_id));
    }
    let second = inertia[1][1] - inertia[1][0] * inertia[1][0] / first;
    if !second.is_finite() || second <= tolerance {
        return Err(degenerate_coupling(constraint_id));
    }
    let lower_20 = inertia[2][0] / first.sqrt();
    let lower_21 = (inertia[2][1] - lower_20 * inertia[1][0] / first.sqrt()) / second.sqrt();
    let third = inertia[2][2] - lower_20 * lower_20 - lower_21 * lower_21;
    if !third.is_finite() || third <= tolerance {
        return Err(degenerate_coupling(constraint_id));
    }
    Ok(())
}

fn degenerate_coupling(constraint_id: &str) -> ConstraintPlanError {
    ConstraintPlanError::Invalid(format!(
        "constraint `{constraint_id}` coupling target must contain non-collinear nodes"
    ))
}

fn rigid_rows(
    mesh: &SolverMeshArtifact,
    weights: &[ConstraintCouplingWeight],
    reference_point: &ReferencePoint,
    reference_identity: Digest,
    constraint_id: &str,
) -> Result<Vec<CanonicalStructuralConstraintRow>, ConstraintPlanError> {
    let coordinates = coordinates_by_identity(mesh);
    let mut rows = Vec::with_capacity(weights.len() * 3);
    for weight in weights {
        let position = coordinates
            .get(&weight.node_identity)
            .copied()
            .ok_or_else(|| {
                ConstraintPlanError::Invalid(format!(
                    "constraint `{constraint_id}` coupling node is absent from solver topology"
                ))
            })?;
        let arm = subtract(position, reference_point.position.meters);
        for component in 0..3 {
            let mut terms = vec![
                mesh_term(weight.node_identity, component, 1.0),
                reference_translation_term(reference_identity, component, -1.0),
            ];
            for rotation in 0..3 {
                let coefficient = -cross(unit_axis(rotation), arm)[component];
                if coefficient != 0.0 {
                    terms.push(reference_rotation_term(
                        reference_identity,
                        rotation,
                        coefficient,
                    ));
                }
            }
            rows.push(row::new_sparse(constraint_id, terms, 0.0)?);
        }
    }
    Ok(rows)
}

fn distributing_rows(
    mesh: &SolverMeshArtifact,
    weights: &[ConstraintCouplingWeight],
    reference_point: &ReferencePoint,
    reference_identity: Digest,
    constraint_id: &str,
) -> Result<Vec<CanonicalStructuralConstraintRow>, ConstraintPlanError> {
    let coordinates = coordinates_by_identity(mesh);
    let arms = weights
        .iter()
        .map(|weight| {
            coordinates
                .get(&weight.node_identity)
                .copied()
                .map(|position| subtract(position, reference_point.position.meters))
                .ok_or_else(|| {
                    ConstraintPlanError::Invalid(format!(
                        "constraint `{constraint_id}` coupling node is absent from solver topology"
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mean_arm = weights
        .iter()
        .zip(&arms)
        .fold([0.0; 3], |sum, (weight, arm)| {
            super::vector::add(sum, scale(*arm, weight.weight))
        });
    let characteristic_length = weights
        .iter()
        .zip(&arms)
        .map(|(weight, arm)| weight.weight * magnitude(*arm).powi(2))
        .sum::<f64>()
        .sqrt();
    if !characteristic_length.is_finite() || characteristic_length <= 1.0e-12 {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` distributing coupling has no rotational lever arm"
        )));
    }
    let mut rows = Vec::with_capacity(6);

    for component in 0..3 {
        let mut terms = weights
            .iter()
            .map(|weight| mesh_term(weight.node_identity, component, weight.weight))
            .collect::<Vec<_>>();
        terms.push(reference_translation_term(
            reference_identity,
            component,
            -1.0,
        ));
        for rotation in 0..3 {
            let coefficient = -cross(unit_axis(rotation), mean_arm)[component];
            if coefficient != 0.0 {
                terms.push(reference_rotation_term(
                    reference_identity,
                    rotation,
                    coefficient,
                ));
            }
        }
        rows.push(row::new_sparse(constraint_id, terms, 0.0)?);
    }

    for moment_component in 0..3 {
        let mut terms = Vec::new();
        for (weight, arm) in weights.iter().zip(&arms) {
            for component in 0..3 {
                let coefficient = weight.weight
                    * cross(*arm, unit_axis(component))[moment_component]
                    / characteristic_length;
                if coefficient != 0.0 {
                    terms.push(mesh_term(weight.node_identity, component, coefficient));
                }
            }
        }
        for component in 0..3 {
            let coefficient =
                -cross(mean_arm, unit_axis(component))[moment_component] / characteristic_length;
            if coefficient != 0.0 {
                terms.push(reference_translation_term(
                    reference_identity,
                    component,
                    coefficient,
                ));
            }
        }
        for rotation in 0..3 {
            let coefficient = -weights
                .iter()
                .zip(&arms)
                .map(|(weight, arm)| {
                    weight.weight * cross(*arm, cross(unit_axis(rotation), *arm))[moment_component]
                })
                .sum::<f64>()
                / characteristic_length;
            if coefficient != 0.0 {
                terms.push(reference_rotation_term(
                    reference_identity,
                    rotation,
                    coefficient,
                ));
            }
        }
        rows.push(row::new_sparse(constraint_id, terms, 0.0)?);
    }
    Ok(rows)
}

fn target_weights(
    mesh: &SolverMeshArtifact,
    target: &ResolvedConstraintTarget,
    constraint_id: &str,
) -> Result<Vec<ConstraintCouplingWeight>, ConstraintPlanError> {
    if target.matched_entities.is_empty() && target.matched_node_identities.len() == 1 {
        return Ok(vec![ConstraintCouplingWeight {
            node_identity: target.matched_node_identities[0],
            weight: 1.0,
        }]);
    }
    let nodes_by_id = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect::<BTreeMap<_, _>>();
    let mut measures = BTreeMap::<StableDigest, f64>::new();
    for face in mesh.topology.boundary_faces.iter().filter(|face| {
        face.provenance
            .iter()
            .any(|entity| target.matched_entities.contains(entity))
    }) {
        let triangles: &[[usize; 3]] = match face.order {
            BoundaryTriangleOrder::Tri3 => &[[0, 1, 2]],
            BoundaryTriangleOrder::Tri6 => &[[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]],
        };
        for indices in triangles {
            let vertices = indices.map(|index| {
                face.node_ids
                    .get(index)
                    .and_then(|node_id| nodes_by_id.get(node_id))
            });
            let [Some(a), Some(b), Some(c)] = vertices else {
                return Err(ConstraintPlanError::Invalid(format!(
                    "constraint `{constraint_id}` coupling facet is malformed"
                )));
            };
            let area = 0.5
                * magnitude(cross(
                    subtract(b.coordinates_m, a.coordinates_m),
                    subtract(c.coordinates_m, a.coordinates_m),
                ));
            if !area.is_finite() || area <= 1.0e-18 {
                return Err(ConstraintPlanError::Invalid(format!(
                    "constraint `{constraint_id}` coupling facet has zero area"
                )));
            }
            for node in [a, b, c] {
                *measures.entry(node.stable_identity).or_insert(0.0) += area / 3.0;
            }
        }
    }
    let total = measures.values().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` coupling target has no measurable boundary"
        )));
    }
    Ok(measures
        .into_iter()
        .map(|(node_identity, measure)| ConstraintCouplingWeight {
            node_identity,
            weight: measure / total,
        })
        .collect())
}

fn coordinates_by_identity(mesh: &SolverMeshArtifact) -> BTreeMap<StableDigest, [f64; 3]> {
    mesh.topology
        .nodes
        .iter()
        .map(|node| (node.stable_identity, node.coordinates_m))
        .collect()
}

fn mesh_term(
    identity: StableDigest,
    component: usize,
    coefficient: f64,
) -> CanonicalStructuralConstraintTerm {
    CanonicalStructuralConstraintTerm {
        owner: CanonicalStructuralDofOwner::StructuralNode(identity),
        dof: CanonicalStructuralDof::TRANSLATIONS[component],
        coefficient,
    }
}

fn reference_translation_term(
    identity: Digest,
    component: usize,
    coefficient: f64,
) -> CanonicalStructuralConstraintTerm {
    CanonicalStructuralConstraintTerm {
        owner: CanonicalStructuralDofOwner::ReferencePoint(identity),
        dof: CanonicalStructuralDof::TRANSLATIONS[component],
        coefficient,
    }
}

fn reference_rotation_term(
    identity: Digest,
    component: usize,
    coefficient: f64,
) -> CanonicalStructuralConstraintTerm {
    let dof = match component {
        0 => StructuralRotationDof::Rx,
        1 => StructuralRotationDof::Ry,
        _ => StructuralRotationDof::Rz,
    };
    CanonicalStructuralConstraintTerm {
        owner: CanonicalStructuralDofOwner::ReferencePoint(identity),
        dof: CanonicalStructuralDof::Rotation(dof),
        coefficient,
    }
}

fn unit_axis(component: usize) -> [f64; 3] {
    let mut axis = [0.0; 3];
    axis[component] = 1.0;
    axis
}
