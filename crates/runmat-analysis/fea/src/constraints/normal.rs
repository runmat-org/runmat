use std::collections::BTreeMap;

use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use super::vector::{add, cross, dot, magnitude, scale, subtract};
use super::{ConstraintPlanError, ResolvedConstraintTarget};

pub(super) fn boundary_node_normals(
    mesh: &SolverMeshArtifact,
    target: &ResolvedConstraintTarget,
    constraint_id: &str,
) -> Result<BTreeMap<StableDigest, [f64; 3]>, ConstraintPlanError> {
    let [entity] = target.matched_entities.as_slice() else {
        return Err(ConstraintPlanError::AmbiguousNormal {
            constraint_id: constraint_id.into(),
            entity_count: target.matched_entities.len(),
        });
    };
    let nodes_by_id = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect::<BTreeMap<_, _>>();
    let elements = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| (element.element_id, element))
        .collect::<BTreeMap<_, _>>();
    let mut accumulated = BTreeMap::<u64, [f64; 3]>::new();
    for face in mesh
        .topology
        .boundary_faces
        .iter()
        .filter(|face| face.provenance.contains(entity))
    {
        let corners =
            face.node_ids
                .get(0..3)
                .ok_or_else(|| ConstraintPlanError::NormalUnavailable {
                    constraint_id: constraint_id.into(),
                })?;
        let points = corners
            .iter()
            .map(|node_id| nodes_by_id.get(node_id).map(|node| node.coordinates_m))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            })?;
        let mut area_vector = cross(
            subtract(points[1], points[0]),
            subtract(points[2], points[0]),
        );
        let face_center = scale(add(add(points[0], points[1]), points[2]), 1.0 / 3.0);
        let [element_id] = face.adjacent_volume_element_ids.as_slice() else {
            return Err(ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            });
        };
        let element =
            elements
                .get(element_id)
                .ok_or_else(|| ConstraintPlanError::NormalUnavailable {
                    constraint_id: constraint_id.into(),
                })?;
        let element_points = element
            .node_ids
            .iter()
            .take(4)
            .map(|node_id| nodes_by_id.get(node_id).map(|node| node.coordinates_m))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            })?;
        if element_points.len() != 4 {
            return Err(ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            });
        }
        let center = scale(element_points.into_iter().fold([0.0; 3], add), 0.25);
        if dot(area_vector, subtract(center, face_center)) > 0.0 {
            area_vector = scale(area_vector, -1.0);
        }
        if magnitude(area_vector) <= 1.0e-18 {
            return Err(ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            });
        }
        for node_id in &face.node_ids {
            let normal = accumulated.entry(*node_id).or_insert([0.0; 3]);
            *normal = add(*normal, area_vector);
        }
    }
    let mut normals = BTreeMap::new();
    for identity in &target.matched_node_identities {
        let node = mesh
            .topology
            .nodes
            .iter()
            .find(|node| node.stable_identity == *identity)
            .ok_or_else(|| ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            })?;
        let vector = accumulated.get(&node.node_id).copied().ok_or_else(|| {
            ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            }
        })?;
        let length = magnitude(vector);
        if !length.is_finite() || length <= 1.0e-12 {
            return Err(ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            });
        }
        normals.insert(*identity, scale(vector, length.recip()));
    }
    Ok(normals)
}
