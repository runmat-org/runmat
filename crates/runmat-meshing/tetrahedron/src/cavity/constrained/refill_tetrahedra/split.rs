use std::collections::{BTreeMap, BTreeSet};

use super::*;

pub fn split_refill_tetrahedra_across_shared_face_at_barycentric(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    nodes: &[ConstrainedCavityNode],
    face: [u32; 3],
    barycentric: [f64; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    (
        Vec<ConstrainedCavityRefillTetrahedron>,
        ConstrainedCavityNode,
    ),
    ConstrainedCavityRefillTetrahedronSplitError,
> {
    let barycentric_sum = barycentric.iter().sum::<f64>();
    if barycentric
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || (barycentric_sum - 1.0).abs() > 1.0e-12
    {
        return Err(
            ConstrainedCavityRefillTetrahedronSplitError::InvalidBarycentricCoordinates {
                barycentric,
            },
        );
    }
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tetrahedron in tetrahedra {
        for node_id in tetrahedron.node_ids {
            if !node_map.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillTetrahedronSplitError::MissingNode { node_id });
            }
        }
    }
    let target_face = sorted_face(face);
    for node_id in target_face {
        if !node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityRefillTetrahedronSplitError::MissingNode { node_id });
        }
    }
    let incident_tetrahedron_indices = tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            tetrahedron_faces(tetrahedron.node_ids)
                .map(sorted_face)
                .contains(&target_face)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if incident_tetrahedron_indices.len() != 2 {
        return Err(
            ConstrainedCavityRefillTetrahedronSplitError::FaceIncidenceNotTwo {
                node_ids: target_face,
                incident_tetrahedron_count: incident_tetrahedron_indices.len(),
            },
        );
    }
    let mut split_node_id = node_map
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    while node_map.contains_key(&split_node_id) {
        split_node_id = split_node_id.saturating_add(1);
    }
    let face_points = target_face.map(|node_id| node_map[&node_id]);
    let split_node = ConstrainedCavityNode {
        node_id: split_node_id,
        coordinates_m: [
            face_points[0][0] * barycentric[0]
                + face_points[1][0] * barycentric[1]
                + face_points[2][0] * barycentric[2],
            face_points[0][1] * barycentric[0]
                + face_points[1][1] * barycentric[1]
                + face_points[2][1] * barycentric[2],
            face_points[0][2] * barycentric[0]
                + face_points[1][2] * barycentric[1]
                + face_points[2][2] * barycentric[2],
        ],
    };
    let mut split_node_map = node_map;
    split_node_map.insert(split_node.node_id, split_node.coordinates_m);
    let incident_tetrahedron_indices = incident_tetrahedron_indices
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut split_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for (index, tetrahedron) in tetrahedra.iter().enumerate() {
        if !incident_tetrahedron_indices.contains(&index) {
            split_tetrahedra.push(tetrahedron.clone());
            continue;
        }
        let opposite_node = tetrahedron
            .node_ids
            .into_iter()
            .find(|node_id| !target_face.contains(node_id))
            .expect("incident tetrahedron should have an opposite node");
        for child_node_ids in [
            [
                target_face[0],
                target_face[1],
                split_node.node_id,
                opposite_node,
            ],
            [
                target_face[1],
                target_face[2],
                split_node.node_id,
                opposite_node,
            ],
            [
                target_face[2],
                target_face[0],
                split_node.node_id,
                opposite_node,
            ],
        ] {
            let points = child_node_ids.map(|node_id| split_node_map[&node_id]);
            match raw_refill_tetrahedron_with_rejection_reason(child_node_ids, points, options) {
                Ok(child) => split_tetrahedra.push(child),
                Err(reason) => {
                    return Err(
                        ConstrainedCavityRefillTetrahedronSplitError::RejectedChildTetrahedron {
                            node_ids: child_node_ids,
                            reason,
                        },
                    );
                }
            }
        }
    }
    Ok((split_tetrahedra, split_node))
}
