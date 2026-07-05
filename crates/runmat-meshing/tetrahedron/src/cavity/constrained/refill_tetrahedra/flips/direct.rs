use super::*;
use std::collections::{BTreeMap, BTreeSet};

use super::super::super::super::super::reconnect::{
    three_to_two_edge_flip_candidate, two_to_three_face_flip_candidate, LocalTetrahedron,
    LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError,
};

use super::super::super::{
    topology::{sorted_edge, sorted_face, sorted_tetrahedron_nodes, tetrahedron_faces},
    ConstrainedCavityRefillTetrahedronFlipError,
};

pub fn flip_refill_tetrahedra_across_shared_face(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    nodes: &[ConstrainedCavityNode],
    face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<ConstrainedCavityRefillTetrahedron>, ConstrainedCavityRefillTetrahedronFlipError> {
    let node_map = refill_component_node_map(tetrahedra, nodes)?;
    let target_face = sorted_face(face);
    for node_id in target_face {
        if !node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityRefillTetrahedronFlipError::MissingNode { node_id });
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
            ConstrainedCavityRefillTetrahedronFlipError::FaceIncidenceNotTwo {
                node_ids: target_face,
                incident_tetrahedron_count: incident_tetrahedron_indices.len(),
            },
        );
    }
    let left_index = incident_tetrahedron_indices[0];
    let right_index = incident_tetrahedron_indices[1];
    let flip = two_to_three_face_flip_candidate(
        LocalTetrahedron {
            tetrahedron_id: left_index as u32,
            node_ids: tetrahedra[left_index].node_ids,
        },
        LocalTetrahedron {
            tetrahedron_id: right_index as u32,
            node_ids: tetrahedra[right_index].node_ids,
        },
    )
    .map_err(
        |err| ConstrainedCavityRefillTetrahedronFlipError::InvalidFlipTopology {
            reason: local_tetrahedron_flip_error_reason(&err),
        },
    )?;
    refill_tetrahedra_from_flip_candidate(tetrahedra, &node_map, &flip, options)
}

pub fn flip_refill_tetrahedra_around_shared_edge(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<ConstrainedCavityRefillTetrahedron>, ConstrainedCavityRefillTetrahedronFlipError> {
    let node_map = refill_component_node_map(tetrahedra, nodes)?;
    let target_edge = sorted_edge(edge);
    for node_id in target_edge {
        if !node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityRefillTetrahedronFlipError::MissingNode { node_id });
        }
    }
    let incident_tetrahedron_indices = tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            (tetrahedron.node_ids.contains(&target_edge[0])
                && tetrahedron.node_ids.contains(&target_edge[1]))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    if incident_tetrahedron_indices.len() != 3 {
        return Err(
            ConstrainedCavityRefillTetrahedronFlipError::EdgeIncidenceNotThree {
                node_ids: target_edge,
                incident_tetrahedron_count: incident_tetrahedron_indices.len(),
            },
        );
    }
    let flip = three_to_two_edge_flip_candidate(
        [
            LocalTetrahedron {
                tetrahedron_id: incident_tetrahedron_indices[0] as u32,
                node_ids: tetrahedra[incident_tetrahedron_indices[0]].node_ids,
            },
            LocalTetrahedron {
                tetrahedron_id: incident_tetrahedron_indices[1] as u32,
                node_ids: tetrahedra[incident_tetrahedron_indices[1]].node_ids,
            },
            LocalTetrahedron {
                tetrahedron_id: incident_tetrahedron_indices[2] as u32,
                node_ids: tetrahedra[incident_tetrahedron_indices[2]].node_ids,
            },
        ],
        target_edge,
    )
    .map_err(
        |err| ConstrainedCavityRefillTetrahedronFlipError::InvalidFlipTopology {
            reason: local_tetrahedron_flip_error_reason(&err),
        },
    )?;
    refill_tetrahedra_from_flip_candidate(tetrahedra, &node_map, &flip, options)
}

fn refill_component_node_map(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    nodes: &[ConstrainedCavityNode],
) -> Result<BTreeMap<u32, Point3>, ConstrainedCavityRefillTetrahedronFlipError> {
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tetrahedron in tetrahedra {
        for node_id in tetrahedron.node_ids {
            if !node_map.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillTetrahedronFlipError::MissingNode { node_id });
            }
        }
    }
    Ok(node_map)
}

fn refill_tetrahedra_from_flip_candidate(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    node_map: &BTreeMap<u32, Point3>,
    flip: &LocalTetrahedronFlipCandidate,
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<ConstrainedCavityRefillTetrahedron>, ConstrainedCavityRefillTetrahedronFlipError> {
    let removed_indices = flip
        .removed_tetrahedron_ids
        .iter()
        .map(|tetrahedron_id| *tetrahedron_id as usize)
        .collect::<BTreeSet<_>>();
    if removed_indices
        .iter()
        .any(|index| *index >= tetrahedra.len())
    {
        return Err(
            ConstrainedCavityRefillTetrahedronFlipError::InvalidFlipTopology {
                reason: "removed_tetrahedron_out_of_bounds",
            },
        );
    }
    let mut candidate_tetrahedra = tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            (!removed_indices.contains(&index)).then_some(tetrahedron.clone())
        })
        .collect::<Vec<_>>();
    let mut candidate_keys = candidate_tetrahedra
        .iter()
        .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
        .collect::<BTreeSet<_>>();
    for node_ids in &flip.created_tetrahedra {
        let key = sorted_tetrahedron_nodes(*node_ids);
        if !candidate_keys.insert(key) {
            return Err(
                ConstrainedCavityRefillTetrahedronFlipError::InvalidFlipTopology {
                    reason: "duplicate_created_tetrahedron",
                },
            );
        }
        let mut points = [[0.0; 3]; 4];
        for (point, node_id) in points.iter_mut().zip(node_ids) {
            *point = *node_map.get(node_id).ok_or(
                ConstrainedCavityRefillTetrahedronFlipError::MissingNode { node_id: *node_id },
            )?;
        }
        match raw_refill_tetrahedron_with_rejection_reason(*node_ids, points, options) {
            Ok(tetrahedron) => candidate_tetrahedra.push(tetrahedron),
            Err(reason) => {
                return Err(
                    ConstrainedCavityRefillTetrahedronFlipError::RejectedCreatedTetrahedron {
                        node_ids: *node_ids,
                        reason,
                    },
                );
            }
        }
    }
    Ok(candidate_tetrahedra)
}

fn local_tetrahedron_flip_error_reason(error: &LocalTetrahedronFlipError) -> &'static str {
    match error {
        LocalTetrahedronFlipError::DegenerateTetrahedron { .. } => "degenerate_tetrahedron",
        LocalTetrahedronFlipError::NoSharedFace => "no_shared_face",
        LocalTetrahedronFlipError::NoSharedEdge => "no_shared_edge",
        LocalTetrahedronFlipError::InvalidEdgeRing => "invalid_edge_ring",
        LocalTetrahedronFlipError::InvalidQualityThresholds => "invalid_quality_thresholds",
        LocalTetrahedronFlipError::MissingNode { .. } => "missing_node",
        LocalTetrahedronFlipError::NonPositiveVolume { .. } => "non_positive_volume",
        LocalTetrahedronFlipError::VolumeBelowThreshold { .. } => "volume_below_threshold",
        LocalTetrahedronFlipError::ScaledJacobianBelowThreshold { .. } => {
            "scaled_jacobian_below_threshold"
        }
        LocalTetrahedronFlipError::QualityDoesNotImprove => "quality_does_not_improve",
    }
}
