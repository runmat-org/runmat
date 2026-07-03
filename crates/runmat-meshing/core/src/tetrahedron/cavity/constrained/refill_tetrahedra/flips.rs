use super::*;
use std::collections::{BTreeMap, BTreeSet};

use crate::tetrahedron::reconnect::{
    evaluate_local_tetrahedron_flip_quality, three_to_two_edge_flip_candidate,
    two_to_three_face_flip_candidate, LocalTetrahedron, LocalTetrahedronFlipCandidate,
    LocalTetrahedronFlipError, LocalTetrahedronFlipQualityThresholds,
};

use super::super::{
    topology::{
        common_tetrahedron_edges, sorted_edge, sorted_face, sorted_tetrahedron_nodes,
        tetrahedron_faces,
    },
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
    }
}

pub(in super::super) fn improve_refill_with_local_flips(
    cavity: &ConstrainedCavity,
    node_coordinates: &BTreeMap<u32, Point3>,
    refill: &ConstrainedCavityRefill,
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefill> {
    if refill.tetrahedra.len() < 2 {
        return None;
    }
    let mut coordinates = node_coordinates.clone();
    for node in &refill.inserted_nodes {
        coordinates.insert(node.node_id, node.coordinates_m);
    }
    let thresholds = LocalTetrahedronFlipQualityThresholds {
        min_volume_m3: options.min_volume_m3,
        min_scaled_jacobian: options.min_scaled_jacobian,
    };
    let mut best = None::<ConstrainedCavityRefill>;

    for left_index in 0..refill.tetrahedra.len() {
        for right_index in (left_index + 1)..refill.tetrahedra.len() {
            let left = LocalTetrahedron {
                tetrahedron_id: left_index as u32,
                node_ids: refill.tetrahedra[left_index].node_ids,
            };
            let right = LocalTetrahedron {
                tetrahedron_id: right_index as u32,
                node_ids: refill.tetrahedra[right_index].node_ids,
            };
            let Ok(flip) = two_to_three_face_flip_candidate(left, right) else {
                continue;
            };
            if evaluate_local_tetrahedron_flip_quality(&flip, &coordinates, thresholds).is_err() {
                continue;
            }

            let Some(candidate) =
                refill_from_local_flip_candidate(cavity, &coordinates, refill, &flip, options)
            else {
                continue;
            };
            if !refill_is_better(&candidate, refill) {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|current| refill_is_better(&candidate, current))
            {
                best = Some(candidate);
            }
        }
    }

    for left_index in 0..refill.tetrahedra.len() {
        for middle_index in (left_index + 1)..refill.tetrahedra.len() {
            for right_index in (middle_index + 1)..refill.tetrahedra.len() {
                let tetrahedra = [
                    LocalTetrahedron {
                        tetrahedron_id: left_index as u32,
                        node_ids: refill.tetrahedra[left_index].node_ids,
                    },
                    LocalTetrahedron {
                        tetrahedron_id: middle_index as u32,
                        node_ids: refill.tetrahedra[middle_index].node_ids,
                    },
                    LocalTetrahedron {
                        tetrahedron_id: right_index as u32,
                        node_ids: refill.tetrahedra[right_index].node_ids,
                    },
                ];
                for edge in
                    common_tetrahedron_edges(tetrahedra.map(|tetrahedron| tetrahedron.node_ids))
                {
                    let Ok(flip) = three_to_two_edge_flip_candidate(tetrahedra, edge) else {
                        continue;
                    };
                    if evaluate_local_tetrahedron_flip_quality(&flip, &coordinates, thresholds)
                        .is_err()
                    {
                        continue;
                    }
                    let Some(candidate) = refill_from_local_flip_candidate(
                        cavity,
                        &coordinates,
                        refill,
                        &flip,
                        options,
                    ) else {
                        continue;
                    };
                    if !refill_is_better(&candidate, refill) {
                        continue;
                    }
                    if best
                        .as_ref()
                        .is_none_or(|current| refill_is_better(&candidate, current))
                    {
                        best = Some(candidate);
                    }
                }
            }
        }
    }

    best
}

fn refill_from_local_flip_candidate(
    cavity: &ConstrainedCavity,
    coordinates: &BTreeMap<u32, Point3>,
    refill: &ConstrainedCavityRefill,
    flip: &LocalTetrahedronFlipCandidate,
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefill> {
    let removed_indices = flip
        .removed_tetrahedron_ids
        .iter()
        .map(|tetrahedron_id| *tetrahedron_id as usize)
        .collect::<BTreeSet<_>>();
    if removed_indices
        .iter()
        .any(|index| *index >= refill.tetrahedra.len())
    {
        return None;
    }
    let mut candidate_tetrahedra = refill
        .tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            (!removed_indices.contains(&index)).then_some(tetrahedron.clone())
        })
        .collect::<Vec<_>>();
    let mut created_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut created_keys = BTreeSet::<[u32; 4]>::new();
    for node_ids in &flip.created_tetrahedra {
        let key = sorted_tetrahedron_nodes(*node_ids);
        if !created_keys.insert(key)
            || candidate_tetrahedra
                .iter()
                .any(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids) == key)
        {
            return None;
        }
        let mut points = [[0.0; 3]; 4];
        for (point, node_id) in points.iter_mut().zip(node_ids) {
            *point = *coordinates.get(node_id)?;
        }
        let tetrahedron =
            raw_refill_tetrahedron_with_rejection_reason(*node_ids, points, options).ok()?;
        created_tetrahedra.push(tetrahedron);
    }
    candidate_tetrahedra.extend(created_tetrahedra);

    let mut candidate = refill_from_tetrahedra(
        cavity,
        candidate_tetrahedra,
        options.volume_relative_tolerance,
    )
    .ok()?;
    candidate.inserted_nodes = refill.inserted_nodes.clone();
    Some(candidate)
}

pub(in super::super) fn refill_is_better(
    candidate: &ConstrainedCavityRefill,
    current: &ConstrainedCavityRefill,
) -> bool {
    let candidate_min = candidate
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let current_min = current
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    candidate_min > current_min + 1.0e-12
        || ((candidate_min - current_min).abs() <= 1.0e-12
            && candidate.tetrahedra.len() < current.tetrahedra.len())
}
