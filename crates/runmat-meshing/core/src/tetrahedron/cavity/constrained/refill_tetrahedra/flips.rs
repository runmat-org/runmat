use super::*;
use std::collections::{BTreeMap, BTreeSet};

use crate::tetrahedron::reconnect::{
    evaluate_local_tetrahedron_flip_quality, three_to_two_edge_flip_candidate,
    two_to_three_face_flip_candidate, LocalTetrahedron, LocalTetrahedronFlipCandidate,
    LocalTetrahedronFlipQualityThresholds,
};

use super::super::topology::{common_tetrahedron_edges, sorted_tetrahedron_nodes};

mod direct;
pub use direct::{
    flip_refill_tetrahedra_across_shared_face, flip_refill_tetrahedra_around_shared_edge,
};

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
