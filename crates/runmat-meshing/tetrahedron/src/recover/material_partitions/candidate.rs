use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{PlcFacet, TetrahedronMesh, TopologyEntityId};

use super::{
    geometry::{
        element_exists, node_coordinates, orient_partition_tetrahedron, tetrahedron_faces,
        volume_face_counts,
    },
    MaterialPartitionRecovery, MaterialPartitionRecoveryRejection,
};

pub(super) struct CandidateMaterialPartition {
    pub(super) elements: Vec<[TopologyEntityId; 4]>,
}

struct CandidateMaterialPartitionElement {
    node_ids: [TopologyEntityId; 4],
    face_keys: BTreeSet<[TopologyEntityId; 3]>,
}

pub(super) fn select_candidate_partition(
    tetrahedron_mesh: &TetrahedronMesh,
    material_facets: &[&PlcFacet],
    material_facet_face_keys: &BTreeSet<[TopologyEntityId; 3]>,
    recovery: &mut MaterialPartitionRecovery,
) -> Result<CandidateMaterialPartition, MaterialPartitionRecoveryRejection> {
    let material_node_ids = material_partition_node_ids(material_facets)?;
    let candidate_node_sets = candidate_partition_node_sets(&material_node_ids);
    let node_coordinates = node_coordinates(tetrahedron_mesh);
    let mut topology_candidate_count = 0;
    let mut quality_rejected_candidate_count = 0;
    let mut existing_element_candidate_count = 0;
    let mut candidates = Vec::<CandidateMaterialPartitionElement>::new();
    for node_ids in candidate_node_sets {
        let face_keys = tetrahedron_faces(node_ids.clone());
        let boundary_face_count = face_keys.intersection(material_facet_face_keys).count();
        if boundary_face_count < 3 {
            continue;
        }
        topology_candidate_count += 1;
        if element_exists(tetrahedron_mesh, &node_ids) {
            existing_element_candidate_count += 1;
            continue;
        }
        let Some(oriented_node_ids) = orient_partition_tetrahedron(node_ids, &node_coordinates)
        else {
            quality_rejected_candidate_count += 1;
            continue;
        };
        candidates.push(CandidateMaterialPartitionElement {
            node_ids: oriented_node_ids,
            face_keys,
        });
    }
    recovery.topology_candidate_count += topology_candidate_count;
    recovery.usable_candidate_count += candidates.len();
    recovery.rejected_existing_candidate_count += existing_element_candidate_count;
    recovery.rejected_quality_candidate_count += quality_rejected_candidate_count;

    if candidates.is_empty() {
        if existing_element_candidate_count > 0 {
            return Err(MaterialPartitionRecoveryRejection::ElementAlreadyExists);
        }
        if quality_rejected_candidate_count > 0 {
            return Err(MaterialPartitionRecoveryRejection::QualityGate);
        }
        return Err(MaterialPartitionRecoveryRejection::FacetTopology);
    }
    if topology_candidate_count > 12 || candidates.len() > 8 {
        return Err(MaterialPartitionRecoveryRejection::FacetCount);
    }

    let volume_face_counts = volume_face_counts(tetrahedron_mesh);
    for candidate_count in 1..=candidates.len().min(4) {
        let mut selected_indices = Vec::<usize>::new();
        let mut rejected_candidate_set_count = 0;
        if let Some(partition) = select_candidate_partition_with_count(
            &candidates,
            candidate_count,
            0,
            &mut selected_indices,
            material_facet_face_keys,
            &volume_face_counts,
            &mut rejected_candidate_set_count,
        ) {
            recovery.rejected_interior_candidate_set_count += rejected_candidate_set_count;
            return Ok(partition);
        }
        recovery.rejected_interior_candidate_set_count += rejected_candidate_set_count;
    }

    Err(MaterialPartitionRecoveryRejection::InteriorFaceTopology)
}

fn material_partition_node_ids(
    material_facets: &[&PlcFacet],
) -> Result<Vec<TopologyEntityId>, MaterialPartitionRecoveryRejection> {
    let node_ids = material_facets
        .iter()
        .flat_map(|facet| facet.node_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    if !(4..=8).contains(&node_ids.len()) {
        return Err(MaterialPartitionRecoveryRejection::FacetTopology);
    }
    Ok(node_ids.into_iter().collect())
}

fn candidate_partition_node_sets(node_ids: &[TopologyEntityId]) -> Vec<[TopologyEntityId; 4]> {
    let mut candidates = Vec::<[TopologyEntityId; 4]>::new();
    for first in 0..node_ids.len() {
        for second in first + 1..node_ids.len() {
            for third in second + 1..node_ids.len() {
                for fourth in third + 1..node_ids.len() {
                    candidates.push([
                        node_ids[first].clone(),
                        node_ids[second].clone(),
                        node_ids[third].clone(),
                        node_ids[fourth].clone(),
                    ]);
                }
            }
        }
    }
    candidates
}

fn select_candidate_partition_with_count(
    candidates: &[CandidateMaterialPartitionElement],
    candidate_count: usize,
    next_candidate_index: usize,
    selected_indices: &mut Vec<usize>,
    material_facet_face_keys: &BTreeSet<[TopologyEntityId; 3]>,
    volume_face_counts: &BTreeMap<[TopologyEntityId; 3], usize>,
    rejected_candidate_set_count: &mut usize,
) -> Option<CandidateMaterialPartition> {
    if selected_indices.len() == candidate_count {
        let partition = build_candidate_partition(
            candidates,
            selected_indices,
            material_facet_face_keys,
            volume_face_counts,
        );
        if partition.is_none() {
            *rejected_candidate_set_count += 1;
        }
        return partition;
    }
    let remaining_slots = candidate_count - selected_indices.len();
    let max_start = candidates.len().saturating_sub(remaining_slots);
    for candidate_index in next_candidate_index..=max_start {
        selected_indices.push(candidate_index);
        if let Some(partition) = select_candidate_partition_with_count(
            candidates,
            candidate_count,
            candidate_index + 1,
            selected_indices,
            material_facet_face_keys,
            volume_face_counts,
            rejected_candidate_set_count,
        ) {
            return Some(partition);
        }
        selected_indices.pop();
    }
    None
}

fn build_candidate_partition(
    candidates: &[CandidateMaterialPartitionElement],
    selected_indices: &[usize],
    material_facet_face_keys: &BTreeSet<[TopologyEntityId; 3]>,
    volume_face_counts: &BTreeMap<[TopologyEntityId; 3], usize>,
) -> Option<CandidateMaterialPartition> {
    let mut selected_face_counts = BTreeMap::<[TopologyEntityId; 3], usize>::new();
    for candidate_index in selected_indices {
        for face_key in &candidates[*candidate_index].face_keys {
            *selected_face_counts.entry(face_key.clone()).or_default() += 1;
        }
    }
    if material_facet_face_keys.iter().any(|face_key| {
        selected_face_counts
            .get(face_key)
            .copied()
            .unwrap_or_default()
            != 1
    }) {
        return None;
    }
    if selected_face_counts.keys().any(|face_key| {
        !material_facet_face_keys.contains(face_key)
            && selected_face_counts
                .get(face_key)
                .copied()
                .unwrap_or_default()
                != 2
            && volume_face_counts
                .get(face_key)
                .copied()
                .unwrap_or_default()
                != 1
    }) {
        return None;
    }

    Some(CandidateMaterialPartition {
        elements: selected_indices
            .iter()
            .map(|candidate_index| candidates[*candidate_index].node_ids.clone())
            .collect(),
    })
}
