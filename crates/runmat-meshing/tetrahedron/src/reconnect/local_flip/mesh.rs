use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        MeshingStage, Tetrahedron4Element, TetrahedronMesh, TopologyEntityId,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX,
    },
    quality::predicate::{tetrahedron_scaled_jacobian, Point3},
};

use super::{
    evaluate_local_tetrahedron_flip_improvement, two_to_three_face_flip_candidate,
    LocalTetrahedron, LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError,
    LocalTetrahedronFlipQualityThresholds,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetrahedronMeshLocalReconnectionOptions {
    pub quality_thresholds: LocalTetrahedronFlipQualityThresholds,
    pub max_attempted_reconnections: usize,
    pub max_accepted_reconnections: usize,
}

impl Default for TetrahedronMeshLocalReconnectionOptions {
    fn default() -> Self {
        Self {
            quality_thresholds: LocalTetrahedronFlipQualityThresholds::default(),
            max_attempted_reconnections: 32,
            max_accepted_reconnections: 1,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TetrahedronMeshLocalReconnectionReport {
    pub attempted_reconnection_count: usize,
    pub accepted_reconnection_count: usize,
    pub rejected_reconnection_count: usize,
    pub budget_limited_reconnection_count: usize,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

pub fn improve_tetrahedron_mesh_with_local_flips(
    mesh: &mut TetrahedronMesh,
    options: TetrahedronMeshLocalReconnectionOptions,
) -> TetrahedronMeshLocalReconnectionReport {
    let mut report = TetrahedronMeshLocalReconnectionReport::default();
    if options.max_attempted_reconnections == 0
        || options.max_accepted_reconnections == 0
        || mesh.elements.len() < 2
    {
        record_local_reconnection_evidence(mesh, &report);
        return report;
    }

    let node_index = mesh_node_index(mesh);
    let node_coordinates = node_coordinate_map(mesh, &node_index);
    let boundary_faces = mesh
        .boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.to_vec()))
        .collect::<BTreeSet<_>>();
    let mut accepted = 0_usize;

    'search: loop {
        if report.attempted_reconnection_count >= options.max_attempted_reconnections {
            if has_potential_local_reconnection_candidate(
                mesh,
                &node_index,
                &node_coordinates,
                &boundary_faces,
                options,
            ) {
                report.budget_limited_reconnection_count += 1;
            }
            break;
        }
        let Some(candidate) = first_local_reconnection_candidate(
            mesh,
            &node_index,
            &node_coordinates,
            &boundary_faces,
            options,
        ) else {
            break;
        };
        report.attempted_reconnection_count += 1;
        match candidate {
            MeshLocalReconnectionCandidate::Accepted {
                left_index,
                right_index,
                flip,
            } => {
                apply_two_to_three_reconnection(mesh, left_index, right_index, &flip, &node_index);
                report.accepted_reconnection_count += 1;
                accepted += 1;
                if accepted >= options.max_accepted_reconnections {
                    if has_potential_local_reconnection_candidate(
                        mesh,
                        &node_index,
                        &node_coordinates,
                        &boundary_faces,
                        options,
                    ) {
                        report.budget_limited_reconnection_count += 1;
                    }
                    break 'search;
                }
            }
            MeshLocalReconnectionCandidate::Rejected { reason } => {
                report.rejected_reconnection_count += 1;
                *report
                    .rejected_by_reason
                    .entry(reason.to_string())
                    .or_default() += 1;
                break 'search;
            }
        }
    }

    if report.attempted_reconnection_count > 0 {
        mesh.quality_optimized = true;
    }
    record_local_reconnection_evidence(mesh, &report);
    report
}

enum MeshLocalReconnectionCandidate {
    Accepted {
        left_index: usize,
        right_index: usize,
        flip: LocalTetrahedronFlipCandidate,
    },
    Rejected {
        reason: &'static str,
    },
}

fn first_local_reconnection_candidate(
    mesh: &TetrahedronMesh,
    node_index: &BTreeMap<TopologyEntityId, u32>,
    node_coordinates: &BTreeMap<u32, Point3>,
    boundary_faces: &BTreeSet<Vec<TopologyEntityId>>,
    options: TetrahedronMeshLocalReconnectionOptions,
) -> Option<MeshLocalReconnectionCandidate> {
    for left_index in 0..mesh.elements.len() {
        for right_index in (left_index + 1)..mesh.elements.len() {
            let left_element = &mesh.elements[left_index];
            let right_element = &mesh.elements[right_index];
            if left_element.region_id != right_element.region_id {
                continue;
            }
            let Some(shared_face) = shared_element_face(left_element, right_element) else {
                continue;
            };
            if boundary_faces.contains(&shared_face) {
                continue;
            }
            let Some(left) = local_tetrahedron(left_index, left_element, node_index) else {
                return Some(MeshLocalReconnectionCandidate::Rejected {
                    reason: "missing_node",
                });
            };
            let Some(right) = local_tetrahedron(right_index, right_element, node_index) else {
                return Some(MeshLocalReconnectionCandidate::Rejected {
                    reason: "missing_node",
                });
            };
            if local_pair_min_scaled_jacobian([left, right], node_coordinates)
                >= options.quality_thresholds.min_scaled_jacobian
            {
                continue;
            }
            let flip = match two_to_three_face_flip_candidate(left, right) {
                Ok(flip) => flip,
                Err(err) => {
                    return Some(MeshLocalReconnectionCandidate::Rejected {
                        reason: local_tetrahedron_flip_error_reason(&err),
                    });
                }
            };
            if let Err(err) = evaluate_local_tetrahedron_flip_improvement(
                &[left, right],
                &flip,
                node_coordinates,
                options.quality_thresholds,
            ) {
                return Some(MeshLocalReconnectionCandidate::Rejected {
                    reason: local_tetrahedron_flip_error_reason(&err),
                });
            }
            return Some(MeshLocalReconnectionCandidate::Accepted {
                left_index,
                right_index,
                flip,
            });
        }
    }
    None
}

fn has_potential_local_reconnection_candidate(
    mesh: &TetrahedronMesh,
    node_index: &BTreeMap<TopologyEntityId, u32>,
    node_coordinates: &BTreeMap<u32, Point3>,
    boundary_faces: &BTreeSet<Vec<TopologyEntityId>>,
    options: TetrahedronMeshLocalReconnectionOptions,
) -> bool {
    for left_index in 0..mesh.elements.len() {
        for right_index in (left_index + 1)..mesh.elements.len() {
            let left_element = &mesh.elements[left_index];
            let right_element = &mesh.elements[right_index];
            if left_element.region_id != right_element.region_id {
                continue;
            }
            let Some(shared_face) = shared_element_face(left_element, right_element) else {
                continue;
            };
            if boundary_faces.contains(&shared_face) {
                continue;
            }
            let Some(left) = local_tetrahedron(left_index, left_element, node_index) else {
                return true;
            };
            let Some(right) = local_tetrahedron(right_index, right_element, node_index) else {
                return true;
            };
            if local_pair_min_scaled_jacobian([left, right], node_coordinates)
                < options.quality_thresholds.min_scaled_jacobian
            {
                return true;
            }
        }
    }
    false
}

fn mesh_node_index(mesh: &TetrahedronMesh) -> BTreeMap<TopologyEntityId, u32> {
    mesh.nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id.clone(), index as u32))
        .collect()
}

fn node_coordinate_map(
    mesh: &TetrahedronMesh,
    node_index: &BTreeMap<TopologyEntityId, u32>,
) -> BTreeMap<u32, Point3> {
    mesh.nodes
        .iter()
        .filter_map(|node| {
            node_index
                .get(&node.node_id)
                .map(|index| (*index, node.coordinates_m))
        })
        .collect()
}

fn shared_element_face(
    left: &Tetrahedron4Element,
    right: &Tetrahedron4Element,
) -> Option<Vec<TopologyEntityId>> {
    let left_nodes = left.node_ids.iter().cloned().collect::<BTreeSet<_>>();
    let shared = right
        .node_ids
        .iter()
        .filter(|node_id| left_nodes.contains(*node_id))
        .cloned()
        .collect::<Vec<_>>();
    (shared.len() == 3).then(|| sorted_topology_ids(shared))
}

fn sorted_topology_ids(mut ids: Vec<TopologyEntityId>) -> Vec<TopologyEntityId> {
    ids.sort();
    ids
}

fn local_tetrahedron(
    index: usize,
    element: &Tetrahedron4Element,
    node_index: &BTreeMap<TopologyEntityId, u32>,
) -> Option<LocalTetrahedron> {
    Some(LocalTetrahedron {
        tetrahedron_id: index as u32,
        node_ids: [
            *node_index.get(&element.node_ids[0])?,
            *node_index.get(&element.node_ids[1])?,
            *node_index.get(&element.node_ids[2])?,
            *node_index.get(&element.node_ids[3])?,
        ],
    })
}

fn local_pair_min_scaled_jacobian(
    tetrahedra: [LocalTetrahedron; 2],
    node_coordinates: &BTreeMap<u32, Point3>,
) -> f64 {
    tetrahedra
        .into_iter()
        .map(|tetrahedron| {
            tetrahedron_scaled_jacobian(tetrahedron.node_ids.map(|node_id| {
                node_coordinates
                    .get(&node_id)
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0])
            }))
        })
        .fold(f64::INFINITY, f64::min)
}

fn apply_two_to_three_reconnection(
    mesh: &mut TetrahedronMesh,
    left_index: usize,
    right_index: usize,
    flip: &LocalTetrahedronFlipCandidate,
    node_index: &BTreeMap<TopologyEntityId, u32>,
) {
    let mut local_to_topology = BTreeMap::<u32, TopologyEntityId>::new();
    for (node_id, local_id) in node_index {
        local_to_topology.insert(*local_id, node_id.clone());
    }
    let region_id = mesh.elements[left_index].region_id.clone();
    let removed = [left_index, right_index]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut next_elements = mesh
        .elements
        .iter()
        .enumerate()
        .filter_map(|(index, element)| (!removed.contains(&index)).then_some(element.clone()))
        .collect::<Vec<_>>();
    let reconnection_index = mesh
        .evidence
        .entity_counts
        .get(TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT)
        .copied()
        .unwrap_or_default();
    for (created_index, created) in flip.created_tetrahedra.iter().enumerate() {
        next_elements.push(Tetrahedron4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("local_reconnection_{reconnection_index}_{created_index}"),
            },
            node_ids: [
                local_to_topology[&created[0]].clone(),
                local_to_topology[&created[1]].clone(),
                local_to_topology[&created[2]].clone(),
                local_to_topology[&created[3]].clone(),
            ],
            region_id: region_id.clone(),
        });
    }
    mesh.elements = next_elements;
}

fn record_local_reconnection_evidence(
    mesh: &mut TetrahedronMesh,
    report: &TetrahedronMeshLocalReconnectionReport,
) {
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT.to_string())
        .or_default() += report.attempted_reconnection_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT.to_string())
        .or_default() += report.accepted_reconnection_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT.to_string())
        .or_default() += report.rejected_reconnection_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT.to_string())
        .or_default() += report.budget_limited_reconnection_count;
    for (reason, count) in &report.rejected_by_reason {
        *mesh
            .evidence
            .rejection_counts
            .entry(format!(
                "{TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX}{reason}"
            ))
            .or_default() += count;
    }
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
