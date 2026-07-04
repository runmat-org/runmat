use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        MeshingStage, PlcFacet, ProtectedBoundaryComplex, Tetrahedron4Element,
        TetrahedronBoundaryFace, TetrahedronMesh, TopologyEntityId,
    },
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
};

use crate::protected_edges::source_edge_ids_for_face_edges;

use super::{
    topology::sorted_topology_ids, TetrahedronMaterialInterfaceTopology, TetrahedronRecoveryKind,
    TetrahedronRecoveryQueue, TetrahedronRecoveryStatus,
};

const MIN_PARTITION_TETRAHEDRON_VOLUME_M3: f64 = 1.0e-18;
const MIN_PARTITION_TETRAHEDRON_SCALED_JACOBIAN: f64 = 1.0e-8;

pub(super) struct MaterialPartitionRecovery {
    pub attempted_material_interface_count: usize,
    pub inserted_material_interface_count: usize,
    pub inserted_element_count: usize,
    pub inserted_boundary_face_count: usize,
    pub rejected_material_interface_count: usize,
    pub rejection_counts: BTreeMap<&'static str, usize>,
}

pub(super) fn recover_absent_material_interface_partitions(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> MaterialPartitionRecovery {
    let recoverable_material_interfaces = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::MaterialInterface
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.material_interface_topology
                    == Some(TetrahedronMaterialInterfaceTopology::AbsentPartition)
        })
        .filter_map(|item| item.material_interface_id.clone())
        .collect::<BTreeSet<_>>();

    let mut recovery = MaterialPartitionRecovery {
        attempted_material_interface_count: 0,
        inserted_material_interface_count: 0,
        inserted_element_count: 0,
        inserted_boundary_face_count: 0,
        rejected_material_interface_count: 0,
        rejection_counts: BTreeMap::new(),
    };

    for material_interface_id in recoverable_material_interfaces {
        recovery.attempted_material_interface_count += 1;
        match insert_absent_material_interface_partition(
            plc,
            tetrahedron_mesh,
            &material_interface_id,
        ) {
            Ok(inserted_partition) => {
                recovery.inserted_material_interface_count += 1;
                recovery.inserted_element_count += inserted_partition.element_count;
                recovery.inserted_boundary_face_count += inserted_partition.boundary_face_count;
            }
            Err(rejection) => {
                recovery.rejected_material_interface_count += 1;
                *recovery
                    .rejection_counts
                    .entry(rejection.evidence_key())
                    .or_default() += 1;
            }
        }
    }

    recovery
}

fn insert_absent_material_interface_partition(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
    material_interface_id: &str,
) -> Result<InsertedMaterialPartition, MaterialPartitionRecoveryRejection> {
    let material_facets = plc
        .facets
        .iter()
        .filter(|facet| {
            facet
                .material_interface_ids
                .iter()
                .any(|facet_material_interface_id| {
                    facet_material_interface_id == material_interface_id
                })
        })
        .collect::<Vec<_>>();
    if !(3..=12).contains(&material_facets.len()) {
        return Err(MaterialPartitionRecoveryRejection::FacetCount);
    }

    let material_facet_face_keys = material_facets
        .iter()
        .map(|facet| sorted_topology_ids(facet.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let candidate_partition = select_candidate_partition(
        tetrahedron_mesh,
        &material_facets,
        &material_facet_face_keys,
    )?;
    let inserted_element_count = candidate_partition.elements.len();

    for (element_index, node_ids) in candidate_partition.elements.into_iter().enumerate() {
        tetrahedron_mesh.elements.push(Tetrahedron4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("material_partition:{material_interface_id}:{element_index}"),
            },
            node_ids,
            material_region_id: material_interface_id.to_string(),
        });
    }

    let inserted_boundary_face_count =
        insert_material_partition_boundary_faces(plc, &material_facets, tetrahedron_mesh);
    Ok(InsertedMaterialPartition {
        element_count: inserted_element_count,
        boundary_face_count: inserted_boundary_face_count,
    })
}

struct InsertedMaterialPartition {
    element_count: usize,
    boundary_face_count: usize,
}

struct CandidateMaterialPartition {
    elements: Vec<[TopologyEntityId; 4]>,
}

struct CandidateMaterialPartitionElement {
    node_ids: [TopologyEntityId; 4],
    face_keys: BTreeSet<[TopologyEntityId; 3]>,
}

fn select_candidate_partition(
    tetrahedron_mesh: &TetrahedronMesh,
    material_facets: &[&PlcFacet],
    material_facet_face_keys: &BTreeSet<[TopologyEntityId; 3]>,
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
        if let Some(partition) = select_candidate_partition_with_count(
            &candidates,
            candidate_count,
            0,
            &mut selected_indices,
            material_facet_face_keys,
            &volume_face_counts,
        ) {
            return Ok(partition);
        }
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
) -> Option<CandidateMaterialPartition> {
    if selected_indices.len() == candidate_count {
        return build_candidate_partition(
            candidates,
            selected_indices,
            material_facet_face_keys,
            volume_face_counts,
        );
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

fn volume_face_counts(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<[TopologyEntityId; 3], usize> {
    tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_faces(element.node_ids.clone()))
        .fold(
            BTreeMap::<[TopologyEntityId; 3], usize>::new(),
            |mut counts, face_key| {
                *counts.entry(face_key).or_default() += 1;
                counts
            },
        )
}

fn insert_material_partition_boundary_faces(
    plc: &ProtectedBoundaryComplex,
    material_facets: &[&PlcFacet],
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
    let mut boundary_face_keys = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let mut inserted_boundary_face_count = 0;
    for facet in material_facets {
        let face_key = sorted_topology_ids(facet.node_ids.clone());
        if !boundary_face_keys.insert(face_key) {
            continue;
        }
        tetrahedron_mesh
            .boundary_faces
            .push(TetrahedronBoundaryFace {
                face_id: facet.facet_id.clone(),
                node_ids: facet.node_ids.clone(),
                source_face_id: facet.source_face_id.clone(),
                source_edge_ids: source_edge_ids_for_face_edges(
                    &plc.protected_edges,
                    facet.node_ids.clone(),
                ),
            });
        inserted_boundary_face_count += 1;
    }
    inserted_boundary_face_count
}

fn element_exists(tetrahedron_mesh: &TetrahedronMesh, node_ids: &[TopologyEntityId; 4]) -> bool {
    let mut candidate_node_ids = node_ids.clone();
    candidate_node_ids.sort();
    tetrahedron_mesh.elements.iter().any(|element| {
        let mut element_node_ids = element.node_ids.clone();
        element_node_ids.sort();
        element_node_ids == candidate_node_ids
    })
}

fn orient_partition_tetrahedron(
    mut node_ids: [TopologyEntityId; 4],
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<[TopologyEntityId; 4]> {
    let mut points = points_for_nodes(node_ids.clone(), node_coordinates)?;
    let mut signed_volume = tetrahedron_signed_volume(points);
    if signed_volume < 0.0 {
        node_ids.swap(1, 2);
        points = points_for_nodes(node_ids.clone(), node_coordinates)?;
        signed_volume = -signed_volume;
    }
    if !signed_volume.is_finite() || signed_volume <= MIN_PARTITION_TETRAHEDRON_VOLUME_M3 {
        return None;
    }
    let scaled_jacobian = tetrahedron_scaled_jacobian(points);
    if !scaled_jacobian.is_finite() || scaled_jacobian < MIN_PARTITION_TETRAHEDRON_SCALED_JACOBIAN {
        return None;
    }
    Some(node_ids)
}

fn points_for_nodes(
    node_ids: [TopologyEntityId; 4],
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<[[f64; 3]; 4]> {
    Some([
        *node_coordinates.get(&node_ids[0])?,
        *node_coordinates.get(&node_ids[1])?,
        *node_coordinates.get(&node_ids[2])?,
        *node_coordinates.get(&node_ids[3])?,
    ])
}

fn node_coordinates(tetrahedron_mesh: &TetrahedronMesh) -> BTreeMap<TopologyEntityId, [f64; 3]> {
    tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect()
}

fn tetrahedron_faces(node_ids: [TopologyEntityId; 4]) -> BTreeSet<[TopologyEntityId; 3]> {
    [
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
    ]
    .into_iter()
    .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaterialPartitionRecoveryRejection {
    FacetCount,
    FacetTopology,
    ElementAlreadyExists,
    InteriorFaceTopology,
    QualityGate,
}

impl MaterialPartitionRecoveryRejection {
    fn evidence_key(self) -> &'static str {
        match self {
            Self::FacetCount => "rejected_absent_material_partition_facet_count",
            Self::FacetTopology => "rejected_absent_material_partition_facet_topology",
            Self::ElementAlreadyExists => "rejected_absent_material_partition_element_exists",
            Self::InteriorFaceTopology => {
                "rejected_absent_material_partition_interior_face_topology"
            }
            Self::QualityGate => "rejected_absent_material_partition_quality_gate",
        }
    }
}
