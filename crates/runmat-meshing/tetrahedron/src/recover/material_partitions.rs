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
            Ok(inserted_boundary_face_count) => {
                recovery.inserted_material_interface_count += 1;
                recovery.inserted_element_count += 1;
                recovery.inserted_boundary_face_count += inserted_boundary_face_count;
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
) -> Result<usize, MaterialPartitionRecoveryRejection> {
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
    if !(3..=4).contains(&material_facets.len()) {
        return Err(MaterialPartitionRecoveryRejection::FacetCount);
    }

    let candidate_node_ids = candidate_partition_node_ids(&material_facets)
        .ok_or(MaterialPartitionRecoveryRejection::FacetTopology)?;
    let candidate_face_keys = tetrahedron_faces(candidate_node_ids.clone());
    let material_facet_face_keys = material_facets
        .iter()
        .map(|facet| sorted_topology_ids(facet.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    if !material_facet_face_keys
        .iter()
        .all(|face_key| candidate_face_keys.contains(face_key))
    {
        return Err(MaterialPartitionRecoveryRejection::FacetTopology);
    }
    if element_exists(tetrahedron_mesh, &candidate_node_ids) {
        return Err(MaterialPartitionRecoveryRejection::ElementAlreadyExists);
    }
    validate_partition_interior_support(
        tetrahedron_mesh,
        &candidate_face_keys,
        &material_facet_face_keys,
    )?;

    let node_coordinates = node_coordinates(tetrahedron_mesh);
    let oriented_node_ids = orient_partition_tetrahedron(candidate_node_ids, &node_coordinates)
        .ok_or(MaterialPartitionRecoveryRejection::QualityGate)?;
    tetrahedron_mesh.elements.push(Tetrahedron4Element {
        element_id: TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: format!("material_partition:{material_interface_id}"),
        },
        node_ids: oriented_node_ids,
        material_region_id: material_interface_id.to_string(),
    });

    Ok(insert_material_partition_boundary_faces(
        plc,
        &material_facets,
        tetrahedron_mesh,
    ))
}

fn candidate_partition_node_ids(material_facets: &[&PlcFacet]) -> Option<[TopologyEntityId; 4]> {
    let node_ids = material_facets
        .iter()
        .flat_map(|facet| facet.node_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    let node_ids = node_ids.into_iter().collect::<Vec<_>>();
    (node_ids.len() == 4).then(|| {
        [
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]
    })
}

fn validate_partition_interior_support(
    tetrahedron_mesh: &TetrahedronMesh,
    candidate_face_keys: &BTreeSet<[TopologyEntityId; 3]>,
    material_facet_face_keys: &BTreeSet<[TopologyEntityId; 3]>,
) -> Result<(), MaterialPartitionRecoveryRejection> {
    let interior_face_keys = candidate_face_keys
        .difference(material_facet_face_keys)
        .collect::<Vec<_>>();
    if interior_face_keys.len() > 1 {
        return Err(MaterialPartitionRecoveryRejection::InteriorFaceTopology);
    }

    let volume_face_counts = tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_faces(element.node_ids.clone()))
        .fold(
            BTreeMap::<[TopologyEntityId; 3], usize>::new(),
            |mut counts, face_key| {
                *counts.entry(face_key).or_default() += 1;
                counts
            },
        );
    if interior_face_keys.iter().any(|face_key| {
        volume_face_counts
            .get(*face_key)
            .copied()
            .unwrap_or_default()
            != 1
    }) {
        return Err(MaterialPartitionRecoveryRejection::InteriorFaceTopology);
    }

    Ok(())
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
