use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TetrahedronMesh, TopologyEntityId};

use super::{
    topology::sorted_topology_ids, TetrahedronMaterialInterfaceTopology, TetrahedronRecoveryKind,
    TetrahedronRecoveryQueue, TetrahedronRecoveryStatus,
};

pub(super) fn recover_material_interface_regions(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> MaterialInterfaceRecovery {
    let missing_material_interface_inputs =
        missing_material_interface_inputs(initial_recovery_queue);
    let missing_material_interfaces = missing_material_interface_inputs.all.clone();
    let recoverable_material_interfaces = missing_material_interface_inputs
        .boundary_owned
        .union(&missing_material_interface_inputs.interior_face)
        .cloned()
        .collect::<BTreeSet<_>>();
    let plc_material_interfaces = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut recovery = MaterialInterfaceRecovery {
        attempted_material_interface_count: missing_material_interfaces.len(),
        repaired_element_count: 0,
        rejected_material_interface_count: 0,
        global_material_interface_count: 0,
        boundary_owned_material_interface_count: 0,
        interior_material_interface_count: 0,
        absent_partition_material_interface_count: missing_material_interface_inputs
            .absent_partition
            .len(),
        missing_boundary_ownership_count: 0,
        missing_interior_ownership_count: 0,
        ambiguous_boundary_ownership_count: 0,
        absent_partition_rejection_count: missing_material_interface_inputs.absent_partition.len(),
    };
    recovery.rejected_material_interface_count +=
        missing_material_interface_inputs.absent_partition.len();

    if recoverable_material_interfaces.is_empty() {
        return recovery;
    }

    if recoverable_material_interfaces.len() != 1
        || !missing_material_interface_inputs
            .absent_partition
            .is_empty()
    {
        recover_boundary_facet_material_interface_regions(
            plc,
            &plc_material_interfaces,
            &recoverable_material_interfaces,
            &missing_material_interface_inputs.boundary_owned,
            &missing_material_interface_inputs.interior_face,
            tetrahedron_mesh,
            &mut recovery,
        );
        return recovery;
    }

    if plc_material_interfaces.len() != 1
        || plc_material_interfaces != recoverable_material_interfaces
    {
        recover_boundary_facet_material_interface_regions(
            plc,
            &plc_material_interfaces,
            &recoverable_material_interfaces,
            &missing_material_interface_inputs.boundary_owned,
            &missing_material_interface_inputs.interior_face,
            tetrahedron_mesh,
            &mut recovery,
        );
        return recovery;
    }

    let material_interface_id = plc_material_interfaces
        .into_iter()
        .next()
        .expect("single material interface checked above");
    recovery.global_material_interface_count = 1;
    for element in &mut tetrahedron_mesh.elements {
        if element.material_region_id != material_interface_id {
            element.material_region_id = material_interface_id.clone();
            recovery.repaired_element_count += 1;
        }
    }
    recovery
}

pub(super) fn material_interface_recovery_topology(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &TetrahedronMesh,
    material_interface_id: &str,
    recovered_material_interfaces: &BTreeSet<String>,
    plc_material_interfaces: &BTreeSet<String>,
) -> TetrahedronMaterialInterfaceTopology {
    let plc_boundary_faces = plc
        .facets
        .iter()
        .map(|facet| sorted_topology_ids(facet.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_boundary_faces = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let material_interfaces_by_boundary_face = plc
        .facets
        .iter()
        .map(|facet| {
            (
                sorted_topology_ids(facet.node_ids.clone()),
                single_owner_material_interfaces(facet.material_interface_ids.as_slice()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let material_interfaces_by_source_face = material_interfaces_by_source_face(plc);
    let boundary_source_faces_by_face = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| {
            (
                sorted_topology_ids(face.node_ids.clone()),
                face.source_face_id.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for element in &tetrahedron_mesh.elements {
        if element.material_region_id == material_interface_id {
            continue;
        }
        for face in tetrahedron_element_faces(element.node_ids.clone()) {
            let material_interfaces =
                material_interfaces_by_boundary_face.get(&face).or_else(|| {
                    boundary_source_faces_by_face
                        .get(&face)
                        .and_then(|source_face_id| {
                            material_interfaces_by_source_face.get(source_face_id)
                        })
                });
            if material_interfaces.is_some_and(|material_interfaces| {
                material_interfaces.contains(material_interface_id)
            }) {
                return TetrahedronMaterialInterfaceTopology::BoundaryOwned;
            }
        }
    }

    let mut material_regions_by_interior_face = BTreeMap::<_, Vec<String>>::new();
    for element in &tetrahedron_mesh.elements {
        for face in tetrahedron_element_faces(element.node_ids.clone()) {
            if !plc_boundary_faces.contains(&face) && !recovered_boundary_faces.contains(&face) {
                material_regions_by_interior_face
                    .entry(face)
                    .or_default()
                    .push(element.material_region_id.clone());
            }
        }
    }

    if material_regions_by_interior_face
        .values()
        .any(|material_regions| {
            let [left, right] = material_regions.as_slice() else {
                return false;
            };
            recovered_material_interfaces.contains(material_interface_id)
                && ((left == material_interface_id && !plc_material_interfaces.contains(right))
                    || (right == material_interface_id && !plc_material_interfaces.contains(left)))
        })
    {
        TetrahedronMaterialInterfaceTopology::InteriorFace
    } else {
        TetrahedronMaterialInterfaceTopology::AbsentPartition
    }
}

pub(super) struct MaterialInterfaceRecovery {
    pub attempted_material_interface_count: usize,
    pub repaired_element_count: usize,
    pub rejected_material_interface_count: usize,
    pub global_material_interface_count: usize,
    pub boundary_owned_material_interface_count: usize,
    pub interior_material_interface_count: usize,
    pub absent_partition_material_interface_count: usize,
    pub missing_boundary_ownership_count: usize,
    pub missing_interior_ownership_count: usize,
    pub ambiguous_boundary_ownership_count: usize,
    pub absent_partition_rejection_count: usize,
}

struct MissingMaterialInterfaceInputs {
    all: BTreeSet<String>,
    boundary_owned: BTreeSet<String>,
    interior_face: BTreeSet<String>,
    absent_partition: BTreeSet<String>,
}

fn missing_material_interface_inputs(
    initial_recovery_queue: &TetrahedronRecoveryQueue,
) -> MissingMaterialInterfaceInputs {
    let mut inputs = MissingMaterialInterfaceInputs {
        all: BTreeSet::new(),
        boundary_owned: BTreeSet::new(),
        interior_face: BTreeSet::new(),
        absent_partition: BTreeSet::new(),
    };
    for item in initial_recovery_queue.items.iter().filter(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Missing
    }) {
        let Some(material_interface_id) = item.material_interface_id.clone() else {
            continue;
        };
        inputs.all.insert(material_interface_id.clone());
        match item.material_interface_topology {
            Some(TetrahedronMaterialInterfaceTopology::BoundaryOwned) => {
                inputs.boundary_owned.insert(material_interface_id);
            }
            Some(TetrahedronMaterialInterfaceTopology::InteriorFace) => {
                inputs.interior_face.insert(material_interface_id);
            }
            Some(TetrahedronMaterialInterfaceTopology::AbsentPartition) => {
                inputs.absent_partition.insert(material_interface_id);
            }
            None => {}
        }
    }
    inputs
}

fn recover_boundary_facet_material_interface_regions(
    plc: &ProtectedBoundaryComplex,
    plc_material_interfaces: &BTreeSet<String>,
    recoverable_material_interfaces: &BTreeSet<String>,
    boundary_owned_material_interface_inputs: &BTreeSet<String>,
    interior_face_material_interface_inputs: &BTreeSet<String>,
    tetrahedron_mesh: &mut TetrahedronMesh,
    recovery: &mut MaterialInterfaceRecovery,
) {
    let material_interfaces_by_facet = plc
        .facets
        .iter()
        .map(|facet| {
            (
                sorted_topology_ids(facet.node_ids.clone()),
                single_owner_material_interfaces(facet.material_interface_ids.as_slice()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let material_interfaces_by_source_face = material_interfaces_by_source_face(plc);
    let boundary_source_faces_by_face = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| {
            (
                sorted_topology_ids(face.node_ids.clone()),
                face.source_face_id.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut repaired_material_interfaces = BTreeSet::<String>::new();
    let mut ambiguous_material_interfaces = BTreeSet::<String>::new();
    recovery.boundary_owned_material_interface_count =
        boundary_owned_material_interface_inputs.len();
    for element in &mut tetrahedron_mesh.elements {
        let boundary_material_interfaces = tetrahedron_element_faces(element.node_ids.clone())
            .iter()
            .filter_map(|face| {
                material_interfaces_by_facet.get(face).or_else(|| {
                    boundary_source_faces_by_face
                        .get(face)
                        .and_then(|source_face_id| {
                            material_interfaces_by_source_face.get(source_face_id)
                        })
                })
            })
            .flat_map(|material_interfaces| material_interfaces.iter().cloned())
            .collect::<BTreeSet<_>>();
        if boundary_material_interfaces.len() != 1 {
            for material_interface_id in boundary_material_interfaces {
                if boundary_owned_material_interface_inputs.contains(&material_interface_id) {
                    ambiguous_material_interfaces.insert(material_interface_id);
                }
            }
            continue;
        }
        let material_interface_id = boundary_material_interfaces
            .into_iter()
            .next()
            .expect("single material interface checked above");
        if !boundary_owned_material_interface_inputs.contains(&material_interface_id) {
            continue;
        }
        if element.material_region_id != material_interface_id {
            element.material_region_id = material_interface_id.clone();
            recovery.repaired_element_count += 1;
            repaired_material_interfaces.insert(material_interface_id);
        } else {
            repaired_material_interfaces.insert(material_interface_id);
        }
    }
    recover_interior_material_interface_regions(
        plc,
        plc_material_interfaces,
        recoverable_material_interfaces,
        tetrahedron_mesh,
        recovery,
        &mut repaired_material_interfaces,
        &mut ambiguous_material_interfaces,
    );
    for material_interface_id in recoverable_material_interfaces {
        if repaired_material_interfaces.contains(material_interface_id) {
            continue;
        }
        recovery.rejected_material_interface_count += 1;
        if ambiguous_material_interfaces.contains(material_interface_id) {
            recovery.ambiguous_boundary_ownership_count += 1;
        } else if interior_face_material_interface_inputs.contains(material_interface_id) {
            recovery.missing_interior_ownership_count += 1;
        } else if !boundary_owned_material_interface_inputs.contains(material_interface_id) {
            recovery.missing_boundary_ownership_count += 1;
        } else {
            recovery.ambiguous_boundary_ownership_count += 1;
        }
    }
}

fn recover_interior_material_interface_regions(
    plc: &ProtectedBoundaryComplex,
    plc_material_interfaces: &BTreeSet<String>,
    recoverable_material_interfaces: &BTreeSet<String>,
    tetrahedron_mesh: &mut TetrahedronMesh,
    recovery: &mut MaterialInterfaceRecovery,
    repaired_material_interfaces: &mut BTreeSet<String>,
    ambiguous_material_interfaces: &mut BTreeSet<String>,
) {
    let plc_boundary_faces = plc
        .facets
        .iter()
        .map(|facet| sorted_topology_ids(facet.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_boundary_faces = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| sorted_topology_ids(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let mut interior_material_interfaces = BTreeSet::<String>::new();

    loop {
        let mut face_element_indices = BTreeMap::<_, Vec<usize>>::new();
        for (element_index, element) in tetrahedron_mesh.elements.iter().enumerate() {
            for face in tetrahedron_element_faces(element.node_ids.clone()) {
                if !plc_boundary_faces.contains(&face) && !recovered_boundary_faces.contains(&face)
                {
                    face_element_indices
                        .entry(face)
                        .or_default()
                        .push(element_index);
                }
            }
        }

        let mut candidates_by_element = BTreeMap::<usize, BTreeSet<String>>::new();
        for element_indices in face_element_indices.values() {
            let [left_index, right_index] = element_indices.as_slice() else {
                continue;
            };
            let left_material_region = tetrahedron_mesh.elements[*left_index]
                .material_region_id
                .clone();
            let right_material_region = tetrahedron_mesh.elements[*right_index]
                .material_region_id
                .clone();
            record_interior_material_candidate(
                *left_index,
                &left_material_region,
                *right_index,
                &right_material_region,
                plc_material_interfaces,
                recoverable_material_interfaces,
                &mut candidates_by_element,
            );
            record_interior_material_candidate(
                *right_index,
                &right_material_region,
                *left_index,
                &left_material_region,
                plc_material_interfaces,
                recoverable_material_interfaces,
                &mut candidates_by_element,
            );
        }

        let mut edits = Vec::<(usize, String)>::new();
        for (element_index, material_interface_candidates) in candidates_by_element {
            if material_interface_candidates.len() == 1 {
                let material_interface_id = material_interface_candidates
                    .into_iter()
                    .next()
                    .expect("single material interface candidate checked above");
                if tetrahedron_mesh.elements[element_index].material_region_id
                    != material_interface_id
                {
                    edits.push((element_index, material_interface_id));
                }
            } else {
                ambiguous_material_interfaces.extend(material_interface_candidates);
            }
        }
        if edits.is_empty() {
            break;
        }
        for (element_index, material_interface_id) in edits {
            tetrahedron_mesh.elements[element_index].material_region_id =
                material_interface_id.clone();
            recovery.repaired_element_count += 1;
            repaired_material_interfaces.insert(material_interface_id.clone());
            interior_material_interfaces.insert(material_interface_id);
        }
    }

    recovery.interior_material_interface_count = interior_material_interfaces.len();
}

pub(super) fn material_interfaces_by_source_face(
    plc: &ProtectedBoundaryComplex,
) -> BTreeMap<TopologyEntityId, BTreeSet<String>> {
    let mut material_interfaces_by_source_face =
        BTreeMap::<TopologyEntityId, BTreeSet<String>>::new();
    for facet in &plc.facets {
        material_interfaces_by_source_face
            .entry(facet.source_face_id.clone())
            .or_default()
            .extend(single_owner_material_interfaces(
                facet.material_interface_ids.as_slice(),
            ));
    }
    material_interfaces_by_source_face
}

fn single_owner_material_interfaces(material_interface_ids: &[String]) -> BTreeSet<String> {
    let [material_interface_id] = material_interface_ids else {
        return BTreeSet::new();
    };
    BTreeSet::from([material_interface_id.clone()])
}

fn record_interior_material_candidate(
    owned_element_index: usize,
    owned_material_region: &str,
    unowned_element_index: usize,
    unowned_material_region: &str,
    plc_material_interfaces: &BTreeSet<String>,
    missing_material_interfaces: &BTreeSet<String>,
    candidates_by_element: &mut BTreeMap<usize, BTreeSet<String>>,
) {
    if owned_element_index == unowned_element_index
        || !missing_material_interfaces.contains(owned_material_region)
        || plc_material_interfaces.contains(unowned_material_region)
    {
        return;
    }
    candidates_by_element
        .entry(unowned_element_index)
        .or_default()
        .insert(owned_material_region.to_string());
}

fn tetrahedron_element_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
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
}
