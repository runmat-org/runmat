use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TetrahedronMesh, TopologyEntityId};

use super::{
    topology::sorted_topology_ids, TetrahedronRecoveryKind, TetrahedronRecoveryQueue,
    TetrahedronRecoveryStatus,
};

pub(super) fn recover_material_interface_regions(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> MaterialInterfaceRecovery {
    let missing_material_interfaces = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::MaterialInterface
                && item.status == TetrahedronRecoveryStatus::Missing
        })
        .filter_map(|item| item.material_interface_id.clone())
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
        missing_boundary_ownership_count: 0,
        ambiguous_boundary_ownership_count: 0,
    };
    if missing_material_interfaces.len() != 1 {
        recover_boundary_facet_material_interface_regions(
            plc,
            &plc_material_interfaces,
            &missing_material_interfaces,
            tetrahedron_mesh,
            &mut recovery,
        );
        return recovery;
    }

    if plc_material_interfaces.len() != 1 || plc_material_interfaces != missing_material_interfaces
    {
        recover_boundary_facet_material_interface_regions(
            plc,
            &plc_material_interfaces,
            &missing_material_interfaces,
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

pub(super) struct MaterialInterfaceRecovery {
    pub attempted_material_interface_count: usize,
    pub repaired_element_count: usize,
    pub rejected_material_interface_count: usize,
    pub global_material_interface_count: usize,
    pub boundary_owned_material_interface_count: usize,
    pub interior_material_interface_count: usize,
    pub missing_boundary_ownership_count: usize,
    pub ambiguous_boundary_ownership_count: usize,
}

fn recover_boundary_facet_material_interface_regions(
    plc: &ProtectedBoundaryComplex,
    plc_material_interfaces: &BTreeSet<String>,
    missing_material_interfaces: &BTreeSet<String>,
    tetrahedron_mesh: &mut TetrahedronMesh,
    recovery: &mut MaterialInterfaceRecovery,
) {
    let material_interfaces_by_facet = plc
        .facets
        .iter()
        .map(|facet| {
            (
                sorted_topology_ids(facet.node_ids.clone()),
                facet
                    .material_interface_ids
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut repaired_material_interfaces = BTreeSet::<String>::new();
    let mut ambiguous_material_interfaces = BTreeSet::<String>::new();
    let mut boundary_owned_material_interfaces = BTreeSet::<String>::new();
    for element in &mut tetrahedron_mesh.elements {
        let boundary_material_interfaces = tetrahedron_element_faces(element.node_ids.clone())
            .iter()
            .filter_map(|face| material_interfaces_by_facet.get(face))
            .flat_map(|material_interfaces| material_interfaces.iter().cloned())
            .collect::<BTreeSet<_>>();
        for material_interface_id in &boundary_material_interfaces {
            if missing_material_interfaces.contains(material_interface_id) {
                boundary_owned_material_interfaces.insert(material_interface_id.clone());
            }
        }
        if boundary_material_interfaces.len() != 1 {
            for material_interface_id in boundary_material_interfaces {
                if missing_material_interfaces.contains(&material_interface_id) {
                    ambiguous_material_interfaces.insert(material_interface_id);
                }
            }
            continue;
        }
        let material_interface_id = boundary_material_interfaces
            .into_iter()
            .next()
            .expect("single material interface checked above");
        if !missing_material_interfaces.contains(&material_interface_id) {
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
    recovery.boundary_owned_material_interface_count = boundary_owned_material_interfaces.len();
    recover_interior_material_interface_regions(
        plc,
        plc_material_interfaces,
        missing_material_interfaces,
        tetrahedron_mesh,
        recovery,
        &mut repaired_material_interfaces,
        &mut ambiguous_material_interfaces,
    );
    for material_interface_id in missing_material_interfaces {
        if repaired_material_interfaces.contains(material_interface_id) {
            continue;
        }
        recovery.rejected_material_interface_count += 1;
        if ambiguous_material_interfaces.contains(material_interface_id) {
            recovery.ambiguous_boundary_ownership_count += 1;
        } else if !boundary_owned_material_interfaces.contains(material_interface_id) {
            recovery.missing_boundary_ownership_count += 1;
        } else {
            recovery.ambiguous_boundary_ownership_count += 1;
        }
    }
}

fn recover_interior_material_interface_regions(
    plc: &ProtectedBoundaryComplex,
    plc_material_interfaces: &BTreeSet<String>,
    missing_material_interfaces: &BTreeSet<String>,
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
    let mut interior_material_interfaces = BTreeSet::<String>::new();

    loop {
        let mut face_element_indices = BTreeMap::<_, Vec<usize>>::new();
        for (element_index, element) in tetrahedron_mesh.elements.iter().enumerate() {
            for face in tetrahedron_element_faces(element.node_ids.clone()) {
                if !plc_boundary_faces.contains(&face) {
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
                missing_material_interfaces,
                &mut candidates_by_element,
            );
            record_interior_material_candidate(
                *right_index,
                &right_material_region,
                *left_index,
                &left_material_region,
                plc_material_interfaces,
                missing_material_interfaces,
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
