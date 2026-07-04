use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TetrahedronMesh, TopologyEntityId};

use super::{
    topology::sorted_topology_ids, TetrahedronRecoveryKind, TetrahedronRecoveryQueue,
    TetrahedronRecoveryStatus,
};

pub(super) fn recover_single_material_interface_region(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
    let missing_material_interfaces = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::MaterialInterface
                && item.status == TetrahedronRecoveryStatus::Missing
        })
        .filter_map(|item| item.material_interface_id.clone())
        .collect::<BTreeSet<_>>();
    if missing_material_interfaces.len() != 1 {
        return recover_boundary_facet_material_interface_regions(
            plc,
            &missing_material_interfaces,
            tetrahedron_mesh,
        );
    }

    let plc_material_interfaces = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    if plc_material_interfaces.len() != 1 || plc_material_interfaces != missing_material_interfaces
    {
        return recover_boundary_facet_material_interface_regions(
            plc,
            &missing_material_interfaces,
            tetrahedron_mesh,
        );
    }

    let material_interface_id = plc_material_interfaces
        .into_iter()
        .next()
        .expect("single material interface checked above");
    let mut repaired_element_count = 0;
    for element in &mut tetrahedron_mesh.elements {
        if element.material_region_id != material_interface_id {
            element.material_region_id = material_interface_id.clone();
            repaired_element_count += 1;
        }
    }
    repaired_element_count
}

fn recover_boundary_facet_material_interface_regions(
    plc: &ProtectedBoundaryComplex,
    missing_material_interfaces: &BTreeSet<String>,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> usize {
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

    let mut repaired_element_count = 0;
    for element in &mut tetrahedron_mesh.elements {
        let boundary_material_interfaces = tetrahedron_element_faces(element.node_ids.clone())
            .iter()
            .filter_map(|face| material_interfaces_by_facet.get(face))
            .flat_map(|material_interfaces| material_interfaces.iter().cloned())
            .collect::<BTreeSet<_>>();
        if boundary_material_interfaces.len() != 1 {
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
            element.material_region_id = material_interface_id;
            repaired_element_count += 1;
        }
    }
    repaired_element_count
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
