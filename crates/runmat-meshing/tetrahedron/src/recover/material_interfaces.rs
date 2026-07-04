use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TetrahedronMesh};

use super::{TetrahedronRecoveryKind, TetrahedronRecoveryQueue, TetrahedronRecoveryStatus};

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
        return 0;
    }

    let plc_material_interfaces = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    if plc_material_interfaces.len() != 1 || plc_material_interfaces != missing_material_interfaces
    {
        return 0;
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
