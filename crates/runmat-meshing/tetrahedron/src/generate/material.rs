use std::collections::BTreeSet;

use runmat_meshing_core::contracts::ProtectedBoundaryComplex;

const DEFAULT_MATERIAL_REGION_ID: &str = "solid_body";
const UNCLASSIFIED_MATERIAL_REGION_ID: &str = "unclassified";

pub(super) fn plc_material_region_id(plc: &ProtectedBoundaryComplex) -> String {
    let material_region_ids = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    match material_region_ids.as_slice() {
        [] => DEFAULT_MATERIAL_REGION_ID.to_string(),
        [material_region_id] => material_region_id.clone(),
        _ => UNCLASSIFIED_MATERIAL_REGION_ID.to_string(),
    }
}
