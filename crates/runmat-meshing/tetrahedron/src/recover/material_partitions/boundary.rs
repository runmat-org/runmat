use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{
    PlcFacet, ProtectedBoundaryComplex, TetrahedronBoundaryFace, TetrahedronMesh,
};

use crate::{
    protected_edges::source_edge_ids_for_face_edges, recover::topology::sorted_topology_ids,
};

pub(super) fn material_partition_boundary_contract_is_satisfied(
    plc: &ProtectedBoundaryComplex,
    material_facets: &[&PlcFacet],
    tetrahedron_mesh: &TetrahedronMesh,
) -> bool {
    let boundary_faces_by_key = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| (sorted_topology_ids(face.node_ids.clone()), face))
        .collect::<BTreeMap<_, _>>();
    material_facets.iter().all(|facet| {
        boundary_faces_by_key
            .get(&sorted_topology_ids(facet.node_ids.clone()))
            .is_some_and(|boundary_face| {
                let expected_source_edge_ids =
                    source_edge_ids_for_face_edges(&plc.protected_edges, facet.node_ids.clone());
                boundary_face.face_id == facet.facet_id
                    && boundary_face.source_face_id == facet.source_face_id
                    && boundary_face.source_edge_ids == expected_source_edge_ids
            })
    })
}

pub(super) fn insert_material_partition_boundary_faces(
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
