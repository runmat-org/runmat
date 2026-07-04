use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        PlcFacet, ProtectedBoundaryComplex, Tetrahedron4Element, TetrahedronBoundaryFace,
        TetrahedronMesh, TopologyEntityId,
    },
    quality::{
        predicate::{point_in_closed_triangle_surface, PointInClosedSurface, Triangle3},
        tolerance::MeshingTolerance,
    },
};

use crate::protected_edges::source_edge_ids_for_face_edges;

use super::{
    topology::sorted_topology_ids, TetrahedronRecoveryKind, TetrahedronRecoveryQueue,
    TetrahedronRecoveryStatus, TetrahedronSourceFaceTopology,
};

pub(super) struct BoundaryLeakRecovery {
    pub removed_element_count: usize,
    pub exposed_source_face_count: usize,
    pub inserted_boundary_face_count: usize,
}

pub(super) fn remove_exterior_elements_across_interior_source_faces(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> BoundaryLeakRecovery {
    let recoverable_source_faces = interior_source_faces(initial_recovery_queue);
    if recoverable_source_faces.is_empty() {
        return empty_recovery();
    }
    let node_coordinates = node_coordinates(tetrahedron_mesh);
    let Some(plc_triangles) = plc_boundary_triangles(plc, &node_coordinates) else {
        return empty_recovery();
    };
    let element_indices_by_face = element_indices_by_face(tetrahedron_mesh);
    let mut leaked_element_indices = BTreeSet::<usize>::new();
    let mut exposed_facets = BTreeMap::<[TopologyEntityId; 3], PlcFacet>::new();

    for facet in plc.facets.iter().filter(|facet| {
        recoverable_source_faces.contains(&(
            facet.source_face_id.clone(),
            sorted_topology_ids(facet.node_ids.clone()),
        ))
    }) {
        let face_key = sorted_topology_ids(facet.node_ids.clone());
        let Some(element_indices) = element_indices_by_face.get(&face_key) else {
            continue;
        };
        if element_indices.len() != 2 {
            continue;
        }
        if element_indices
            .iter()
            .map(|element_index| {
                tetrahedron_mesh.elements[*element_index]
                    .material_region_id
                    .as_str()
            })
            .collect::<BTreeSet<_>>()
            .len()
            != 1
        {
            continue;
        }
        let outside_element_indices = element_indices
            .iter()
            .copied()
            .filter(|element_index| {
                element_opposite_point(
                    &tetrahedron_mesh.elements[*element_index],
                    &face_key,
                    &node_coordinates,
                )
                .is_some_and(|point| {
                    point_in_closed_triangle_surface(
                        point,
                        &plc_triangles,
                        MeshingTolerance::default(),
                    ) == PointInClosedSurface::Outside
                })
            })
            .collect::<Vec<_>>();
        if outside_element_indices.len() != 1 {
            continue;
        }
        leaked_element_indices.insert(outside_element_indices[0]);
        exposed_facets.insert(face_key, facet.clone());
    }

    if leaked_element_indices.is_empty() {
        return empty_recovery();
    }
    let removed_element_count = remove_elements(tetrahedron_mesh, &leaked_element_indices);
    let (exposed_source_face_count, inserted_boundary_face_count) =
        materialize_exposed_boundary_faces(plc, tetrahedron_mesh, exposed_facets);

    BoundaryLeakRecovery {
        removed_element_count,
        exposed_source_face_count,
        inserted_boundary_face_count,
    }
}

fn empty_recovery() -> BoundaryLeakRecovery {
    BoundaryLeakRecovery {
        removed_element_count: 0,
        exposed_source_face_count: 0,
        inserted_boundary_face_count: 0,
    }
}

fn interior_source_faces(
    initial_recovery_queue: &TetrahedronRecoveryQueue,
) -> BTreeSet<(TopologyEntityId, [TopologyEntityId; 3])> {
    initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceFace
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.source_face_topology == Some(TetrahedronSourceFaceTopology::InteriorFace)
        })
        .filter_map(|item| {
            Some((
                item.source_entity_id.clone()?,
                item.source_face_node_ids.clone()?,
            ))
        })
        .collect()
}

fn plc_boundary_triangles(
    plc: &ProtectedBoundaryComplex,
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<Vec<Triangle3>> {
    plc.facets
        .iter()
        .map(|facet| {
            Some([
                *node_coordinates.get(&facet.node_ids[0])?,
                *node_coordinates.get(&facet.node_ids[1])?,
                *node_coordinates.get(&facet.node_ids[2])?,
            ])
        })
        .collect()
}

fn element_indices_by_face(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<[TopologyEntityId; 3], Vec<usize>> {
    tetrahedron_mesh
        .elements
        .iter()
        .enumerate()
        .flat_map(|(element_index, element)| {
            tetrahedron_element_faces(element.node_ids.clone())
                .into_iter()
                .map(move |face| (face, element_index))
        })
        .fold(
            BTreeMap::<[TopologyEntityId; 3], Vec<usize>>::new(),
            |mut indices_by_face, (face, element_index)| {
                indices_by_face.entry(face).or_default().push(element_index);
                indices_by_face
            },
        )
}

fn element_opposite_point(
    element: &Tetrahedron4Element,
    face_key: &[TopologyEntityId; 3],
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<[f64; 3]> {
    let opposite_nodes = element
        .node_ids
        .iter()
        .filter(|node_id| !face_key.contains(node_id))
        .collect::<Vec<_>>();
    let [opposite_node] = opposite_nodes.as_slice() else {
        return None;
    };
    node_coordinates.get(*opposite_node).copied()
}

fn remove_elements(
    tetrahedron_mesh: &mut TetrahedronMesh,
    leaked_element_indices: &BTreeSet<usize>,
) -> usize {
    let original_count = tetrahedron_mesh.elements.len();
    let elements = std::mem::take(&mut tetrahedron_mesh.elements);
    tetrahedron_mesh.elements = elements
        .into_iter()
        .enumerate()
        .filter_map(|(element_index, element)| {
            (!leaked_element_indices.contains(&element_index)).then_some(element)
        })
        .collect();
    original_count - tetrahedron_mesh.elements.len()
}

fn materialize_exposed_boundary_faces(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
    exposed_facets: BTreeMap<[TopologyEntityId; 3], PlcFacet>,
) -> (usize, usize) {
    let element_face_counts = element_face_counts(tetrahedron_mesh);
    let mut boundary_face_indices = tetrahedron_mesh
        .boundary_faces
        .iter()
        .enumerate()
        .map(|(boundary_face_index, boundary_face)| {
            (
                sorted_topology_ids(boundary_face.node_ids.clone()),
                boundary_face_index,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut exposed_source_face_count = 0;
    let mut inserted_boundary_face_count = 0;

    for (face_key, facet) in exposed_facets {
        if element_face_counts.get(&face_key).copied() != Some(1) {
            continue;
        }
        exposed_source_face_count += 1;
        let source_edge_ids =
            source_edge_ids_for_face_edges(&plc.protected_edges, facet.node_ids.clone());
        if let Some(boundary_face_index) = boundary_face_indices.get(&face_key).copied() {
            let boundary_face = &mut tetrahedron_mesh.boundary_faces[boundary_face_index];
            boundary_face.face_id = facet.facet_id;
            boundary_face.node_ids = facet.node_ids;
            boundary_face.source_face_id = facet.source_face_id;
            boundary_face.source_edge_ids = source_edge_ids;
            continue;
        }
        boundary_face_indices.insert(face_key, tetrahedron_mesh.boundary_faces.len());
        tetrahedron_mesh
            .boundary_faces
            .push(TetrahedronBoundaryFace {
                face_id: facet.facet_id,
                node_ids: facet.node_ids,
                source_face_id: facet.source_face_id,
                source_edge_ids,
            });
        inserted_boundary_face_count += 1;
    }

    (exposed_source_face_count, inserted_boundary_face_count)
}

fn node_coordinates(tetrahedron_mesh: &TetrahedronMesh) -> BTreeMap<TopologyEntityId, [f64; 3]> {
    tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect()
}

fn element_face_counts(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<[TopologyEntityId; 3], usize> {
    tetrahedron_mesh
        .elements
        .iter()
        .flat_map(|element| tetrahedron_element_faces(element.node_ids.clone()))
        .fold(
            BTreeMap::<[TopologyEntityId; 3], usize>::new(),
            |mut counts, face| {
                *counts.entry(face).or_default() += 1;
                counts
            },
        )
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
