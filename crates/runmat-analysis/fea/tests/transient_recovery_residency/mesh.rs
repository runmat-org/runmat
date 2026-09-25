//! Physical fixture meshes used by the isolated recovery-residency test.

use runmat_analysis_core::{AnalysisModel, ExcitationTarget};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, solver_volume_element_identity, ElementOrder,
    FieldTopologyLocation, MeshNeighbor, PersistentEntityId, PersistentEntityKind,
    SolverMeshArtifact, SolverMeshNode, StableDigest,
};

#[derive(Clone, Copy)]
pub(super) enum RecoveryMesh {
    Tet4,
    Tet10,
    ConnectedTet4,
}

fn add_face_region(mesh: &mut SolverMeshArtifact, face: usize, region: &str) {
    let provenance = &mut mesh.topology.boundary_faces[face].provenance;
    let entity = PersistentEntityId {
        kind: PersistentEntityKind::Face,
        source_topology_id: region.to_owned(),
        assembly_path: vec!["root".to_owned()],
    };
    if !provenance.contains(&entity) {
        provenance.push(entity);
        provenance.sort();
    }
}

pub(super) fn physical_fixture_mesh(
    model: &AnalysisModel,
    kind: RecoveryMesh,
) -> SolverMeshArtifact {
    let order = match kind {
        RecoveryMesh::Tet10 => ElementOrder::Tet10,
        RecoveryMesh::Tet4 | RecoveryMesh::ConnectedTet4 => ElementOrder::Tet4,
    };
    let mut mesh = canonical_tetrahedron_solver_mesh(order);
    if let Some(assignment) = model.material_assignments.first() {
        let region = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: assignment.selected_region_id().unwrap().to_owned(),
            assembly_path: vec!["root".to_owned()],
        };
        mesh.topology.volume_elements[0].region_id = region.clone();
        mesh.topology.regions[0].region_id = region;
    }
    for region in model
        .constraints
        .iter()
        .filter_map(|condition| condition.target())
    {
        match region.region_id.as_str() {
            "free_vertex_y_support" => add_face_region(&mut mesh, 1, &region.region_id),
            "free_vertex_z_support" => add_face_region(&mut mesh, 2, &region.region_id),
            other => add_face_region(&mut mesh, 0, other),
        }
    }
    for action in &model.excitations {
        if let ExcitationTarget::Region(target) = &action.excitation().target {
            if target.region_id != "dynamic_force" {
                add_face_region(&mut mesh, 1, &target.region_id);
            }
        }
    }
    for assignment in &model.material_assignments {
        add_face_region(&mut mesh, 1, assignment.selected_region_id().unwrap());
        add_face_region(&mut mesh, 1, &assignment.assigned_material_id);
    }
    add_face_region(&mut mesh, 1, "tip");
    add_face_region(&mut mesh, 1, "dynamic_force");
    add_face_region(&mut mesh, 3, "dynamic_force");
    if matches!(kind, RecoveryMesh::ConnectedTet4) {
        subdivide_tet4_around_centroid(&mut mesh);
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh
}

fn subdivide_tet4_around_centroid(mesh: &mut SolverMeshArtifact) {
    let parent = mesh.topology.volume_elements[0].clone();
    let centroid_id = 5;
    let centroid_identity = StableDigest::from_bytes([42; 32]);
    mesh.topology.nodes.push(SolverMeshNode {
        node_id: centroid_id,
        stable_identity: centroid_identity,
        coordinates_m: [0.25, 0.25, 0.25],
        provenance: parent.provenance.clone(),
        exact_parameters: Vec::new(),
    });
    let children = [[5, 2, 3, 4], [1, 5, 3, 4], [1, 2, 5, 4], [1, 2, 3, 5]];
    mesh.topology.volume_elements = children
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| {
            let mut child = parent.clone();
            child.element_id = index as u64 + 1;
            child.node_ids = node_ids.to_vec();
            child.stable_identity = solver_volume_element_identity(
                node_ids.map(|node_id| mesh.topology.nodes[node_id as usize - 1].stable_identity),
            );
            child
        })
        .collect();
    for face in &mut mesh.topology.boundary_faces {
        let corners = &face.node_ids[..3];
        face.adjacent_volume_element_ids = mesh
            .topology
            .volume_elements
            .iter()
            .filter(|element| corners.iter().all(|node| element.node_ids.contains(node)))
            .map(|element| element.element_id)
            .collect();
        assert_eq!(face.adjacent_volume_element_ids.len(), 1);
    }
    mesh.topology.neighbors = mesh
        .topology
        .volume_elements
        .iter()
        .flat_map(|element| {
            (0..4).map(|local_face_index| {
                let shared = element
                    .node_ids
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != local_face_index)
                    .map(|(_, node)| node)
                    .collect::<Vec<_>>();
                MeshNeighbor {
                    element_id: element.element_id,
                    local_face_index: local_face_index as u8,
                    adjacent_element_id: mesh
                        .topology
                        .volume_elements
                        .iter()
                        .find(|other| {
                            other.element_id != element.element_id
                                && shared.iter().all(|node| other.node_ids.contains(node))
                        })
                        .map(|other| other.element_id),
                }
            })
        })
        .collect();
    mesh.topology.regions[0].element_ids = (1..=4).collect();
    for topology in &mut mesh.topology.field_topologies {
        match topology.location {
            FieldTopologyLocation::Node => topology.ordered_entity_ids.push(centroid_id),
            FieldTopologyLocation::VolumeElement => topology.ordered_entity_ids = (1..=4).collect(),
            FieldTopologyLocation::BoundaryFace | FieldTopologyLocation::BoundaryEdge => {}
        }
    }
}
