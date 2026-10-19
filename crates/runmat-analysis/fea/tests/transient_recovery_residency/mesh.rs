//! Physical fixture meshes used by the isolated recovery-residency test.

use runmat_analysis_core::{AnalysisModel, ExcitationTarget};
use runmat_meshing_core::{
    fixtures::{canonical_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh},
    ElementOrder, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact, StableDigest,
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
    let mut mesh = match kind {
        RecoveryMesh::Tet4 => canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
        RecoveryMesh::Tet10 => canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
        RecoveryMesh::ConnectedTet4 => connected_tetrahedron_solver_mesh(),
    };
    if let Some(assignment) = model.material_assignments.first() {
        let region = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: assignment.selected_region_id().unwrap().to_owned(),
            assembly_path: vec!["root".to_owned()],
        };
        for element in &mut mesh.topology.volume_elements {
            element.region_id = region.clone();
        }
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
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh
}
