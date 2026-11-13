//! Physical fixture meshes used by the isolated recovery-residency test.

use runmat_analysis_core::{AnalysisModel, ExcitationTarget};
use runmat_meshing_core::{
    fixtures::{
        canonical_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh, fixture_entity,
        fixture_entity_for_selection,
    },
    ElementOrder, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact, StableDigest,
};

#[derive(Clone, Copy)]
pub(super) enum RecoveryMesh {
    Tet4,
    Tet10,
    ConnectedTet4,
}

fn add_face_region(mesh: &mut SolverMeshArtifact, face: usize, entity: PersistentEntityId) {
    assert_eq!(entity.kind, PersistentEntityKind::Face);
    let provenance = &mut mesh.topology.boundary_faces[face].provenance;
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
        let region = fixture_entity_for_selection(assignment.selected_region_id().unwrap())
            .expect("fixture material domain has its registered persistent entity");
        assert_eq!(region.kind, PersistentEntityKind::Region);
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
        let entity = fixture_entity_for_selection(&region.region_id)
            .expect("fixture constraint has its registered persistent face");
        let face = if entity == fixture_entity(PersistentEntityKind::Face, "free_vertex_y_support")
        {
            1
        } else if entity == fixture_entity(PersistentEntityKind::Face, "free_vertex_z_support") {
            2
        } else {
            assert_eq!(entity, fixture_entity(PersistentEntityKind::Face, "root"));
            0
        };
        add_face_region(&mut mesh, face, entity);
    }
    for action in &model.excitations {
        if let ExcitationTarget::Region(target) = &action.excitation().target {
            let entity = fixture_entity_for_selection(&target.region_id)
                .expect("fixture excitation has its registered persistent face");
            if entity != fixture_entity(PersistentEntityKind::Face, "dynamic_force") {
                add_face_region(&mut mesh, 1, entity);
            }
        }
    }
    add_face_region(
        &mut mesh,
        1,
        fixture_entity(PersistentEntityKind::Face, "tip"),
    );
    for face in [1, 3] {
        add_face_region(
            &mut mesh,
            face,
            fixture_entity(PersistentEntityKind::Face, "dynamic_force"),
        );
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh
}
