use super::*;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
    PersistentEntityKind,
};

#[test]
fn shared_cad_face_provenance_is_deduplicated_without_changing_selection() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        let entity = PersistentEntityId::new(
            PersistentEntityKind::Face,
            "shared-face".into(),
            vec!["root".into()],
        );
        for face in &mut mesh.topology.boundary_faces {
            face.provenance.push(entity.clone());
        }
        let mut target =
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("shared-face"));
        target.cardinality = TargetCardinality::One;
        let expected = resolve_nodal_target(&mesh, &target, "fixed").unwrap();
        assert_eq!(expected.matched_entities, vec![entity]);
        assert_eq!(
            expected.matched_node_identities.len(),
            mesh.topology.nodes.len()
        );
        // Exercise the selection owner independently of mesh validation: repeated references
        // must not alter canonical target identity or count triangles as distinct CAD faces.
        let faces = mesh.topology.boundary_faces.clone();
        for _ in 0..1024 {
            mesh.topology.boundary_faces.extend(faces.iter().cloned());
        }
        assert_eq!(
            resolve_nodal_target(&mesh, &target, "fixed").unwrap(),
            expected
        );
    }
}
