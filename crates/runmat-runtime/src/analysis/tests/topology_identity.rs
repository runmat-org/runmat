use super::*;

#[test]
fn render_topology_preserves_repeated_occurrence_selection_identities() {
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let first = PersistentEntityId::new(
        PersistentEntityKind::Face,
        "shared-face".into(),
        vec!["root".into(), "instance-a".into()],
    );
    let second = {
        let __base = &(first.clone());
        PersistentEntityId::new(
            __base.kind,
            __base.source_topology_id.clone(),
            vec!["root".into(), "instance-b".into()],
        )
    };
    mesh.topology.boundary_faces[0].provenance = vec![first.clone()];
    mesh.topology.boundary_faces[1].provenance = vec![second.clone()];

    let topology = render_topology_from_solver_mesh(Some(&mesh)).unwrap();
    let regions = &topology.meshes[0].regions;
    let first_id = first.analysis_selection_id().unwrap();
    let second_id = second.analysis_selection_id().unwrap();
    let first_region = regions
        .iter()
        .find(|region| region.region_id == first_id)
        .unwrap();
    let second_region = regions
        .iter()
        .find(|region| region.region_id == second_id)
        .unwrap();
    assert_ne!(first_region.triangle_ranges, second_region.triangle_ranges);
    assert!(!regions
        .iter()
        .any(|region| region.region_id == "shared-face"));
}
