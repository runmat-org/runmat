use super::*;

#[test]
fn connected_tet10_patch_preserves_affine_fields_with_curved_interior_elements() {
    let tet4 = refine(connected_tetrahedron_solver_mesh(), 1).unwrap();
    let corner_count = tet4.topology.nodes.len() as u64;
    let mut mesh = elevate_connected_tet4_fixture(tet4);
    let exterior_nodes = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    let bowed = mesh
        .topology
        .nodes
        .iter_mut()
        .find(|node| node.node_id > corner_count && !exterior_nodes.contains(&node.node_id))
        .expect("a connected refined patch has an interior midside node");
    assert!(bowed.exact_parameters.is_empty());
    let bowed_id = bowed.node_id;
    {
        let quantity_change = 0.01;
        let quantity_index = 2;
        let quantity_slot = &mut bowed.coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(
        mesh.topology
            .volume_elements
            .iter()
            .filter(|element| element.node_ids.contains(&bowed_id))
            .count()
            > 1
    );
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();

    let model = patch_model(&mesh);
    let serial = solve_patch(&model, &mesh, LocalExecutionPlan::serial()).unwrap();
    let scoped = solve_patch(
        &model,
        &mesh,
        LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap(),
    )
    .unwrap();
    assert_static_field_parity(&serial, &scoped);
    assert_affine_displacement(&scoped, &mesh, 1.0e-6);
    assert_affine_element_fields(&scoped, &mesh, 1.0e-5, 1.0e-3);
}
