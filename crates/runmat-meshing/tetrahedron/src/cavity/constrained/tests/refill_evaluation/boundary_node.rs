use super::super::*;

#[test]
fn boundary_node_completion_repairs_missing_cavity_boundary_faces() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let incomplete_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, options)
            .expect("fixture tetrahedron should pass quality gates");

    assert!(refill_from_tetrahedra(
        &cavity,
        vec![incomplete_tetrahedron.clone()],
        options.volume_relative_tolerance
    )
    .is_err());

    let (_, completed, inserted_nodes) = complete_missing_boundary_face_tetrahedra(
        &cavity,
        &boundary_nodes,
        vec![incomplete_tetrahedron],
        &boundary_triangles,
        options,
    )
    .expect("completion should evaluate")
    .expect("completion should add the missing tetrahedron");
    let refill = refill_from_tetrahedra(&cavity, completed, options.volume_relative_tolerance)
        .expect("completed refill should validate");

    assert!(inserted_nodes.is_empty());
    assert_eq!(refill.tetrahedra.len(), 2);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("completed refill should preserve the cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("completed refill should preserve volume");
}

#[test]
fn boundary_node_completion_reports_when_no_cap_tetrahedron_passes_quality() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let initial_options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let incomplete_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, initial_options)
            .expect("fixture tetrahedron should pass initial quality gates");

    let rejected = complete_missing_boundary_face_tetrahedra(
        &cavity,
        &boundary_nodes,
        vec![incomplete_tetrahedron],
        &boundary_triangles,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.95,
            ..initial_options
        },
    )
    .expect("completion should evaluate")
    .expect_err("strict quality should reject every cap tetrahedron");

    assert_eq!(rejected, "boundary_node_completion_no_candidate");
}

#[test]
fn boundary_node_exact_cover_recovers_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let refill = boundary_node_exact_cover_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        options,
    )
    .expect("exact cover should evaluate")
    .expect("exact cover should recover the cavity");

    assert_eq!(refill.tetrahedra.len(), 2);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("exact cover should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("exact cover should preserve volume");
}
