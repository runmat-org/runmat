use super::super::*;

#[test]
fn two_interior_node_refill_preserves_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let interior_candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, -0.25],
        },
    ];
    let options = refill_options();

    let refill = two_interior_node_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &interior_candidates,
        options,
    )
    .expect("two-interior refill should evaluate")
    .expect("two-interior refill should recover the cavity");

    assert_eq!(refill.inserted_nodes, interior_candidates);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("two-interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("two-interior refill should preserve volume");
}

#[test]
fn multi_interior_node_refill_preserves_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let interior_candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, -0.25],
        },
        ConstrainedCavityNode {
            node_id: 12,
            coordinates_m: [0.50, 0.25, 0.0],
        },
    ];
    let options = refill_options();

    let refill = multi_interior_node_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &interior_candidates,
        options,
    )
    .expect("multi-interior refill should evaluate")
    .expect("multi-interior refill should recover the cavity");

    assert!(!refill.inserted_nodes.is_empty());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("multi-interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("multi-interior refill should preserve volume");
}
