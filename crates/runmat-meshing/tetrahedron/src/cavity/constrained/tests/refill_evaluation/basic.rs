use super::super::*;

#[test]
fn refill_candidates_preserve_split_boundary_face() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.boundary_faces =
        split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [0, 1, 2], 4)
            .expect("fixture face should split");
    let mut nodes = unit_tetrahedron_nodes();
    nodes.push(ConstrainedCavityNode {
        node_id: 4,
        coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
    });

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("split boundary cavity should refill");

    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("refill should preserve split boundary faces");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("split boundary refill should preserve volume");
    assert!(
        refill
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&4))
            .count()
            >= 3
    );
}

#[test]
fn refill_candidates_preserve_single_tetrahedron_cavity_boundary_and_volume() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("single tetrahedron cavity should refill");

    assert_eq!(refill.tetrahedra.len(), 1);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("refill boundary should match cavity boundary");
    assert!((refill.total_volume_m3 - cavity.target_volume_m3).abs() < 1.0e-12);
}

#[test]
fn single_tetrahedron_refill_ignores_non_boundary_nodes_in_coordinate_table() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();
    nodes.push(ConstrainedCavityNode {
        node_id: 99,
        coordinates_m: [4.0, 4.0, 4.0],
    });

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("coordinate table may contain nodes outside the cavity boundary");

    assert_eq!(refill.tetrahedra.len(), 1);
    assert!((refill.total_volume_m3 - cavity.target_volume_m3).abs() < 1.0e-12);
}

#[test]
fn star_refill_candidates_preserve_cavity_boundary_and_volume() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let interior = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [0.25, 0.25, 0.25],
    }];

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &interior, refill_options())
            .expect("interior star refill should generate");

    assert_eq!(refill.tetrahedra.len(), 4);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("star refill boundary should match cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("star refill should preserve cavity volume");
}

#[test]
fn refill_evaluation_uses_boundary_nodes_for_multi_face_cavity_without_interior_point() {
    let cavity = octahedron_cavity();
    let nodes = octahedron_nodes();

    let evaluation =
        evaluate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("evaluation should complete");

    let refill = evaluation
        .refill
        .expect("boundary-node refill should support closed multi-face cavities");
    assert!(evaluation.rejected_by_reason.is_empty());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("boundary-node refill should preserve the cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("boundary-node refill should preserve volume");
}
