use super::*;

#[test]
fn boundary_cap_nodes_are_empty_when_solid_boundary_coverage_exists() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let cap_nodes =
        generate_constrained_cavity_boundary_cap_nodes(&cavity, &nodes, refill_options(), 4)
            .expect("boundary cap node generation should evaluate");

    assert!(cap_nodes.is_empty());
}

#[test]
fn solid_empty_boundary_faces_are_empty_when_boundary_coverage_exists() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let faces = constrained_cavity_solid_empty_boundary_faces(&cavity, &nodes, refill_options())
        .expect("solid-empty boundary face detection should evaluate");

    assert!(faces.is_empty());
}

#[test]
fn solid_empty_boundary_face_classification_is_empty_when_coverage_exists() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let classification =
        constrained_cavity_classified_solid_empty_boundary_faces(&cavity, &nodes, refill_options())
            .expect("solid-empty boundary face classification should evaluate");

    assert_eq!(
        classification,
        ConstrainedCavitySolidEmptyBoundaryFaces {
            faces: Vec::new(),
            true_exterior_faces: Vec::new(),
            expandable_faces: Vec::new(),
        }
    );
}

#[test]
fn solid_empty_boundary_recovery_is_noop_when_coverage_exists() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let source_tetrahedra = [
        candidate_tetrahedron(1, [0, 1, 2, 3], 1.0 / 6.0, &["body"]),
        candidate_tetrahedron(2, [0, 2, 1, 4], 1.0 / 6.0, &["body"]),
    ];

    let recovery = recover_constrained_cavity_solid_empty_boundaries(
        &cavity,
        &nodes,
        &source_tetrahedra,
        &nodes,
        refill_options(),
    )
    .expect("solid-empty boundary recovery should evaluate");

    assert_eq!(recovery.cavity, cavity);
    assert!(recovery.split_nodes.is_empty());
    assert!(recovery.split_steps.is_empty());
    assert!(recovery.rejected_splits.is_empty());
    assert!(recovery.expanded_removed_tetrahedron_ids.is_empty());
    assert!(recovery.classification.faces.is_empty());
}
