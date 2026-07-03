use super::*;

#[test]
fn boundary_steiner_exact_cover_diagnostic_reports_centroid_candidate_coverage() {
    let mut cavity = unit_tetrahedron_cavity();
    let split_specs = [
        ([0, 2, 1], 4),
        ([0, 1, 3], 5),
        ([1, 2, 3], 6),
        ([2, 0, 3], 7),
    ];
    for (face, split_node_id) in split_specs {
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node_id)
                .expect("fixture face should split");
    }
    let mut nodes = unit_tetrahedron_nodes();
    nodes.extend([
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 7,
            coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
        },
    ]);

    let diagnostic = diagnostic_boundary_steiner_exact_cover(&cavity, &nodes, refill_options())
        .expect("Steiner exact-cover diagnostic should evaluate");

    assert!(diagnostic.candidate_count > 0);
    assert_eq!(diagnostic.zero_candidate_boundary_face_count, 0);
    assert!(diagnostic.search_attempt_count > 0);
    assert_eq!(diagnostic.reason, "cover_found");
    assert!(diagnostic.selected_tetrahedron_count > 0);
}

#[test]
fn boundary_patch_steiner_exact_cover_diagnostic_reports_boundary_complete_fixture() {
    let mut cavity = unit_tetrahedron_cavity();
    let split_specs = [
        ([0, 2, 1], 4),
        ([0, 1, 3], 5),
        ([1, 2, 3], 6),
        ([2, 0, 3], 7),
    ];
    for (face, split_node_id) in split_specs {
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node_id)
                .expect("fixture face should split");
    }
    let mut nodes = unit_tetrahedron_nodes();
    nodes.extend([
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 7,
            coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
        },
    ]);

    let diagnostic =
        diagnostic_boundary_patch_steiner_exact_cover(&cavity, &nodes, refill_options())
            .expect("patch Steiner exact-cover diagnostic should evaluate");

    assert_eq!(diagnostic.boundary_node_count, 8);
    assert_eq!(diagnostic.boundary_face_count, 12);
    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert_eq!(diagnostic.steiner_node_count, 0);
    assert_eq!(diagnostic.candidate_count, 0);
    assert_eq!(diagnostic.search_attempt_count, 0);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}
