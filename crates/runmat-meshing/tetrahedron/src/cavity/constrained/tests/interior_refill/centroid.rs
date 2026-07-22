use super::super::*;

#[test]
fn candidate_orphan_interior_face_counts_report_global_orphans() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let lower = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let upper = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 2, 1, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };

    assert_eq!(
        candidate_orphan_interior_face_counts(&cavity, std::slice::from_ref(&lower)),
        (1, 0)
    );
    assert_eq!(
        candidate_orphan_interior_face_counts(&cavity, &[lower, upper]),
        (0, 2)
    );
}

#[test]
fn centroid_interior_refill_candidate_recovers_split_boundary_tetrahedron_cavity() {
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
    validate_constrained_cavity(&cavity).expect("split boundary fixture should be valid");
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

    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let refill = centroid_interior_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        refill_options(),
    )
    .expect("centroid interior refill should evaluate")
    .expect("centroid interior refill should recover the split boundary cavity");

    assert_eq!(refill.inserted_nodes.len(), 1);
    assert_eq!(refill.inserted_nodes[0].node_id, 8);
    assert_eq!(refill.tetrahedra.len(), cavity.boundary_faces.len());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("centroid interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        refill_options().volume_relative_tolerance,
    )
    .expect("centroid interior refill should preserve volume");
}
