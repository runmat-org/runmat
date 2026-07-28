use super::super::*;

#[test]
fn multi_interior_exact_cover_failure_reports_boundary_face_without_addable_candidate() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let options = refill_options();
    let lower_points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let lower_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], lower_points, options)
            .expect("fixture tetrahedron should pass quality gates");

    assert_eq!(
        multi_interior_exact_cover_failure_reason(&cavity, &[lower_tetrahedron], options),
        "multi_interior_exact_cover_boundary_face_no_addable_candidate"
    );
}

#[test]
fn exact_cover_trace_reports_boundary_face_without_addable_candidate() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: vec![
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 2],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 3],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
        ],
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 4],
            volume_m3: 0.1,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 3, 4],
            volume_m3: 0.1,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
    ];
    let mut search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);

    let (selected, trace) = search.search_with_trace();

    assert!(selected.is_none());
    assert_eq!(
        trace.dead_end,
        Some(BoundaryExactCoverDeadEnd {
            reason: "boundary_face_no_addable_candidate",
            face: Some([0, 1, 2]),
            depth: 0,
            selected_tetrahedra: Vec::new(),
            selected_roles: Vec::new(),
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.0,
            target_volume_m3: 1.0,
        })
    );
    assert_eq!(
        trace.dead_end_reason_counts,
        BTreeMap::from([("boundary_face_no_addable_candidate", 1)])
    );
    assert_eq!(
        trace.dead_end_faces_by_reason,
        BTreeMap::from([(
            "boundary_face_no_addable_candidate",
            BTreeSet::from([[0, 1, 2]])
        )])
    );
}
