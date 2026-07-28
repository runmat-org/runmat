use super::*;

#[test]
fn exact_cover_search_targets_unpaired_interior_faces_after_boundary_faces() {
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
                node_ids: [3, 4, 5],
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
            node_ids: [0, 1, 2, 6],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [3, 4, 5, 7],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 6, 8],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 8, 9],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [1, 6, 8, 10],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 6, 8, 11],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
    ];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for face in [[0, 1, 2], [3, 4, 5], sorted_face([0, 1, 6])] {
        face_counts.insert(face, 1);
    }

    let candidates = search
        .next_cover_candidates(&face_counts, &[0, 1])
        .expect("unpaired interior face should request connector candidates");

    assert_eq!(candidates, vec![2]);
}

#[test]
fn exact_cover_search_prunes_orphan_interior_faces() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: vec![ConstrainedCavityBoundaryFace {
            node_ids: [0, 1, 2],
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        }],
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 3],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 4],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 4, 5],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [1, 2, 4, 6],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 2, 4, 7],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
    ];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);

    let candidates = search
        .next_cover_candidates(&BTreeMap::new(), &[])
        .expect("boundary face should request cover candidates");

    assert_eq!(candidates, vec![1]);
}

#[test]
fn exact_cover_search_forces_single_interior_mate() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 3, 4],
        volume_m3: 0.2,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let mut face_counts = BTreeMap::from([
        (sorted_face([0, 1, 3]), 1),
        (sorted_face([0, 1, 4]), 1),
        (sorted_face([0, 3, 4]), 1),
        (sorted_face([1, 3, 4]), 1),
    ]);
    let mut selected = Vec::<usize>::new();

    let propagated = search
        .propagate_forced_interior_mates(0.0, &mut face_counts, &mut selected)
        .expect("single interior mate should be forced");

    assert_eq!(propagated, (0.2, vec![0]));
    assert_eq!(selected, vec![0]);
    assert!(face_counts.values().all(|count| *count == 2));
}

#[test]
fn exact_cover_search_rolls_back_forced_mates_on_volume_failure() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 0.1,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 3, 4],
        volume_m3: 0.2,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts = BTreeMap::from([
        (sorted_face([0, 1, 3]), 1),
        (sorted_face([0, 1, 4]), 1),
        (sorted_face([0, 3, 4]), 1),
        (sorted_face([1, 3, 4]), 1),
    ]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();

    assert!(search
        .propagate_forced_interior_mates(0.0, &mut face_counts, &mut selected)
        .is_none());
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
}
