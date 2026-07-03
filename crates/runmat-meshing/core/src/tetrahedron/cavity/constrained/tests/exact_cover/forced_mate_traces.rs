use super::*;

#[test]
fn exact_cover_trace_reports_forced_mate_without_addable_mate() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [4, 5, 6, 7],
        volume_m3: 0.1,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts = BTreeMap::from([(sorted_face([0, 1, 2]), 1)]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();
    let mut selected_roles = Vec::<&'static str>::new();

    let result = search.propagate_forced_interior_mates_traced(
        0.0,
        &mut face_counts,
        &mut selected,
        &mut selected_roles,
    );

    assert_eq!(
        result,
        Err(ForcedInteriorMateFailure::NoAddableMate {
            face: Some([0, 1, 2]),
            reason: ForcedInteriorMateNoAddableReason::NoCandidateContainsFace
        })
    );
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
    assert!(selected_roles.is_empty());
}

#[test]
fn exact_cover_trace_reports_forced_mate_face_count_conflict() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 0.1,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts =
        BTreeMap::from([(sorted_face([0, 1, 2]), 1), (sorted_face([0, 1, 3]), 2)]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();
    let mut selected_roles = Vec::<&'static str>::new();

    let result = search.propagate_forced_interior_mates_traced(
        0.0,
        &mut face_counts,
        &mut selected,
        &mut selected_roles,
    );

    assert_eq!(
        result,
        Err(ForcedInteriorMateFailure::NoAddableMate {
            face: Some([0, 1, 2]),
            reason: ForcedInteriorMateNoAddableReason::FaceCountConflict
        })
    );
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
    assert!(selected_roles.is_empty());
}

#[test]
fn exact_cover_trace_reports_forced_mate_future_mate_conflict() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 0.1,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts = BTreeMap::from([(sorted_face([0, 1, 2]), 1)]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();
    let mut selected_roles = Vec::<&'static str>::new();

    let result = search.propagate_forced_interior_mates_traced(
        0.0,
        &mut face_counts,
        &mut selected,
        &mut selected_roles,
    );

    assert_eq!(
        result,
        Err(ForcedInteriorMateFailure::NoAddableMate {
            face: Some([0, 1, 2]),
            reason: ForcedInteriorMateNoAddableReason::FutureMateConflict
        })
    );
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
    assert!(selected_roles.is_empty());
}

#[test]
fn exact_cover_trace_reports_forced_mate_volume_overflow() {
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
    let mut selected_roles = Vec::<&'static str>::new();

    let result = search.propagate_forced_interior_mates_traced(
        0.0,
        &mut face_counts,
        &mut selected,
        &mut selected_roles,
    );

    assert_eq!(
        result,
        Err(ForcedInteriorMateFailure::VolumeOverflow {
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.2,
            target_volume_m3: 0.1,
        })
    );
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
    assert!(selected_roles.is_empty());
}
