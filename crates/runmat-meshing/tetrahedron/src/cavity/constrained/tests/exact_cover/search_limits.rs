use super::*;

#[test]
fn exact_cover_search_uses_configured_attempt_limit() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 3],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 2, 1, 4],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
    ];
    let mut low_limit_search =
        BoundaryExactCoverSearch::with_attempt_limit(&cavity, &candidates, 1.0e-9, 1);

    assert!(low_limit_search.search().is_none());
    assert!(low_limit_search.attempts > 1);

    let mut sufficient_limit_search =
        BoundaryExactCoverSearch::with_attempt_limit(&cavity, &candidates, 1.0e-9, 2);

    assert_eq!(sufficient_limit_search.search(), Some(vec![0, 1]));
    assert_eq!(sufficient_limit_search.attempts, 2);
}

#[test]
fn exact_cover_trace_reports_volume_overflow_dead_end() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let candidates = [
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 3],
            volume_m3: 10.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 2, 1, 4],
            volume_m3: 10.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
    ];
    let mut search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-12);

    let (selected, trace) = search.search_with_trace();

    assert!(selected.is_none());
    assert_eq!(
        trace.dead_end,
        Some(BoundaryExactCoverDeadEnd {
            reason: "volume_overflow",
            face: None,
            depth: 1,
            selected_tetrahedra: vec![[0, 1, 2, 3]],
            selected_roles: vec!["branch"],
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.0,
            target_volume_m3: 1.0 / 3.0,
        })
    );
}

#[test]
fn exact_cover_recursive_search_checks_cancellation() {
    struct Cancelled;
    impl runmat_meshing_core::MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let cavity = two_tetrahedron_bipyramid_cavity();
    let candidates = [ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    }];
    let mut search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-12);

    assert!(search.search_with_trace_controlled(&Cancelled, 1).is_err());
    assert_eq!(search.attempts, 1);
}
