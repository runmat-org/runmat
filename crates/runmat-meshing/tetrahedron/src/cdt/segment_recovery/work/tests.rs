use super::*;
use runmat_meshing_core::NeverCancelled;

#[test]
fn phase_inventory_counts_every_attempt_against_one_unchanged_search_limit() {
    let options = DelaunaySegmentRecoveryOptions {
        maximum_search_steps: 7,
        ..Default::default()
    };
    let mut work = RecoveryWork::new(options, &NeverCancelled);
    work.search_step(7).unwrap();
    work.search_location_step(7).unwrap();
    work.search_encroachment_step(7).unwrap();
    work.search_flip_step(7).unwrap();
    work.search_point_index_step(7).unwrap();
    work.search_point_handoff_step(7).unwrap();
    work.search_cell_index_step(7).unwrap();
    let error = work.search_cell_query_step(7).unwrap_err();
    assert_eq!(error.kind, DelaunaySegmentRecoveryErrorKind::ResourceLimit);
    assert_eq!(work.search_steps, 8);
    assert_eq!(work.search_by_phase, [1; 8]);
    assert!(error.to_string().contains("total=8, maximum=7"));
    assert!(error
        .to_string()
        .contains("topology=1, location=1, encroachment=1, flips=1"));
}
