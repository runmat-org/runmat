use super::*;
use crate::cdt::facet_recovery::{
    construct_delaunay_facet_recovery,
    tests::cancellation::{two_small_facets, CancelOnSecondProbe},
};
use runmat_meshing_core::NeverCancelled;

#[test]
fn terminal_evidence_observes_cancellation_between_short_facets() {
    let (constraints, segments) = two_small_facets();
    let options = DelaunayFacetRecoveryOptions::default();
    let recovery =
        construct_delaunay_facet_recovery(segments, &constraints, options, &NeverCancelled)
            .unwrap();
    let effective = effective_segment_recovery(&recovery, &constraints);
    evidence::validate_recovered_facets(
        &recovery,
        &effective,
        &constraints,
        options,
        &NeverCancelled,
    )
    .unwrap();

    let cancellation = CancelOnSecondProbe::new();
    let error = evidence::validate_recovered_facets(
        &recovery,
        &effective,
        &constraints,
        options,
        &cancellation,
    )
    .unwrap_err();
    assert_eq!(error.kind, DelaunayFacetRecoveryErrorKind::Cancelled);
    assert_eq!(error.constraint_index, Some(1));
    assert_eq!(cancellation.probes(), 2);
}
