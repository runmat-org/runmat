use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

pub(in crate::cdt::facet_recovery) struct CancelOnSecondProbe(AtomicUsize);

impl CancelOnSecondProbe {
    pub(in crate::cdt::facet_recovery) fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    pub(in crate::cdt::facet_recovery) fn probes(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

impl MeshingCancellationSignal for CancelOnSecondProbe {
    fn is_cancelled(&self) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed) >= 1
    }
}

pub(in crate::cdt::facet_recovery) fn two_small_facets(
) -> (DelaunayConstraints, DelaunaySegmentRecovery) {
    let mut constraints = constraints(false);
    let mut side = constraints.facets[0].clone();
    side.facet_id = StableDigest::from_bytes([92; 32]);
    side.chart_id = StableDigest::from_bytes([93; 32]);
    side.vertex_indices = [0, 1, 3];
    side.source_face_id = entity(PersistentEntityKind::Face, "face:side");
    constraints.facets.push(side);
    for vertex_indices in [[0, 3], [1, 3]] {
        constraints.segments.push(DelaunayConstraintSegment {
            vertex_indices,
            source_edge_id: None,
            source_edge_parameters: None,
        });
    }
    constraints
        .segments
        .sort_by_key(|segment| segment.vertex_indices);
    let segments = segment_recovery(false, &constraints);
    (constraints, segments)
}

#[test]
fn construction_observes_cancellation_between_short_facets() {
    let (constraints, segments) = two_small_facets();
    let options = DelaunayFacetRecoveryOptions::default();
    let complete =
        construct_delaunay_facet_recovery(segments.clone(), &constraints, options, &NeverCancelled)
            .unwrap();
    assert_eq!(complete.facets.len(), 2);
    validate_delaunay_facet_recovery(&complete, &constraints, options, &NeverCancelled).unwrap();

    let cancellation = CancelOnSecondProbe::new();
    let error = construct_delaunay_facet_recovery(segments, &constraints, options, &cancellation)
        .unwrap_err();
    assert_eq!(error.kind, DelaunayFacetRecoveryErrorKind::Cancelled);
    assert_eq!(error.constraint_index, Some(1));
    assert_eq!(cancellation.probes(), 2);
}
