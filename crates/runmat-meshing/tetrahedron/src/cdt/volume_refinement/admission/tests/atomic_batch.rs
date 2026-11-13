use super::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

struct Stop(Arc<AtomicBool>);
impl runmat_meshing_core::MeshingCancellationSignal for Stop {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}
struct StopAfterPoint(Arc<AtomicBool>);
impl tracing::Subscriber for StopAfterPoint {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == "runmat_meshing_tetrahedron::refinement::provisional"
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if self.enabled(event.metadata()) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cancellation after a real provisional point restores the complete original topology
//   and quality under uniform and scoped metrics. It does not certify every cancellation boundary,
//   CAD publication or whole-task quiescence.
#[test]
fn cancellation_after_provisional_point_restores_complete_topology_and_quality() {
    for scoped in [false, true] {
        let (topology, provenance) = fixture_scaled(false, 0, [1.0; 3]);
        let original_topology = topology.clone();
        let request = metric(&provenance, scoped);
        let options = DelaunayVolumeQualityOptions::default();
        let original_quality = crate::cdt::evaluate_delaunay_volume_quality(
            &topology,
            &request,
            &provenance,
            options,
            &NeverCancelled,
        )
        .unwrap();
        let prepared = PreparedLocalQuality::prepare(&request, &provenance, options).unwrap();
        let mut state = WorkingState::new(
            topology,
            original_quality.clone(),
            options,
            &prepared,
            &[],
            &NeverCancelled,
        )
        .unwrap();
        let stopped = Arc::new(AtomicBool::new(false));
        let signal = Stop(stopped.clone());
        let controls = DelaunayVolumeRefinementStepOptions::default();
        let result = tracing::subscriber::with_default(StopAfterPoint(stopped.clone()), || {
            crate::cdt::volume_refinement::working::WorkingRefinement::default()
                .step(&mut state, controls, 64, &signal)
        });
        assert!(
            stopped.load(Ordering::Relaxed),
            "must cancel after real provisional progress"
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("cancelled batch returned progress"),
        };
        assert_eq!(error.kind, DelaunayVolumeRefinementStepErrorKind::Cancelled);
        let (topology, quality) = state.finish(controls, &NeverCancelled).unwrap();
        assert_eq!(topology, original_topology);
        assert_eq!(quality, original_quality);
    }
}
