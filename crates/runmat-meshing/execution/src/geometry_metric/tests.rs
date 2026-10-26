mod persistent;
use super::*;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_core::{GeometryEvaluationError, GeometryEvaluationErrorKind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Default)]
struct Control {
    cancelled: AtomicBool,
    allocations: AtomicUsize,
    cancel_on_iteration: AtomicBool,
    deny_allocations: AtomicBool,
}
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.cancelled.load(Ordering::Relaxed) {
            Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "cancel preparation",
            ))
        } else {
            Ok(())
        }
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        if self.cancel_on_iteration.swap(false, Ordering::Relaxed) {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        self.checkpoint()
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        self.allocations
            .fetch_add(bytes as usize, Ordering::Relaxed);
        if self.deny_allocations.load(Ordering::Relaxed) {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                "deny cached metric inventory",
            ));
        }
        self.checkpoint()
    }
}
#[derive(Clone, Debug)]
struct CountingProvider(Arc<AtomicUsize>);
impl ExactMeshingEvaluatorProvider for CountingProvider {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<
        Box<dyn ExactMeshingGeometryEvaluation + 'a>,
        runmat_geometry_core::GeometryContractError,
    > {
        crate::PortableMeshingEvaluatorProvider.evaluator(geometry, control)
    }
    fn geometry_metric(
        &self,
        geometry: &PreparedExactGeometryObjects,
        evaluator: &dyn ExactMeshingGeometryEvaluation,
        control: &dyn GeometryEvaluationControl,
        request: &MeshingRequest,
    ) -> Result<Arc<MetricFieldRequest>, SharedCurveError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        crate::PortableMeshingEvaluatorProvider
            .geometry_metric(geometry, evaluator, control, request)
    }
}
fn geometry() -> PreparedExactGeometryObjects {
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_tetrahedron();
    crate::prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Eight concurrent consumers share one actual prepared metric and match independent
//   derivation. Changes to requested sizing or curve/surface controls require fresh preparation.
//   This covers one worker-scoped immutable authority, not cross-process caching or native meshing
//   acceptance.
#[test]
fn concurrent_consumers_prepare_once_preserve_authority_and_admit_each_borrow() {
    let geometry = geometry();
    let request = crate::serial_tests::request();
    let count = Arc::new(AtomicUsize::new(0));
    let provider = PreparedMetricProvider::new(CountingProvider(Arc::clone(&count)));
    let control = Control::default();
    let fields = std::thread::scope(|scope| {
        let handles = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    let evaluator = provider.evaluator(&geometry, &control).unwrap();
                    provider
                        .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(count.load(Ordering::Relaxed), 1);
    assert!(fields.iter().all(|field| Arc::ptr_eq(field, &fields[0])));
    let evaluator = provider.evaluator(&geometry, &control).unwrap();
    let independent = crate::PortableMeshingEvaluatorProvider
        .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
        .unwrap();
    assert_eq!(fields[0], independent);
    assert!(control.allocations.load(Ordering::Relaxed) > 0);

    let mut changed = request.clone();
    changed.metric.maximum_grading_ratio += 0.1;
    let next = provider
        .geometry_metric(&geometry, evaluator.as_ref(), &control, &changed)
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 2);
    assert!(!Arc::ptr_eq(&next, &fields[0]));
    changed.quality.surface.maximum_chordal_deviation_m = runmat_types::physical::Quantity::new(
        changed
            .quality
            .surface
            .maximum_chordal_deviation_m
            .canonical_value()
            * (0.5),
    );
    provider
        .geometry_metric(&geometry, evaluator.as_ref(), &control, &changed)
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 3);
    changed.quality.curve.maximum_chordal_deviation_m = runmat_types::physical::Quantity::new(
        changed
            .quality
            .curve
            .maximum_chordal_deviation_m
            .canonical_value()
            * (0.5),
    );
    provider
        .geometry_metric(&geometry, evaluator.as_ref(), &control, &changed)
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 4);
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cancellation before and during derivation does not create a successful cached field;
//   later preparation succeeds. Cached reuse still rejects cancellation and denied storage. This
//   does not establish whole-task memory bounds or scientific accuracy.
#[test]
fn cancelled_preparation_is_not_cached_and_cache_hits_still_observe_cancellation() {
    let geometry = geometry();
    let request = crate::serial_tests::request();
    let count = Arc::new(AtomicUsize::new(0));
    let provider = PreparedMetricProvider::new(CountingProvider(Arc::clone(&count)));
    let control = Control::default();
    let evaluator = provider.evaluator(&geometry, &control).unwrap();
    control.cancelled.store(true, Ordering::Relaxed);
    assert_eq!(
        provider
            .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
            .unwrap_err()
            .kind,
        SharedCurveErrorKind::GeometryEvaluation(GeometryEvaluationErrorKind::Cancelled)
    );
    assert_eq!(count.load(Ordering::Relaxed), 0);
    control.cancelled.store(false, Ordering::Relaxed);
    control.cancel_on_iteration.store(true, Ordering::Relaxed);
    assert!(provider
        .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
        .is_err());
    assert_eq!(count.load(Ordering::Relaxed), 1);
    control.cancelled.store(false, Ordering::Relaxed);
    provider
        .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 2);
    control.deny_allocations.store(true, Ordering::Relaxed);
    assert_eq!(
        provider
            .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
            .unwrap_err()
            .kind,
        SharedCurveErrorKind::GeometryEvaluation(
            GeometryEvaluationErrorKind::AllocationBudgetExceeded
        )
    );
    assert_eq!(count.load(Ordering::Relaxed), 2);
    control.deny_allocations.store(false, Ordering::Relaxed);
    control.cancelled.store(true, Ordering::Relaxed);
    assert!(provider
        .geometry_metric(&geometry, evaluator.as_ref(), &control, &request)
        .is_err());
    assert_eq!(count.load(Ordering::Relaxed), 2);
}
