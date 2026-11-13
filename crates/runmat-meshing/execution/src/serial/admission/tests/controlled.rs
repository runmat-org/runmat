//! Cold-input class checks; these synthetic metadata fixtures do not qualify CAD or physics.
use super::*;
use crate::{MeshingExecutionResult, PreparedMeshingInput};
use runmat_geometry_core::{GeometryEvaluationControl, GeometryEvaluationErrorKind};
use std::sync::{atomic::AtomicUsize, Mutex};
mod fixtures;
mod scale;
#[derive(Clone, Copy, Debug)]
enum Kind {
    Exact,
    Faceted,
    Evidence,
}
pub(super) struct RecordingControl {
    pub(super) inner: crate::MeshingGeometryEvaluationControl<'static>,
    pub(super) allocations: Mutex<Vec<u64>>,
    pub(super) checkpoints: AtomicUsize,
    fail_allocation: Option<usize>,
    fail_checkpoint: Option<usize>,
    kind: GeometryEvaluationErrorKind,
}
impl RecordingControl {
    pub(super) fn new(
        fail_allocation: Option<usize>,
        fail_checkpoint: Option<usize>,
        kind: GeometryEvaluationErrorKind,
    ) -> Self {
        Self {
            inner: bounded_control(64 << 20),
            allocations: Mutex::new(Vec::new()),
            checkpoints: AtomicUsize::new(0),
            fail_allocation,
            fail_checkpoint,
            kind,
        }
    }
    fn failure(&self) -> GeometryEvaluationError {
        GeometryEvaluationError::new(self.kind, "injected original input control failure")
    }
}
impl GeometryEvaluationControl for RecordingControl {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        let index = self.checkpoints.fetch_add(1, Ordering::Relaxed);
        if self.fail_checkpoint == Some(index) {
            return Err(self.failure());
        }
        self.inner.checkpoint()
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        let mut entries = self.allocations.lock().unwrap();
        let index = entries.len();
        entries.push(bytes);
        if self.fail_allocation == Some(index) {
            return Err(self.failure());
        }
        self.inner.consume_allocation_bytes(bytes)
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        self.inner.release_allocation_bytes(bytes);
    }
    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        self.inner.consume_iterations(count)
    }
    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        self.inner.consume_search_work(count)
    }
}
fn run(
    kind: Kind,
    fail_allocation: Option<usize>,
    fail_checkpoint: Option<usize>,
    error_kind: GeometryEvaluationErrorKind,
) -> (usize, usize) {
    let fixture = fixtures::Fixture::new(kind);
    let mut source = crate::tests::MemoryCache::default();
    source.insert_all(&fixture.objects);
    let reads = crate::tests::read_buffers::ReadBuffers::new(&source);
    let control = RecordingControl::new(fail_allocation, fail_checkpoint, error_kind);
    {
        let scope = GeometryAllocationScope::new(&control);
        let reader = InputReader::new(&reads, &scope);
        let codec = crate::codec_control::CodecControl::new(&scope);
        let limits = runmat_execution_artifact::object::ObjectInventoryLimits::default();
        let result: MeshingExecutionResult<PreparedMeshingInput> = match kind {
            Kind::Exact => crate::geometry_objects::import_exact_geometry_input_controlled(
                &reader,
                fixture.document.clone().unwrap(),
                &fixture.root,
                fixture.access.clone(),
                limits,
                Some(&codec),
            )
            .map(Box::new)
            .map(PreparedMeshingInput::ExactGeometry),
            Kind::Faceted => {
                crate::faceted_geometry_objects::import_faceted_geometry_input_controlled(
                    &reader,
                    fixture.document.clone().unwrap(),
                    &fixture.root,
                    fixture.access.clone(),
                    limits,
                    Some(&codec),
                )
                .map(Box::new)
                .map(PreparedMeshingInput::FacetedGeometry)
            }
            Kind::Evidence => crate::evidence_objects::import_evidence_input_controlled(
                &reader,
                &fixture.root,
                fixture.access.clone(),
                limits,
                Some(&codec),
            )
            .map(Box::new)
            .map(PreparedMeshingInput::Evidence),
        };
        if fail_allocation.is_some() || fail_checkpoint.is_some() {
            assert!(
                result.is_err(),
                "{kind:?}: injected denial did not reach input import"
            );
            let original = codec
                .take_failure()
                .or_else(|| reader.take_failure())
                .expect("original failure must survive import");
            assert_eq!(original.kind, error_kind);
            assert_eq!(original.reason, "injected original input control failure");
        } else {
            let value = result.as_ref().unwrap();
            reads.assert_retained(value.objects());
            assert_eq!(value.objects(), fixture.objects);
            assert!(control.inner.usage().live_allocation_bytes > 0);
        }
        drop(result);
    }
    assert_eq!(
        control.inner.usage().live_allocation_bytes,
        0,
        "{kind:?}: input scope leaked storage"
    );
    let count = control.allocations.lock().unwrap().len();
    (count, control.checkpoints.load(Ordering::Relaxed))
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: All exact/faceted/evidence import allocation boundaries reject injected denial before
//   storage allocation and retain original cancellation/deadline classes. Successful reads preserve
//   every original provider buffer; scope completion releases declared input storage. Synthetic
//   metadata is not original CAD, physics, whole-process RSS or native throughput evidence.
#[test]
fn all_cold_input_classes_admit_each_storage_boundary_and_preserve_original_failures() {
    for kind in [Kind::Exact, Kind::Faceted, Kind::Evidence] {
        let (allocations, checkpoints) = run(
            kind,
            None,
            None,
            GeometryEvaluationErrorKind::AllocationBudgetExceeded,
        );
        assert!(allocations > 5 && checkpoints > 5);
        for index in 0..allocations {
            run(
                kind,
                Some(index),
                None,
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
            );
        }
        for error in [
            GeometryEvaluationErrorKind::Cancelled,
            GeometryEvaluationErrorKind::TimeBudgetExceeded,
        ] {
            for index in [0, checkpoints / 2, checkpoints - 1] {
                run(kind, None, Some(index), error);
            }
        }
    }
}
