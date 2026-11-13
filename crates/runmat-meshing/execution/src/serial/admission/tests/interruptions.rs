//! Provider I/O is bracketed by the original execution authority.
use super::*;
use runmat_geometry_core::{GeometryEvaluationControl, GeometryEvaluationErrorKind};
use std::sync::atomic::AtomicBool;
struct Control {
    inner: crate::MeshingGeometryEvaluationControl<'static>,
    stopped: AtomicBool,
    kind: GeometryEvaluationErrorKind,
}
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.stopped.load(Ordering::Relaxed) {
            return Err(GeometryEvaluationError::new(
                self.kind,
                "original read interruption",
            ));
        }
        self.inner.checkpoint()
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
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
struct InterruptedRead<'a> {
    control: &'a Control,
    calls: AtomicU64,
}
impl CacheImport for InterruptedRead<'_> {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("unbounded read");
    }
    fn read_verified_bounded_with_admission(
        &self,
        _: Digest,
        maximum: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert!(maximum >= 16);
        admit(16)?;
        let bytes = vec![7; 16];
        self.control.stopped.store(true, Ordering::Relaxed);
        Ok(Some(bytes))
    }
}
#[test]
fn original_interruptions_precede_provider_and_release_a_completed_read_buffer() {
    for kind in [
        GeometryEvaluationErrorKind::Cancelled,
        GeometryEvaluationErrorKind::TimeBudgetExceeded,
    ] {
        for before in [true, false] {
            let control = Control {
                inner: bounded_control(32),
                stopped: AtomicBool::new(before),
                kind,
            };
            let source = InterruptedRead {
                control: &control,
                calls: AtomicU64::new(0),
            };
            {
                let scope = GeometryAllocationScope::new(&control);
                let reader = InputReader::new(&source, &scope);
                assert!(reader
                    .read_verified_bounded(Digest::sha256([7; 16]), 1 << 30)
                    .is_err());
                let error = reader.take_failure().unwrap();
                assert_eq!(error.kind, kind);
                assert_eq!(error.reason, "original read interruption");
                assert_eq!(source.calls.load(Ordering::Relaxed), u64::from(!before));
                assert_eq!(control.inner.usage().live_allocation_bytes, 0);
            }
            assert_eq!(control.inner.usage().live_allocation_bytes, 0);
        }
    }
}
