//! Preserve the original geometry failure through cooperative codec interfaces.
use runmat_geometry_core::{GeometryEvaluationControl, GeometryEvaluationError};
use std::sync::Mutex;
pub(crate) struct CodecControl<'a> {
    geometry: &'a dyn GeometryEvaluationControl,
    failure: Mutex<Option<GeometryEvaluationError>>,
}
impl<'a> CodecControl<'a> {
    pub(crate) fn new(geometry: &'a dyn GeometryEvaluationControl) -> Self {
        Self {
            geometry,
            failure: Mutex::new(None),
        }
    }
    fn interrupted(&self) -> bool {
        let mut failure = self.failure.lock().expect("codec failure lock");
        if failure.is_some() {
            return true;
        }
        if let Err(error) = self.geometry.checkpoint() {
            *failure = Some(error);
            return true;
        }
        false
    }
    pub(crate) fn take_failure(&self) -> Option<GeometryEvaluationError> {
        self.failure.lock().expect("codec failure lock").take()
    }
    fn retain_failure(&self, error: GeometryEvaluationError) -> GeometryEvaluationError {
        let mut first = self.failure.lock().expect("codec failure lock");
        first.get_or_insert(error).clone()
    }
}
impl GeometryEvaluationControl for CodecControl<'_> {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.interrupted() {
            Err(self
                .failure
                .lock()
                .expect("codec failure lock")
                .as_ref()
                .expect("interruption retains its original failure")
                .clone())
        } else {
            Ok(())
        }
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()?;
        self.geometry
            .consume_allocation_bytes(bytes)
            .map_err(|error| self.retain_failure(error))
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        self.geometry.release_allocation_bytes(bytes);
    }
    fn consume_search_work(&self, work: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()?;
        self.geometry
            .consume_search_work(work)
            .map_err(|error| self.retain_failure(error))
    }
    fn consume_iterations(&self, iterations: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()?;
        self.geometry
            .consume_iterations(iterations)
            .map_err(|error| self.retain_failure(error))
    }
}
impl runmat_canonical_codec::CanonicalCodecControl for CodecControl<'_> {
    fn is_cancelled(&self) -> bool {
        self.interrupted()
    }
}
impl runmat_meshing_core::MeshingCancellationSignal for CodecControl<'_> {
    fn is_cancelled(&self) -> bool {
        self.interrupted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::GeometryEvaluationErrorKind;
    struct Control(GeometryEvaluationErrorKind);
    impl GeometryEvaluationControl for Control {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Err(GeometryEvaluationError::new(
                self.0,
                "original geometry failure",
            ))
        }
        fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            self.checkpoint()
        }
        fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            self.checkpoint()
        }
        fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            self.checkpoint()
        }
    }
    #[test]
    fn allocation_failures_remain_latched_and_scope_release_still_reaches_the_owner() {
        let geometry = crate::geometry_control::tests::bounded_control(100);
        let codec = CodecControl::new(&geometry);
        {
            let scope = runmat_geometry_core::GeometryAllocationScope::new(&codec);
            scope.consume_allocation_bytes(80).unwrap();
            assert_eq!(
                scope.consume_allocation_bytes(21).unwrap_err().kind,
                GeometryEvaluationErrorKind::AllocationBudgetExceeded
            );
            assert_eq!(
                scope.consume_iterations(1).unwrap_err().kind,
                GeometryEvaluationErrorKind::AllocationBudgetExceeded
            );
        }
        assert_eq!(geometry.usage().live_allocation_bytes, 0);
        assert_eq!(
            codec.take_failure().unwrap().kind,
            GeometryEvaluationErrorKind::AllocationBudgetExceeded
        );
    }
    #[test]
    fn codec_probes_preserve_deadline_and_cancellation_failure_types() {
        for kind in [
            GeometryEvaluationErrorKind::TimeBudgetExceeded,
            GeometryEvaluationErrorKind::Cancelled,
        ] {
            let geometry = Control(kind);
            let codec = CodecControl::new(&geometry);
            let _error = runmat_canonical_codec::profile_canonical_encoding_with_control(
                &vec![1; 8],
                runmat_canonical_codec::CanonicalLimits::new(4096, 1024, 1024, 32),
                &codec,
            )
            .unwrap_err();
            let error = codec
                .take_failure()
                .expect("original geometry failure must survive codec");
            assert_eq!(error.kind, kind);
            assert_eq!(error.reason, "original geometry failure");
        }
    }
}
