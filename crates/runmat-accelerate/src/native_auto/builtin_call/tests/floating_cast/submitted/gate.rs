//! Holds an actual progress reservation, not a synthetic receipt completion.
use runmat_accelerate_api::observation::{
    WgpuObservationProgress, WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use std::sync::{Arc, Mutex};

struct Pending {
    reservation: Box<dyn WgpuObservationProgressReservation>,
    submission: wgpu::SubmissionIndex,
    cleanup: runmat_accelerate_api::observation::WgpuObservationCompletion,
}

pub(super) struct Gate {
    inner: Arc<dyn WgpuObservationProgress>,
    pending: Arc<Mutex<Option<Pending>>>,
}

impl Gate {
    pub(super) fn new(inner: Arc<dyn WgpuObservationProgress>) -> Self {
        Self {
            inner,
            pending: Arc::new(Mutex::new(None)),
        }
    }
    pub(super) fn has_submission(&self) -> bool {
        self.pending.lock().unwrap().is_some()
    }
    pub(super) fn release(
        &self,
    ) -> (
        wgpu::SubmissionIndex,
        futures::channel::oneshot::Receiver<()>,
    ) {
        let pending = self
            .pending
            .lock()
            .unwrap()
            .take()
            .expect("real submitted cast ticket");
        let index = pending.submission.clone();
        let (retired, retirement) = futures::channel::oneshot::channel();
        let completion = runmat_accelerate_api::observation::WgpuObservationCompletion::new(
            Box::new(move |status| {
                pending.cleanup.finish(status);
                let _ = retired.send(());
            }),
            [None, None],
        );
        pending
            .reservation
            .submit(pending.submission, completion)
            .expect("native progress accepts exact ticket");
        (index, retirement)
    }
}

struct Reservation {
    inner: Box<dyn WgpuObservationProgressReservation>,
    pending: Arc<Mutex<Option<Pending>>>,
}
impl WgpuObservationProgress for Gate {
    fn reserve(
        &self,
    ) -> Result<Box<dyn WgpuObservationProgressReservation>, WgpuObservationProgressError> {
        Ok(Box::new(Reservation {
            inner: self.inner.reserve()?,
            pending: self.pending.clone(),
        }))
    }
}
impl WgpuObservationProgressReservation for Reservation {
    fn submit(
        self: Box<Self>,
        submission: wgpu::SubmissionIndex,
        cleanup: runmat_accelerate_api::observation::WgpuObservationCompletion,
    ) -> Result<(), WgpuObservationProgressError> {
        let mut pending = self.pending.lock().unwrap();
        assert!(
            pending.is_none(),
            "small cast must submit exactly one command buffer"
        );
        *pending = Some(Pending {
            reservation: self.inner,
            submission,
            cleanup,
        });
        Ok(())
    }
}
