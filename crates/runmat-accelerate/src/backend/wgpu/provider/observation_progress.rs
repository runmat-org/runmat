//! One bounded sleeping progress owner per native device. Runtime callers never
//! wait or join. Timings describe prompt completion delivery, not GPU timestamps.
use runmat_accelerate_api::observation::{
    WgpuObservationProgress, WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use std::num::NonZeroU32;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    mpsc::{sync_channel, SyncSender},
    Arc,
};

struct State {
    used: AtomicU32,
    limit: u32,
    stopped: Arc<AtomicBool>,
}
struct Slot(Arc<State>);
impl Drop for Slot {
    fn drop(&mut self) {
        self.0.used.fetch_sub(1, Ordering::AcqRel);
    }
}

struct FailureCleanup(Option<Box<dyn FnOnce() + Send>>);
impl Drop for FailureCleanup {
    fn drop(&mut self) {
        if let Some(cleanup) = self.0.take() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(cleanup));
        }
    }
}
struct Job<I = wgpu::SubmissionIndex> {
    submission: I,
    failure: FailureCleanup,
    _slot: Slot,
}

fn drive<I>(receiver: std::sync::mpsc::Receiver<Job<I>>, state: &State, mut poll: impl FnMut(&I)) {
    for mut job in receiver {
        if state.stopped.load(Ordering::Acquire) {
            continue;
        }
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| poll(&job.submission)));
        if result.is_ok() && !state.stopped.load(Ordering::Acquire) {
            job.failure.0 = None;
        } else {
            state.stopped.store(true, Ordering::Release);
        }
    }
    state.stopped.store(true, Ordering::Release);
}

pub(super) struct NativeObservationProgress {
    sender: SyncSender<Job>,
    state: Arc<State>,
}

impl NativeObservationProgress {
    pub(super) fn is_available(&self) -> bool {
        !self.state.stopped.load(Ordering::Acquire)
    }

    pub(super) fn start(
        device: Arc<wgpu::Device>,
        limit: NonZeroU32,
        stopped: Arc<AtomicBool>,
    ) -> std::io::Result<Arc<Self>> {
        let (sender, receiver) = sync_channel::<Job>(limit.get() as usize);
        let state = Arc::new(State {
            used: AtomicU32::new(0),
            limit: limit.get(),
            stopped,
        });
        let worker_state = state.clone();
        // Detached worker owns Device, not Provider or a sender. Closing all
        // senders drains queued work and exits; runtime Drop never joins it.
        let worker = std::thread::Builder::new()
            .name("runmat-gpu-observation".into())
            .spawn(move || {
                drive(receiver, &worker_state, |submission| {
                    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission.clone()));
                });
            })?;
        drop(worker);
        Ok(Arc::new(Self { sender, state }))
    }
}

impl WgpuObservationProgress for NativeObservationProgress {
    fn reserve(
        &self,
    ) -> Result<Box<dyn WgpuObservationProgressReservation>, WgpuObservationProgressError> {
        if self.state.stopped.load(Ordering::Acquire) {
            return Err(WgpuObservationProgressError::Stopped);
        }
        self.state
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                (used < self.state.limit).then(|| used + 1)
            })
            .map_err(|_| WgpuObservationProgressError::Capacity)?;
        let slot = Slot(self.state.clone());
        if self.state.stopped.load(Ordering::Acquire) {
            return Err(WgpuObservationProgressError::Stopped);
        }
        Ok(Box::new(Reservation {
            sender: self.sender.clone(),
            slot,
        }))
    }
}

struct Reservation {
    sender: SyncSender<Job>,
    slot: Slot,
}
impl WgpuObservationProgressReservation for Reservation {
    fn submit(
        self: Box<Self>,
        submission: wgpu::SubmissionIndex,
        failure_cleanup: Box<dyn FnOnce() + Send>,
    ) -> Result<(), WgpuObservationProgressError> {
        let Self { sender, slot } = *self;
        let job = Job {
            submission,
            failure: FailureCleanup(Some(failure_cleanup)),
            _slot: slot,
        };
        if job._slot.0.stopped.load(Ordering::Acquire) {
            return Err(WgpuObservationProgressError::Stopped);
        }
        // Reservation accounts for queued AND active jobs. Disconnect drops Job,
        // running cleanup and releasing the slot, without a blocking send.
        sender
            .try_send(job)
            .map_err(|_| WgpuObservationProgressError::Stopped)
    }
}

#[cfg(test)]
#[path = "observation_progress/tests.rs"]
mod tests;
