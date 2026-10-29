//! One bounded sleeping progress owner per native device. Runtime callers never
//! wait or join; process shutdown retires workers before driver finalization.
//! Timings describe prompt completion delivery, not GPU timestamps.
use runmat_accelerate_api::observation::{
    WgpuObservationProgress, WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::num::NonZeroU32;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    mpsc::{sync_channel, SyncSender},
    Arc,
};

#[path = "observation_progress/shutdown.rs"]
mod shutdown;

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

type CompletionCallback = Box<dyn FnOnce(Result<(), WgpuObservationProgressError>) + Send>;
pub(super) struct Completion {
    callback: Option<CompletionCallback>,
    // The consuming callback deallocates its Box before this external claim
    // retires. This field is inline in the existing preallocated queue Job.
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}
impl Completion {
    fn ordinary(callback: CompletionCallback) -> Self {
        Self::funded(callback, None)
    }

    pub(super) fn funded(
        callback: CompletionCallback,
        claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Self {
        Self {
            callback: Some(callback),
            claim,
        }
    }

    pub(super) fn finish(&mut self, status: Result<(), WgpuObservationProgressError>) {
        let completion = self.callback.take();
        let claim = self.claim.take();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            if let Some(completion) = completion {
                completion(status);
            }
            // This unboxed outer guard retires only after the callback Box's
            // consuming call (or its unwind). Claim-drop panics stay contained.
            drop(claim);
        }));
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        self.finish(Err(WgpuObservationProgressError::Stopped));
    }
}
struct Job<I = Option<wgpu::SubmissionIndex>> {
    submission: I,
    completion: Completion,
    _slot: Slot,
}

enum Command<I = Option<wgpu::SubmissionIndex>> {
    Observe(Job<I>),
    Shutdown,
}

struct Control {
    sender: SyncSender<Command>,
    state: Arc<State>,
}

impl Control {
    fn stop(&self) {
        self.state.stopped.store(true, Ordering::Release);
        // A full queue already wakes the receiver. The iterator checks stopped
        // before every receive, so a failed wake cannot strand an idle worker.
        let _ = self.sender.try_send(Command::Shutdown);
    }
}

fn commands<'a, I: 'a>(
    receiver: std::sync::mpsc::Receiver<Command<I>>,
    state: &'a State,
) -> impl Iterator<Item = Job<I>> + 'a {
    std::iter::from_fn(move || {
        if state.stopped.load(Ordering::Acquire) {
            return None;
        }
        match receiver.recv().ok()? {
            Command::Observe(job) => Some(job),
            Command::Shutdown => None,
        }
    })
}

fn drive<I>(receiver: impl IntoIterator<Item = Job<I>>, state: &State, mut poll: impl FnMut(&I)) {
    for mut job in receiver {
        if state.stopped.load(Ordering::Acquire) {
            continue;
        }
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| poll(&job.submission)));
        if result.is_ok() && !state.stopped.load(Ordering::Acquire) {
            job.completion.finish(Ok(()));
        } else {
            state.stopped.store(true, Ordering::Release);
        }
    }
    state.stopped.store(true, Ordering::Release);
}

pub(super) struct NativeObservationProgress {
    control: Arc<Control>,
    state: Arc<State>,
}

impl NativeObservationProgress {
    pub(super) fn is_available(&self) -> bool {
        !self.state.stopped.load(Ordering::Acquire)
    }

    pub(super) fn start(
        access: runmat_accelerate_api::WgpuDeviceAccess,
        limit: NonZeroU32,
        stopped: Arc<AtomicBool>,
    ) -> std::io::Result<Arc<Self>> {
        shutdown::prepare()?;
        let (sender, receiver) = sync_channel::<Command>(limit.get() as usize);
        let state = Arc::new(State {
            used: AtomicU32::new(0),
            limit: limit.get(),
            stopped,
        });
        let control = Arc::new(Control {
            sender,
            state: state.clone(),
        });
        let worker_state = state.clone();
        // The worker owns device access, not Provider or a sender. Closing all
        // senders drains queued work; only the process exit boundary joins it.
        shutdown::spawn(&control, move || {
            let jobs = commands(receiver, &worker_state);
            drive(jobs, &worker_state, |submission| {
                // Poll drives callbacks and may wait: never hold an emission
                // permit here. A native poll fatal is caught by `drive`,
                // separately from the device-lost callback's state.
                access.poll(match submission {
                    Some(index) => wgpu::Maintain::WaitForSubmissionIndex(index.clone()),
                    // Device control synchronizes existing queue work; it
                    // does not manufacture an ordinary submission while
                    // the session's emission admission is fenced. This
                    // wait runs only on this existing device worker.
                    None => wgpu::Maintain::Wait,
                });
            });
        })?;
        Ok(Arc::new(Self { control, state }))
    }

    pub(super) fn complete_queue_tail(
        &self,
        completion: CompletionCallback,
    ) -> Result<(), WgpuObservationProgressError> {
        self.reserve_slot()?.complete_queue_tail(completion)
    }

    pub(super) fn reserve_slot(&self) -> Result<Reservation, WgpuObservationProgressError> {
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
        Ok(Reservation {
            control: self.control.clone(),
            slot,
        })
    }
}

impl WgpuObservationProgress for NativeObservationProgress {
    fn reserve(
        &self,
    ) -> Result<Box<dyn WgpuObservationProgressReservation>, WgpuObservationProgressError> {
        Ok(Box::new(self.reserve_slot()?))
    }
}

pub(super) struct Reservation {
    control: Arc<Control>,
    slot: Slot,
}

impl Reservation {
    /// Commit already-reserved capacity after map registration, outside the
    /// emission gate. Rejection drops the job and completes its owner once.
    pub(super) fn complete_queue_tail(
        self,
        completion: CompletionCallback,
    ) -> Result<(), WgpuObservationProgressError> {
        self.complete_queue_tail_owner(Completion::ordinary(completion))
    }

    pub(super) fn complete_queue_tail_owner(
        self,
        completion: Completion,
    ) -> Result<(), WgpuObservationProgressError> {
        let Self { control, slot } = self;
        commit_job(
            &control.sender,
            Job {
                submission: None,
                completion,
                _slot: slot,
            },
        )
    }
}

pub(super) fn reservation_host_upper_bytes() -> Option<u64> {
    std::mem::size_of::<Reservation>()
        .checked_add(std::mem::align_of::<Reservation>())?
        .try_into()
        .ok()
}

fn commit_job<I>(
    sender: &SyncSender<Command<I>>,
    job: Job<I>,
) -> Result<(), WgpuObservationProgressError> {
    if job._slot.0.stopped.load(Ordering::Acquire) {
        return Err(WgpuObservationProgressError::Stopped);
    }
    // On rejection the owned job runs completion outside the emission gate.
    sender
        .try_send(Command::Observe(job))
        .map_err(|_| WgpuObservationProgressError::Stopped)
}

impl WgpuObservationProgressReservation for Reservation {
    fn submit(
        self: Box<Self>,
        submission: wgpu::SubmissionIndex,
        completion: CompletionCallback,
    ) -> Result<(), WgpuObservationProgressError> {
        let Self { control, slot } = *self;
        let job = Job {
            submission: Some(submission),
            completion: Completion::ordinary(completion),
            _slot: slot,
        };
        // Reservation accounts for queued AND active jobs. Disconnect drops Job,
        // running cleanup and releasing the slot, without a blocking send.
        commit_job(&control.sender, job)
    }
}

#[cfg(test)]
#[path = "observation_progress/tests.rs"]
mod tests;
