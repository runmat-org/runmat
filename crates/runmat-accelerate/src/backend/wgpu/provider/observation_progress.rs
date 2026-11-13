//! One bounded sleeping progress owner per native device. Runtime callers never
//! wait or join; process shutdown retires workers before driver finalization.
//! Timings describe prompt completion delivery, not GPU timestamps.
use runmat_accelerate_api::observation::{
    WgpuObservationCompletion, WgpuObservationProgress, WgpuObservationProgressError,
    WgpuObservationProgressReservation,
};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::num::NonZeroU32;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    mpsc::{sync_channel, SyncSender},
    Arc, Mutex,
};

#[path = "observation_progress/scoped_retirement.rs"]
mod scoped_retirement;
#[path = "observation_progress/shutdown.rs"]
mod shutdown;
use scoped_retirement::ScopedWork;

struct State {
    used: AtomicU32,
    limit: u32,
    stopped: Arc<AtomicBool>,
    scoped_emission: Mutex<()>,
    // Fixed per-device backing, sized at construction to the same slot limit.
    // Physical ownership lives here; channel commands only wake the worker.
    scoped_work: Mutex<Vec<Option<ScopedWork>>>,
    worker_closed: AtomicBool,
    retained_uncertain: AtomicBool,
    shutdown_requested: AtomicBool,
    access: Option<runmat_accelerate_api::WgpuDeviceAccess>,
    health: Option<Arc<super::WgpuDeviceHealth>>,
}

struct Slot(Arc<State>);
impl Drop for Slot {
    fn drop(&mut self) {
        self.0.used.fetch_sub(1, Ordering::AcqRel);
    }
}

type CompletionCallback = Box<dyn FnOnce(Result<(), WgpuObservationProgressError>) + Send>;
pub(super) struct Completion {
    callback: Option<WgpuObservationCompletion>,
    scoped: Option<ScopedCompletion>,
}
struct ScopedCompletion {
    owner: Arc<State>,
    index: usize,
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
            callback: Some(WgpuObservationCompletion::new(callback, [claim, None])),
            scoped: None,
        }
    }

    fn scoped(owner: Arc<State>, index: usize) -> Self {
        Self {
            callback: None,
            scoped: Some(ScopedCompletion { owner, index }),
        }
    }

    pub(super) fn finish(&mut self, status: Result<(), WgpuObservationProgressError>) {
        let completion = self.callback.take();
        let scoped = self.scoped.take();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            if let Some(scoped) = scoped {
                scoped.owner.finish_scoped(scoped.index);
            }
            if let Some(completion) = completion {
                completion.finish(status);
            }
        }));
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        self.finish(Err(WgpuObservationProgressError::Stopped));
    }
}
enum WaitTarget {
    Existing(Option<wgpu::SubmissionIndex>),
    Scoped(crate::backend::wgpu::dispatch::observation::ScopedRetirement),
}

struct Job<I = WaitTarget> {
    submission: I,
    completion: Completion,
    _slot: Option<Slot>,
    retire_after_stop: bool,
}

enum Command<I = WaitTarget> {
    Observe(Job<I>),
    Shutdown,
    Wake,
}

struct Control {
    sender: SyncSender<Command>,
    state: Arc<State>,
}

impl Control {
    fn stop(&self) {
        let _closing = self
            .state
            .scoped_emission
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.state.stopped.store(true, Ordering::Release);
        self.state.shutdown_requested.store(true, Ordering::Release);
        // A full queue already wakes the receiver. The iterator checks stopped
        // before every receive, so a failed wake cannot strand an idle worker.
        let _ = self.sender.try_send(Command::Shutdown);
    }
}

impl Drop for Control {
    fn drop(&mut self) {
        // Normal sender exhaustion still drains accepted work. A catastrophically
        // failed worker must also be able to close without another driver call.
        self.state.shutdown_requested.store(true, Ordering::Release);
        let _ = self.sender.try_send(Command::Wake);
    }
}

fn commands<'a, I: 'a>(
    receiver: std::sync::mpsc::Receiver<Command<I>>,
    state: &'a State,
    mut scoped: impl FnMut() -> Option<Job<I>> + 'a,
) -> impl Iterator<Item = Job<I>> + 'a {
    std::iter::from_fn(move || {
        loop {
            if let Some(job) = scoped() {
                return Some(job);
            }
            let command = if state.stopped.load(Ordering::Acquire) {
                // The closing gate already prevented new scoped emission. Only
                // drain accepted jobs; never wait for unused reservations.
                let _closing = state
                    .scoped_emission
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(job) = scoped() {
                    return Some(job);
                }
                match receiver.try_recv() {
                    Ok(command) => command,
                    Err(_) => {
                        state.worker_closed.store(true, Ordering::Release);
                        return None;
                    }
                }
            } else {
                match receiver.recv() {
                    Ok(command) => command,
                    Err(_) => {
                        let _closing = state
                            .scoped_emission
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        if let Some(job) = scoped() {
                            return Some(job);
                        }
                        state.worker_closed.store(true, Ordering::Release);
                        return None;
                    }
                }
            };
            match command {
                Command::Observe(job) => return Some(job),
                Command::Shutdown | Command::Wake => {}
            }
        }
    })
}

fn drive<I>(
    receiver: impl IntoIterator<Item = Job<I>>,
    state: &State,
    mut poll: impl FnMut(&I) -> Result<(), WgpuObservationProgressError>,
) {
    for mut job in receiver {
        if state.stopped.load(Ordering::Acquire) && !job.retire_after_stop {
            continue;
        }
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| poll(&job.submission)));
        if matches!(result, Ok(Ok(())))
            && (!state.stopped.load(Ordering::Acquire) || job.retire_after_stop)
        {
            job.completion.finish(Ok(()));
        } else {
            let _closing = state
                .scoped_emission
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.stopped.store(true, Ordering::Release);
            if !matches!(result, Ok(Ok(()))) {
                state.retain_uncertain();
            }
        }
    }
    let _closing = state
        .scoped_emission
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    state.stopped.store(true, Ordering::Release);
}

pub(super) struct NativeObservationProgress {
    control: Arc<Control>,
    state: Arc<State>,
}

impl NativeObservationProgress {
    pub(super) fn is_available(&self) -> bool {
        !self.state.stopped.load(Ordering::Acquire)
            && !self.state.worker_closed.load(Ordering::Acquire)
    }

    pub(super) fn start(
        access: runmat_accelerate_api::WgpuDeviceAccess,
        limit: NonZeroU32,
        stopped: Arc<AtomicBool>,
        health: Arc<super::WgpuDeviceHealth>,
    ) -> std::io::Result<Arc<Self>> {
        shutdown::prepare()?;
        let (sender, receiver) = sync_channel::<Command>(limit.get() as usize);
        let mut scoped_work = Vec::new();
        scoped_work
            .try_reserve_exact(limit.get() as usize)
            .map_err(|error| {
                std::io::Error::other(format!("scoped retirement backing: {error}"))
            })?;
        scoped_work.resize_with(limit.get() as usize, || None);
        let state = Arc::new(State {
            used: AtomicU32::new(0),
            limit: limit.get(),
            stopped,
            scoped_emission: Mutex::new(()),
            scoped_work: Mutex::new(scoped_work),
            worker_closed: AtomicBool::new(false),
            retained_uncertain: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            access: Some(access.clone()),
            health: Some(health),
        });
        let control = Arc::new(Control {
            sender,
            state: state.clone(),
        });
        let worker_state = state.clone();
        // The worker owns device access, not Provider or a sender. Closing all
        // senders drains queued work; only the process exit boundary joins it.
        shutdown::spawn(&control, move || {
            let owner = worker_state.clone();
            let jobs = commands(receiver, &worker_state, move || owner.next_scoped());
            drive(jobs, &worker_state, |submission| {
                // Poll drives callbacks and may wait: never hold an emission
                // permit here. A native poll fatal is caught by `drive`,
                // separately from the device-lost callback's state.
                match submission {
                    WaitTarget::Existing(Some(index)) => {
                        access.poll(wgpu::Maintain::WaitForSubmissionIndex(index.clone()));
                    }
                    // Device control synchronizes existing queue work; it
                    // does not manufacture an ordinary submission while
                    // the session's emission admission is fenced. This
                    // wait runs only on this existing device worker.
                    WaitTarget::Existing(None) => {
                        access.poll(wgpu::Maintain::Wait);
                    }
                    WaitTarget::Scoped(signal) => {
                        scoped_retirement::poll_scoped(&access, &worker_state, signal)?;
                    }
                }
                Ok(())
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
                submission: WaitTarget::Existing(None),
                completion,
                _slot: Some(slot),
                retire_after_stop: false,
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
    if job
        ._slot
        .as_ref()
        .expect("ordinary reservation")
        .0
        .stopped
        .load(Ordering::Acquire)
    {
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
        completion: WgpuObservationCompletion,
    ) -> Result<(), WgpuObservationProgressError> {
        let Self { control, slot } = *self;
        let job = Job {
            submission: WaitTarget::Existing(Some(submission)),
            completion: Completion {
                callback: Some(completion),
                scoped: None,
            },
            _slot: Some(slot),
            retire_after_stop: false,
        };
        // Reservation accounts for queued AND active jobs. Disconnect drops Job,
        // running cleanup and releasing the slot, without a blocking send.
        commit_job(&control.sender, job)
    }
}

#[cfg(test)]
#[path = "observation_progress/tests.rs"]
mod tests;
