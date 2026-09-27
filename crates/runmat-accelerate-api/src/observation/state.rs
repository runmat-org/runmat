use super::*;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

struct State {
    start: Option<Duration>,
    publication: Option<Duration>,
    completion: Option<Duration>,
    registered: u64,
    pending: u64,
    failure: Option<ProviderObservationFailure>,
    taken: bool,
    stages: [ProviderStageMeasurement; 8],
    pipelines: ProviderPipelineObservations,
    waiter: Option<Waker>,
}

impl State {
    fn status(&self) -> ProviderObservationStatus {
        if let Some(failure) = self.failure {
            return ProviderObservationStatus::Failed(failure);
        }
        match (self.start, self.publication) {
            (Some(start), Some(publication)) if self.pending == 0 => {
                match publication
                    .max(self.completion.unwrap_or(publication))
                    .checked_sub(start)
                {
                    Some(total_elapsed) => ProviderObservationStatus::Completed {
                        total_elapsed,
                        submissions: self.registered,
                    },
                    None => {
                        ProviderObservationStatus::Failed(ProviderObservationFailure::InvalidTiming)
                    }
                }
            }
            _ => ProviderObservationStatus::Pending,
        }
    }

    fn fail(&mut self, reason: ProviderObservationFailure) {
        if !self.taken && self.failure.is_none() {
            self.failure = Some(reason);
        }
    }

    fn timestamp(&mut self, time: Option<Duration>) -> Option<Duration> {
        let time = time.filter(|time| self.start.is_some_and(|start| *time >= start));
        if time.is_none() {
            self.fail(ProviderObservationFailure::InvalidTiming);
        }
        time
    }
}

struct Shared {
    state: Mutex<State>,
    clock: Arc<dyn ProviderObservationClock>,
}

impl Shared {
    fn with<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = f(&mut state);
        let wake = if state.status() != ProviderObservationStatus::Pending {
            state.waiter.take()
        } else {
            None
        };
        drop(state);
        if let Some(wake) = wake {
            wake.wake();
        }
        result
    }
}

/// Unique session-side owner. Dropping an unconsumed receipt abandons it;
/// provider callbacks cannot resurrect it or deliver feedback themselves.
pub struct ProviderOperationReceipt(Arc<Shared>);

impl Default for ProviderOperationReceipt {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderOperationReceipt {
    pub fn new() -> Self {
        Self::with_clock(clock())
    }

    pub fn with_clock(clock: Arc<dyn ProviderObservationClock>) -> Self {
        let start = clock.now();
        Self(Arc::new(Shared {
            state: Mutex::new(State {
                start,
                publication: None,
                completion: None,
                registered: 0,
                pending: 0,
                failure: start
                    .is_none()
                    .then_some(ProviderObservationFailure::InvalidTiming),
                taken: false,
                stages: [ProviderStageMeasurement::default(); 8],
                pipelines: ProviderPipelineObservations::default(),
                waiter: None,
            }),
            clock,
        }))
    }

    pub fn recorder(&self) -> ProviderOperationRecorder {
        ProviderOperationRecorder(self.0.clone())
    }

    /// Seal submission registration after all selected-route work and successful
    /// output publication. Positively identified synchronous/empty work with no
    /// submissions completes here. Missing instrumentation is not such evidence;
    /// an opaque provider must use ProviderObservationSupport::Unsupported.
    pub fn publish(&mut self) -> Result<(), ProviderObservationFailure> {
        let now = self.0.clock.now();
        self.0.with(|state| {
            if let Some(reason) = state.failure {
                return Err(reason);
            }
            if state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::PublicationFailed);
                return Err(ProviderObservationFailure::PublicationFailed);
            }
            state.publication = state.timestamp(now);
            state.failure.map_or(Ok(()), Err)
        })
    }

    pub fn fail(&mut self, reason: ProviderObservationFailure) {
        self.0.with(|state| state.fail(reason));
    }

    /// Readiness only: does not consume evidence or call any session service.
    pub fn status(&self) -> ProviderObservationStatus {
        self.0.with(|state| state.status())
    }

    /// Deliver at most once on the session's owning thread. The total was fixed
    /// at publication/completion, never at this potentially delayed drain.
    pub fn take_completed(&mut self) -> Option<ProviderOperationObservation> {
        self.0.with(|state| {
            let status = state.status();
            if state.taken || status == ProviderObservationStatus::Pending {
                return None;
            }
            state.taken = true;
            Some(ProviderOperationObservation {
                status,
                stages: state.stages,
                pipelines: state.pipelines,
            })
        })
    }

    /// Race-safe single-owner wait. Callback threads only wake this caller;
    /// session feedback is still delivered by the caller on its owning thread.
    /// Ready(None) means evidence was already consumed. No custom Waker method
    /// (including dropping a replaced waker) runs while the state lock is held.
    pub fn poll_completed(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<ProviderOperationObservation>> {
        let mut next = Some(cx.waker().clone());
        let (result, old) = self.0.with(|state| {
            if state.taken {
                return (Poll::Ready(None), None);
            }
            let status = state.status();
            if status == ProviderObservationStatus::Pending {
                return (
                    Poll::Pending,
                    std::mem::replace(&mut state.waiter, next.take()),
                );
            }
            state.taken = true;
            (
                Poll::Ready(Some(ProviderOperationObservation {
                    status,
                    stages: state.stages,
                    pipelines: state.pipelines,
                })),
                None,
            )
        });
        drop(old);
        result
    }
}

impl Drop for ProviderOperationReceipt {
    fn drop(&mut self) {
        self.0
            .with(|state| state.fail(ProviderObservationFailure::Abandoned));
    }
}

/// Thread-safe callback authority; cannot publish outputs or consume feedback.
#[derive(Clone)]
pub struct ProviderOperationRecorder(Arc<Shared>);

impl ProviderOperationRecorder {
    /// Register before submission. Each token is completed exactly once, or its
    /// drop marks observation failure. Registration after publication is rejected.
    pub fn submission(&self) -> Result<ProviderSubmission, ProviderObservationFailure> {
        self.0.with(|state| {
            if let Some(reason) = state.failure {
                return Err(reason);
            }
            if state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::PublicationFailed);
                return Err(ProviderObservationFailure::PublicationFailed);
            }
            let Some(registered) = state.registered.checked_add(1) else {
                state.fail(ProviderObservationFailure::TrackingOverflow);
                return Err(ProviderObservationFailure::TrackingOverflow);
            };
            state.registered = registered;
            state.pending += 1; // pending <= registered, whose increment was checked
            Ok(ProviderSubmission {
                shared: self.0.clone(),
                consumed: false,
            })
        })
    }

    /// Explicit provider measurement only. Do not supply queue-submit latency as
    /// execution, or sum these overlapping stage aggregates into total elapsed.
    /// Record a submission's stages before completing its token. Evidence freezes
    /// once publication and all submissions complete (or failure occurs), not at
    /// session delivery; late measurements are ignored.
    pub fn record_stage(&self, stage: ProviderObservationStage, elapsed: Duration) {
        self.0.with(|state| {
            if state.taken || state.status() != ProviderObservationStatus::Pending {
                return;
            }
            let slot = &mut state.stages[stage as usize];
            match (
                slot.samples.checked_add(1),
                slot.elapsed.checked_add(elapsed),
            ) {
                (Some(samples), Some(elapsed)) => {
                    *slot = ProviderStageMeasurement { samples, elapsed }
                }
                _ => state.fail(ProviderObservationFailure::TrackingOverflow),
            }
        });
    }

    /// Record readiness at the actual pipeline owner/cache lookup. This does not
    /// establish pre-execution cost-query readiness or replace exact cache keys.
    /// Record measured preparation duration separately via record_stage. As with
    /// stage measurements, events after terminal completion are ignored.
    pub fn record_pipeline(&self, preparation: runmat_execution::CandidatePreparationState) {
        use runmat_execution::CandidatePreparationState;
        self.0.with(|state| {
            if state.taken || state.status() != ProviderObservationStatus::Pending {
                return;
            }
            let count = match preparation {
                CandidatePreparationState::Ready => &mut state.pipelines.ready,
                CandidatePreparationState::Warm => &mut state.pipelines.warm,
                CandidatePreparationState::Cold => &mut state.pipelines.cold,
                CandidatePreparationState::Preparing => &mut state.pipelines.preparing,
            };
            match count.checked_add(1) {
                Some(next) => *count = next,
                None => state.fail(ProviderObservationFailure::TrackingOverflow),
            }
        });
    }
}

pub struct ProviderSubmission {
    shared: Arc<Shared>,
    consumed: bool,
}

impl ProviderSubmission {
    /// Call only at actual completion of this submission, never at enqueue.
    pub fn complete(mut self) {
        let now = self.shared.clock.now();
        self.shared.with(|state| {
            if state.failure.is_some() || state.taken {
                return;
            }
            if let Some(now) = state.timestamp(now) {
                state.completion = Some(state.completion.map_or(now, |previous| previous.max(now)));
                state.pending -= 1;
            }
        });
        self.consumed = true;
    }

    pub fn fail(mut self) {
        self.shared
            .with(|state| state.fail(ProviderObservationFailure::ProviderFailed));
        self.consumed = true;
    }
}

impl Drop for ProviderSubmission {
    fn drop(&mut self) {
        if !self.consumed {
            self.shared
                .with(|state| state.fail(ProviderObservationFailure::Abandoned));
        }
    }
}
