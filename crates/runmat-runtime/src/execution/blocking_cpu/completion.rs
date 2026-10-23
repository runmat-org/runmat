//! Exact-layout completion for this existing owned CPU lane. Unlike an opaque
//! library oneshot, every request-local allocation has a canonical quote.
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use crate::execution::ExecutionServiceError;
use futures::task::AtomicWaker;
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit};
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

struct Slot<T, G> {
    value: Option<Result<T, ExecutionServiceError>>,
    closed: bool,
    receiver: bool,
    _retained: G,
}
struct State<T, G> {
    slot: Mutex<Slot<T, G>>,
    waker: AtomicWaker,
    // ExactShared deallocates its physical header BEFORE dropping State, so
    // this claim cannot refund the still-live shared allocation on final drop.
    _claim: Option<ExactHostAllocationClaim>,
}
pub(in crate::execution) struct Sender<T, G> {
    state: ExactShared<State<T, G>>,
    published: bool,
}
pub(in crate::execution) struct Receiver<T, G> {
    state: ExactShared<State<T, G>>,
}
type Endpoints<T, G> = (Sender<T, G>, Receiver<T, G>);

pub(in crate::execution) fn channel<T: Send + 'static, G: Send + 'static>(
    runtime: Option<&RuntimeContext>,
    retained: G,
) -> Result<Endpoints<T, G>, ExecutionServiceError> {
    let bytes = ExactShared::<State<T, G>>::requested_bytes();
    let bytes64 = u64::try_from(bytes)
        .map_err(|_| ExecutionServiceError::HostAllocation(HostAllocationError::Overflow))?;
    let retained_bytes = HostAllocationPlan::retained_exact_output_bytes(bytes64, 1)
        .map_err(ExecutionServiceError::HostAllocation)?;
    let receipt = HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, retained_bytes)
            .map_err(ExecutionServiceError::HostAllocation)?,
    )
    .map_err(ExecutionServiceError::HostAllocation)?;
    let claim = match &receipt {
        Some(receipt) => Some(
            receipt
                .output_permit(runtime.expect("managed completion has a context"))
                .map_err(ExecutionServiceError::HostAllocation)?
                .reserve_exact_claim(bytes)
                .map_err(|cause| ExecutionServiceError::HostAllocation(cause.into()))?,
        ),
        None => None,
    };
    let state = ExactShared::try_new(State {
        slot: Mutex::new(Slot {
            value: None,
            closed: false,
            receiver: true,
            _retained: retained,
        }),
        waker: AtomicWaker::new(),
        _claim: claim,
    })
    .map_err(|_| {
        ExecutionServiceError::HostAllocation(HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
        ))
    })?;
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(runtime.expect("managed completion has a context"))
            .map_err(ExecutionServiceError::HostAllocation)?;
    }
    Ok((
        Sender {
            state: state.clone(),
            published: false,
        },
        Receiver { state },
    ))
}

impl<T, G> Sender<T, G> {
    pub(in crate::execution) fn send(mut self, value: Result<T, ExecutionServiceError>) {
        let mut value = Some(value);
        {
            let mut slot = self
                .state
                .get()
                .slot
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if slot.receiver {
                slot.value = value.take();
            }
            slot.closed = true;
        }
        self.published = true;
        self.state.get().waker.wake();
        drop(value);
    }
}
impl<T, G> Drop for Sender<T, G> {
    fn drop(&mut self) {
        if !self.published {
            self.state
                .get()
                .slot
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .closed = true;
            self.state.get().waker.wake();
        }
    }
}
impl<T, G> Future for Receiver<T, G> {
    type Output = Result<T, ExecutionServiceError>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.state.get().waker.register(context.waker());
        let ready = {
            let mut slot = self
                .state
                .get()
                .slot
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(value) = slot.value.take() {
                Some(value)
            } else if slot.closed {
                Some(Err(ExecutionServiceError::WorkerLost(
                    "owned blocking CPU job ended before publication".into(),
                )))
            } else {
                None
            }
        };
        match ready {
            Some(value) => {
                drop(self.state.get().waker.take());
                Poll::Ready(value)
            }
            None => Poll::Pending,
        }
    }
}
impl<T, G> Drop for Receiver<T, G> {
    fn drop(&mut self) {
        let value = {
            let mut slot = self
                .state
                .get()
                .slot
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            slot.receiver = false;
            slot.value.take()
        };
        drop(self.state.get().waker.take());
        drop(value);
    }
}
