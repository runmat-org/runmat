//! Ordered host RNG turn ownership.
//!
//! The turn is non-Clone and remains held across cooperative yields. The
//! authoritative stream itself is borrowed only for short synchronous chunks.

use std::future::Future;
use std::mem::{align_of, size_of};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll, Waker};

use futures::future::{select, Either};

use super::{random_error, with_rng_state_in, RuntimeRng};
use crate::context::host_allocation::HostRetainedSessionAllocation;
use crate::context::RuntimeContext;
use crate::BuiltinResult;

#[derive(Default)]
struct Queue {
    held: bool,
    head: Option<Arc<Waiter>>,
    tail: Option<Arc<Waiter>>,
}

#[derive(Default)]
pub(crate) struct RandomTurnGate {
    queue: Mutex<Queue>,
}

impl std::fmt::Debug for RandomTurnGate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RandomTurnGate")
            .finish_non_exhaustive()
    }
}

struct Waiter {
    notified: AtomicBool,
    waker: Mutex<Option<Waker>>,
    next: Mutex<Option<Arc<Waiter>>>,
    _allocation: Option<HostRetainedSessionAllocation>,
}

impl Waiter {
    fn requested_bytes() -> BuiltinResult<u64> {
        let bytes = size_of::<Self>()
            .checked_add(2 * size_of::<usize>())
            .and_then(|n| n.checked_add(align_of::<Self>()))
            .ok_or_else(|| random_error("rng", "rng: turn queue metadata size overflow"))?;
        u64::try_from(bytes)
            .map_err(|_| random_error("rng", "rng: turn queue metadata size overflow"))
    }

    fn new(owner: Option<&RuntimeContext>) -> BuiltinResult<Arc<Self>> {
        let allocation = HostRetainedSessionAllocation::admit(owner, Self::requested_bytes()?)
            .map_err(|source| {
                crate::build_runtime_error("rng: ordered turn admission failed")
                    .with_builtin("rng")
                    .with_source(source)
                    .build()
            })?;
        Ok(Arc::new(Self {
            notified: AtomicBool::new(false),
            waker: Mutex::new(None),
            next: Mutex::new(None),
            _allocation: allocation,
        }))
    }

    fn notify(&self) {
        self.notified.store(true, Ordering::Release);
        if let Some(waker) = self.waker.lock().expect("RNG waiter lock poisoned").take() {
            waker.wake();
        }
    }
}

struct WaitForTurn(Arc<Waiter>);

/// Removing a suspended acquisition must also pass on a grant that was
/// already dequeued but whose future was never polled again.
struct PendingRegistration {
    gate: Arc<RandomTurnGate>,
    waiter: Arc<Waiter>,
    completed: bool,
}

impl Drop for PendingRegistration {
    fn drop(&mut self) {
        if !self.completed && !self.gate.remove_waiter(&self.waiter) {
            self.gate.release();
        }
    }
}

impl Future for WaitForTurn {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.0.notified.load(Ordering::Acquire) {
            return Poll::Ready(());
        }
        *self.0.waker.lock().expect("RNG waiter lock poisoned") = Some(cx.waker().clone());
        if self.0.notified.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl RandomTurnGate {
    fn release(&self) {
        let next = {
            let mut queue = self.queue.lock().expect("RNG turn queue poisoned");
            let next = queue.head.take();
            if let Some(waiter) = next.as_ref() {
                queue.head = waiter.next.lock().expect("RNG waiter link poisoned").take();
                if queue.head.is_none() {
                    queue.tail = None;
                }
            } else {
                queue.held = false;
            }
            next
        };
        if let Some(next) = next {
            next.notify();
        }
    }

    fn remove_waiter(&self, target: &Arc<Waiter>) -> bool {
        let mut queue = self.queue.lock().expect("RNG turn queue poisoned");
        let mut previous: Option<Arc<Waiter>> = None;
        let mut current = queue.head.clone();
        while let Some(waiter) = current {
            let next = waiter
                .next
                .lock()
                .expect("RNG waiter link poisoned")
                .clone();
            if Arc::ptr_eq(&waiter, target) {
                if let Some(previous) = previous.as_ref() {
                    *previous.next.lock().expect("RNG waiter link poisoned") = next.clone();
                } else {
                    queue.head = next.clone();
                }
                if queue
                    .tail
                    .as_ref()
                    .is_some_and(|tail| Arc::ptr_eq(tail, target))
                {
                    queue.tail = previous;
                }
                *waiter.next.lock().expect("RNG waiter link poisoned") = None;
                return true;
            }
            previous = Some(waiter);
            current = next;
        }
        false
    }
}

fn fallback_gate() -> Arc<RandomTurnGate> {
    static GATE: OnceLock<Arc<RandomTurnGate>> = OnceLock::new();
    Arc::clone(GATE.get_or_init(|| Arc::new(RandomTurnGate::default())))
}

/// Passed by mutable reference to nested samplers; never reacquired there.
pub(crate) struct RandomTurn {
    gate: Arc<RandomTurnGate>,
    owner: Option<RuntimeContext>,
    generation: Option<(Arc<AtomicU64>, u64)>,
}

impl RandomTurn {
    fn new(
        gate: Arc<RandomTurnGate>,
        owner: Option<RuntimeContext>,
        generation: Option<(Arc<AtomicU64>, u64)>,
    ) -> Self {
        Self {
            gate,
            owner,
            generation,
        }
    }

    pub(crate) fn with_state<T>(
        &mut self,
        label: &str,
        operation: impl FnOnce(&mut RuntimeRng) -> T,
    ) -> BuiltinResult<T> {
        self.check_authority(label)?;
        with_rng_state_in(self.owner.as_ref(), label, operation)
    }

    fn check_authority(&self, label: &str) -> BuiltinResult<()> {
        if let Some(owner) = self.owner.as_ref() {
            if owner.is_cancelled() {
                return Err(cancelled_error(label, owner));
            }
            if owner
                .parallel_random_scope()
                .is_some_and(|scope| !scope.matches_current_owner(owner))
            {
                return Err(random_error(
                    label,
                    format!("{label}: parallel random task owner changed"),
                ));
            }
        }
        if self
            .generation
            .as_ref()
            .is_some_and(|(counter, expected)| counter.load(Ordering::Acquire) != *expected)
        {
            return Err(random_error(
                label,
                format!("{label}: random turn authority changed"),
            ));
        }
        Ok(())
    }

    pub(crate) fn owner(&self) -> Option<&RuntimeContext> {
        self.owner.as_ref()
    }
}

fn cancelled_error(label: &str, owner: &RuntimeContext) -> crate::RuntimeError {
    crate::build_runtime_error(format!("{label}: random turn was cancelled"))
        .with_builtin(label)
        .with_identifier(format!("{}:Cancelled", owner.error_namespace()))
        .with_source(crate::execution::ExecutionServiceError::Cancelled)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

impl Drop for RandomTurn {
    fn drop(&mut self) {
        self.gate.release();
    }
}

fn owner_and_gate(
    context: Option<&RuntimeContext>,
) -> (Option<RuntimeContext>, Arc<RandomTurnGate>) {
    let owner = context.cloned().or_else(crate::context::legacy::active);
    let gate = gate_for_owner(owner.as_ref());
    (owner, gate)
}

fn gate_for_owner(owner: Option<&RuntimeContext>) -> Arc<RandomTurnGate> {
    owner.map_or_else(fallback_gate, |runtime| {
        runtime.parallel_random_scope().map_or_else(
            || Arc::clone(&runtime.state().random_turn),
            |scope| Arc::clone(&scope.gate),
        )
    })
}

pub(crate) async fn acquire(
    context: Option<&RuntimeContext>,
    label: &str,
) -> BuiltinResult<RandomTurn> {
    if let Some(runtime) = context.cloned().or_else(crate::context::legacy::active) {
        runtime
            .service_ports()
            .require_selected_local_effect(runmat_types::EffectKind::Randomness)?;
    }
    let (owner, gate) = owner_and_gate(context);
    acquire_selected(owner, gate, label).await
}

/// Select only the explicit original owner, including context-free None.
/// The legacy entry keeps its existing ambient-selection behavior.
pub(crate) async fn acquire_original(
    context: Option<&RuntimeContext>,
    label: &str,
) -> BuiltinResult<RandomTurn> {
    if let Some(runtime) = context {
        runtime
            .service_ports()
            .require_selected_local_effect(runmat_types::EffectKind::Randomness)?;
    }
    let owner = context.cloned();
    let gate = gate_for_owner(owner.as_ref());
    acquire_selected(owner, gate, label).await
}

async fn acquire_selected(
    owner: Option<RuntimeContext>,
    gate: Arc<RandomTurnGate>,
    label: &str,
) -> BuiltinResult<RandomTurn> {
    let generation = owner
        .as_ref()
        .map(RuntimeContext::host_effect_session_generation);
    if let Some(runtime) = owner.as_ref() {
        if runtime.is_cancelled() {
            return Err(cancelled_error(label, runtime));
        }
    }
    {
        let mut queue = gate.queue.lock().expect("RNG turn queue poisoned");
        if !queue.held {
            queue.held = true;
            return Ok(RandomTurn::new(Arc::clone(&gate), owner, generation));
        }
    }
    let waiter = Waiter::new(owner.as_ref())?;
    {
        let mut queue = gate.queue.lock().expect("RNG turn queue poisoned");
        if !queue.held {
            queue.held = true;
            return Ok(RandomTurn::new(Arc::clone(&gate), owner, generation));
        }
        if let Some(tail) = queue.tail.as_ref() {
            *tail.next.lock().expect("RNG waiter link poisoned") = Some(Arc::clone(&waiter));
        } else {
            queue.head = Some(Arc::clone(&waiter));
        }
        queue.tail = Some(Arc::clone(&waiter));
    }
    let mut registration = PendingRegistration {
        gate: Arc::clone(&gate),
        waiter: Arc::clone(&waiter),
        completed: false,
    };
    if let Some(runtime) = owner.as_ref() {
        let notified = WaitForTurn(waiter);
        let cancelled = runtime.cancellation().cancelled();
        futures::pin_mut!(notified, cancelled);
        match select(notified, cancelled).await {
            Either::Left(((), _)) if !runtime.is_cancelled() => {}
            _ => {
                return Err(cancelled_error(label, runtime));
            }
        }
    } else {
        WaitForTurn(waiter).await;
    }
    registration.completed = true;
    let turn = RandomTurn::new(gate, owner, generation);
    turn.check_authority(label)?;
    Ok(turn)
}

/// Direct synchronous embedding can use an idle stream, never block its own
/// single-threaded executor while another invocation is suspended.
pub(crate) fn try_acquire(
    context: Option<&RuntimeContext>,
    label: &str,
) -> BuiltinResult<RandomTurn> {
    if let Some(runtime) = context.cloned().or_else(crate::context::legacy::active) {
        runtime
            .service_ports()
            .require_selected_local_effect(runmat_types::EffectKind::Randomness)?;
    }
    let (owner, gate) = owner_and_gate(context);
    let generation = owner
        .as_ref()
        .map(RuntimeContext::host_effect_session_generation);
    if let Some(runtime) = owner.as_ref() {
        if runtime.is_cancelled() {
            return Err(cancelled_error(label, runtime));
        }
    }
    let mut queue = gate.queue.lock().expect("RNG turn queue poisoned");
    if queue.held {
        return Err(random_error(
            label,
            format!("{label}: random stream is already in use"),
        ));
    }
    queue.held = true;
    drop(queue);
    Ok(RandomTurn::new(gate, owner, generation))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_original_turn_selection_does_not_borrow_foreign_ambient_stream() {
        let ambient = crate::user_functions::test_support::exact_fixture_runtime();
        let original = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = ambient.enter();
        futures::executor::block_on(async {
            let strict = acquire_original(None, "rng").await.unwrap();
            assert!(strict.owner().is_none());
            drop(strict);
            let strict = acquire_original(Some(&original), "rng").await.unwrap();
            assert_eq!(
                strict.owner().unwrap().local_identity(),
                original.local_identity()
            );
            drop(strict);
            let legacy = acquire(None, "rng").await.unwrap();
            assert_eq!(
                legacy.owner().unwrap().local_identity(),
                ambient.local_identity()
            );
        });
    }

    fn waiting_gate() -> (Arc<RandomTurnGate>, Arc<Waiter>) {
        let gate = Arc::new(RandomTurnGate::default());
        let waiter = Waiter::new(None).expect("unbounded waiter");
        {
            let mut queue = gate.queue.lock().expect("queue");
            queue.held = true;
            queue.head = Some(Arc::clone(&waiter));
            queue.tail = Some(Arc::clone(&waiter));
        }
        (gate, waiter)
    }

    #[test]
    fn abandoned_queued_turn_removes_node_without_consuming_grant() {
        let (gate, waiter) = waiting_gate();
        drop(PendingRegistration {
            gate: Arc::clone(&gate),
            waiter,
            completed: false,
        });
        gate.release();
        let queue = gate.queue.lock().expect("queue");
        assert!(!queue.held);
        assert!(queue.head.is_none());
        assert!(queue.tail.is_none());
    }

    #[test]
    fn abandoned_granted_turn_passes_grant_to_next_waiter() {
        let (gate, first) = waiting_gate();
        let second = Waiter::new(None).expect("unbounded waiter");
        *first.next.lock().expect("link") = Some(Arc::clone(&second));
        gate.queue.lock().expect("queue").tail = Some(Arc::clone(&second));
        let abandoned = PendingRegistration {
            gate: Arc::clone(&gate),
            waiter: first,
            completed: false,
        };
        gate.release();
        drop(abandoned);
        assert!(second.notified.load(Ordering::Acquire));
        gate.release();
        assert!(!gate.queue.lock().expect("queue").held);
    }
}
