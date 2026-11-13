//! Poll-local selection of an already-owned command. This is never a session
//! counter or a substitute for the explicit receipt carried by an operation.

use super::CommandProfilingReceipt;
use super::CommandThreadCpuSampler;
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

#[derive(Default)]
struct Selection {
    receipt: Option<CommandProfilingReceipt>,
    baseline: Option<Duration>,
}

thread_local! {
    static ACTIVE: RefCell<Selection> = RefCell::new(Selection::default());
    static SAMPLING: Cell<bool> = const { Cell::new(false) };
    static REENTERED: Cell<bool> = const { Cell::new(false) };
}

pub fn current_command_profiling() -> Option<CommandProfilingReceipt> {
    ACTIVE.with(|active| active.borrow().receipt.clone())
}

enum Restore {
    Previous(Option<CommandProfilingReceipt>),
    Noop,
}

#[must_use]
pub struct CommandProfilingSelectionGuard {
    _restore: Restore,
    // The TLS slot must be restored on the same origin thread.
    _origin_thread: PhantomData<Rc<()>>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        if let Self::Previous(previous) = self {
            transition(previous.take());
        }
    }
}

fn same_owner(
    left: Option<&CommandProfilingReceipt>,
    right: Option<&CommandProfilingReceipt>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => Arc::ptr_eq(&left.0, &right.0),
        _ => false,
    }
}

fn sample(sampler: &dyn CommandThreadCpuSampler) -> Option<Duration> {
    struct SamplingGuard;
    impl Drop for SamplingGuard {
        fn drop(&mut self) {
            SAMPLING.with(|sampling| sampling.set(false));
        }
    }
    SAMPLING.with(|sampling| sampling.set(true));
    let _guard = SamplingGuard;
    // A host-supplied clock must not unwind through a selection guard's
    // destructor: restoration still needs to run on cancellation or a user
    // panic. Treat the failed read as unknown CPU evidence instead.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sampler.sample_current_thread()
    }))
    .ok()
    .flatten()
}

fn transition(next: Option<CommandProfilingReceipt>) {
    let (previous, baseline) = ACTIVE.with(|active| {
        let selected = active.borrow();
        (selected.receipt.clone(), selected.baseline)
    });
    if same_owner(previous.as_ref(), next.as_ref()) {
        return;
    }
    struct TransitionGuard {
        previous: Option<CommandProfilingReceipt>,
        next: Option<CommandProfilingReceipt>,
        complete: bool,
    }
    impl Drop for TransitionGuard {
        fn drop(&mut self) {
            if !self.complete {
                if let Some(previous) = &self.previous {
                    previous.invalidate_thread_cpu();
                }
                if let Some(next) = &self.next {
                    next.invalidate_thread_cpu();
                }
            }
        }
    }
    let mut transition_guard = TransitionGuard {
        previous: previous.clone(),
        next: next.clone(),
        complete: false,
    };
    let old_sampler = previous
        .as_ref()
        .and_then(CommandProfilingReceipt::thread_cpu_sampler);
    let new_sampler = next
        .as_ref()
        .and_then(CommandProfilingReceipt::thread_cpu_sampler);
    REENTERED.with(|reentered| reentered.set(false));
    let old_end = old_sampler
        .as_ref()
        .and_then(|sampler| sample(sampler.as_ref()));
    if let Some(previous) = &previous {
        match (baseline, old_end) {
            (Some(start), Some(end)) => previous.record_thread_cpu_segment(start, end),
            _ if old_sampler.is_some() => previous.invalidate_thread_cpu(),
            _ => {}
        }
    }
    let new_start = match (&old_sampler, old_end, &new_sampler) {
        (Some(old), Some(end), Some(new)) if Arc::ptr_eq(old, new) => Some(end),
        (_, _, Some(new)) => sample(new.as_ref()),
        _ => None,
    };
    if REENTERED.with(Cell::get) {
        if let Some(previous) = &previous {
            previous.invalidate_thread_cpu();
        }
        if let Some(next) = &next {
            next.invalidate_thread_cpu();
        }
    }
    if new_sampler.is_some() && new_start.is_none() {
        if let Some(next) = &next {
            next.invalidate_thread_cpu();
        }
    }
    ACTIVE.with(|active| {
        *active.borrow_mut() = Selection {
            receipt: next,
            baseline: new_start,
        };
    });
    transition_guard.complete = true;
}

/// Select only for a synchronous poll/callback extent. Explicit None masks a
/// parent command while a foreign or independent child operation runs.
pub fn with_command_profiling<T>(
    receipt: Option<CommandProfilingReceipt>,
    f: impl FnOnce() -> T,
) -> T {
    let _restore = select_command_profiling(receipt);
    f()
}

pub fn select_command_profiling(
    receipt: Option<CommandProfilingReceipt>,
) -> CommandProfilingSelectionGuard {
    if SAMPLING.with(Cell::get) {
        REENTERED.with(|reentered| reentered.set(true));
        if let Some(current) = current_command_profiling() {
            current.invalidate_thread_cpu();
        }
        if let Some(next) = &receipt {
            next.invalidate_thread_cpu();
        }
        return CommandProfilingSelectionGuard {
            _restore: Restore::Noop,
            _origin_thread: PhantomData,
        };
    }
    let previous = ACTIVE.with(|active| active.borrow().receipt.clone());
    if same_owner(previous.as_ref(), receipt.as_ref()) {
        return CommandProfilingSelectionGuard {
            _restore: Restore::Noop,
            _origin_thread: PhantomData,
        };
    }
    transition(receipt);
    CommandProfilingSelectionGuard {
        _restore: Restore::Previous(previous),
        _origin_thread: PhantomData,
    }
}

struct CommandFuture<F> {
    inner: Option<Pin<Box<F>>>,
    receipt: CommandProfilingReceipt,
}

impl<F: Future> Future for CommandFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        with_command_profiling(Some(this.receipt.clone()), || {
            let result = this
                .inner
                .as_mut()
                .expect("command future polled after completion")
                .as_mut()
                .poll(cx);
            if result.is_ready() {
                drop(this.inner.take());
            }
            result
        })
    }
}

impl<F> Drop for CommandFuture<F> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            with_command_profiling(Some(self.receipt.clone()), || drop(inner));
        }
    }
}

/// Wrap only an already-owned command future. No-owner routes should use
/// their original future, avoiding a box or activation on that hot path.
pub fn observe_command_future<F: Future>(
    receipt: CommandProfilingReceipt,
    future: F,
) -> impl Future<Output = F::Output> {
    CommandFuture {
        inner: Some(Box::pin(future)),
        receipt,
    }
}

#[cfg(test)]
#[path = "activation/tests.rs"]
mod tests;
