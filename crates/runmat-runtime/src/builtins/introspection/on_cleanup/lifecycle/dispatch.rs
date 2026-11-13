//! Host-supervised callback execution and exact typed completion receipts.

use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use runmat_gc::RootId;

use crate::context::{RuntimeContext, RuntimeLocalTaskOutcome};
use crate::{BuiltinResult, RuntimeError};

use super::registry::{
    CleanupAdmission, CleanupCompletionId, CleanupEntry, CleanupReceipt, CleanupRegistry,
};

enum CleanupCallbackOutcome {
    Skipped,
    Completed,
}

/// The outer owner destroys the scoped callback future, including its feval
/// frame and captured Values, before retiring roots or profiling admission.
/// TaskSlot may announce Cancelled while a poll is active; that announcement
/// is deliberately not this future's retirement boundary.
struct OwnedCleanupFuture {
    inner: Option<Pin<Box<dyn Future<Output = BuiltinResult<CleanupCallbackOutcome>>>>>,
    entry: Rc<CleanupEntry>,
    registry: Option<Rc<CleanupRegistry>>,
    runtime: Option<RuntimeContext>,
}

/// A captured resource can panic from Drop. Before destroying the inner
/// future, transfer terminal authority here so an unwind cannot strand its
/// callback roots or the admitted deferred lease. Only the inner destruction
/// boundary is armed; the ordinary completion path takes over afterward.
struct InnerDropRetirementGuard {
    entry: Rc<CleanupEntry>,
    registry: Rc<CleanupRegistry>,
    runtime: RuntimeContext,
    armed: bool,
}

impl Drop for InnerDropRetirementGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let _selected = self.runtime.enter();
        self.entry.abandon();
        self.registry.mark_terminal(&self.entry);
        self.entry.note_future_retired();
        self.entry.release_terminal_after_future_drop();
        self.entry.settle_deferred_from_receipt();
        schedule_eligible(&self.runtime, &self.registry, &[]);
        self.entry.finish_future_retirement();
    }
}

impl OwnedCleanupFuture {
    fn retire(&mut self, result: Option<BuiltinResult<CleanupCallbackOutcome>>) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        let registry = self
            .registry
            .take()
            .expect("cleanup registry retained by future");
        let runtime = self
            .runtime
            .take()
            .expect("cleanup runtime retained by future");
        let mut retirement = InnerDropRetirementGuard {
            entry: Rc::clone(&self.entry),
            registry,
            runtime,
            armed: true,
        };
        let _ = schedule_before_retirement(&retirement.runtime, &retirement.registry, &[]);
        drop(inner);
        retirement.armed = false;
        // The scoped inner has retired; terminal root release and its
        // synchronous GC observers still belong to the queue-time origin,
        // including an explicit None that masks a later command drainer.
        let _selected = retirement.runtime.enter();
        let callback_result = match result {
            Some(Ok(CleanupCallbackOutcome::Completed)) => Some(Ok(())),
            Some(Err(error)) => Some(Err(error)),
            Some(Ok(CleanupCallbackOutcome::Skipped)) | None => None,
        };
        if let Some(result) = callback_result {
            if self
                .entry
                .lease()
                .is_some_and(|lease| lease.outcome() == Some(RuntimeLocalTaskOutcome::Cancelled))
            {
                // TaskSlot may receive cancel during its current poll, then
                // allow that poll to return Ready. The cancellation intent
                // wins, but only this post-inner-Drop path retires roots.
                self.entry.abandon();
                retirement.registry.mark_terminal(&self.entry);
            } else {
                self.entry.complete(result);
                retirement.registry.mark_terminal(&self.entry);
            }
        }
        let settled = self.entry.note_future_retired();
        if settled && self.entry.receipt().is_none() {
            self.entry.abandon();
            retirement.registry.mark_terminal(&self.entry);
        }
        self.entry.release_terminal_after_future_drop();
        if matches!(
            self.entry.phase(),
            super::registry::CleanupPhase::Completed
                | super::registry::CleanupPhase::Cancelled
                | super::registry::CleanupPhase::Abandoned
                | super::registry::CleanupPhase::DispatchFailed
        ) {
            // Any terminal root retirement, including cancellation, can
            // expose a chained callback. The queue-origin runtime owns this
            // safepoint; a later drainer cannot reattribute it.
            schedule_eligible(&retirement.runtime, &retirement.registry, &[]);
        }
        if settled {
            self.entry.settle_deferred_from_receipt();
        }
        self.entry.finish_future_retirement();
    }
}

impl Future for OwnedCleanupFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let result = this
            .inner
            .as_mut()
            .expect("cleanup future polled after retirement")
            .as_mut()
            .poll(context);
        match result {
            Poll::Ready(result) => {
                this.retire(Some(result));
                Poll::Ready(())
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for OwnedCleanupFuture {
    fn drop(&mut self) {
        self.retire(None);
    }
}

fn settle_retired_after_ack(
    entry: &Rc<CleanupEntry>,
    registry: &Rc<CleanupRegistry>,
    runtime: &RuntimeContext,
) {
    let missing_receipt = entry.receipt().is_none();
    let queue_runtime = runtime
        .clone()
        .with_cleanup_command_origin(entry.queued_command_origin())
        .with_cleanup_completion(entry.completion());
    let _selected = queue_runtime.enter();
    let _ = schedule_before_retirement(&queue_runtime, registry, &[]);
    if missing_receipt {
        entry.abandon();
        registry.mark_terminal(entry);
    }
    entry.release_terminal_after_future_drop();
    entry.settle_deferred_from_receipt();
    if missing_receipt {
        schedule_eligible(&queue_runtime, registry, &[]);
    }
    entry.finish_future_retirement();
}

fn lifecycle_error(identifier: &'static str, message: &'static str) -> RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin("onCleanup")
        .with_identifier(identifier)
        .build()
}

/// Queue admission has already rooted the callback and target. The existing
/// session-local supervisor owns the future after submission, so dropping an
/// explicit clear/delete waiter cannot cancel or replay it.
pub(crate) fn start_queued(
    runtime: &RuntimeContext,
    registry: &Rc<CleanupRegistry>,
    entry: Rc<CleanupEntry>,
) -> BuiltinResult<()> {
    let tasks = runtime
        .service_ports()
        .require_local_tasks("onCleanup")
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error);
    let tasks = match tasks {
        Ok(tasks) => tasks,
        Err(error) => {
            entry.dispatch_failed(error);
            registry.mark_terminal(&entry);
            return receipt_result_selected(registry, &entry, false);
        }
    };
    let Some(callback) = entry.callback() else {
        let error = lifecycle_error(
            "RunMat:onCleanup:CallbackUnavailable",
            "queued cleanup callback is unavailable",
        );
        entry.dispatch_failed(error);
        registry.mark_terminal(&entry);
        return receipt_result_selected(registry, &entry, false);
    };
    let task_entry = Rc::clone(&entry);
    // Queue admission, not the eventual drain, fixed this callback's origin.
    // Explicit None masks a drainer that belongs to a later command.
    let task_runtime = runtime
        .clone()
        .with_cleanup_command_origin(entry.queued_command_origin())
        .with_cleanup_completion(entry.completion());
    let callback_runtime = task_runtime.clone();
    let inner = task_runtime.clone().scope(async move {
        if !task_entry.begin_running() {
            return Ok(CleanupCallbackOutcome::Skipped);
        }
        crate::call_feval_async_with_outputs_in_context(&callback_runtime, callback, &[], 0)
            .await
            .map(|_| CleanupCallbackOutcome::Completed)
    });
    entry.prepare_future_handoff();
    let future = OwnedCleanupFuture {
        inner: Some(Box::pin(inner)),
        entry: Rc::clone(&entry),
        registry: Some(Rc::clone(registry)),
        runtime: Some(task_runtime),
    };
    match tasks.spawn_local(runtime.local_identity(), Box::pin(future)) {
        Ok(lease) => {
            if entry.acknowledge_future_handoff(true) {
                settle_retired_after_ack(&entry, registry, runtime);
            }
            entry.attach_lease(lease);
            Ok(())
        }
        Err(error) => {
            let retired = entry.acknowledge_future_handoff(false);
            if entry.phase() == super::registry::CleanupPhase::Queued {
                entry.dispatch_failed(error);
                registry.mark_terminal(&entry);
                if retired {
                    entry.settle_deferred_from_receipt();
                }
                receipt_result_selected(registry, &entry, false)
            } else {
                // A custom port may have polled the callback through Ready
                // before returning Err. The callback's completed receipt
                // and linear retirement remain true; the submitter still
                // receives the original typed port failure, never success.
                if retired {
                    entry.settle_deferred_from_receipt();
                }
                entry.note_submission_failure(error);
                registry.mark_terminal(&entry);
                receipt_result_selected(registry, &entry, false)
            }
        }
    }
}

/// Admit as many independent source components as can be proven in fresh
/// phase-aware snapshots. Spawned callbacks are owned by the existing host
/// supervisor; no task waits on an Active cleanup object.
pub(crate) fn schedule_eligible(
    runtime: &RuntimeContext,
    registry: &Rc<CleanupRegistry>,
    temporary: &[RootId],
) {
    let _ = schedule_selected(runtime, registry, temporary, runtime.cleanup_completion());
}

pub(crate) fn schedule_before_retirement(
    runtime: &RuntimeContext,
    registry: &Rc<CleanupRegistry>,
    temporary: &[RootId],
) -> BuiltinResult<()> {
    schedule_selected(runtime, registry, temporary, None)
}

pub(crate) fn schedule_completion(
    runtime: &RuntimeContext,
    registry: &Rc<CleanupRegistry>,
    temporary: &[RootId],
    completion: CleanupCompletionId,
) -> BuiltinResult<()> {
    schedule_selected(runtime, registry, temporary, Some(completion))
}

fn schedule_selected(
    runtime: &RuntimeContext,
    registry: &Rc<CleanupRegistry>,
    temporary: &[RootId],
    completion: Option<CleanupCompletionId>,
) -> BuiltinResult<()> {
    // Ordinary numeric execution with no active lifecycle targets performs no
    // root census or callback allocation at a value-retirement safepoint.
    if !registry.has_active() {
        return Ok(());
    }
    match registry.admit_batch_for_completion(temporary, runtime.command_profiling(), completion) {
        CleanupAdmission::Queued(entries) => {
            for entry in entries {
                // Failed submissions retain their typed errors on the entry.
                // Only its completion owner (or host drain) reports them.
                let _ = start_queued(runtime, registry, entry);
            }
            Ok(())
        }
        CleanupAdmission::NoneEligible => Ok(()),
        CleanupAdmission::Indeterminate => Err(lifecycle_error(
            "RunMat:onCleanup:ReachabilityUnavailable",
            "cleanup reachability is incomplete; callbacks remain rooted for retry",
        )),
    }
}

/// A function joins only its own admitted callbacks and cascades. It never
/// joins a current/enclosing callback or an unrelated pending task.
pub(crate) async fn drain_completion(
    _runtime: &RuntimeContext,
    registry: &Rc<CleanupRegistry>,
    completion: CleanupCompletionId,
) -> BuiltinResult<()> {
    let mut result = Ok(());
    while let Some(entry) = registry.pending_completion(completion) {
        let settled = wait_queued(registry, &entry).await;
        // A cancelled lease can acknowledge before its currently polling
        // future is destroyed. Join the exact entry's physical retirement;
        // cascades are admitted under the same queue-time completion ID.
        entry.wait_future_retired().await;
        result = crate::runtime_error::complete_with_cleanup(result, settled);
    }
    while let Some(entry) = registry.completion_errors(completion) {
        result =
            crate::runtime_error::complete_with_cleanup(result, receipt_result(registry, &entry));
    }
    result
}

pub(crate) async fn wait_queued(
    registry: &CleanupRegistry,
    entry: &Rc<CleanupEntry>,
) -> BuiltinResult<()> {
    if let Some(lease) = entry.lease() {
        if lease.join().await == RuntimeLocalTaskOutcome::Cancelled && entry.receipt().is_none() {
            entry.abandon();
            registry.mark_terminal(entry);
            // A Cancelled lease is not proof that its future has been
            // destroyed: shutdown can race a currently polling callback.
            // OwnedCleanupFuture owns the actual release boundary.
        }
    }
    receipt_result(registry, entry)
}

fn receipt_result(registry: &CleanupRegistry, entry: &CleanupEntry) -> BuiltinResult<()> {
    receipt_result_selected(registry, entry, true)
}

fn receipt_result_selected(
    registry: &CleanupRegistry,
    entry: &CleanupEntry,
    report: bool,
) -> BuiltinResult<()> {
    let callback = match entry.receipt() {
        Some(CleanupReceipt::Completed(Ok(()))) | Some(CleanupReceipt::Cancelled) => Ok(()),
        Some(CleanupReceipt::Completed(Err(error)))
        | Some(CleanupReceipt::DispatchFailed(error)) => {
            Err(RuntimeError::from_shared_original(error))
        }
        Some(CleanupReceipt::Abandoned) => Err(lifecycle_error(
            "RunMat:onCleanup:Abandoned",
            "cleanup callback was abandoned during session shutdown",
        )),
        None => Err(lifecycle_error(
            "RunMat:onCleanup:MissingReceipt",
            "cleanup callback has no completion receipt",
        )),
    };
    let result = match entry.submission_failure() {
        Some(error) => crate::runtime_error::complete_with_cleanup(
            Err(RuntimeError::from_shared_original(error)),
            callback,
        ),
        None => callback,
    };
    match result {
        Err(error) if !report => Err(error),
        Err(error) if entry.report_once() => {
            registry.note_reported(entry);
            Err(error)
        }
        _ => Ok(()),
    }
}

/// Drain only proven-eligible objects; Active objects with outside owners
/// remain live. Independent source SCCs share a single graph snapshot;
/// queued callback roots count as owners before the next inspection.
pub(crate) async fn drain_eligible(
    runtime: &RuntimeContext,
    registry: &Rc<CleanupRegistry>,
    temporary: &[RootId],
) -> BuiltinResult<()> {
    let mut first_error = None;
    loop {
        let pending = registry.pending_entries();
        if !pending.is_empty() {
            for entry in pending {
                if let Err(error) = wait_queued(registry, &entry).await {
                    first_error.get_or_insert(error);
                }
            }
            continue;
        }
        match registry.admit_batch(temporary, runtime.command_profiling()) {
            CleanupAdmission::Queued(entries) => {
                for entry in entries {
                    if let Err(error) = start_queued(runtime, registry, Rc::clone(&entry)) {
                        first_error.get_or_insert(error);
                    }
                }
            }
            CleanupAdmission::NoneEligible => break,
            CleanupAdmission::Indeterminate => {
                return Err(first_error.unwrap_or_else(|| {
                    lifecycle_error(
                        "RunMat:onCleanup:ReachabilityUnavailable",
                        "cleanup reachability is incomplete; callbacks remain rooted for retry",
                    )
                }));
            }
        }
    }
    for entry in registry.terminal_errors() {
        if let Err(error) = receipt_result(registry, &entry) {
            first_error.get_or_insert(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "dispatch/tests.rs"]
mod tests;
