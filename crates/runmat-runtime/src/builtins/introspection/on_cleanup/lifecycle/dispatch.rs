//! Host-supervised callback execution and exact typed completion receipts.

use std::rc::Rc;

use runmat_gc::RootId;

use crate::context::{RuntimeContext, RuntimeLocalTaskOutcome};
use crate::{BuiltinResult, RuntimeError};

use super::registry::{CleanupAdmission, CleanupEntry, CleanupReceipt, CleanupRegistry};

struct CallbackFutureRootRelease(Rc<CleanupEntry>);

impl Drop for CallbackFutureRootRelease {
    fn drop(&mut self) {
        self.0.release_terminal_after_future_drop();
    }
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
            return receipt_result(registry, &entry);
        }
    };
    let Some(callback) = entry.callback() else {
        let error = lifecycle_error(
            "RunMat:onCleanup:CallbackUnavailable",
            "queued cleanup callback is unavailable",
        );
        entry.dispatch_failed(error);
        registry.mark_terminal(&entry);
        return receipt_result(registry, &entry);
    };
    let task_entry = Rc::clone(&entry);
    let task_registry = Rc::clone(registry);
    let task_runtime = runtime.clone();
    let release_on_drop = CallbackFutureRootRelease(Rc::clone(&entry));
    let future = runtime.clone().scope(async move {
        let _release_on_drop = release_on_drop;
        if !task_entry.begin_running() {
            return;
        }
        let result = crate::call_feval_async_with_outputs(callback, &[], 0)
            .await
            .map(|_| ());
        task_entry.complete(result);
        task_registry.mark_terminal(&task_entry);
        // After A releases its queued/running callback root, B may become
        // eligible. This is a safepoint, not a second executor or timer.
        schedule_eligible(&task_runtime, &task_registry, &[]);
    });
    match tasks.spawn_local(runtime.local_identity(), Box::pin(future)) {
        Ok(lease) => {
            entry.attach_lease(lease);
            Ok(())
        }
        Err(error) => {
            entry.dispatch_failed(error);
            registry.mark_terminal(&entry);
            receipt_result(registry, &entry)
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
    if let CleanupAdmission::Queued(entries) = registry.admit_batch(temporary) {
        for entry in entries {
            let _ = start_queued(runtime, registry, entry);
        }
    }
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
            // CallbackFutureRootRelease owns the actual release boundary.
        }
    }
    receipt_result(registry, entry)
}

fn receipt_result(registry: &CleanupRegistry, entry: &CleanupEntry) -> BuiltinResult<()> {
    match entry.receipt() {
        Some(CleanupReceipt::Completed(Ok(()))) | Some(CleanupReceipt::Cancelled) => Ok(()),
        Some(CleanupReceipt::Completed(Err(error)))
        | Some(CleanupReceipt::DispatchFailed(error)) => {
            if entry.report_once() {
                registry.note_reported(entry);
                Err(RuntimeError::from_shared_original(error))
            } else {
                Ok(())
            }
        }
        Some(CleanupReceipt::Abandoned) => {
            if entry.report_once() {
                registry.note_reported(entry);
                Err(lifecycle_error(
                    "RunMat:onCleanup:Abandoned",
                    "cleanup callback was abandoned during session shutdown",
                ))
            } else {
                Ok(())
            }
        }
        None => Err(lifecycle_error(
            "RunMat:onCleanup:MissingReceipt",
            "cleanup callback has no completion receipt",
        )),
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
        match registry.admit_batch(temporary) {
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
