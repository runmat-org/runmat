//! Per-session cleanup identity, GC-root ownership, and phase-aware admission.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use runmat_execution::profiling::{CommandProfilingReceipt, DeferredWorkTerminal};
use runmat_gc::{GcHandle, RootId, ValueRootRegistration};
use runmat_value::{HandleRef, Value};

use crate::context::RuntimeLocalTaskLease;
use crate::runtime_error::RuntimeErrorOwner;

use super::super::ON_CLEANUP_CLASS;
use super::dependencies::{mark_admissible_batch, ActiveCleanupIdentity};

#[path = "registry/deferred.rs"]
mod deferred;
#[path = "registry/retirement.rs"]
mod retirement;

use deferred::{CleanupDeferredOrigin, CleanupFutureHandoff};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CleanupPhase {
    Active,
    Queued,
    Running,
    Completed,
    DispatchFailed,
    Cancelled,
    Abandoned,
}

/// Errors retain the original typed RuntimeError owner, including nested
/// source and Never-replay disposition. They are never reduced to a string.
#[derive(Clone, Debug)]
pub(crate) enum CleanupReceipt {
    Completed(Result<(), RuntimeErrorOwner>),
    DispatchFailed(RuntimeErrorOwner),
    Cancelled,
    Abandoned,
}

pub(crate) struct CleanupEntry {
    completion: Cell<Option<CleanupCompletionId>>,
    submission_failure: RefCell<Option<RuntimeErrorOwner>>,
    target: GcHandle,
    ordinal: u64,
    callback: RefCell<Option<Value>>,
    root: RefCell<Option<ValueRootRegistration>>,
    target_root: RefCell<Option<ValueRootRegistration>>,
    phase: Cell<CleanupPhase>,
    receipt: RefCell<Option<CleanupReceipt>>,
    lease: RefCell<Option<RuntimeLocalTaskLease>>,
    // Captured only on Active→Queued. Object creation and later drain do not
    // establish the callback's command attribution.
    queued_command_origin: RefCell<Option<CommandProfilingReceipt>>,
    deferred_origin: RefCell<CleanupDeferredOrigin>,
    future_handoff: Cell<CleanupFutureHandoff>,
    retirement_waker: RefCell<Option<std::task::Waker>>,
    retirement_settled: Cell<bool>,
    reported: Cell<bool>,
}

impl CleanupEntry {
    pub(crate) fn submission_failure(&self) -> Option<RuntimeErrorOwner> {
        self.submission_failure.borrow().clone()
    }

    pub(crate) fn note_submission_failure(&self, error: crate::RuntimeError) {
        *self.submission_failure.borrow_mut() = Some(RuntimeErrorOwner::new(error));
    }

    pub(crate) fn completion(&self) -> Option<CleanupCompletionId> {
        self.completion.get()
    }

    pub(crate) fn phase(&self) -> CleanupPhase {
        self.phase.get()
    }

    #[cfg(test)]
    pub(crate) fn test_has_callback_roots_and_deferred_admission(&self) -> bool {
        self.root.borrow().is_some()
            && self.target_root.borrow().is_some()
            && matches!(
                &*self.deferred_origin.borrow(),
                CleanupDeferredOrigin::Admitted(_)
            )
    }

    #[cfg(test)]
    pub(crate) fn test_callback_root_released(&self) -> bool {
        self.root.borrow().is_none()
    }

    pub(crate) fn callback(&self) -> Option<Value> {
        self.callback.borrow().clone()
    }

    pub(crate) fn receipt(&self) -> Option<CleanupReceipt> {
        self.receipt.borrow().clone()
    }

    pub(crate) fn report_once(&self) -> bool {
        !self.reported.replace(true)
    }

    pub(crate) fn lease(&self) -> Option<RuntimeLocalTaskLease> {
        self.lease.borrow().clone()
    }

    pub(crate) fn queued_command_origin(&self) -> Option<CommandProfilingReceipt> {
        self.queued_command_origin.borrow().clone()
    }

    pub(crate) fn has_task_lease(&self) -> bool {
        self.lease.borrow().is_some()
    }

    pub(crate) fn attach_lease(&self, lease: RuntimeLocalTaskLease) -> bool {
        if self.phase.get() != CleanupPhase::Queued {
            lease.cancel();
            return false;
        }
        *self.lease.borrow_mut() = Some(lease);
        true
    }

    pub(crate) fn begin_running(&self) -> bool {
        if self.phase.get() != CleanupPhase::Queued {
            return false;
        }
        self.phase.set(CleanupPhase::Running);
        true
    }

    pub(crate) fn complete(&self, result: Result<(), crate::RuntimeError>) {
        if self.phase.get() != CleanupPhase::Running {
            return;
        }
        self.phase.set(CleanupPhase::Completed);
        *self.receipt.borrow_mut() = Some(CleanupReceipt::Completed(
            result.map_err(RuntimeErrorOwner::new),
        ));
        let callback = self.callback.borrow_mut().take();
        let root = self.root.borrow_mut().take();
        let target_root = self.target_root.borrow_mut().take();
        let lease = self.lease.borrow_mut().take();
        let origin = self.queued_command_origin.borrow_mut().take();
        drop((callback, root, target_root, lease, origin));
    }

    pub(crate) fn dispatch_failed(&self, error: crate::RuntimeError) {
        if self.phase.get() != CleanupPhase::Queued {
            return;
        }
        self.phase.set(CleanupPhase::DispatchFailed);
        *self.receipt.borrow_mut() = Some(CleanupReceipt::DispatchFailed(RuntimeErrorOwner::new(
            error,
        )));
        if !self.future_may_own_resources() {
            let callback = self.callback.borrow_mut().take();
            let root = self.root.borrow_mut().take();
            let target_root = self.target_root.borrow_mut().take();
            let origin = self.queued_command_origin.borrow_mut().take();
            drop((callback, root, target_root, origin));
            self.settle_deferred(DeferredWorkTerminal::Failed);
        }
    }

    pub(crate) fn abandon(&self) {
        if matches!(
            self.phase.get(),
            CleanupPhase::Active | CleanupPhase::Queued | CleanupPhase::Running
        ) {
            self.phase.set(CleanupPhase::Abandoned);
            *self.receipt.borrow_mut() = Some(CleanupReceipt::Abandoned);
            self.queued_command_origin.borrow_mut().take();
        }
    }

    pub(crate) fn release_terminal_after_future_drop(&self) {
        if matches!(
            self.phase.get(),
            CleanupPhase::Abandoned | CleanupPhase::Cancelled | CleanupPhase::DispatchFailed
        ) {
            let callback = self.callback.borrow_mut().take();
            let root = self.root.borrow_mut().take();
            let target_root = self.target_root.borrow_mut().take();
            let lease = self.lease.borrow_mut().take();
            let origin = self.queued_command_origin.borrow_mut().take();
            drop((callback, root, target_root, lease, origin));
            self.settle_deferred_from_receipt();
        }
    }
}

pub(crate) enum CleanupAdmission {
    Queued(Vec<Rc<CleanupEntry>>),
    NoneEligible,
    Indeterminate,
}

/// Issued by one session registry, with no per-call heap ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CleanupCompletionId(u64);

/// The host owns this registry for exactly one RuntimeContextLocalId. A weak
/// reference is published into cloneable RuntimeContext state; neither it nor
/// a GC finalizer confers host ownership.
pub(crate) struct CleanupRegistry {
    index: RefCell<CleanupIndex>,
    next_ordinal: Cell<u64>,
    next_completion: Cell<u64>,
    last_pruned_collection: Cell<u64>,
    closed: Cell<bool>,
}

#[derive(Default)]
struct CleanupIndex {
    /// Exact allocation epoch is part of GcHandle's Hash/Eq identity.
    entries: HashMap<GcHandle, Rc<CleanupEntry>>,
    active: HashMap<GcHandle, Rc<CleanupEntry>>,
    pending: HashMap<GcHandle, Rc<CleanupEntry>>,
    terminal: HashMap<GcHandle, Rc<CleanupEntry>>,
    unreported_errors: HashMap<GcHandle, Rc<CleanupEntry>>,
    retirement_retry: HashMap<GcHandle, Rc<CleanupEntry>>,
}

impl CleanupRegistry {
    pub(crate) fn has_active(&self) -> bool {
        !self.index.borrow().active.is_empty()
    }

    pub(crate) fn has_completion_work(&self, completion: CleanupCompletionId) -> bool {
        let index = self.index.borrow();
        index
            .pending
            .values()
            .chain(index.unreported_errors.values())
            .chain(
                index
                    .terminal
                    .values()
                    .filter(|entry| entry.completion_retirement_pending()),
            )
            .any(|entry| {
                entry.completion() == Some(completion)
                    && (!entry.reported.get() || entry.completion_retirement_pending())
            })
    }

    pub(crate) fn new() -> Self {
        Self {
            index: RefCell::new(CleanupIndex::default()),
            next_ordinal: Cell::new(0),
            next_completion: Cell::new(0),
            last_pruned_collection: Cell::new(runmat_gc::gc_collection_epoch()),
            closed: Cell::new(false),
        }
    }

    pub(crate) fn register(
        &self,
        target: GcHandle,
        callback: Value,
    ) -> Result<Rc<CleanupEntry>, runmat_gc::GcError> {
        self.prune_terminal_after_collection();
        if self.closed.get() {
            return Err(runmat_gc::GcError::RootRegistrationFailed(
                "cleanup lifecycle host is closed".into(),
            ));
        }
        let ordinal = self.next_ordinal.get();
        self.next_ordinal.set(ordinal.checked_add(1).ok_or_else(|| {
            runmat_gc::GcError::RootRegistrationFailed("cleanup identity exhausted".into())
        })?);
        let root = ValueRootRegistration::new(Box::new(runmat_gc::GlobalRoot::new(
            vec![callback.clone()],
            "active onCleanup callback".into(),
        )))?;
        let target_root = ValueRootRegistration::new(Box::new(runmat_gc::GlobalRoot::new(
            vec![Value::HandleObject(HandleRef {
                class_name: ON_CLEANUP_CLASS.into(),
                target,
                valid: true,
            })],
            "session onCleanup target".into(),
        )))?;
        let entry = Rc::new(CleanupEntry {
            completion: Cell::new(None),
            submission_failure: RefCell::new(None),
            target,
            ordinal,
            callback: RefCell::new(Some(callback)),
            root: RefCell::new(Some(root)),
            target_root: RefCell::new(Some(target_root)),
            phase: Cell::new(CleanupPhase::Active),
            receipt: RefCell::new(None),
            lease: RefCell::new(None),
            queued_command_origin: RefCell::new(None),
            deferred_origin: RefCell::new(CleanupDeferredOrigin::Unowned),
            future_handoff: Cell::new(CleanupFutureHandoff::NotStarted),
            retirement_waker: RefCell::new(None),
            retirement_settled: Cell::new(false),
            reported: Cell::new(false),
        });
        let mut index = self.index.borrow_mut();
        index.entries.insert(target, Rc::clone(&entry));
        index.active.insert(target, Rc::clone(&entry));
        Ok(entry)
    }

    pub(crate) fn find(&self, target: GcHandle) -> Option<Rc<CleanupEntry>> {
        self.prune_terminal_after_collection();
        self.index.borrow().entries.get(&target).cloned()
    }

    pub(crate) fn pending_entries(&self) -> Vec<Rc<CleanupEntry>> {
        self.pending_entries_selected(None)
    }

    pub(crate) fn pending_completion(
        &self,
        completion: CleanupCompletionId,
    ) -> Option<Rc<CleanupEntry>> {
        let index = self.index.borrow();
        index
            .pending
            .values()
            .chain(
                index
                    .terminal
                    .values()
                    .filter(|entry| entry.completion_retirement_pending()),
            )
            .filter(|entry| entry.completion() == Some(completion))
            .min_by_key(|entry| entry.ordinal)
            .cloned()
    }

    fn pending_entries_selected(
        &self,
        completion: Option<CleanupCompletionId>,
    ) -> Vec<Rc<CleanupEntry>> {
        let mut pending = self
            .index
            .borrow()
            .pending
            .values()
            .filter(|entry| completion.is_none_or(|id| entry.completion() == Some(id)))
            .cloned()
            .collect::<Vec<_>>();
        pending.sort_unstable_by_key(|entry| entry.ordinal);
        pending
    }

    pub(crate) fn terminal_errors(&self) -> Vec<Rc<CleanupEntry>> {
        self.terminal_errors_selected(None)
    }

    pub(crate) fn completion_errors(
        &self,
        completion: CleanupCompletionId,
    ) -> Option<Rc<CleanupEntry>> {
        self.index
            .borrow()
            .unreported_errors
            .values()
            .filter(|entry| !entry.reported.get() && entry.completion() == Some(completion))
            .min_by_key(|entry| entry.ordinal)
            .cloned()
    }

    fn terminal_errors_selected(
        &self,
        completion: Option<CleanupCompletionId>,
    ) -> Vec<Rc<CleanupEntry>> {
        let mut index = self.index.borrow_mut();
        index
            .unreported_errors
            .retain(|_, entry| !entry.reported.get());
        let mut errors = index
            .unreported_errors
            .values()
            .filter(|entry| completion.is_none_or(|id| entry.completion() == Some(id)))
            .cloned()
            .collect::<Vec<_>>();
        errors.sort_unstable_by_key(|entry| entry.ordinal);
        errors
    }

    /// Snapshot active IDs, then atomically admit independent source SCCs
    /// while the GC root-publication guard still protects that proof.
    /// The caller must retain detached Values until this method returns.
    pub(crate) fn admit_batch(
        &self,
        temporary: &[RootId],
        origin: Option<&CommandProfilingReceipt>,
    ) -> CleanupAdmission {
        self.admit_batch_for_completion(temporary, origin, None)
    }

    pub(crate) fn issue_completion(&self) -> Result<CleanupCompletionId, crate::RuntimeError> {
        let id = self.next_completion.get();
        self.next_completion.set(id.checked_add(1).ok_or_else(|| {
            crate::runtime_error::semantic_error(
                "RunMat:onCleanup:CompletionExhausted",
                "cleanup completion identity is exhausted",
            )
        })?);
        Ok(CleanupCompletionId(id))
    }

    pub(crate) fn admit_batch_for_completion(
        &self,
        temporary: &[RootId],
        origin: Option<&CommandProfilingReceipt>,
        completion: Option<CleanupCompletionId>,
    ) -> CleanupAdmission {
        self.prune_terminal_after_collection();
        let mut active = self
            .index
            .borrow()
            .active
            .values()
            .cloned()
            .collect::<Vec<_>>();
        active.sort_unstable_by_key(|entry| entry.ordinal);
        if active.is_empty() {
            return CleanupAdmission::NoneEligible;
        }
        let targets = active.iter().map(|entry| entry.target).collect::<Vec<_>>();
        let roots = active
            .iter()
            .map(|entry| entry.root.borrow().as_ref().map(ValueRootRegistration::id))
            .collect::<Option<Vec<_>>>();
        let Some(roots) = roots else {
            return CleanupAdmission::Indeterminate;
        };
        let identities = active
            .iter()
            .map(|entry| ActiveCleanupIdentity {
                ordinal: entry.ordinal,
            })
            .collect::<Vec<_>>();
        let mut incomplete = false;
        let active_targets = active
            .iter()
            .map(|entry| {
                entry
                    .target_root
                    .borrow()
                    .as_ref()
                    .map(ValueRootRegistration::id)
            })
            .collect::<Option<Vec<_>>>();
        let Some(active_targets) = active_targets else {
            return CleanupAdmission::Indeterminate;
        };
        let excluded = temporary
            .iter()
            .copied()
            .chain(active_targets)
            .collect::<Vec<_>>();
        let mut selected = vec![false; active.len()];
        let any =
            runmat_gc::gc_with_registered_root_reachability(&targets, &roots, &excluded, |graph| {
                match graph {
                    Some(graph) => {
                        let any = mark_admissible_batch(&identities, &graph, &mut selected);
                        if any {
                            for (entry, queued) in active.iter().zip(&selected) {
                                if *queued {
                                    entry.completion.set(completion);
                                    entry.phase.set(CleanupPhase::Queued);
                                }
                            }
                        }
                        any
                    }
                    None => {
                        incomplete = true;
                        false
                    }
                }
            });
        if any {
            let queued = active
                .into_iter()
                .zip(selected)
                .filter_map(|(entry, selected)| selected.then_some(entry))
                .collect::<Vec<_>>();
            // The GC reachability decision held ROOT_ADMISSION.read. Admit
            // profiling only after it returns: this may take the command
            // receipt mutex, and no callback can start on this thread yet.
            for entry in &queued {
                entry.queue_command_origin(origin);
            }
            let mut index = self.index.borrow_mut();
            for entry in &queued {
                index.active.remove(&entry.target);
                index.pending.insert(entry.target, Rc::clone(entry));
            }
            CleanupAdmission::Queued(queued)
        } else if incomplete {
            CleanupAdmission::Indeterminate
        } else {
            CleanupAdmission::NoneEligible
        }
    }

    #[cfg(test)]
    pub(crate) fn queue_explicit(
        &self,
        entry: &CleanupEntry,
        origin: Option<&CommandProfilingReceipt>,
    ) -> bool {
        self.queue_explicit_for_completion(entry, origin, None)
    }

    pub(crate) fn queue_explicit_for_completion(
        &self,
        entry: &CleanupEntry,
        origin: Option<&CommandProfilingReceipt>,
        completion: Option<CleanupCompletionId>,
    ) -> bool {
        if entry.phase.get() == CleanupPhase::Active {
            entry.completion.set(completion);
            entry.queue_command_origin(origin);
            entry.phase.set(CleanupPhase::Queued);
            let mut index = self.index.borrow_mut();
            if let Some(entry) = index.active.remove(&entry.target) {
                index.pending.insert(entry.target, entry);
            }
            true
        } else {
            false
        }
    }

    pub(crate) fn cancel(&self, entry: &CleanupEntry) -> bool {
        if matches!(
            entry.phase.get(),
            CleanupPhase::Active | CleanupPhase::Queued
        ) {
            entry.phase.set(CleanupPhase::Cancelled);
            *entry.receipt.borrow_mut() = Some(CleanupReceipt::Cancelled);
            entry.queued_command_origin.borrow_mut().take();
            if let Some(lease) = entry.lease() {
                // The child may still be retained by its task slot. Its drop
                // guard releases captured roots at actual future retirement.
                lease.cancel();
            } else if !entry.future_may_own_resources() {
                entry.release_terminal_after_future_drop();
            }
            let owned = { self.index.borrow().entries.get(&entry.target).cloned() };
            if let Some(entry) = owned {
                self.mark_terminal(&entry);
            }
            true
        } else {
            false
        }
    }

    /// First mark terminal outcomes while callback futures may still exist;
    /// the host drops/cancels its supervisor before releasing their roots.
    pub(crate) fn force_abandon(&self) {
        self.closed.set(true);
        let entries = self
            .index
            .borrow()
            .entries
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for entry in entries {
            entry.abandon();
            self.mark_terminal(&entry);
        }
    }

    pub(crate) fn release_abandoned_after_task_shutdown(&self) {
        let entries = self
            .index
            .borrow()
            .terminal
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for entry in entries {
            // A currently polling child may remain alive until that poll
            // unwinds. Its future's drop guard releases roots at the actual
            // destruction boundary, not at the lease's Cancelled signal.
            if !entry.has_task_lease() && !entry.future_may_own_resources() {
                entry.release_terminal_after_future_drop();
            }
        }
    }
}

impl Default for CleanupRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "registry/tests.rs"]
mod tests;
