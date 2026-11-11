use std::collections::HashMap;
use std::rc::Rc;

use super::{CleanupEntry, CleanupPhase, CleanupReceipt, CleanupRegistry};

impl CleanupRegistry {
    /// Terminal records are needed while an exact GC target is still live:
    /// later delete/cancel through an alias must resolve to this session and
    /// remain a no-op. A completed collection is only a prompt to inspect;
    /// actual InvalidPointer is the retirement proof. Outstanding waiters
    /// retain their own Rc receipt after this index releases its owner.
    pub(crate) fn prune_terminal_after_collection(&self) {
        let epoch = runmat_gc::gc_collection_epoch();
        let new_collection = epoch != self.last_pruned_collection.get();
        let mut candidates = HashMap::new();
        {
            let index = self.index.borrow();
            for entry in index.retirement_retry.values() {
                if !entry.completion_retirement_pending() {
                    candidates.insert(entry.target, Rc::clone(entry));
                }
            }
            if new_collection {
                for entry in index.terminal.values() {
                    if entry.completion_retirement_pending() {
                        continue;
                    }
                    let unreported_failure = matches!(
                        entry.receipt(),
                        Some(CleanupReceipt::Completed(Err(_)))
                            | Some(CleanupReceipt::DispatchFailed(_))
                            | Some(CleanupReceipt::Abandoned)
                    ) || entry.submission_failure().is_some();
                    if !unreported_failure || entry.reported.get() {
                        candidates.insert(entry.target, Rc::clone(entry));
                    }
                }
            }
        }
        if candidates.is_empty() {
            self.last_pruned_collection.set(epoch);
            return;
        }
        let mut retired = Vec::new();
        let mut retry = Vec::new();
        for entry in candidates.into_values() {
            match runmat_gc::gc_with_value(&entry.target, |_| ()) {
                Err(runmat_gc::GcError::InvalidPointer(_)) => retired.push(entry),
                Ok(()) => {}
                Err(_) => retry.push(entry),
            }
        }
        let mut index = self.index.borrow_mut();
        index.retirement_retry.clear();
        for entry in retry {
            if index
                .terminal
                .get(&entry.target)
                .is_some_and(|current| Rc::ptr_eq(current, &entry))
            {
                index.retirement_retry.insert(entry.target, entry);
            }
        }
        for entry in retired {
            let exact = index
                .terminal
                .get(&entry.target)
                .is_some_and(|current| Rc::ptr_eq(current, &entry));
            if exact {
                index.terminal.remove(&entry.target);
                index.entries.remove(&entry.target);
                index.unreported_errors.remove(&entry.target);
                index.retirement_retry.remove(&entry.target);
            }
        }
        drop(index);
        self.last_pruned_collection.set(epoch);
    }

    pub(crate) fn mark_terminal(&self, entry: &Rc<CleanupEntry>) {
        if !matches!(
            entry.phase(),
            CleanupPhase::Completed
                | CleanupPhase::DispatchFailed
                | CleanupPhase::Cancelled
                | CleanupPhase::Abandoned
        ) {
            return;
        }
        let mut index = self.index.borrow_mut();
        if !index
            .entries
            .get(&entry.target)
            .is_some_and(|current| Rc::ptr_eq(current, entry))
        {
            return;
        }
        index.active.remove(&entry.target);
        index.pending.remove(&entry.target);
        index.terminal.insert(entry.target, Rc::clone(entry));
        if (matches!(
            entry.receipt(),
            Some(CleanupReceipt::Completed(Err(_)))
                | Some(CleanupReceipt::DispatchFailed(_))
                | Some(CleanupReceipt::Abandoned)
        ) || entry.submission_failure().is_some())
            && !entry.reported.get()
        {
            index
                .unreported_errors
                .insert(entry.target, Rc::clone(entry));
        }
    }

    pub(crate) fn note_reported(&self, entry: &CleanupEntry) {
        {
            let mut index = self.index.borrow_mut();
            index.unreported_errors.remove(&entry.target);
        }
        // An earlier collection may already have retired this target while
        // its error was deliberately held for reporting. Check this exact
        // identity now; a live alias still keeps the terminal entry indexed.
        let probe = runmat_gc::gc_with_value(&entry.target, |_| ());
        let mut index = self.index.borrow_mut();
        let exact = index
            .terminal
            .get(&entry.target)
            .filter(|current| std::ptr::eq(current.as_ref(), entry))
            .cloned();
        let Some(exact) = exact else {
            return;
        };
        if exact.completion_retirement_pending() {
            return;
        }
        match probe {
            Err(runmat_gc::GcError::InvalidPointer(_)) => {
                index.retirement_retry.remove(&entry.target);
                index.terminal.remove(&entry.target);
                index.entries.remove(&entry.target);
            }
            Ok(()) => {
                index.retirement_retry.remove(&entry.target);
            }
            Err(_) => {
                index.retirement_retry.insert(entry.target, exact);
            }
        }
    }
}
