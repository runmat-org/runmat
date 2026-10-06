use std::cell::RefCell;
use std::rc::Rc;

use runmat_value::Value;

use super::journal::{WorkspaceJournal, DEFAULT_HISTORY_CHANGE_LIMIT};
use super::values::RootedWorkspaceEntries;
use super::{BaseWorkspaceError, BaseWorkspaceProjection, WorkspaceDelta, WorkspaceRevision};

#[derive(Clone, Debug)]
pub struct BaseWorkspaceSnapshot {
    revision: WorkspaceRevision,
    pub(super) entries: RootedWorkspaceEntries,
}

impl BaseWorkspaceSnapshot {
    pub(super) fn new(revision: WorkspaceRevision, entries: RootedWorkspaceEntries) -> Self {
        Self { revision, entries }
    }

    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.entries.get(name).map(|entry| entry.value())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries
            .iter()
            .map(|(name, entry)| (name.as_str(), entry.value()))
    }

    /// Materialize independent language values for an explicit host/export
    /// boundary. Executor admission should use [`BaseWorkspaceOwner::begin_projection`]
    /// so unchanged entries remain shared.
    pub fn to_values(&self) -> Vec<(String, Value)> {
        self.entries
            .iter()
            .map(|(name, entry)| (name.clone(), entry.clone_value()))
            .collect()
    }
}

#[derive(Debug)]
pub struct BaseWorkspaceCommit {
    base_revision: WorkspaceRevision,
    previous_revision: WorkspaceRevision,
    revision: WorkspaceRevision,
    changes: WorkspaceDelta,
}

impl BaseWorkspaceCommit {
    pub(super) fn new(
        base_revision: WorkspaceRevision,
        previous_revision: WorkspaceRevision,
        revision: WorkspaceRevision,
        changes: WorkspaceDelta,
    ) -> Self {
        Self {
            base_revision,
            previous_revision,
            revision,
            changes,
        }
    }

    pub fn base_revision(&self) -> WorkspaceRevision {
        self.base_revision
    }

    pub fn previous_revision(&self) -> WorkspaceRevision {
        self.previous_revision
    }

    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Canonical effect after ordered input changes were applied. Intermediate
    /// assigned values that were overwritten, removed, or cleared are absent.
    pub fn changes(&self) -> &WorkspaceDelta {
        &self.changes
    }

    pub fn merged_concurrent_changes(&self) -> bool {
        self.base_revision != self.previous_revision
    }
}

#[derive(Debug)]
pub enum BaseWorkspaceReconciliation {
    UpToDate {
        revision: WorkspaceRevision,
    },
    Delta {
        from: WorkspaceRevision,
        to: WorkspaceRevision,
        changes: WorkspaceDelta,
    },
    Reset(BaseWorkspaceSnapshot),
}

#[derive(Clone, Debug)]
pub struct BaseWorkspaceOwner {
    pub(super) state: Rc<RefCell<BaseWorkspaceState>>,
}

#[derive(Debug)]
pub(super) struct BaseWorkspaceState {
    pub(super) closed: bool,
    pub(super) revision: WorkspaceRevision,
    pub(super) entries: RootedWorkspaceEntries,
    pub(super) journal: WorkspaceJournal,
}

impl Default for BaseWorkspaceOwner {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseWorkspaceOwner {
    pub fn new() -> Self {
        Self::with_history_limit(DEFAULT_HISTORY_CHANGE_LIMIT)
    }

    pub fn snapshot(&self) -> Result<BaseWorkspaceSnapshot, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(state.snapshot())
    }

    pub fn lookup(&self, name: &str) -> Result<Option<Value>, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(state.entries.get(name).map(|entry| entry.clone_value()))
    }

    /// Visit canonical retained values without materializing a workspace
    /// snapshot. `None` means the visitor stopped before completing the view.
    pub fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Result<Option<()>, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        for (_, entry) in state.entries.iter() {
            if visitor(entry.value()).is_none() {
                return Ok(None);
            }
        }
        Ok(Some(()))
    }

    pub fn revision(&self) -> Result<WorkspaceRevision, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(state.revision)
    }

    /// Confirm that this owner can still authorize a newly admitted external
    /// effect without materializing or exposing its workspace contents.
    pub fn ensure_effect_authority(&self) -> Result<(), BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()
    }

    pub fn begin_projection(&self) -> Result<BaseWorkspaceProjection, BaseWorkspaceError> {
        BaseWorkspaceProjection::new(self)
    }

    pub fn commit(
        &self,
        base_revision: WorkspaceRevision,
        delta: WorkspaceDelta,
    ) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        self.state
            .try_borrow_mut()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?
            .commit(base_revision, delta)
    }

    /// Apply a synchronous host mutation against the current canonical
    /// revision. Executor projections use [`Self::commit`] so their older base
    /// revision remains visible in the receipt.
    pub fn apply(&self, delta: WorkspaceDelta) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        let revision = state.revision;
        state.commit(revision, delta)
    }

    pub fn reconcile(
        &self,
        revision: WorkspaceRevision,
    ) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        self.state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?
            .reconcile(revision)
    }

    pub fn close(&self) -> Result<bool, BaseWorkspaceError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        if state.closed {
            return Ok(false);
        }
        state.closed = true;
        state.entries.clear();
        let revision = state.revision;
        state.journal.clear_at(revision);
        Ok(true)
    }

    pub fn is_closed(&self) -> Result<bool, BaseWorkspaceError> {
        self.state
            .try_borrow()
            .map(|state| state.closed)
            .map_err(|_| BaseWorkspaceError::BorrowConflict)
    }

    fn with_history_limit(history_change_limit: usize) -> Self {
        Self {
            state: Rc::new(RefCell::new(BaseWorkspaceState {
                closed: false,
                revision: WorkspaceRevision::INITIAL,
                entries: RootedWorkspaceEntries::new(),
                journal: WorkspaceJournal::new(history_change_limit),
            })),
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(history_change_limit: usize) -> Self {
        Self::with_history_limit(history_change_limit)
    }

    #[cfg(test)]
    pub(super) fn set_revision_for_test(&self, revision: WorkspaceRevision) {
        let mut state = self.state.borrow_mut();
        state.revision = revision;
        state.journal.clear_at(revision);
    }
}

impl BaseWorkspaceState {
    pub(super) fn ensure_open(&self) -> Result<(), BaseWorkspaceError> {
        if self.closed {
            Err(BaseWorkspaceError::Closed)
        } else {
            Ok(())
        }
    }

    pub(super) fn snapshot(&self) -> BaseWorkspaceSnapshot {
        BaseWorkspaceSnapshot::new(self.revision, self.entries.clone())
    }

    pub(super) fn reconcile(
        &self,
        revision: WorkspaceRevision,
    ) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        self.ensure_open()?;
        self.journal
            .reconcile(revision, self.revision, &self.entries)
    }
}
