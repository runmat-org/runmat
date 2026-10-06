use std::rc::{Rc, Weak};

use runmat_value::Value;

use super::delta::WorkspaceChangeData;
use super::owner::BaseWorkspaceState;
use super::values::{RootedWorkspaceEntries, RootedWorkspaceEntry};
use super::{
    BaseWorkspaceCommit, BaseWorkspaceError, BaseWorkspaceOwner, BaseWorkspaceReconciliation,
    WorkspaceDelta, WorkspaceRevision,
};

#[derive(Debug)]
pub struct BaseWorkspaceProjection {
    owner: Weak<std::cell::RefCell<BaseWorkspaceState>>,
    revision: WorkspaceRevision,
    entries: RootedWorkspaceEntries,
    pending: WorkspaceDelta,
}

impl BaseWorkspaceProjection {
    pub(super) fn new(owner: &BaseWorkspaceOwner) -> Result<Self, BaseWorkspaceError> {
        let state = owner
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(Self {
            owner: Rc::downgrade(&owner.state),
            revision: state.revision,
            entries: state.entries.clone(),
            pending: WorkspaceDelta::new(),
        })
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

    /// Iterate the projection's retained local view. The view remains valid
    /// after its owner closes so diagnostics and unwind code can inspect
    /// already-admitted values; it does not confer executable authority.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries
            .iter()
            .map(|(name, entry)| (name.as_str(), entry.value()))
    }

    /// Read the projection's retained local view. Call
    /// [`Self::ensure_effect_authority`] before admitting new execution or an
    /// external effect against this projection.
    pub fn lookup(&self, name: &str) -> Option<&Value> {
        self.entries.get(name).map(|entry| entry.value())
    }

    /// Verify that the canonical owner which admitted this projection is
    /// still live. Local reads remain available after closure, but publication,
    /// reconciliation, and new executor/effect admission must pass this check.
    pub fn ensure_effect_authority(&self) -> Result<(), BaseWorkspaceError> {
        let owner = self.owner.upgrade().ok_or(BaseWorkspaceError::Closed)?;
        let state = owner
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()
    }

    pub fn has_pending_changes(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn assign(
        &mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<(), BaseWorkspaceError> {
        let name = name.into();
        let entry = RootedWorkspaceEntry::new(value)?;
        self.pending.push_assign_entry(name.clone(), entry.clone());
        self.entries.insert(name, entry);
        Ok(())
    }

    pub fn remove(&mut self, name: &str) {
        self.pending.remove(name.to_string());
        self.entries.remove(name);
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.entries.clear();
    }

    /// Publish this projection's ordered mutations over the latest canonical
    /// state. A later publisher wins for names both projections changed; names
    /// absent from this delta are never overwritten.
    pub fn publish(&mut self) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        let owner = self.owner.upgrade().ok_or(BaseWorkspaceError::Closed)?;
        let (commit, reconciliation) = {
            let mut state = owner
                .try_borrow_mut()
                .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
            let revision = state.preflight_commit(self.revision, &self.pending)?;
            let base_revision = self.revision;
            let delta = std::mem::take(&mut self.pending);
            let commit = state.commit_preflighted(base_revision, revision, delta);
            let reconciliation =
                state
                    .journal
                    .reconcile_known(base_revision, state.revision, &state.entries);
            (commit, reconciliation)
        };
        self.apply_reconciliation(&reconciliation);
        Ok(commit)
    }

    /// Refresh a clean projection from the canonical owner. Dirty projections
    /// must publish first so event order, rather than snapshot replacement,
    /// decides same-name writes.
    pub fn reconcile(&mut self) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        if self.has_pending_changes() {
            return Err(BaseWorkspaceError::PendingChanges);
        }
        let owner = self.owner.upgrade().ok_or(BaseWorkspaceError::Closed)?;
        let reconciliation = owner
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?
            .reconcile(self.revision)?;
        self.apply_reconciliation(&reconciliation);
        Ok(reconciliation)
    }

    fn apply_reconciliation(&mut self, reconciliation: &BaseWorkspaceReconciliation) {
        match reconciliation {
            BaseWorkspaceReconciliation::UpToDate { revision } => {
                self.revision = *revision;
            }
            BaseWorkspaceReconciliation::Delta { to, changes, .. } => {
                apply_delta(&mut self.entries, changes);
                self.revision = *to;
            }
            BaseWorkspaceReconciliation::Reset(snapshot) => {
                self.entries = snapshot.entries.clone();
                self.revision = snapshot.revision();
            }
        }
    }
}

fn apply_delta(entries: &mut RootedWorkspaceEntries, delta: &WorkspaceDelta) {
    for change in delta.changes() {
        match change.data() {
            WorkspaceChangeData::Assign { name, entry } => {
                entries.insert(name.clone(), entry.clone());
            }
            WorkspaceChangeData::Remove { name } => {
                entries.remove(name);
            }
            WorkspaceChangeData::Clear => entries.clear(),
        }
    }
}
