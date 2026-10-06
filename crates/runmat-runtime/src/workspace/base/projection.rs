use std::collections::BTreeMap;
use std::rc::{Rc, Weak};

use runmat_value::Value;

use super::owner::BaseWorkspaceState;
use super::values::RootedWorkspaceValues;
use super::{
    BaseWorkspaceCommit, BaseWorkspaceError, BaseWorkspaceOwner, BaseWorkspaceReconciliation,
    BaseWorkspaceSnapshot, WorkspaceChange, WorkspaceDelta, WorkspaceRevision,
};

#[derive(Debug)]
pub struct BaseWorkspaceProjection {
    owner: Weak<std::cell::RefCell<BaseWorkspaceState>>,
    revision: WorkspaceRevision,
    values: RootedWorkspaceValues,
    pending: WorkspaceDelta,
}

impl BaseWorkspaceProjection {
    pub(super) fn new(owner: &BaseWorkspaceOwner) -> Result<Self, BaseWorkspaceError> {
        let snapshot = owner.snapshot()?;
        Ok(Self {
            owner: Rc::downgrade(&owner.state),
            revision: snapshot.revision(),
            values: RootedWorkspaceValues::new(snapshot.values().clone())?,
            pending: WorkspaceDelta::new(),
        })
    }

    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn values(&self) -> &BTreeMap<String, Value> {
        self.values.values()
    }

    pub fn lookup(&self, name: &str) -> Option<&Value> {
        self.values.values().get(name)
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
        let mut candidate = self.values.values().clone();
        candidate.insert(name.clone(), value.clone());
        super::owner::validate_entry_count(candidate.len())?;
        let candidate = RootedWorkspaceValues::new(candidate)?;
        self.pending.assign(name, value)?;
        self.values = candidate;
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Result<(), BaseWorkspaceError> {
        self.pending.remove(name.to_string())?;
        self.values.remove(name);
        Ok(())
    }

    pub fn clear(&mut self) -> Result<(), BaseWorkspaceError> {
        self.pending.clear()?;
        self.values.clear();
        Ok(())
    }

    /// Publish this projection's ordered mutations over the latest canonical
    /// state. A later publisher wins for names both projections changed; names
    /// absent from this delta are never overwritten.
    pub fn publish(&mut self) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        let owner = self.owner.upgrade().ok_or(BaseWorkspaceError::Closed)?;
        let delta = self.pending.clone();
        let commit = {
            let mut state = owner
                .try_borrow_mut()
                .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
            state.commit(self.revision, delta)?
        };
        self.adopt(commit.snapshot())?;
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
        self.apply_reconciliation(&reconciliation)?;
        Ok(reconciliation)
    }

    fn apply_reconciliation(
        &mut self,
        reconciliation: &BaseWorkspaceReconciliation,
    ) -> Result<(), BaseWorkspaceError> {
        match reconciliation {
            BaseWorkspaceReconciliation::UpToDate { revision } => {
                self.revision = *revision;
            }
            BaseWorkspaceReconciliation::Delta { to, changes, .. } => {
                let mut values = self.values.values().clone();
                for change in changes.changes() {
                    match change {
                        WorkspaceChange::Assign { name, value } => {
                            values.insert(name.clone(), value.clone());
                        }
                        WorkspaceChange::Remove { name } => {
                            values.remove(name);
                        }
                        WorkspaceChange::Clear => values.clear(),
                    }
                }
                self.values = RootedWorkspaceValues::new(values)?;
                self.revision = *to;
            }
            BaseWorkspaceReconciliation::Reset(snapshot) => self.adopt(snapshot)?,
        }
        Ok(())
    }

    fn adopt(&mut self, snapshot: &BaseWorkspaceSnapshot) -> Result<(), BaseWorkspaceError> {
        self.values = RootedWorkspaceValues::new(snapshot.values().clone())?;
        self.revision = snapshot.revision();
        self.pending = WorkspaceDelta::new();
        Ok(())
    }
}
