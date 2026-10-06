use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;

use runmat_value::Value;

use super::values::RootedWorkspaceValues;
use super::{
    BaseWorkspaceError, BaseWorkspaceProjection, WorkspaceChange, WorkspaceDelta, WorkspaceRevision,
};

pub const MAX_BASE_WORKSPACE_ENTRIES: usize = u16::MAX as usize;
const DEFAULT_HISTORY_LIMIT: usize = 1_024;

#[derive(Clone, Debug)]
pub struct BaseWorkspaceSnapshot {
    revision: WorkspaceRevision,
    values: RootedWorkspaceValues,
}

impl BaseWorkspaceSnapshot {
    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn values(&self) -> &BTreeMap<String, Value> {
        self.values.values()
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.values().get(name)
    }

    pub fn into_values(self) -> BTreeMap<String, Value> {
        self.values.into_values()
    }
}

#[derive(Clone, Debug)]
pub struct BaseWorkspaceCommit {
    base_revision: WorkspaceRevision,
    previous_revision: WorkspaceRevision,
    snapshot: BaseWorkspaceSnapshot,
    change_count: usize,
}

impl BaseWorkspaceCommit {
    pub fn base_revision(&self) -> WorkspaceRevision {
        self.base_revision
    }

    pub fn previous_revision(&self) -> WorkspaceRevision {
        self.previous_revision
    }

    pub fn revision(&self) -> WorkspaceRevision {
        self.snapshot.revision()
    }

    pub fn change_count(&self) -> usize {
        self.change_count
    }

    pub fn merged_concurrent_changes(&self) -> bool {
        self.base_revision != self.previous_revision
    }

    pub fn snapshot(&self) -> &BaseWorkspaceSnapshot {
        &self.snapshot
    }
}

#[derive(Clone, Debug)]
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
    closed: bool,
    revision: WorkspaceRevision,
    values: RootedWorkspaceValues,
    history_floor: WorkspaceRevision,
    history: VecDeque<WorkspaceHistoryRecord>,
    history_change_count: usize,
    history_change_limit: usize,
}

#[derive(Clone, Debug)]
struct WorkspaceHistoryRecord {
    revision: WorkspaceRevision,
    impacts: Vec<WorkspaceImpact>,
}

#[derive(Clone, Debug)]
enum WorkspaceImpact {
    Name(String),
    Clear,
}

impl BaseWorkspaceOwner {
    pub fn new() -> Result<Self, BaseWorkspaceError> {
        Self::with_history_limit(DEFAULT_HISTORY_LIMIT)
    }

    pub fn snapshot(&self) -> Result<BaseWorkspaceSnapshot, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(state.snapshot())
    }

    pub fn revision(&self) -> Result<WorkspaceRevision, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(state.revision)
    }

    pub fn begin_projection(&self) -> Result<BaseWorkspaceProjection, BaseWorkspaceError> {
        BaseWorkspaceProjection::new(self)
    }

    pub fn commit(
        &self,
        base_revision: WorkspaceRevision,
        delta: WorkspaceDelta,
    ) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.commit(base_revision, delta)
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
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.reconcile(revision)
    }

    pub fn close(&self) -> Result<bool, BaseWorkspaceError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        if state.closed {
            return Ok(false);
        }
        let empty = RootedWorkspaceValues::empty()?;
        state.closed = true;
        state.values = empty;
        state.history.clear();
        state.history_change_count = 0;
        state.history_floor = state.revision;
        Ok(true)
    }

    pub fn is_closed(&self) -> Result<bool, BaseWorkspaceError> {
        self.state
            .try_borrow()
            .map(|state| state.closed)
            .map_err(|_| BaseWorkspaceError::BorrowConflict)
    }

    fn with_history_limit(history_change_limit: usize) -> Result<Self, BaseWorkspaceError> {
        Ok(Self {
            state: Rc::new(RefCell::new(BaseWorkspaceState {
                closed: false,
                revision: WorkspaceRevision::INITIAL,
                values: RootedWorkspaceValues::empty()?,
                history_floor: WorkspaceRevision::INITIAL,
                history: VecDeque::new(),
                history_change_count: 0,
                history_change_limit,
            })),
        })
    }

    #[cfg(test)]
    pub(super) fn for_test(history_limit: usize) -> Result<Self, BaseWorkspaceError> {
        Self::with_history_limit(history_limit)
    }

    #[cfg(test)]
    pub(super) fn set_revision_for_test(&self, revision: WorkspaceRevision) {
        let mut state = self.state.borrow_mut();
        state.revision = revision;
        state.history_floor = revision;
        state.history.clear();
        state.history_change_count = 0;
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
        BaseWorkspaceSnapshot {
            revision: self.revision,
            values: self.values.clone(),
        }
    }

    pub(super) fn commit(
        &mut self,
        base_revision: WorkspaceRevision,
        delta: WorkspaceDelta,
    ) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        self.ensure_open()?;
        if base_revision > self.revision {
            return Err(BaseWorkspaceError::FutureRevision {
                requested: base_revision.get(),
                current: self.revision.get(),
            });
        }
        let previous_revision = self.revision;
        if delta.is_empty() {
            return Ok(BaseWorkspaceCommit {
                base_revision,
                previous_revision,
                snapshot: self.snapshot(),
                change_count: 0,
            });
        }

        let revision = self
            .revision
            .next()
            .ok_or(BaseWorkspaceError::RevisionExhausted)?;
        let mut candidate = self.values.values().clone();
        for change in delta.changes() {
            apply_change(&mut candidate, change);
        }
        validate_entry_count(candidate.len())?;
        let candidate = RootedWorkspaceValues::new(candidate)?;
        let impacts = impacts(delta.changes());

        self.values = candidate;
        self.revision = revision;
        self.push_history(WorkspaceHistoryRecord { revision, impacts });

        Ok(BaseWorkspaceCommit {
            base_revision,
            previous_revision,
            snapshot: self.snapshot(),
            change_count: delta.len(),
        })
    }

    pub(super) fn reconcile(
        &self,
        revision: WorkspaceRevision,
    ) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        self.ensure_open()?;
        if revision > self.revision {
            return Err(BaseWorkspaceError::FutureRevision {
                requested: revision.get(),
                current: self.revision.get(),
            });
        }
        if revision == self.revision {
            return Ok(BaseWorkspaceReconciliation::UpToDate { revision });
        }
        if revision < self.history_floor {
            return Ok(BaseWorkspaceReconciliation::Reset(self.snapshot()));
        }

        let mut clear = false;
        let mut names = BTreeSet::new();
        for record in self
            .history
            .iter()
            .filter(|entry| entry.revision > revision)
        {
            for impact in &record.impacts {
                match impact {
                    WorkspaceImpact::Name(name) => {
                        names.insert(name.clone());
                    }
                    WorkspaceImpact::Clear => {
                        clear = true;
                        names.clear();
                    }
                }
            }
        }
        let extra = usize::from(clear);
        if names.len().saturating_add(extra) > super::MAX_WORKSPACE_DELTA_CHANGES {
            return Ok(BaseWorkspaceReconciliation::Reset(self.snapshot()));
        }
        let mut changes = Vec::with_capacity(names.len() + extra);
        if clear {
            changes.push(WorkspaceChange::Clear);
        }
        for name in names {
            match self.values.values().get(&name) {
                Some(value) => changes.push(WorkspaceChange::Assign {
                    name,
                    value: value.clone(),
                }),
                None if !clear => changes.push(WorkspaceChange::Remove { name }),
                None => {}
            }
        }
        Ok(BaseWorkspaceReconciliation::Delta {
            from: revision,
            to: self.revision,
            changes: WorkspaceDelta::from_changes(changes)?,
        })
    }

    fn push_history(&mut self, record: WorkspaceHistoryRecord) {
        let change_count = record.impacts.len();
        if change_count > self.history_change_limit || self.history_change_limit == 0 {
            self.history.clear();
            self.history_change_count = 0;
            self.history_floor = record.revision;
            return;
        }
        self.history_change_count += change_count;
        self.history.push_back(record);
        while self.history_change_count > self.history_change_limit {
            if let Some(retired) = self.history.pop_front() {
                self.history_change_count -= retired.impacts.len();
                self.history_floor = retired.revision;
            }
        }
    }
}

fn apply_change(values: &mut BTreeMap<String, Value>, change: &WorkspaceChange) {
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

fn impacts(changes: &[WorkspaceChange]) -> Vec<WorkspaceImpact> {
    changes
        .iter()
        .map(|change| match change {
            WorkspaceChange::Assign { name, .. } | WorkspaceChange::Remove { name } => {
                WorkspaceImpact::Name(name.clone())
            }
            WorkspaceChange::Clear => WorkspaceImpact::Clear,
        })
        .collect()
}

pub(super) fn validate_entry_count(count: usize) -> Result<(), BaseWorkspaceError> {
    if count > MAX_BASE_WORKSPACE_ENTRIES {
        return Err(BaseWorkspaceError::TooManyEntries {
            actual: count,
            maximum: MAX_BASE_WORKSPACE_ENTRIES,
        });
    }
    Ok(())
}
