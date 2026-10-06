use runmat_value::Value;

use super::values::RootedWorkspaceEntry;
use super::BaseWorkspaceError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceChangeKind {
    Assign,
    Remove,
    Clear,
}

#[derive(Clone, Debug)]
pub(super) enum WorkspaceChangeData {
    Assign {
        name: String,
        entry: RootedWorkspaceEntry,
    },
    Remove {
        name: String,
    },
    Clear,
}

/// One checked workspace mutation. Its representation is private so an
/// assigning change cannot carry an unrooted or transient value.
#[derive(Clone, Debug)]
pub struct WorkspaceChange(WorkspaceChangeData);

impl WorkspaceChange {
    pub fn assign(name: impl Into<String>, value: Value) -> Result<Self, BaseWorkspaceError> {
        Ok(Self::assign_entry(
            name.into(),
            RootedWorkspaceEntry::new(value)?,
        ))
    }

    pub fn remove(name: impl Into<String>) -> Self {
        Self(WorkspaceChangeData::Remove { name: name.into() })
    }

    pub const fn clear() -> Self {
        Self(WorkspaceChangeData::Clear)
    }

    pub fn kind(&self) -> WorkspaceChangeKind {
        match &self.0 {
            WorkspaceChangeData::Assign { .. } => WorkspaceChangeKind::Assign,
            WorkspaceChangeData::Remove { .. } => WorkspaceChangeKind::Remove,
            WorkspaceChangeData::Clear => WorkspaceChangeKind::Clear,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match &self.0 {
            WorkspaceChangeData::Assign { name, .. } | WorkspaceChangeData::Remove { name } => {
                Some(name)
            }
            WorkspaceChangeData::Clear => None,
        }
    }

    pub fn value(&self) -> Option<&Value> {
        match &self.0 {
            WorkspaceChangeData::Assign { entry, .. } => Some(entry.value()),
            WorkspaceChangeData::Remove { .. } | WorkspaceChangeData::Clear => None,
        }
    }

    pub(super) fn assign_entry(name: String, entry: RootedWorkspaceEntry) -> Self {
        Self(WorkspaceChangeData::Assign { name, entry })
    }

    pub(super) fn data(&self) -> &WorkspaceChangeData {
        &self.0
    }
}

#[derive(Debug, Default)]
pub struct WorkspaceDelta {
    changes: Vec<WorkspaceChange>,
}

impl WorkspaceDelta {
    pub const fn new() -> Self {
        Self {
            changes: Vec::new(),
        }
    }

    pub fn from_changes(changes: Vec<WorkspaceChange>) -> Self {
        Self { changes }
    }

    pub fn assign(
        &mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<(), BaseWorkspaceError> {
        self.changes.push(WorkspaceChange::assign(name, value)?);
        Ok(())
    }

    pub fn remove(&mut self, name: impl Into<String>) {
        self.changes.push(WorkspaceChange::remove(name));
    }

    pub fn clear(&mut self) {
        self.changes.push(WorkspaceChange::clear());
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.changes.len()
    }

    pub fn changes(&self) -> &[WorkspaceChange] {
        &self.changes
    }

    pub(super) fn push_assign_entry(&mut self, name: String, entry: RootedWorkspaceEntry) {
        self.changes
            .push(WorkspaceChange::assign_entry(name, entry));
    }
}
