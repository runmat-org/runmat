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
    BindGlobal {
        name: String,
    },
    UnbindGlobal {
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
            WorkspaceChangeData::Remove { .. }
            | WorkspaceChangeData::BindGlobal { .. }
            | WorkspaceChangeData::UnbindGlobal { .. } => WorkspaceChangeKind::Remove,
            WorkspaceChangeData::Clear => WorkspaceChangeKind::Clear,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match &self.0 {
            WorkspaceChangeData::Assign { name, .. }
            | WorkspaceChangeData::Remove { name }
            | WorkspaceChangeData::BindGlobal { name }
            | WorkspaceChangeData::UnbindGlobal { name } => Some(name),
            WorkspaceChangeData::Clear => None,
        }
    }

    pub fn value(&self) -> Option<&Value> {
        match &self.0 {
            WorkspaceChangeData::Assign { entry, .. } => Some(entry.value()),
            WorkspaceChangeData::Remove { .. }
            | WorkspaceChangeData::BindGlobal { .. }
            | WorkspaceChangeData::UnbindGlobal { .. }
            | WorkspaceChangeData::Clear => None,
        }
    }

    pub(super) fn assign_entry(name: String, entry: RootedWorkspaceEntry) -> Self {
        Self(WorkspaceChangeData::Assign { name, entry })
    }

    pub(super) fn data(&self) -> &WorkspaceChangeData {
        &self.0
    }
}

/// Ordered, unpublished workspace mutations.
///
/// Every assigned value remains rooted until the delta is published or
/// dropped, including an assignment later overwritten, removed, or cleared in
/// the same delta. This is intentional: the pending record is replayable in
/// order. A successful commit returns a canonical-effect delta which omits
/// intermediate values that no longer contribute to workspace state.
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

    /// Bind a base-workspace name to the separate session-global value store.
    /// The canonical base owner records this alongside removing any local
    /// value so both namespace changes share one ordered revision.
    pub fn bind_global(&mut self, name: impl Into<String>) {
        self.changes
            .push(WorkspaceChange(WorkspaceChangeData::BindGlobal {
                name: name.into(),
            }));
    }

    /// Remove a base-workspace global binding and any hidden local value.
    /// A later local assignment must be explicit; stale projections cannot
    /// leave a value that resurfaces when the binding is cleared.
    pub fn unbind_global(&mut self, name: impl Into<String>) {
        self.changes
            .push(WorkspaceChange(WorkspaceChangeData::UnbindGlobal {
                name: name.into(),
            }));
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
