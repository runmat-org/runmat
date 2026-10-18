use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use runmat_value::Value;

use crate::execution::RootedValueHandles;
use crate::workspace::retained::{WorkspaceValueLease, WorkspaceValueReadLease};
use crate::workspace::WorkspaceSlotReadLease;

use super::{BaseWorkspaceError, PreparedWorkspaceWrite, WorkspaceName, WorkspaceRetainedValue};

#[derive(Clone, Debug)]
pub(super) enum WorkspaceEntryName {
    Owned(String),
    Prepared(WorkspaceName),
}

impl WorkspaceEntryName {
    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Owned(name) => name,
            Self::Prepared(name) => name.as_str(),
        }
    }
}

impl From<String> for WorkspaceEntryName {
    fn from(name: String) -> Self {
        Self::Owned(name)
    }
}

impl Borrow<str> for WorkspaceEntryName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq for WorkspaceEntryName {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for WorkspaceEntryName {}

impl PartialOrd for WorkspaceEntryName {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkspaceEntryName {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl Hash for WorkspaceEntryName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

/// One immutable workspace binding. Clones share both the language value and
/// its managed-handle roots; mutation replaces the complete binding.
#[derive(Clone, Debug)]
pub(super) struct RootedWorkspaceEntry {
    value: WorkspaceValueLease,
    _roots: Option<RootedValueHandles>,
}

impl RootedWorkspaceEntry {
    pub(super) fn new(value: Value) -> Result<Self, BaseWorkspaceError> {
        runmat_value::validate_no_transient_sequence(&value)?;
        let roots = RootedValueHandles::for_values(std::iter::once(&value))?;
        Ok(Self {
            value: WorkspaceValueLease::unbounded(value),
            _roots: Some(roots),
        })
    }

    /// Reuse an already-admitted allocation-free workspace alias. Eligibility
    /// excludes managed handles, so no second GC-root owner is
    /// needed for the compatibility delta/table entry.
    pub(super) fn from_prepared(write: &PreparedWorkspaceWrite) -> Self {
        Self::from_retained(write.value().clone())
    }

    pub(super) fn from_retained(value: WorkspaceRetainedValue) -> Self {
        Self {
            value: WorkspaceValueLease::from_retained(value),
            _roots: None,
        }
    }

    pub(super) fn from_read_lease(source: WorkspaceSlotReadLease) -> Self {
        Self {
            value: WorkspaceValueLease::from_shared_generation(
                source.shared_generation(),
                source.shared_roots(),
            ),
            _roots: None,
        }
    }

    pub(super) fn is_prepared(&self) -> bool {
        self.value.is_admitted()
    }

    pub(super) fn value(&self) -> &Value {
        self.value.value()
    }

    pub(super) fn read_lease(&self) -> WorkspaceValueReadLease {
        WorkspaceValueReadLease::with_roots(self.value.clone(), self._roots.clone())
    }

    pub(super) fn clone_value(&self) -> Value {
        self.value().clone()
    }

    #[cfg(test)]
    pub(super) fn shares_value_with(&self, other: &Self) -> bool {
        self.value.shares_owner_with(&other.value)
    }
}

pub(super) type RootedWorkspaceEntries = BTreeMap<WorkspaceEntryName, RootedWorkspaceEntry>;
