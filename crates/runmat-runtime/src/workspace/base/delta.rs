use runmat_value::Value;

use crate::execution::RootedValueHandles;

use super::BaseWorkspaceError;

pub const MAX_WORKSPACE_DELTA_CHANGES: usize = u16::MAX as usize;

#[derive(Clone, Debug, PartialEq)]
pub enum WorkspaceChange {
    Assign { name: String, value: Value },
    Remove { name: String },
    Clear,
}

#[derive(Clone, Debug)]
pub struct WorkspaceDelta {
    changes: Vec<WorkspaceChange>,
    _roots: Option<RootedValueHandles>,
}

impl Default for WorkspaceDelta {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkspaceDelta {
    pub const fn new() -> Self {
        Self {
            changes: Vec::new(),
            _roots: None,
        }
    }

    pub fn from_changes(changes: Vec<WorkspaceChange>) -> Result<Self, BaseWorkspaceError> {
        validate_change_count(changes.len())?;
        validate_changes(&changes)?;
        let roots = roots_for_changes(&changes)?;
        Ok(Self {
            changes,
            _roots: roots,
        })
    }

    pub fn assign(
        &mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<(), BaseWorkspaceError> {
        runmat_value::validate_no_transient_sequence(&value)?;
        validate_change_count(self.changes.len().saturating_add(1))?;
        let mut changes = self.changes.clone();
        changes.push(WorkspaceChange::Assign {
            name: name.into(),
            value,
        });
        let roots = roots_for_changes(&changes)?;
        self.changes = changes;
        self._roots = roots;
        Ok(())
    }

    pub fn remove(&mut self, name: impl Into<String>) -> Result<(), BaseWorkspaceError> {
        self.push_unrooted(WorkspaceChange::Remove { name: name.into() })
    }

    pub fn clear(&mut self) -> Result<(), BaseWorkspaceError> {
        self.push_unrooted(WorkspaceChange::Clear)
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

    fn push_unrooted(&mut self, change: WorkspaceChange) -> Result<(), BaseWorkspaceError> {
        validate_change_count(self.changes.len().saturating_add(1))?;
        self.changes.push(change);
        Ok(())
    }
}

pub(super) fn validate_change_count(count: usize) -> Result<(), BaseWorkspaceError> {
    if count > MAX_WORKSPACE_DELTA_CHANGES {
        return Err(BaseWorkspaceError::DeltaTooLarge {
            actual: count,
            maximum: MAX_WORKSPACE_DELTA_CHANGES,
        });
    }
    Ok(())
}

fn validate_changes(changes: &[WorkspaceChange]) -> Result<(), BaseWorkspaceError> {
    for change in changes {
        if let WorkspaceChange::Assign { value, .. } = change {
            runmat_value::validate_no_transient_sequence(value)?;
        }
    }
    Ok(())
}

fn roots_for_changes(
    changes: &[WorkspaceChange],
) -> Result<Option<RootedValueHandles>, BaseWorkspaceError> {
    let values = changes.iter().filter_map(|change| match change {
        WorkspaceChange::Assign { value, .. } => Some(value),
        WorkspaceChange::Remove { .. } | WorkspaceChange::Clear => None,
    });
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(None);
    }
    Ok(Some(RootedValueHandles::for_values(values.into_iter())?))
}
