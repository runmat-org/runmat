//! Borrowed compiler-metadata view of the canonical workspace namespace.

use super::owner::BaseWorkspaceState;
use super::{BaseWorkspaceError, BaseWorkspaceOwner, WorkspaceRevision};
use crate::workspace::GlobalBindingTarget;

/// Names borrowed while the canonical owner remains read-locked. This view
/// neither copies language values nor materializes a detached name collection.
pub struct BaseWorkspaceBindingNames<'a> {
    state: &'a BaseWorkspaceState,
}

impl BaseWorkspaceBindingNames<'_> {
    pub fn revision(&self) -> WorkspaceRevision {
        self.state.revision
    }

    pub fn contains_local(&self, name: &str) -> bool {
        !self.contains_global(name)
            && (self.state.managed_storage.lookup(name).is_some()
                || self.state.entries.contains_key(name))
    }

    pub fn contains_global(&self, name: &str) -> bool {
        self.state
            .managed_storage
            .global(name)
            .unwrap_or_else(|| self.state.global_bindings.contains_key(name))
    }

    pub fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        match self.state.managed_storage.global(name) {
            Some(true) => self.state.managed_storage.global_target(name),
            Some(false) => None,
            None => self.state.global_bindings.get(name),
        }
    }

    pub fn local_names(&self) -> impl Iterator<Item = &str> {
        self.state
            .managed_storage
            .iter()
            .map(|(name, _)| name)
            .chain(self.state.entries.keys().map(|name| name.as_str()))
    }

    pub fn global_names(&self) -> impl Iterator<Item = &str> {
        self.state.managed_storage.global_names().chain(
            self.state
                .global_bindings
                .keys()
                .map(|name| name.as_str())
                .filter(|name| self.state.managed_storage.global(name).is_none()),
        )
    }
}

impl BaseWorkspaceOwner {
    /// Inspect names for metadata preparation without cloning a workspace.
    /// The higher-ranked callback prevents borrowed names escaping the read
    /// guard. Prepare allocation authority before copying into owned metadata;
    /// reentrant mutation of this owner is rejected while the view is active.
    pub fn with_binding_names<R>(
        &self,
        inspect: impl for<'a> FnOnce(BaseWorkspaceBindingNames<'a>) -> R,
    ) -> Result<R, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(inspect(BaseWorkspaceBindingNames { state: &state }))
    }
}
