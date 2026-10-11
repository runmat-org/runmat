use std::cell::{Ref, RefCell, RefMut};
use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_value::Value;

use crate::context::RuntimeWorkspaceService;
use crate::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection};
use crate::RuntimeError;

use super::{
    base_workspace_error_to_runtime, BaseWorkspaceCommit, BaseWorkspaceError,
    BaseWorkspaceProjection, BaseWorkspaceReconciliation, WorkspaceChange, WorkspaceDelta,
    WorkspaceRevision,
};

/// Exact current-workspace authority for one admitted executor invocation.
///
/// The service retains the projection and its managed-value roots for the
/// complete invocation lifetime. Language-visible mutations are staged in
/// order and may be published at executor safepoints without exposing the
/// session's close authority or replacing canonical state from a snapshot.
#[derive(Clone, Debug)]
pub struct BaseWorkspaceProjectionService {
    state: Rc<RefCell<ProjectionServiceState>>,
}

#[derive(Debug)]
struct ProjectionServiceState {
    projection: BaseWorkspaceProjection,
    published_effect: PublishedWorkspaceEffect,
}

#[derive(Debug, Default)]
struct PublishedWorkspaceEffect {
    cleared: bool,
    changes: BTreeMap<String, WorkspaceChange>,
}

impl PublishedWorkspaceEffect {
    fn observe(&mut self, delta: &WorkspaceDelta) -> Vec<WorkspaceChange> {
        let mut retired = Vec::new();
        for change in delta.changes() {
            match change.kind() {
                super::WorkspaceChangeKind::Clear => {
                    self.cleared = true;
                    retired.extend(std::mem::take(&mut self.changes).into_values());
                }
                super::WorkspaceChangeKind::Assign | super::WorkspaceChangeKind::Remove => {
                    if let Some(name) = change.name() {
                        if let Some(previous) =
                            self.changes.insert(name.to_string(), change.clone())
                        {
                            retired.push(previous);
                        }
                    }
                }
            }
        }
        retired
    }

    fn take(&mut self) -> WorkspaceDelta {
        let mut changes = Vec::with_capacity(self.changes.len() + usize::from(self.cleared));
        if self.cleared {
            changes.push(WorkspaceChange::clear());
        }
        changes.extend(std::mem::take(&mut self.changes).into_values());
        self.cleared = false;
        WorkspaceDelta::from_changes(changes)
    }
}

impl BaseWorkspaceProjectionService {
    pub fn new(projection: BaseWorkspaceProjection) -> Self {
        Self {
            state: Rc::new(RefCell::new(ProjectionServiceState {
                projection,
                published_effect: PublishedWorkspaceEffect::default(),
            })),
        }
    }

    pub fn revision(&self) -> Result<WorkspaceRevision, RuntimeError> {
        Ok(self.borrow()?.revision())
    }

    pub fn has_pending_changes(&self) -> Result<bool, RuntimeError> {
        Ok(self.borrow()?.has_pending_changes())
    }

    pub fn lookup(&self, name: &str) -> Result<Option<Value>, RuntimeError> {
        self.with_live_projection(|projection| projection.lookup(name).cloned())
    }

    pub fn snapshot(&self) -> Result<Vec<(String, Value)>, RuntimeError> {
        self.with_live_projection(|projection| {
            projection
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect()
        })
    }

    pub fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        <Self as RuntimeWorkspaceService>::assign(self, name, value)
    }

    pub fn remove(&self, name: &str) -> Result<(), RuntimeError> {
        <Self as RuntimeWorkspaceService>::remove(self, name)
    }

    pub fn clear(&self) -> Result<(), RuntimeError> {
        <Self as RuntimeWorkspaceService>::clear(self)
    }

    pub fn bind_global(&self, name: &str) -> Result<(), RuntimeError> {
        self.with_live_projection_mut(|projection| {
            projection.bind_global(name);
            Ok(())
        })
    }

    pub fn unbind_global(&self, name: &str) -> Result<(), RuntimeError> {
        self.with_live_projection_mut(|projection| {
            projection.unbind_global(name);
            Ok(())
        })
    }

    /// Publish all ordered mutations admitted since the preceding safepoint.
    pub fn publish(&self) -> Result<BaseWorkspaceCommit, RuntimeError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let commit = state
            .projection
            .publish()
            .map_err(base_workspace_error_to_runtime)?;
        let retired = state.published_effect.observe(commit.changes());
        drop(state);
        drop(retired);
        Ok(commit)
    }

    /// Drain the canonical effects published by this invocation, including
    /// commits made at intermediate effect safepoints.
    pub fn take_published_changes(&self) -> Result<WorkspaceDelta, RuntimeError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        Ok(state.published_effect.take())
    }

    /// Refresh a clean invocation projection from canonical session state.
    pub fn reconcile(&self) -> Result<BaseWorkspaceReconciliation, RuntimeError> {
        self.borrow_mut()?
            .reconcile()
            .map_err(base_workspace_error_to_runtime)
    }

    /// Admit a new external or reentrant effect against this exact workspace.
    pub fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        self.synchronize_for_effect()?;
        self.borrow()?
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)
    }

    fn synchronize_for_effect(&self) -> Result<(), RuntimeError> {
        if self.has_pending_changes()? {
            self.publish()?;
        } else {
            self.reconcile()?;
        }
        Ok(())
    }

    fn borrow(&self) -> Result<Ref<'_, BaseWorkspaceProjection>, RuntimeError> {
        self.state
            .try_borrow()
            .map(|state| Ref::map(state, |state| &state.projection))
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))
    }

    fn borrow_mut(&self) -> Result<RefMut<'_, BaseWorkspaceProjection>, RuntimeError> {
        self.state
            .try_borrow_mut()
            .map(|state| RefMut::map(state, |state| &mut state.projection))
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))
    }

    fn with_live_projection<T>(
        &self,
        operation: impl FnOnce(&BaseWorkspaceProjection) -> T,
    ) -> Result<T, RuntimeError> {
        self.synchronize_for_effect()?;
        let projection = self.borrow()?;
        projection
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        Ok(operation(&projection))
    }

    fn with_live_projection_mut<T>(
        &self,
        operation: impl FnOnce(&mut BaseWorkspaceProjection) -> Result<T, BaseWorkspaceError>,
    ) -> Result<T, RuntimeError> {
        let mut projection = self.borrow_mut()?;
        projection
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        operation(&mut projection).map_err(base_workspace_error_to_runtime)
    }
}

impl RuntimeWorkspaceService for BaseWorkspaceProjectionService {
    fn detach_selected(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        self.synchronize_for_effect()?;
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let (detached, commit) = state
            .projection
            .detach_selected(selection)
            .map_err(base_workspace_error_to_runtime)?;
        let retired = state.published_effect.observe(commit.changes());
        drop(state);
        drop(retired);
        Ok(detached)
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        let state = self.state.try_borrow().ok()?;
        for (_, value) in state.projection.iter() {
            visitor(value)?;
        }
        Some(())
    }

    // These infallible methods implement the compatibility surface. Exact
    // executor paths use the checked methods below and retain typed failures.
    fn lookup(&self, name: &str) -> Option<Value> {
        self.state
            .try_borrow()
            .ok()?
            .projection
            .lookup(name)
            .cloned()
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        self.state
            .try_borrow()
            .map(|state| {
                state
                    .projection
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn global_names(&self) -> Vec<String> {
        super::super::session::global_names()
    }

    fn try_lookup(&self, name: &str) -> Result<Option<Value>, RuntimeError> {
        self.with_live_projection(|projection| projection.lookup(name).cloned())
    }

    fn try_snapshot(&self) -> Result<Vec<(String, Value)>, RuntimeError> {
        self.with_live_projection(|projection| {
            projection
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect()
        })
    }

    fn try_global_names(&self) -> Result<Vec<String>, RuntimeError> {
        self.ensure_effect_authority()?;
        Ok(super::super::session::global_names())
    }

    fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        BaseWorkspaceProjectionService::ensure_effect_authority(self)
    }

    fn ensure_mutation_authority(&self) -> Result<(), RuntimeError> {
        self.borrow()?
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)
    }

    fn workspace_revision(&self) -> Result<Option<WorkspaceRevision>, RuntimeError> {
        self.revision().map(Some)
    }

    fn base_global_binding_names(&self) -> Result<Vec<String>, RuntimeError> {
        self.with_live_projection(|projection| {
            let mut names = projection
                .global_names()
                .map(str::to_string)
                .collect::<Vec<_>>();
            names.sort();
            names
        })
    }

    fn is_base_global_binding(&self, name: &str) -> Result<bool, RuntimeError> {
        self.with_live_projection(|projection| projection.is_global(name))
    }

    fn bind_base_global(&self, name: &str) -> Result<(), RuntimeError> {
        self.bind_global(name)
    }

    fn unbind_base_global(&self, name: &str) -> Result<(), RuntimeError> {
        self.unbind_global(name)
    }

    fn assign_base_local(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        self.with_live_projection_mut(|projection| {
            projection.unbind_global(name);
            projection.assign(name, value)
        })
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        self.with_live_projection_mut(|projection| projection.assign(name, value))
    }

    fn clear(&self) -> Result<(), RuntimeError> {
        self.with_live_projection_mut(|projection| {
            projection.clear();
            Ok(())
        })
    }

    fn remove(&self, name: &str) -> Result<(), RuntimeError> {
        self.with_live_projection_mut(|projection| {
            projection.remove(name);
            Ok(())
        })
    }
}
