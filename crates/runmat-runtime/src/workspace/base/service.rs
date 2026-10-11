use runmat_value::Value;

use crate::context::{RuntimeBaseWorkspaceService, RuntimeWorkspaceService};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

use super::{BaseWorkspaceError, BaseWorkspaceOwner, WorkspaceDelta};
use crate::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection};

/// Runtime service port for the session-owned canonical base workspace.
///
/// Caller frames may install other [`RuntimeWorkspaceService`] implementations,
/// but those services do not gain base-workspace authority. Globals and
/// persistents remain in the session-variable service and are only reported
/// here through `global_names`.
///
/// The host invokes this adapter under its owning [`crate::context::RuntimeContext`].
/// Outside a context, session-global lookup still reaches the transitional
/// standalone store; that fallback is not canonical base-workspace state.
#[derive(Clone, Debug)]
pub struct BaseWorkspaceService {
    owner: BaseWorkspaceOwner,
}

impl BaseWorkspaceService {
    pub fn new(owner: BaseWorkspaceOwner) -> Self {
        Self { owner }
    }

    pub fn owner(&self) -> BaseWorkspaceOwner {
        self.owner.clone()
    }

    /// Revoke the canonical authority at session teardown. Existing
    /// projections retain their local diagnostic view, but cannot admit or
    /// publish further effects.
    pub fn close(&self) -> Result<bool, RuntimeError> {
        self.owner.close().map_err(base_workspace_error_to_runtime)
    }

    fn apply(&self, delta: WorkspaceDelta) -> Result<(), RuntimeError> {
        self.owner
            .apply(delta)
            .map(|_| ())
            .map_err(base_workspace_error_to_runtime)
    }
}

impl RuntimeBaseWorkspaceService for BaseWorkspaceService {
    fn owner(&self) -> BaseWorkspaceOwner {
        BaseWorkspaceService::owner(self)
    }
}

impl RuntimeWorkspaceService for BaseWorkspaceService {
    fn detach_selected(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        self.owner
            .detach_selected(selection)
            .map_err(base_workspace_error_to_runtime)
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        self.owner.visit_retained_values(visitor).ok().flatten()
    }

    // These infallible reads implement the pre-existing compatibility port.
    // Canonical consumers use the checked methods below; only those methods can
    // distinguish a closed owner from an empty workspace.
    fn lookup(&self, name: &str) -> Option<Value> {
        self.owner.lookup(name).ok().flatten()
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        self.owner
            .snapshot()
            .map(|snapshot| snapshot.to_values())
            .unwrap_or_default()
    }

    fn global_names(&self) -> Vec<String> {
        super::super::session::global_names()
    }

    fn try_lookup(&self, name: &str) -> Result<Option<Value>, RuntimeError> {
        self.owner
            .lookup(name)
            .map_err(base_workspace_error_to_runtime)
    }

    fn try_snapshot(&self) -> Result<Vec<(String, Value)>, RuntimeError> {
        self.owner
            .snapshot()
            .map(|snapshot| snapshot.to_values())
            .map_err(base_workspace_error_to_runtime)
    }

    fn try_global_names(&self) -> Result<Vec<String>, RuntimeError> {
        Ok(super::super::session::global_names())
    }

    fn begin_workspace_projection(
        &self,
    ) -> Result<Option<super::BaseWorkspaceProjection>, RuntimeError> {
        self.owner
            .begin_projection()
            .map(Some)
            .map_err(base_workspace_error_to_runtime)
    }

    fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        self.owner
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)
    }

    fn workspace_revision(&self) -> Result<Option<super::WorkspaceRevision>, RuntimeError> {
        self.owner
            .revision()
            .map(Some)
            .map_err(base_workspace_error_to_runtime)
    }

    fn base_global_binding_names(&self) -> Result<Vec<String>, RuntimeError> {
        self.owner
            .global_names()
            .map_err(base_workspace_error_to_runtime)
    }

    fn is_base_global_binding(&self, name: &str) -> Result<bool, RuntimeError> {
        self.owner
            .is_global(name)
            .map_err(base_workspace_error_to_runtime)
    }

    fn bind_base_global(&self, name: &str) -> Result<(), RuntimeError> {
        let mut delta = WorkspaceDelta::new();
        delta.bind_global(name);
        self.apply(delta)
    }

    fn unbind_base_global(&self, name: &str) -> Result<(), RuntimeError> {
        let mut delta = WorkspaceDelta::new();
        delta.unbind_global(name);
        self.apply(delta)
    }

    fn assign_base_local(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        let mut delta = WorkspaceDelta::new();
        delta.unbind_global(name);
        delta
            .assign(name, value)
            .map_err(base_workspace_error_to_runtime)?;
        self.apply(delta)
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        let mut delta = WorkspaceDelta::new();
        delta
            .assign(name, value)
            .map_err(base_workspace_error_to_runtime)?;
        self.apply(delta)
    }

    fn clear(&self) -> Result<(), RuntimeError> {
        let mut delta = WorkspaceDelta::new();
        delta.clear();
        self.apply(delta)
    }

    fn remove(&self, name: &str) -> Result<(), RuntimeError> {
        let mut delta = WorkspaceDelta::new();
        delta.remove(name);
        self.apply(delta)
    }
}

/// Preserve a typed base-workspace failure at a Runtime error boundary.
///
/// Execution hosts use this adapter when they operate directly on a canonical
/// owner or one of its projections. Keeping the mapping here prevents hosts
/// from copying identifiers or erasing the original rooting/transient cause.
pub fn base_workspace_error_to_runtime(error: BaseWorkspaceError) -> RuntimeError {
    match error {
        BaseWorkspaceError::Rooting(source) => {
            let message = format!("could not retain a base workspace value: {source}");
            crate::build_runtime_error(message)
                .with_identifier("RunMat:Workspace:Rooting")
                .with_source(source)
                .build()
        }
        BaseWorkspaceError::TransientValue(source) => {
            let message = format!("a workspace value is not storable: {source}");
            crate::build_runtime_error(message)
                .with_identifier("RunMat:TransientSequenceNotStorable")
                .with_source(source)
                .build()
        }
        error @ BaseWorkspaceError::Closed => {
            semantic_error("RunMat:Workspace:Closed", error.to_string())
        }
        error @ BaseWorkspaceError::BorrowConflict => {
            semantic_error("RunMat:Workspace:Busy", error.to_string())
        }
        error @ BaseWorkspaceError::FutureRevision { .. } => {
            semantic_error("RunMat:Workspace:RevisionMismatch", error.to_string())
        }
        error @ BaseWorkspaceError::RevisionExhausted => {
            semantic_error("RunMat:Workspace:RevisionExhausted", error.to_string())
        }
        error @ BaseWorkspaceError::PendingChanges => {
            semantic_error("RunMat:Workspace:PendingChanges", error.to_string())
        }
    }
}
