use runmat_value::Value;

use crate::context::RuntimeWorkspaceService;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

use super::{BaseWorkspaceError, BaseWorkspaceOwner, WorkspaceDelta};

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

impl RuntimeWorkspaceService for BaseWorkspaceService {
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

    fn base_workspace_owner(&self) -> Option<BaseWorkspaceOwner> {
        Some(self.owner())
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

pub(crate) fn base_workspace_error_to_runtime(error: BaseWorkspaceError) -> RuntimeError {
    match error {
        BaseWorkspaceError::Rooting(source) => {
            let message = format!("could not retain a base workspace value: {source}");
            crate::build_runtime_error(message)
                .with_identifier("RunMat:Workspace:Rooting")
                .with_source(source)
                .build()
        }
        error => {
            let identifier = match &error {
                BaseWorkspaceError::Closed => "RunMat:Workspace:Closed",
                BaseWorkspaceError::BorrowConflict => "RunMat:Workspace:Busy",
                BaseWorkspaceError::FutureRevision { .. } => "RunMat:Workspace:RevisionMismatch",
                BaseWorkspaceError::RevisionExhausted => "RunMat:Workspace:RevisionExhausted",
                BaseWorkspaceError::TransientValue(_) => "RunMat:TransientSequenceNotStorable",
                BaseWorkspaceError::PendingChanges => "RunMat:Workspace:PendingChanges",
                BaseWorkspaceError::Rooting(_) => unreachable!("handled above"),
            };
            semantic_error(identifier, error.to_string())
        }
    }
}
