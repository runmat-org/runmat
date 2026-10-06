use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use runmat_value::Value;

use crate::context::RuntimeWorkspaceService;
use crate::RuntimeError;

use super::{
    base_workspace_error_to_runtime, BaseWorkspaceCommit, BaseWorkspaceError,
    BaseWorkspaceProjection, BaseWorkspaceReconciliation, WorkspaceRevision,
};

/// Exact current-workspace authority for one admitted executor invocation.
///
/// The service retains the projection and its managed-value roots for the
/// complete invocation lifetime. Language-visible mutations are staged in
/// order and may be published at executor safepoints without exposing the
/// session's close authority or replacing canonical state from a snapshot.
#[derive(Clone, Debug)]
pub struct BaseWorkspaceProjectionService {
    projection: Rc<RefCell<BaseWorkspaceProjection>>,
}

impl BaseWorkspaceProjectionService {
    pub fn new(projection: BaseWorkspaceProjection) -> Self {
        Self {
            projection: Rc::new(RefCell::new(projection)),
        }
    }

    pub fn revision(&self) -> Result<WorkspaceRevision, RuntimeError> {
        Ok(self.borrow()?.revision())
    }

    pub fn has_pending_changes(&self) -> Result<bool, RuntimeError> {
        Ok(self.borrow()?.has_pending_changes())
    }

    /// Publish all ordered mutations admitted since the preceding safepoint.
    pub fn publish(&self) -> Result<BaseWorkspaceCommit, RuntimeError> {
        self.borrow_mut()?
            .publish()
            .map_err(base_workspace_error_to_runtime)
    }

    /// Refresh a clean invocation projection from canonical session state.
    pub fn reconcile(&self) -> Result<BaseWorkspaceReconciliation, RuntimeError> {
        self.borrow_mut()?
            .reconcile()
            .map_err(base_workspace_error_to_runtime)
    }

    /// Admit a new external or reentrant effect against this exact workspace.
    pub fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        self.borrow()?
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)
    }

    fn borrow(&self) -> Result<Ref<'_, BaseWorkspaceProjection>, RuntimeError> {
        self.projection
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))
    }

    fn borrow_mut(&self) -> Result<RefMut<'_, BaseWorkspaceProjection>, RuntimeError> {
        self.projection
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))
    }

    fn with_live_projection<T>(
        &self,
        operation: impl FnOnce(&BaseWorkspaceProjection) -> T,
    ) -> Result<T, RuntimeError> {
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
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        let projection = self.projection.try_borrow().ok()?;
        for (_, value) in projection.iter() {
            visitor(value)?;
        }
        Some(())
    }

    // These infallible methods implement the compatibility surface. Exact
    // executor paths use the checked methods below and retain typed failures.
    fn lookup(&self, name: &str) -> Option<Value> {
        self.projection.try_borrow().ok()?.lookup(name).cloned()
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        self.projection
            .try_borrow()
            .map(|projection| {
                projection
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
