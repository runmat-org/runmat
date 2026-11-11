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
    invocation_projection: Option<std::rc::Rc<super::BaseWorkspaceProjectionService>>,
}

impl BaseWorkspaceService {
    pub fn new(owner: BaseWorkspaceOwner) -> Self {
        Self {
            owner,
            invocation_projection: None,
        }
    }

    /// Bind base effects to this invocation's already-observed canonical
    /// projection, never to an unrelated source or a second workspace owner.
    pub fn with_invocation_projection(
        mut self,
        projection: std::rc::Rc<super::BaseWorkspaceProjectionService>,
    ) -> Result<Self, RuntimeError> {
        if !projection.canonical_owner()?.same_owner(&self.owner) {
            return Err(semantic_error(
                "RunMat:Workspace:ForeignProjection",
                "base projection belongs to a different canonical owner",
            ));
        }
        self.invocation_projection = Some(projection);
        Ok(self)
    }

    pub fn invocation_projection(
        &self,
    ) -> Option<std::rc::Rc<super::BaseWorkspaceProjectionService>> {
        self.invocation_projection.clone()
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

    fn invocation_projection(&self) -> Option<std::rc::Rc<super::BaseWorkspaceProjectionService>> {
        BaseWorkspaceService::invocation_projection(self)
    }

    fn assign_logical_ref_with_context(
        &self,
        variables: &crate::workspace::session::SessionVariableAuthority,
        context: &crate::context::RuntimeContext,
        name: &str,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        let Some(projection) = self.invocation_projection.as_ref() else {
            return super::assign_logical_base_value_ref_with_context(
                &self.owner,
                variables,
                context,
                name,
                value,
            );
        };
        crate::execution::validate_storable_value(value)?;
        if let Some(target) = projection.global_target(name)? {
            return match target {
                crate::workspace::GlobalBindingTarget::Named(target_name) => {
                    variables.store_global_ref_with_context(context, target_name.as_str(), value)
                }
                crate::workspace::GlobalBindingTarget::LegacySlot(slot) => {
                    variables.store_global_slot_ref_with_context(context, slot, value)
                }
            };
        }
        crate::context::RuntimeWorkspaceService::assign_ref_with_context(
            projection.as_ref(),
            context,
            name,
            value,
        )?;
        projection.publish_with_context(context).map(|_| ())
    }
}

impl RuntimeWorkspaceService for BaseWorkspaceService {
    fn prepare_workspace_completion_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        identity: crate::workspace::WorkspaceCompletionIdentity,
        source: &dyn crate::workspace::WorkspaceCompletionSource,
    ) -> Result<crate::workspace::PreparedWorkspaceCompletion, RuntimeError> {
        crate::workspace::PreparedWorkspaceCompletion::for_direct_owner(
            &self.owner,
            context,
            identity,
            source,
        )
    }

    fn capture_workspace_generation_with_context(
        &self,
        context: &crate::context::RuntimeContext,
    ) -> Result<Option<crate::workspace::LogicalWorkspaceGeneration>, RuntimeError> {
        crate::workspace::LogicalWorkspaceGeneration::capture(
            context,
            &self.owner,
            &context.session_variable_authority(),
        )
        .map(Some)
    }

    fn capture_dynamic_workspace_target_with_context(
        self: std::rc::Rc<Self>,
        context: &crate::context::RuntimeContext,
    ) -> Result<Option<crate::workspace::DynamicWorkspaceTargetCapture>, RuntimeError> {
        let identity = self
            .capture_workspace_completion_identity_with_context(context)?
            .ok_or_else(|| {
                semantic_error(
                    "RunMat:Workspace:ExactCompletionUnavailable",
                    "base workspace did not issue transactional completion authority",
                )
            })?;
        let generation = crate::workspace::LogicalWorkspaceGeneration::capture(
            context,
            &self.owner,
            &context.session_variable_authority(),
        )?;
        let issuer: std::rc::Rc<dyn RuntimeWorkspaceService> = self;
        Ok(Some(
            crate::workspace::DynamicWorkspaceTargetCapture::issued(
                context.local_identity(),
                identity,
                generation,
                issuer,
            ),
        ))
    }

    fn capture_workspace_completion_identity_with_context(
        &self,
        context: &crate::context::RuntimeContext,
    ) -> Result<Option<crate::workspace::WorkspaceCompletionIdentity>, RuntimeError> {
        let plan =
            crate::context::host_allocation::HostAllocationPlan::new(0, 0).map_err(|cause| {
                super::projection_service::workspace_admission_error(context, cause)
            })?;
        if let Some(receipt) =
            crate::context::host_allocation::HostAllocationReceipt::admit(Some(context), plan)
                .map_err(|cause| {
                    super::projection_service::workspace_admission_error(context, cause)
                })?
        {
            receipt.checkpoint(context).map_err(|cause| {
                super::projection_service::workspace_admission_error(context, cause)
            })?;
        }
        self.owner
            .workspace_completion_identity()
            .map(Some)
            .map_err(base_workspace_error_to_runtime)
    }

    fn apply_workspace_completion_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        identity: crate::workspace::WorkspaceCompletionIdentity,
        source: &dyn crate::workspace::WorkspaceCompletionSource,
    ) -> Result<(), RuntimeError> {
        self.owner
            .apply_workspace_completion(context, identity, source)
            .map_err(base_workspace_error_to_runtime)
    }

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

    fn publish_admitted_mutations_with_context(
        &self,
        context: &crate::context::RuntimeContext,
    ) -> Result<(), RuntimeError> {
        match &self.invocation_projection {
            Some(projection) => projection.publish_admitted_mutations_with_context(context),
            None => self.ensure_mutation_authority_with_context(context),
        }
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

    fn bind_base_global_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        name: &str,
    ) -> Result<(), RuntimeError> {
        let service = super::BaseWorkspaceProjectionService::new_for_context(
            self.owner
                .begin_projection()
                .map_err(base_workspace_error_to_runtime)?,
            context,
        );
        RuntimeWorkspaceService::bind_base_global_with_context(&service, context, name)?;
        service.publish_with_context(context).map(|_| ())
    }

    fn bind_base_global_target_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        name: &str,
        target: &crate::workspace::GlobalBindingTarget,
    ) -> Result<(), RuntimeError> {
        let service = super::BaseWorkspaceProjectionService::new_for_context(
            self.owner
                .begin_projection()
                .map_err(base_workspace_error_to_runtime)?,
            context,
        );
        RuntimeWorkspaceService::bind_base_global_target_with_context(
            &service, context, name, target,
        )?;
        service.publish_with_context(context).map(|_| ())
    }

    fn unbind_base_global(&self, name: &str) -> Result<(), RuntimeError> {
        let mut delta = WorkspaceDelta::new();
        delta.unbind_global(name);
        self.apply(delta)
    }

    fn unbind_base_global_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        name: &str,
    ) -> Result<(), RuntimeError> {
        let service = super::BaseWorkspaceProjectionService::new_for_context(
            self.owner
                .begin_projection()
                .map_err(base_workspace_error_to_runtime)?,
            context,
        );
        RuntimeWorkspaceService::unbind_base_global_with_context(&service, context, name)?;
        service.publish_with_context(context).map(|_| ())
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
        BaseWorkspaceError::HostAllocation(source) => {
            let identifier = format!("RunMat:Workspace:{}", source.runtime_identifier_suffix());
            crate::context::host_allocation::host_allocation_runtime_error(
                None,
                &identifier,
                format!("could not admit base-workspace storage: {source}"),
                source,
            )
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
        error @ BaseWorkspaceError::CompletionConflict => {
            semantic_error("RunMat:Workspace:CompletionConflict", error.to_string())
        }
        error @ BaseWorkspaceError::RevisionExhausted => {
            semantic_error("RunMat:Workspace:RevisionExhausted", error.to_string())
        }
        error @ BaseWorkspaceError::PendingChanges => {
            semantic_error("RunMat:Workspace:PendingChanges", error.to_string())
        }
        error @ BaseWorkspaceError::PrivateProjectionRequired => semantic_error(
            "RunMat:Workspace:PrivateProjectionRequired",
            error.to_string(),
        ),
    }
}
