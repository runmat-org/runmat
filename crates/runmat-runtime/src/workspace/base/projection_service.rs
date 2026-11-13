use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use runmat_value::Value;

use crate::context::RuntimeWorkspaceService;
use crate::context::{
    host_allocation::{
        HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection, WorkspaceSlotReadLease};
use crate::RuntimeError;

use super::{
    base_workspace_error_to_runtime, BaseWorkspaceCommit, BaseWorkspaceError,
    BaseWorkspaceProjection, BaseWorkspaceReconciliation, PreparedWorkspaceWrite, WorkspaceDelta,
    WorkspaceRevision,
};

mod completion_token;
mod detach;
mod private_live;
mod prospective;
mod publication;
mod terminal_publication;
pub use completion_token::PreparedWorkspaceCompletion;
use publication::{PreparedPublishedWorkspaceCapacity, PublishedWorkspaceEffect};
pub use publication::{
    WorkspacePublicationBatch, WorkspacePublicationMutation, WorkspacePublicationStamp,
};
pub use publication::{
    WorkspacePublicationObserver, WorkspacePublicationOrigin, WorkspacePublicationProtocol,
    WorkspacePublicationToken,
};
pub use terminal_publication::WorkspaceProjectionPublication;

/// Exact current-workspace authority for one admitted executor invocation.
///
/// The service retains the projection and its managed-value roots for the
/// complete invocation lifetime. Language-visible mutations are staged in
/// order and may be published at executor safepoints without exposing the
/// session's close authority or replacing canonical state from a snapshot.
#[derive(Clone, Debug)]
pub struct BaseWorkspaceProjectionService {
    state: Rc<RefCell<ProjectionServiceState>>,
    publication_observer: Option<WorkspacePublicationObserver>,
}

#[derive(Debug)]
struct ProjectionServiceState {
    projection: BaseWorkspaceProjection,
    published_effect: PublishedWorkspaceEffect,
    runtime: Option<crate::context::RuntimeContextLocalId>,
    last_publication_stamp: Option<WorkspacePublicationStamp>,
}

impl BaseWorkspaceProjectionService {
    pub fn canonical_owner(&self) -> Result<super::BaseWorkspaceOwner, RuntimeError> {
        self.borrow()?
            .canonical_owner()
            .map_err(super::base_workspace_error_to_runtime)
    }

    pub fn global_target(
        &self,
        name: &str,
    ) -> Result<Option<crate::workspace::GlobalBindingTarget>, RuntimeError> {
        Ok(self.borrow()?.global_target(name).cloned())
    }

    pub fn new(projection: BaseWorkspaceProjection) -> Self {
        Self {
            state: Rc::new(RefCell::new(ProjectionServiceState {
                projection,
                published_effect: PublishedWorkspaceEffect::default(),
                runtime: None,
                last_publication_stamp: None,
            })),
            publication_observer: None,
        }
    }

    /// Bind this invocation's mutation authority without retaining its context
    /// or creating a service/context ownership cycle.
    pub fn new_for_context(projection: BaseWorkspaceProjection, context: &RuntimeContext) -> Self {
        Self {
            state: Rc::new(RefCell::new(ProjectionServiceState {
                projection,
                published_effect: PublishedWorkspaceEffect::default(),
                runtime: Some(context.local_identity()),
                last_publication_stamp: None,
            })),
            publication_observer: None,
        }
    }

    pub fn with_publication_observer(mut self, observer: WorkspacePublicationObserver) -> Self {
        self.publication_observer = Some(observer);
        self
    }

    pub fn revision(&self) -> Result<WorkspaceRevision, RuntimeError> {
        Ok(self.borrow()?.revision())
    }

    /// A clean projection has no private suffix to preserve. Report the
    /// canonical owner's current revision without publishing or mutating the
    /// projection; a dirty projection still reports its own live revision.
    fn live_read_revision(&self) -> Result<WorkspaceRevision, RuntimeError> {
        let projection = self.borrow()?;
        if projection.has_pending_changes() {
            return Ok(projection.revision());
        }
        let owner = projection
            .canonical_owner()
            .map_err(base_workspace_error_to_runtime)?;
        drop(projection);
        owner.revision().map_err(base_workspace_error_to_runtime)
    }

    pub fn has_pending_changes(&self) -> Result<bool, RuntimeError> {
        Ok(self.borrow()?.has_pending_changes())
    }

    pub fn lookup(&self, name: &str) -> Result<Option<Value>, RuntimeError> {
        self.with_live_projection(|projection| projection.lookup(name).cloned())
    }

    pub fn snapshot(&self) -> Result<Vec<(String, Value)>, RuntimeError> {
        if self.is_private()? {
            self.ensure_effect_authority()?;
            return self.private_snapshot();
        }
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
        if self.is_private()? {
            return self.private_global_unbounded(name);
        }
        self.ensure_unbounded_compatibility()?;
        self.prepare_unbounded_appended_effect(1)?;
        self.with_live_publication_mut(|projection| {
            let target = crate::workspace::GlobalBindingTarget::Named(
                crate::workspace::WorkspaceName::unbounded(name),
            );
            Ok((
                (),
                projection.namespace_unbounded_retired(
                    name,
                    super::storage::WorkspaceNamespaceAction::BindGlobal,
                    Some(&target),
                ),
            ))
        })
    }

    pub fn unbind_global(&self, name: &str) -> Result<(), RuntimeError> {
        if self.is_private()? {
            return self.private_remove(None, name);
        }
        self.ensure_unbounded_compatibility()?;
        self.prepare_unbounded_appended_effect(1)?;
        self.with_live_publication_mut(|projection| {
            Ok((
                (),
                projection.namespace_unbounded_retired(
                    name,
                    super::storage::WorkspaceNamespaceAction::UnbindGlobal,
                    None,
                ),
            ))
        })
    }

    /// Publish all ordered mutations admitted since the preceding safepoint.
    pub fn publish(&self) -> Result<BaseWorkspaceCommit, RuntimeError> {
        self.publish_internal(None)
    }

    /// Publish through this invocation's exact runtime authority.
    pub fn publish_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<BaseWorkspaceCommit, RuntimeError> {
        self.ensure_mutation_context(context)?;
        self.publish_internal(Some(context))
    }

    fn publish_internal(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<BaseWorkspaceCommit, RuntimeError> {
        if self.is_private()? {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:PrivatePublicationUnavailable",
                "private executor workspaces install each prepared mutation atomically",
            ));
        }
        {
            let state = self
                .state
                .try_borrow()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            if state.runtime.is_some() && context.is_none() {
                return Err(crate::runtime_error::semantic_error(
                    "RunMat:Workspace:PublicationContextRequired",
                    "context-bound workspace publication requires its live runtime context",
                ));
            }
        }
        {
            let mut state = self
                .state
                .try_borrow_mut()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            let upper_bound = state
                .projection
                .publication_upper_bound()
                .map_err(base_workspace_error_to_runtime)?;
            if !state.published_effect.can_observe(upper_bound) {
                if state.runtime.is_some() {
                    return Err(publication_preparation_error());
                }
                state
                    .published_effect
                    .prepare_unbounded_for_upper_bound(upper_bound)?;
            }
            let ProjectionServiceState {
                projection,
                published_effect,
                ..
            } = &mut *state;
            published_effect.prepare_publication_indices(
                projection
                    .pending_publication()
                    .map_err(base_workspace_error_to_runtime)?,
            );
        }
        let state = self
            .state
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let pending = state
            .projection
            .pending_publication()
            .map_err(base_workspace_error_to_runtime)?;
        let observer_preparation = self
            .publication_observer
            .as_ref()
            .map(|observer| {
                let batch = WorkspacePublicationBatch::for_changes(
                    pending,
                    &state.projection,
                    state.published_effect.publication_indices(),
                );
                observer.prepare(
                    context,
                    state
                        .last_publication_stamp
                        .map_or(batch, |stamp| batch.with_stamp(stamp)),
                )
            })
            .transpose();
        let mut observer_preparation = match observer_preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                drop(state);
                self.finish_publication_indices()?;
                return Err(error);
            }
        };
        if let Some(preparation) = &mut observer_preparation {
            if let Err(error) = preparation.checkpoint(context) {
                drop(state);
                self.finish_publication_indices()?;
                return Err(error);
            }
        }
        drop(state);
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let ProjectionServiceState {
            projection,
            published_effect,
            ..
        } = &mut *state;
        let effect = published_effect.take_canonical_effect(
            projection
                .pending_publication()
                .map_err(base_workspace_error_to_runtime)?,
        );
        let (commit, projection_retired) = match projection.publish_with_effect(effect, context) {
            Ok(result) => result,
            Err((error, effect)) => {
                state.published_effect.restore_canonical_effect(effect);
                state.published_effect.finish_publication();
                return Err(base_workspace_error_to_runtime(error));
            }
        };
        let retired = state.published_effect.observe(commit.changes());
        state.published_effect.finish_publication();
        state.last_publication_stamp = None;
        drop(state);
        if let Some(preparation) = observer_preparation {
            preparation.commit();
        }
        drop(projection_retired);
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
        if state.runtime.is_some() {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:PublicationContextRequired",
                "context-bound workspace drain requires its live runtime context",
            ));
        }
        Ok(state.published_effect.take())
    }

    /// Drain the backing prepared before the effects were published. A drain
    /// must not request fresh allocation after cancellation or another source
    /// error: the already-committed prefix still belongs to this invocation.
    pub fn take_published_changes_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<WorkspaceDelta, RuntimeError> {
        self.ensure_mutation_context(context)?;
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        Ok(state.published_effect.take())
    }

    /// Refresh a clean invocation projection from canonical session state.
    pub fn reconcile(&self) -> Result<BaseWorkspaceReconciliation, RuntimeError> {
        if self.is_private()? {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:PrivateReconciliationUnavailable",
                "private executor workspaces do not consume the public ordered journal",
            ));
        }
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
        if self.is_private()? {
            self.borrow()?
                .ensure_effect_authority()
                .map_err(base_workspace_error_to_runtime)?;
            return Ok(());
        }
        if self.has_pending_changes()? {
            self.publish()?;
        } else {
            self.reconcile()?;
        }
        Ok(())
    }

    fn synchronize_for_effect_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            HostAllocationReceipt::policy(Some(context))
                .map_err(|cause| workspace_admission_error(context, cause))?;
            self.borrow()?
                .ensure_effect_authority()
                .map_err(base_workspace_error_to_runtime)?;
            return Ok(());
        }
        if self.has_pending_changes()? {
            self.publish_with_context(context)?;
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

    fn capture_canonical_prefix(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        let owner = self
            .borrow()?
            .canonical_owner()
            .map_err(base_workspace_error_to_runtime)?;
        match context {
            Some(context) => owner.capture_stable_with_context(context),
            None => owner.capture_stable_unbounded(),
        }
        .map(|_| ())
        .map_err(base_workspace_error_to_runtime)
    }

    fn install_unbounded_compat(
        &self,
        prepared: super::projection::PreparedUnboundedProjectionMutation,
    ) -> Result<(), RuntimeError> {
        let admission = {
            let mut projection = self.borrow_mut()?;
            projection.reserve_unbounded_mutation_capacity(&prepared)
        };
        admission.map_err(base_workspace_error_to_runtime)?;
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        state
            .projection
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        let retired = state
            .projection
            .ordered_view_mut()
            .map_err(base_workspace_error_to_runtime)?
            .install_unbounded_mutation(prepared);
        drop(state);
        drop(retired);
        Ok(())
    }

    fn assign_unbounded_owned(
        &self,
        name: &str,
        value: Value,
        slot: Option<usize>,
        local: bool,
        coalesce: bool,
    ) -> Result<(), RuntimeError> {
        let entry = super::values::RootedWorkspaceEntry::new(value)
            .map_err(base_workspace_error_to_runtime)?;
        self.capture_canonical_prefix(None)?;
        let prepared = self
            .borrow()?
            .prepare_unbounded_assignment_entry_with_order(name, &entry, slot, local, coalesce)
            .map_err(base_workspace_error_to_runtime)?;
        self.install_unbounded_compat(prepared)
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

    fn with_live_projection_with_context<T>(
        &self,
        context: &RuntimeContext,
        operation: impl FnOnce(&BaseWorkspaceProjection) -> T,
    ) -> Result<T, RuntimeError> {
        self.synchronize_for_effect_with_context(context)?;
        let projection = self.borrow()?;
        projection
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        Ok(operation(&projection))
    }

    fn with_live_publication_mut<T, R>(
        &self,
        operation: impl FnOnce(&mut BaseWorkspaceProjection) -> Result<(T, R), BaseWorkspaceError>,
    ) -> Result<T, RuntimeError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        if state.runtime.is_some() {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:PublicationContextRequired",
                "context-bound workspace mutation requires its live runtime context",
            ));
        }
        state
            .projection
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        let (value, retired) =
            operation(&mut state.projection).map_err(base_workspace_error_to_runtime)?;
        drop(state);
        drop(retired);
        Ok(value)
    }

    fn ensure_mutation_context(&self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        if state.runtime == Some(context.local_identity()) {
            Ok(())
        } else {
            Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:MutationContextMismatch",
                "workspace mutation context does not match the projection invocation",
            ))
        }
    }

    fn ensure_unbounded_compatibility(&self) -> Result<(), RuntimeError> {
        if self
            .state
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?
            .runtime
            .is_some()
        {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:PublicationContextRequired",
                "context-bound workspace mutation requires its live runtime context",
            ));
        }
        Ok(())
    }

    fn stage_namespace_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
        action: super::storage::WorkspaceNamespaceAction,
        target: Option<&crate::workspace::GlobalBindingTarget>,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return match action {
                super::storage::WorkspaceNamespaceAction::BindGlobal => {
                    let default_target;
                    let target = match target {
                        Some(target) => target,
                        None => {
                            default_target = crate::workspace::GlobalBindingTarget::Named(
                                crate::workspace::WorkspaceName::unbounded(name),
                            );
                            &default_target
                        }
                    };
                    self.private_global(context, name, target)
                }
                super::storage::WorkspaceNamespaceAction::UnbindGlobal
                | super::storage::WorkspaceNamespaceAction::Remove => {
                    self.private_remove(Some(context), name)
                }
            };
        }
        self.capture_canonical_prefix(Some(context))?;
        let projection = self.borrow()?;
        let plan = projection
            .plan_namespace_mutation(name)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let publication_upper_bound = projection
            .publication_upper_bound_after_appended_change()
            .map_err(base_workspace_error_to_runtime)?;
        drop(projection);
        let publication_retained =
            self.published_effect_retained_bytes(context, publication_upper_bound)?;
        let target_retained = target
            .map(crate::workspace::GlobalBindingTarget::additional_retained_bytes)
            .transpose()
            .map_err(|cause| workspace_admission_error(context, cause))?
            .unwrap_or(0);
        let retained = plan
            .retained_bytes()
            .checked_add(publication_retained)
            .and_then(|bytes| bytes.checked_add(target_retained))
            .ok_or_else(|| workspace_admission_error(context, HostAllocationError::Overflow))?;
        let allocation_plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), allocation_plan)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            // `None` is the selected unbounded host policy, not an admission
            // failure. Preserve the compatibility representation explicitly.
            self.prepare_unbounded_published_effect(publication_upper_bound)?;
            let prepared = self
                .borrow()?
                .prepare_unbounded_namespace(
                    name,
                    action,
                    target,
                    super::delta::WorkspaceMutationOrigin::Executor,
                )
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        };
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared_name = super::WorkspaceName::prepare(name, &permit).map_err(|cause| {
            workspace_admission_error(context, HostAllocationError::from(cause))
        })?;
        let prepared_target = target
            .map(|target| target.prepare_retained(&permit))
            .transpose()
            .map_err(|cause| {
                workspace_admission_error(context, HostAllocationError::from(cause))
            })?;
        let publication =
            self.prepare_published_effect(context, publication_upper_bound, &permit)?;
        receipt
            .checkpoint(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared = self
            .borrow()?
            .prepare_namespace_mutation(
                context,
                plan,
                prepared_name,
                action,
                prepared_target,
                &permit,
            )
            .map_err(base_workspace_error_to_runtime)?;
        self.install_published_effect(publication)?;
        self.install_observed_mutation(context, prepared)
    }

    fn stage_clear_with_context(&self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        self.capture_canonical_prefix(Some(context))?;
        let projection = self.borrow()?;
        let retained = projection
            .plan_clear()
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let publication_upper_bound = projection
            .publication_upper_bound_after_appended_change()
            .map_err(base_workspace_error_to_runtime)?;
        drop(projection);
        let retained = retained
            .checked_add(self.published_effect_retained_bytes(context, publication_upper_bound)?)
            .ok_or_else(|| workspace_admission_error(context, HostAllocationError::Overflow))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            self.prepare_unbounded_published_effect(publication_upper_bound)?;
            let prepared = self
                .borrow()?
                .prepare_unbounded_clear(super::delta::WorkspaceMutationOrigin::Executor)
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        };
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let publication =
            self.prepare_published_effect(context, publication_upper_bound, &permit)?;
        receipt
            .checkpoint(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared = self
            .borrow()?
            .prepare_clear_mutation(&permit)
            .map_err(base_workspace_error_to_runtime)?;
        self.install_published_effect(publication)?;
        self.install_observed_mutation(context, prepared)
    }

    fn published_effect_retained_bytes(
        &self,
        context: &RuntimeContext,
        upper_bound: usize,
    ) -> Result<u64, RuntimeError> {
        self.state
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?
            .published_effect
            .retained_bytes_for_upper_bound(upper_bound)
            .map_err(|cause| workspace_admission_error(context, cause))
    }

    fn prepare_published_effect(
        &self,
        context: &RuntimeContext,
        upper_bound: usize,
        permit: &crate::context::host_allocation::HostOutputPermit,
    ) -> Result<PreparedPublishedWorkspaceCapacity, RuntimeError> {
        self.state
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?
            .published_effect
            .prepare_for_upper_bound(upper_bound, permit)
            .map_err(|cause| workspace_admission_error(context, cause))
    }

    fn install_published_effect(
        &self,
        prepared: PreparedPublishedWorkspaceCapacity,
    ) -> Result<(), RuntimeError> {
        self.state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?
            .published_effect
            .install_prepared_capacity(prepared);
        Ok(())
    }

    fn finish_publication_indices(&self) -> Result<(), RuntimeError> {
        self.state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?
            .published_effect
            .finish_publication();
        Ok(())
    }

    fn prepare_unbounded_published_effect(&self, upper_bound: usize) -> Result<(), RuntimeError> {
        self.state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?
            .published_effect
            .prepare_unbounded_for_upper_bound(upper_bound)
    }

    fn prepare_unbounded_appended_effect(&self, additional: usize) -> Result<(), RuntimeError> {
        let upper_bound = self
            .borrow()?
            .publication_upper_bound()
            .map_err(base_workspace_error_to_runtime)?
            .checked_add(additional)
            .ok_or_else(publication_preparation_error)?;
        self.prepare_unbounded_published_effect(upper_bound)
    }

    fn prepare_unbounded_executor_effect(&self, name: &str) -> Result<(), RuntimeError> {
        let upper_bound = self
            .borrow()?
            .publication_upper_bound_after_executor(name)
            .map_err(base_workspace_error_to_runtime)?;
        self.prepare_unbounded_published_effect(upper_bound)
    }

    /// A borrowed value without an exact managed write plan may use the
    /// compatibility owner only under the explicit unbounded policy. Refuse a
    /// finite invocation before cloning or publishing that value.
    fn require_unbounded_borrowed_write(
        &self,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        if HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?
            == HostAllocationPolicy::Managed
        {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:ExactWriteAdmissionUnavailable",
                "finite workspace assignment requires an admitted value owner",
            ));
        }
        Ok(())
    }

    fn detach_selected_internal(
        &self,
        context: Option<&RuntimeContext>,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        match context {
            Some(context) => self.synchronize_for_effect_with_context(context)?,
            None => {
                if self
                    .state
                    .try_borrow()
                    .map_err(|_| {
                        base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict)
                    })?
                    .runtime
                    .is_some()
                {
                    return Err(crate::runtime_error::semantic_error(
                        "RunMat:Workspace:PublicationContextRequired",
                        "context-bound workspace detachment requires its live runtime context",
                    ));
                }
                self.synchronize_for_effect()?;
            }
        }
        self.capture_canonical_prefix(context)?;
        let workspace_len = self.borrow()?.len();
        let selection_bound = match selection {
            WorkspaceDetachSelection::All => 1,
            WorkspaceDetachSelection::Names(names) => names.len(),
            WorkspaceDetachSelection::Except(_) => workspace_len,
        };
        let detach_upper_bound = workspace_len
            .max(selection_bound)
            .checked_add(1)
            .ok_or_else(publication_preparation_error)?;
        self.prepare_unbounded_published_effect(detach_upper_bound)?;
        let detached = self.logical_detach_values(context, selection)?;
        let delta = self
            .borrow()?
            .prepare_detach_selected(selection)
            .map_err(base_workspace_error_to_runtime)?;
        let completion = self
            .borrow()?
            .prepare_detachment_completion(delta.clone_unbounded())
            .map_err(base_workspace_error_to_runtime)?;
        let stamp = {
            let mut state = self
                .state
                .try_borrow_mut()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            let ProjectionServiceState {
                projection,
                published_effect,
                ..
            } = &mut *state;
            let view = completion
                .publication_view(projection)
                .map_err(base_workspace_error_to_runtime)?;
            published_effect.prepare_prospective_indices(&view);
            WorkspacePublicationStamp::issued(
                Rc::as_ptr(&self.state) as usize,
                view.revision(),
                view.before_generation(),
                view.generation(),
            )
        };
        let state = self
            .state
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let view = completion
            .publication_view(&state.projection)
            .map_err(base_workspace_error_to_runtime)?;
        let observer_preparation = self
            .publication_observer
            .as_ref()
            .map(|observer| {
                observer.prepare_completion_publication(
                    context,
                    WorkspacePublicationBatch::prospective(
                        &view,
                        state.published_effect.publication_indices(),
                    )
                    .with_stamp(stamp),
                )
            })
            .transpose();
        let mut observer_preparation = match observer_preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                drop(state);
                self.finish_publication_indices()?;
                return Err(error);
            }
        };
        if let Some(preparation) = &mut observer_preparation {
            if let Err(error) = preparation.checkpoint(context) {
                drop(state);
                self.finish_publication_indices()?;
                return Err(error);
            }
        }
        drop(state);
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let ProjectionServiceState {
            projection,
            published_effect,
            ..
        } = &mut *state;
        let mut ordered = projection
            .ordered_view_mut()
            .map_err(base_workspace_error_to_runtime)?;
        let installed_retired = ordered.install_completion(completion);
        let effect = published_effect.take_canonical_effect(ordered.pending());
        let (commit, publication_retired) = match ordered.publish_with_effect(effect, context) {
            Ok(result) => result,
            Err((error, effect)) => {
                published_effect.restore_canonical_effect(effect);
                published_effect.finish_publication();
                let staged = ordered.rollback_completion(installed_retired);
                drop(state);
                drop(staged);
                return Err(base_workspace_error_to_runtime(error));
            }
        };
        let retired = state.published_effect.observe(commit.changes());
        state.published_effect.finish_publication();
        state.last_publication_stamp = None;
        drop(state);
        if let Some(preparation) = observer_preparation {
            preparation.commit();
        }
        drop(publication_retired);
        drop(installed_retired);
        drop(retired);
        Ok(detached)
    }
}

impl RuntimeWorkspaceService for BaseWorkspaceProjectionService {
    fn prepare_workspace_completion_with_context(
        &self,
        context: &RuntimeContext,
        identity: crate::workspace::WorkspaceCompletionIdentity,
        source: &dyn crate::workspace::WorkspaceCompletionSource,
    ) -> Result<crate::workspace::PreparedWorkspaceCompletion, RuntimeError> {
        if self.is_private()? {
            return PreparedWorkspaceCompletion::for_private_projection(
                self, context, identity, source,
            );
        }
        PreparedWorkspaceCompletion::for_projection(self, context, identity, source)
    }

    fn capture_workspace_generation_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<Option<crate::workspace::LogicalWorkspaceGeneration>, RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return self.private_generation(context).map(Some);
        }
        let state = self.borrow()?;
        let snapshot = if state.has_pending_changes() {
            let policy = HostAllocationReceipt::policy(Some(context))
                .map_err(|cause| workspace_admission_error(context, cause))?;
            match policy {
                HostAllocationPolicy::Managed => state.retained_live_snapshot(),
                HostAllocationPolicy::Unbounded => state.retained_live_snapshot_unbounded(),
            }
            .map_err(base_workspace_error_to_runtime)?
        } else {
            let owner = state
                .canonical_owner()
                .map_err(base_workspace_error_to_runtime)?;
            drop(state);
            owner
                .capture_stable_with_context(context)
                .map_err(base_workspace_error_to_runtime)?
        };
        crate::workspace::LogicalWorkspaceGeneration::from_base_snapshot(
            context,
            snapshot,
            &context.session_variable_authority(),
        )
        .map(Some)
    }

    fn capture_dynamic_workspace_target_with_context(
        self: Rc<Self>,
        context: &RuntimeContext,
    ) -> Result<Option<crate::workspace::DynamicWorkspaceTargetCapture>, RuntimeError> {
        let identity = self
            .capture_workspace_completion_identity_with_context(context)?
            .ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "RunMat:Workspace:ExactCompletionUnavailable",
                    "workspace projection did not issue transactional completion authority",
                )
            })?;
        let generation = if self.is_private()? {
            self.private_generation(context)?
        } else {
            let owner = self
                .borrow()?
                .canonical_owner()
                .map_err(base_workspace_error_to_runtime)?;
            crate::workspace::LogicalWorkspaceGeneration::capture(
                context,
                &owner,
                &context.session_variable_authority(),
            )?
        };
        let issuer: Rc<dyn RuntimeWorkspaceService> = self;
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
        context: &RuntimeContext,
    ) -> Result<Option<crate::workspace::WorkspaceCompletionIdentity>, RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            let _ = self.private_generation(context)?;
            let state = self.borrow()?;
            let storage = state.private_storage().ok_or_else(|| {
                base_workspace_error_to_runtime(BaseWorkspaceError::PrivateProjectionRequired)
            })?;
            return Ok(Some(crate::workspace::WorkspaceCompletionIdentity::issued(
                state.revision(),
                storage.epoch(),
                Rc::as_ptr(&self.state) as usize,
            )));
        }
        let plan = HostAllocationPlan::new(0, 0)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        if let Some(receipt) = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| workspace_admission_error(context, cause))?
        {
            receipt
                .checkpoint(context)
                .map_err(|cause| workspace_admission_error(context, cause))?;
        }
        // Reads made by the child may synchronize completed executor writes.
        // Establish that prefix before issuing its mutation identity, so a
        // read alone cannot invalidate the unchanged target.
        self.synchronize_for_effect_with_context(context)?;
        self.capture_canonical_prefix(Some(context))?;
        let state = self.borrow()?;
        // Final-publication observers run outside the service borrow. If one
        // staged another write, refuse this capture rather than issuing a
        // token that the child's first contextual read would invalidate.
        if state.has_pending_changes() {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:CompletionConflict",
                "workspace changed while capturing transactional completion",
            ));
        }
        state
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        Ok(Some(crate::workspace::WorkspaceCompletionIdentity::issued(
            state.revision(),
            state.pending_generation(),
            Rc::as_ptr(&self.state) as usize,
        )))
    }

    fn apply_workspace_completion_with_context(
        &self,
        context: &RuntimeContext,
        identity: crate::workspace::WorkspaceCompletionIdentity,
        source: &dyn crate::workspace::WorkspaceCompletionSource,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return self.private_apply_completion(context, identity, source);
        }
        let issuer = Rc::as_ptr(&self.state) as usize;
        let validate_identity = |projection: &BaseWorkspaceProjection| {
            if identity.issuer() != issuer
                || identity.revision() != projection.revision()
                || identity.pending_generation() != projection.pending_generation()
            {
                return Err(crate::runtime_error::semantic_error(
                    "RunMat:Workspace:CompletionConflict",
                    "workspace changed after transactional completion capture",
                ));
            }
            projection
                .ensure_effect_authority()
                .map_err(base_workspace_error_to_runtime)
        };
        if source.is_empty() {
            let state = self.borrow()?;
            validate_identity(&state)?;
            drop(state);
            let plan = HostAllocationPlan::new(0, 0)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            if let Some(receipt) = HostAllocationReceipt::admit(Some(context), plan)
                .map_err(|cause| workspace_admission_error(context, cause))?
            {
                receipt
                    .checkpoint(context)
                    .map_err(|cause| workspace_admission_error(context, cause))?;
            }
            return Ok(());
        }
        self.capture_canonical_prefix(Some(context))?;
        if HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?
            == HostAllocationPolicy::Unbounded
        {
            let state = self.borrow()?;
            validate_identity(&state)?;
            let completion = state
                .prepare_completion_unbounded(source)
                .map_err(base_workspace_error_to_runtime)?;
            let publication_upper_bound = state
                .publication_upper_bound()
                .map_err(base_workspace_error_to_runtime)?
                .checked_add(source.len())
                .ok_or_else(publication_preparation_error)?;
            drop(state);
            let mut state = self
                .state
                .try_borrow_mut()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            validate_identity(&state.projection)?;
            state
                .published_effect
                .prepare_unbounded_for_upper_bound(publication_upper_bound)?;
            drop(state);
            return self.install_observed_completion(context, completion);
        }
        let state = self.borrow()?;
        validate_identity(&state)?;
        let completion_retained = state
            .completion_retained_bytes(source)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let publication_upper_bound = state
            .publication_upper_bound()
            .map_err(base_workspace_error_to_runtime)?
            .checked_add(source.len())
            .ok_or_else(publication_preparation_error)?;
        drop(state);
        let publication_retained =
            self.published_effect_retained_bytes(context, publication_upper_bound)?;
        let retained = completion_retained
            .checked_add(publication_retained)
            .ok_or_else(|| workspace_admission_error(context, HostAllocationError::Overflow))?;
        let allocation_plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), allocation_plan)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            let state = self.borrow()?;
            validate_identity(&state)?;
            let completion = state
                .prepare_completion_unbounded(source)
                .map_err(base_workspace_error_to_runtime)?;
            drop(state);
            let mut state = self
                .state
                .try_borrow_mut()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            validate_identity(&state.projection)?;
            state
                .published_effect
                .prepare_unbounded_for_upper_bound(publication_upper_bound)?;
            drop(state);
            return self.install_observed_completion(context, completion);
        };
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let state = self.borrow()?;
        validate_identity(&state)?;
        let completion = state
            .prepare_completion(context, source, &permit)
            .map_err(base_workspace_error_to_runtime)?;
        drop(state);
        let publication =
            self.prepare_published_effect(context, publication_upper_bound, &permit)?;
        receipt
            .checkpoint(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        validate_identity(&state.projection)?;
        state
            .published_effect
            .install_prepared_capacity(publication);
        drop(state);
        self.install_observed_completion(context, completion)
    }

    fn detach_selected(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        if self.is_private()? {
            return self.private_detach_unbounded(None, selection);
        }
        self.detach_selected_internal(None, selection)
    }

    fn detach_selected_with_context(
        &self,
        context: &RuntimeContext,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return self.private_detach_unbounded(Some(context), selection);
        }
        if HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?
            == HostAllocationPolicy::Managed
        {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:ExactDetachAdmissionUnavailable",
                "finite workspace detachment requires admitted output and root storage",
            ));
        }
        self.detach_selected_internal(Some(context), selection)
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
        if self.is_private().unwrap_or(false) {
            return self.private_snapshot().unwrap_or_default();
        }
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

    fn try_lookup_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
    ) -> Result<Option<Value>, RuntimeError> {
        self.with_live_projection_with_context(context, |projection| {
            projection.lookup(name).cloned()
        })
    }

    fn try_snapshot(&self) -> Result<Vec<(String, Value)>, RuntimeError> {
        if self.is_private()? {
            self.ensure_effect_authority()?;
            return self.private_snapshot();
        }
        self.with_live_projection(|projection| {
            projection
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect()
        })
    }

    fn try_snapshot_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<Vec<(String, Value)>, RuntimeError> {
        if self.is_private()? {
            self.ensure_effect_authority_with_context(context)?;
            return self.private_snapshot();
        }
        self.with_live_projection_with_context(context, |projection| {
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

    fn try_global_names_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<Vec<String>, RuntimeError> {
        self.synchronize_for_effect_with_context(context)?;
        self.borrow()?
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        Ok(super::super::session::global_names())
    }

    fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        BaseWorkspaceProjectionService::ensure_effect_authority(self)
    }

    fn ensure_effect_authority_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.synchronize_for_effect_with_context(context)?;
        self.borrow()?
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)
    }

    fn ensure_mutation_authority(&self) -> Result<(), RuntimeError> {
        self.borrow()?
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)
    }

    fn ensure_mutation_authority_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        self.ensure_mutation_authority()
    }

    fn publish_admitted_mutations_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_authority_with_context(context)?;
        if !self.is_private()? && self.has_pending_changes()? {
            self.publish_with_context(context)?;
        }
        Ok(())
    }

    fn workspace_revision(&self) -> Result<Option<WorkspaceRevision>, RuntimeError> {
        self.live_read_revision().map(Some)
    }

    fn workspace_revision_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<Option<WorkspaceRevision>, RuntimeError> {
        self.ensure_mutation_context(context)?;
        HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;
        self.live_read_revision().map(Some)
    }

    fn base_global_binding_names(&self) -> Result<Vec<String>, RuntimeError> {
        if self.is_private()? {
            self.ensure_effect_authority()?;
            return self.private_global_names();
        }
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

    fn bind_base_global_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.stage_namespace_with_context(
            context,
            name,
            super::storage::WorkspaceNamespaceAction::BindGlobal,
            None,
        )
    }

    fn bind_base_global_target_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
        target: &crate::workspace::GlobalBindingTarget,
    ) -> Result<(), RuntimeError> {
        self.stage_namespace_with_context(
            context,
            name,
            super::storage::WorkspaceNamespaceAction::BindGlobal,
            Some(target),
        )
    }

    fn unbind_base_global(&self, name: &str) -> Result<(), RuntimeError> {
        self.unbind_global(name)
    }

    fn unbind_base_global_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.stage_namespace_with_context(
            context,
            name,
            super::storage::WorkspaceNamespaceAction::UnbindGlobal,
            None,
        )
    }

    fn assign_base_local(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        if self.is_private()? {
            return self.private_assign_unbounded(name, value);
        }
        self.ensure_unbounded_compatibility()?;
        self.prepare_unbounded_appended_effect(2)?;
        self.assign_unbounded_owned(name, value, None, true, false)
    }

    fn assign_base_local_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return self.private_assign_owned(context, name, value);
        }
        self.capture_canonical_prefix(Some(context))?;
        if HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?
            == HostAllocationPolicy::Unbounded
            && self.borrow()?.global_target(name).is_some()
        {
            // Root and validate the owned value before the existing global
            // alias is retired. One prospective final-local origin covers the
            // two ordered compatibility changes atomically for the observer.
            self.prepare_unbounded_appended_effect(2)?;
            let entry = super::values::RootedWorkspaceEntry::new(value)
                .map_err(base_workspace_error_to_runtime)?;
            let prepared = self
                .borrow()?
                .prepare_unbounded_assignment_entry_with_order(name, &entry, None, true, true)
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        }
        self.assign_with_context(context, name, value)
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        if self.is_private()? {
            return self.private_assign_unbounded(name, value);
        }
        self.ensure_unbounded_compatibility()?;
        self.prepare_unbounded_appended_effect(1)?;
        self.assign_unbounded_owned(name, value, None, false, false)
    }

    fn assign_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return self.private_assign_owned(context, name, value);
        }
        self.assign_ref_with_context(context, name, &value)
    }

    fn assign_ref_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return self.private_assign_ref(context, name, value);
        }
        self.capture_canonical_prefix(Some(context))?;
        let projection = self.borrow()?;
        let Some(plan) = projection
            .plan_executor_write(name, value)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            drop(projection);
            self.require_unbounded_borrowed_write(context)?;
            self.prepare_unbounded_executor_effect(name)?;
            let prepared = self
                .borrow()?
                .prepare_unbounded_assignment(name, value.clone(), None, false)
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        };
        let publication_upper_bound = projection
            .publication_upper_bound_after_executor(name)
            .map_err(base_workspace_error_to_runtime)?;
        drop(projection);
        let publication_retained =
            self.published_effect_retained_bytes(context, publication_upper_bound)?;
        let retained = plan
            .retained_bytes()
            .checked_add(publication_retained)
            .ok_or_else(|| workspace_admission_error(context, HostAllocationError::Overflow))?;
        let allocation_plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), allocation_plan)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            self.prepare_unbounded_executor_effect(name)?;
            let prepared = self
                .borrow()?
                .prepare_unbounded_assignment(name, value.clone(), None, false)
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        };
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared = PreparedWorkspaceWrite::prepare(0, name, value, &permit)
            .map_err(|cause| workspace_admission_error(context, HostAllocationError::from(cause)))?
            .expect("workspace overlay eligibility was checked before admission");
        let publication =
            self.prepare_published_effect(context, publication_upper_bound, &permit)?;
        receipt
            .checkpoint(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared = self
            .borrow()?
            .prepare_assignment(context, plan, &prepared, &permit, false)
            .map_err(base_workspace_error_to_runtime)?;
        self.install_published_effect(publication)?;
        self.install_observed_mutation(context, prepared)
    }

    fn assign_executor_slot(
        &self,
        slot: usize,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        if self.is_private()? {
            return self.private_assign_unbounded(name, value);
        }
        self.ensure_unbounded_compatibility()?;
        self.prepare_unbounded_executor_effect(name)?;
        self.assign_unbounded_owned(name, value, Some(slot), false, true)
    }

    fn assign_executor_slot_ref(
        &self,
        context: &RuntimeContext,
        slot: usize,
        name: &str,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return self.private_assign_ref(context, name, value);
        }
        self.capture_canonical_prefix(Some(context))?;
        let projection = self.borrow()?;
        let Some(plan) = projection
            .plan_executor_write(name, value)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            drop(projection);
            // Aggregate and managed-handle values enter through the general
            // rooted publication owner until that owner's exact admission is
            // available; this typed fast path never claims to cover them.
            self.require_unbounded_borrowed_write(context)?;
            self.prepare_unbounded_executor_effect(name)?;
            let prepared = self
                .borrow()?
                .prepare_unbounded_assignment(name, value.clone(), Some(slot), false)
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        };
        let publication_upper_bound = projection
            .publication_upper_bound_after_executor(name)
            .map_err(base_workspace_error_to_runtime)?;
        drop(projection);
        let publication_retained =
            self.published_effect_retained_bytes(context, publication_upper_bound)?;
        let retained = plan
            .retained_bytes()
            .checked_add(publication_retained)
            .ok_or_else(|| workspace_admission_error(context, HostAllocationError::Overflow))?;
        let allocation_plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), allocation_plan)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            self.prepare_unbounded_executor_effect(name)?;
            let prepared = self
                .borrow()?
                .prepare_unbounded_assignment(name, value.clone(), Some(slot), false)
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        };
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared = PreparedWorkspaceWrite::prepare(slot, name, value, &permit)
            .map_err(|cause| workspace_admission_error(context, HostAllocationError::from(cause)))?
            .expect("workspace overlay eligibility was checked before admission");
        let publication =
            self.prepare_published_effect(context, publication_upper_bound, &permit)?;
        receipt
            .checkpoint(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared = self
            .borrow()?
            .prepare_assignment(context, plan, &prepared, &permit, true)
            .map_err(base_workspace_error_to_runtime)?;
        self.install_published_effect(publication)?;
        self.install_observed_mutation(context, prepared)
    }

    fn assign_executor_slot_lease_with_context(
        &self,
        context: &RuntimeContext,
        slot: usize,
        name: &str,
        value: WorkspaceSlotReadLease,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        if self.is_private()? {
            return self.private_assign_lease(context, name, value);
        }
        self.capture_canonical_prefix(Some(context))?;
        let projection = self.borrow()?;
        let Some(plan) = projection
            .plan_executor_retained_write(name, &value)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            drop(projection);
            // A legacy unclaimed source cannot acquire a finite backing
            // certificate by being wrapped in a retained read lease.
            self.require_unbounded_borrowed_write(context)?;
            self.prepare_unbounded_executor_effect(name)?;
            let entry = super::values::RootedWorkspaceEntry::from_read_lease(value);
            let prepared = self
                .borrow()?
                .prepare_unbounded_assignment_entry_with_order(
                    name,
                    &entry,
                    Some(slot),
                    false,
                    true,
                )
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        };
        let publication_upper_bound = projection
            .publication_upper_bound_after_executor(name)
            .map_err(base_workspace_error_to_runtime)?;
        drop(projection);
        let publication_retained =
            self.published_effect_retained_bytes(context, publication_upper_bound)?;
        let retained = plan
            .retained_bytes()
            .checked_add(publication_retained)
            .ok_or_else(|| workspace_admission_error(context, HostAllocationError::Overflow))?;
        let allocation_plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), allocation_plan)
            .map_err(|cause| workspace_admission_error(context, cause))?
        else {
            self.prepare_unbounded_executor_effect(name)?;
            let entry = super::values::RootedWorkspaceEntry::from_read_lease(value);
            let prepared = self
                .borrow()?
                .prepare_unbounded_assignment_entry_with_order(
                    name,
                    &entry,
                    Some(slot),
                    false,
                    true,
                )
                .map_err(base_workspace_error_to_runtime)?;
            return self.install_unbounded_observed_mutation(context, prepared);
        };
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared = PreparedWorkspaceWrite::prepare_retained(slot, name, value, &permit)
            .map_err(|cause| workspace_admission_error(context, HostAllocationError::from(cause)))?
            .expect("retained Value owner was checked before admission");
        let publication =
            self.prepare_published_effect(context, publication_upper_bound, &permit)?;
        receipt
            .checkpoint(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let prepared = self
            .borrow()?
            .prepare_assignment(context, plan, &prepared, &permit, true)
            .map_err(base_workspace_error_to_runtime)?;
        self.install_published_effect(publication)?;
        self.install_observed_mutation(context, prepared)
    }

    fn declare_global_binding_with_context(
        &self,
        context: &RuntimeContext,
        _slot: usize,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.bind_base_global_with_context(context, name)
    }

    fn clear(&self) -> Result<(), RuntimeError> {
        if self.is_private()? {
            return self.private_clear(None);
        }
        self.ensure_unbounded_compatibility()?;
        self.prepare_unbounded_appended_effect(1)?;
        self.with_live_publication_mut(|projection| Ok(((), projection.clear_unbounded_retired())))
    }

    fn clear_with_context(&self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        if self.is_private()? {
            return self.private_clear(Some(context));
        }
        self.stage_clear_with_context(context)
    }

    fn remove_with_context(
        &self,
        context: &RuntimeContext,
        name: &str,
    ) -> Result<(), RuntimeError> {
        if self.is_private()? {
            return self.private_remove(Some(context), name);
        }
        self.stage_namespace_with_context(
            context,
            name,
            super::storage::WorkspaceNamespaceAction::Remove,
            None,
        )
    }

    fn remove(&self, name: &str) -> Result<(), RuntimeError> {
        if self.is_private()? {
            return self.private_remove(None, name);
        }
        self.ensure_unbounded_compatibility()?;
        self.prepare_unbounded_appended_effect(1)?;
        self.with_live_publication_mut(|projection| {
            Ok((
                (),
                projection.namespace_unbounded_retired(
                    name,
                    super::storage::WorkspaceNamespaceAction::Remove,
                    None,
                ),
            ))
        })
    }
}

pub(super) fn workspace_admission_error(
    context: &RuntimeContext,
    cause: HostAllocationError,
) -> RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        Some(context),
        &format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ),
        format!("workspace publication admission failed: {cause}"),
        cause,
    )
}

fn publication_preparation_error() -> RuntimeError {
    crate::build_runtime_error("workspace publication effect was not prepared before commit")
        .with_identifier("RunMat:Workspace:PublicationPreparation")
        .build()
}
