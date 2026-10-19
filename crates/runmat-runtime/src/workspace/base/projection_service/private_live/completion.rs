use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::workspace::WorkspaceName;
use crate::RuntimeError;

use super::super::super::storage::{BindingTreeValue, PreparedPrivateTransition};
use super::super::super::values::RootedWorkspaceEntry;
use super::super::super::{
    base_workspace_error_to_runtime, BaseWorkspaceError, BaseWorkspaceOwner,
    BaseWorkspaceProjection, PreparedWorkspaceWrite,
};
use super::super::{workspace_admission_error, BaseWorkspaceProjectionService};
use super::{
    begin_private_operation, private_name_retained_bytes, private_projection_required,
    private_storage_error,
};

impl BaseWorkspaceProjectionService {
    pub(in crate::workspace::base) fn prepare_private_completion_token(
        &self,
        context: &RuntimeContext,
        identity: crate::workspace::WorkspaceCompletionIdentity,
        source: &dyn crate::workspace::WorkspaceCompletionSource,
    ) -> Result<
        (
            BaseWorkspaceOwner,
            PreparedPrivateTransition,
            PreparedPrivateTransition,
            Option<HostAllocationReceipt>,
        ),
        RuntimeError,
    > {
        debug_assert!(!source.is_empty());
        self.ensure_mutation_context(context)?;
        let operation = begin_private_operation(context)?;
        let issuer = std::rc::Rc::as_ptr(&self.state) as usize;
        let validate = |projection: &BaseWorkspaceProjection| -> Result<(), RuntimeError> {
            let storage = projection
                .private_storage()
                .ok_or_else(private_projection_required)?;
            if identity.issuer() != issuer
                || identity.revision() != projection.revision()
                || identity.pending_generation() != storage.epoch()
            {
                return Err(base_workspace_error_to_runtime(
                    BaseWorkspaceError::CompletionConflict,
                ));
            }
            projection
                .ensure_effect_authority()
                .map_err(base_workspace_error_to_runtime)
        };
        {
            let projection = self.borrow()?;
            validate(&projection)?;
            projection
                .validate_completion_source(source)
                .map_err(private_storage_error)?;
        }
        let policy = HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let mut retained = 0_u64;
        if policy == HostAllocationPolicy::Managed {
            for index in 0..source.len() {
                let action = source
                    .action(index)
                    .ok_or_else(private_projection_required)?;
                let bytes = (|| -> Result<u64, HostAllocationError> {
                    match &action {
                        crate::workspace::WorkspaceCompletionAction::AssignLocal {
                            name,
                            value,
                        } => Ok(match value {
                            crate::workspace::WorkspaceCompletionValue::Borrowed(value) => {
                                PreparedWorkspaceWrite::retained_bytes(name, value)
                            }
                            crate::workspace::WorkspaceCompletionValue::Retained(value) => {
                                PreparedWorkspaceWrite::retained_generation_bytes(name, value)
                            }
                        }
                        .map_err(HostAllocationError::from)?
                        .ok_or(HostAllocationError::Resource(
                            crate::context::host_effect::HostEffectAllocationError::Unavailable,
                        ))?),
                        crate::workspace::WorkspaceCompletionAction::BindGlobal {
                            name,
                            target,
                        } => Ok(private_name_retained_bytes(name)?
                            .checked_add(target.additional_retained_bytes()?)
                            .ok_or(HostAllocationError::Overflow)?),
                        crate::workspace::WorkspaceCompletionAction::UnbindGlobal { name }
                        | crate::workspace::WorkspaceCompletionAction::Remove { name } => {
                            private_name_retained_bytes(name)
                        }
                    }
                })()
                .map_err(|cause| workspace_admission_error(context, cause))?;
                retained = retained.checked_add(bytes).ok_or_else(|| {
                    workspace_admission_error(context, HostAllocationError::Overflow)
                })?;
            }
        }
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, retained)
                .map_err(|cause| workspace_admission_error(context, cause))?,
        )
        .map_err(|cause| workspace_admission_error(context, cause))?;
        let permit = receipt
            .as_ref()
            .map(|receipt| receipt.output_permit(context))
            .transpose()
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let projection_prepared = projection_storage
            .prepare_batch_with(Some(context), source.len(), |index| {
                let action = source
                    .action(index)
                    .ok_or(HostAllocationError::StaleAuthority)?;
                match action {
                    crate::workspace::WorkspaceCompletionAction::AssignLocal { name, value } => {
                        let (name, entry) = match (permit.as_ref(), value) {
                            (
                                Some(permit),
                                crate::workspace::WorkspaceCompletionValue::Borrowed(value),
                            ) => {
                                let write =
                                    PreparedWorkspaceWrite::prepare(0, name, value, permit)?
                                        .ok_or(HostAllocationError::StaleAuthority)?;
                                (
                                    write.name().clone(),
                                    RootedWorkspaceEntry::from_prepared(&write),
                                )
                            }
                            (
                                Some(permit),
                                crate::workspace::WorkspaceCompletionValue::Retained(value),
                            ) => {
                                let write = PreparedWorkspaceWrite::prepare_retained(
                                    0, name, value, permit,
                                )?
                                .ok_or(HostAllocationError::StaleAuthority)?;
                                (
                                    write.name().clone(),
                                    RootedWorkspaceEntry::from_prepared(&write),
                                )
                            }
                            (None, crate::workspace::WorkspaceCompletionValue::Borrowed(value)) => {
                                (
                                    WorkspaceName::unbounded(name),
                                    RootedWorkspaceEntry::new(value.clone()).map_err(|error| {
                                        match error {
                                            BaseWorkspaceError::HostAllocation(cause) => cause,
                                            _ => HostAllocationError::StaleAuthority,
                                        }
                                    })?,
                                )
                            }
                            (None, crate::workspace::WorkspaceCompletionValue::Retained(value)) => {
                                (
                                    WorkspaceName::unbounded(name),
                                    RootedWorkspaceEntry::from_read_lease(value),
                                )
                            }
                        };
                        Ok((name, Some(BindingTreeValue::Local(entry))))
                    }
                    crate::workspace::WorkspaceCompletionAction::BindGlobal { name, target } => {
                        let name = match permit.as_ref() {
                            Some(permit) => WorkspaceName::prepare(name, permit)?,
                            None => WorkspaceName::unbounded(name),
                        };
                        let target = match permit.as_ref() {
                            Some(permit) => target.prepare_retained(permit)?,
                            None => target.clone(),
                        };
                        Ok((name, Some(BindingTreeValue::Global(target))))
                    }
                    crate::workspace::WorkspaceCompletionAction::UnbindGlobal { name }
                    | crate::workspace::WorkspaceCompletionAction::Remove { name } => {
                        let name = match permit.as_ref() {
                            Some(permit) => WorkspaceName::prepare(name, permit)?,
                            None => WorkspaceName::unbounded(name),
                        };
                        Ok((name, None))
                    }
                }
            })
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(Some(context), &projection_prepared)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(context)
                .map_err(|cause| workspace_admission_error(context, cause))?;
        }
        let projection = self.borrow()?;
        validate(&projection)?;
        drop(projection);
        Ok((owner, owner_prepared, projection_prepared, operation))
    }

    pub(in crate::workspace::base) fn private_apply_completion(
        &self,
        context: &RuntimeContext,
        identity: crate::workspace::WorkspaceCompletionIdentity,
        source: &dyn crate::workspace::WorkspaceCompletionSource,
    ) -> Result<(), RuntimeError> {
        if source.is_empty() {
            self.ensure_mutation_context(context)?;
            let projection = self.borrow()?;
            let storage = projection
                .private_storage()
                .ok_or_else(private_projection_required)?;
            if identity.issuer() != std::rc::Rc::as_ptr(&self.state) as usize
                || identity.revision() != projection.revision()
                || identity.pending_generation() != storage.epoch()
            {
                return Err(base_workspace_error_to_runtime(
                    BaseWorkspaceError::CompletionConflict,
                ));
            }
            drop(projection);
            let operation = begin_private_operation(context)?;
            if let Some(receipt) = operation {
                receipt.checkpoint(context).map_err(private_storage_error)?;
            }
            return Ok(());
        }
        let (owner, owner_prepared, projection_prepared, operation) =
            self.prepare_private_completion_token(context, identity, source)?;
        self.install_private_transition(
            Some(context),
            operation.as_ref(),
            owner,
            owner_prepared,
            projection_prepared,
        )
    }
}
