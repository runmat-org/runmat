use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::workspace::{GlobalBindingTarget, WorkspaceName, WorkspaceSlotReadLease};
use crate::RuntimeError;

use super::super::super::values::RootedWorkspaceEntry;
use super::super::super::{base_workspace_error_to_runtime, PreparedWorkspaceWrite};
use super::super::{workspace_admission_error, BaseWorkspaceProjectionService};
use super::{begin_private_operation, prepare_private_name, private_storage_error};

impl BaseWorkspaceProjectionService {
    /// Move an owned input into its retained generation before publishing it.
    /// Borrowed writes cannot take this path: cloning an arbitrary aggregate
    /// would lose its allocation provenance and duplicate its owner.
    pub(in crate::workspace::base) fn private_assign_owned(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: runmat_value::Value,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        let value = WorkspaceSlotReadLease::prepare_owned(value, context)?;
        self.private_assign_lease(context, name, value)
    }

    pub(in crate::workspace::base) fn private_assign_unbounded(
        &self,
        name: &str,
        value: runmat_value::Value,
    ) -> Result<(), RuntimeError> {
        self.ensure_unbounded_compatibility()?;
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let projection_prepared = projection_storage
            .prepare_local(
                None,
                WorkspaceName::unbounded(name),
                RootedWorkspaceEntry::new(value).map_err(base_workspace_error_to_runtime)?,
            )
            .map_err(private_storage_error)?;
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(None, &projection_prepared)
            .map_err(private_storage_error)?;
        self.install_private_transition(None, None, owner, owner_prepared, projection_prepared)
    }

    pub(in crate::workspace::base) fn private_assign_ref(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: &runmat_value::Value,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        let operation = begin_private_operation(context)?;
        if HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?
            == HostAllocationPolicy::Unbounded
        {
            let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
            let projection_prepared = projection_storage
                .prepare_local(
                    Some(context),
                    WorkspaceName::unbounded(name),
                    RootedWorkspaceEntry::new(value.clone())
                        .map_err(base_workspace_error_to_runtime)?,
                )
                .map_err(private_storage_error)?;
            let owner_prepared = owner_storage
                .prepare_adopt_pruned(Some(context), &projection_prepared)
                .map_err(private_storage_error)?;
            return self.install_private_transition(
                Some(context),
                operation.as_ref(),
                owner,
                owner_prepared,
                projection_prepared,
            );
        }
        let retained = PreparedWorkspaceWrite::retained_bytes(name, value)
            .map_err(HostAllocationError::from)
            .map_err(|cause| workspace_admission_error(context, cause))?
            .ok_or_else(|| {
                workspace_admission_error(
                    context,
                    HostAllocationError::Resource(
                        crate::context::host_effect::HostEffectAllocationError::Unavailable,
                    ),
                )
            })?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, retained)
                .map_err(|cause| workspace_admission_error(context, cause))?,
        )
        .map_err(|cause| workspace_admission_error(context, cause))?
        .expect("managed private workspace policy returns a receipt");
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let write = PreparedWorkspaceWrite::prepare(0, name, value, &permit)
            .map_err(HostAllocationError::from)
            .map_err(|cause| workspace_admission_error(context, cause))?
            .expect("private workspace eligibility was quoted before admission");
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let projection_prepared = projection_storage
            .prepare_local(
                Some(context),
                write.name().clone(),
                RootedWorkspaceEntry::from_prepared(&write),
            )
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(Some(context), &projection_prepared)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        receipt
            .checkpoint(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        self.install_private_transition(
            Some(context),
            operation.as_ref(),
            owner,
            owner_prepared,
            projection_prepared,
        )
    }

    pub(in crate::workspace::base) fn private_assign_lease(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: WorkspaceSlotReadLease,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        let operation = begin_private_operation(context)?;
        if HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?
            == HostAllocationPolicy::Unbounded
        {
            let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
            let projection_prepared = projection_storage
                .prepare_local(
                    Some(context),
                    WorkspaceName::unbounded(name),
                    RootedWorkspaceEntry::from_read_lease(value),
                )
                .map_err(private_storage_error)?;
            let owner_prepared = owner_storage
                .prepare_adopt_pruned(Some(context), &projection_prepared)
                .map_err(private_storage_error)?;
            return self.install_private_transition(
                Some(context),
                operation.as_ref(),
                owner,
                owner_prepared,
                projection_prepared,
            );
        }
        let retained = PreparedWorkspaceWrite::retained_generation_bytes(name, &value)
            .map_err(HostAllocationError::from)
            .map_err(|cause| workspace_admission_error(context, cause))?
            .ok_or_else(|| {
                workspace_admission_error(
                    context,
                    HostAllocationError::Resource(
                        crate::context::host_effect::HostEffectAllocationError::Unavailable,
                    ),
                )
            })?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, retained)
                .map_err(|cause| workspace_admission_error(context, cause))?,
        )
        .map_err(|cause| workspace_admission_error(context, cause))?
        .expect("managed private workspace policy returns a receipt");
        let permit = receipt
            .output_permit(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let write = PreparedWorkspaceWrite::prepare_retained(0, name, value, &permit)
            .map_err(HostAllocationError::from)
            .map_err(|cause| workspace_admission_error(context, cause))?
            .expect("retained private workspace value was quoted before admission");
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let projection_prepared = projection_storage
            .prepare_local(
                Some(context),
                write.name().clone(),
                RootedWorkspaceEntry::from_prepared(&write),
            )
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(Some(context), &projection_prepared)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        receipt
            .checkpoint(context)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        self.install_private_transition(
            Some(context),
            operation.as_ref(),
            owner,
            owner_prepared,
            projection_prepared,
        )
    }

    pub(in crate::workspace::base) fn private_remove(
        &self,
        context: Option<&RuntimeContext>,
        name: &str,
    ) -> Result<(), RuntimeError> {
        match context {
            Some(context) => self.ensure_mutation_context(context)?,
            None => self.ensure_unbounded_compatibility()?,
        }
        let operation = context.map(begin_private_operation).transpose()?;
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let name = match context {
            Some(context) => prepare_private_name(context, name)?,
            None => WorkspaceName::unbounded(name),
        };
        let projection_prepared = projection_storage
            .prepare_remove(context, name)
            .map_err(private_storage_error)?;
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(context, &projection_prepared)
            .map_err(private_storage_error)?;
        self.install_private_transition(
            context,
            operation.as_ref().and_then(Option::as_ref),
            owner,
            owner_prepared,
            projection_prepared,
        )
    }

    pub(in crate::workspace::base) fn private_clear(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        match context {
            Some(context) => self.ensure_mutation_context(context)?,
            None => self.ensure_unbounded_compatibility()?,
        }
        let operation = context.map(begin_private_operation).transpose()?;
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let projection_prepared = projection_storage
            .prepare_clear(context)
            .map_err(private_storage_error)?;
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(context, &projection_prepared)
            .map_err(private_storage_error)?;
        self.install_private_transition(
            context,
            operation.as_ref().and_then(Option::as_ref),
            owner,
            owner_prepared,
            projection_prepared,
        )
    }

    pub(in crate::workspace::base) fn private_global(
        &self,
        context: &RuntimeContext,
        name: &str,
        target: &GlobalBindingTarget,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_context(context)?;
        let operation = begin_private_operation(context)?;
        let policy = HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let (name, target, receipt) = match policy {
            HostAllocationPolicy::Unbounded => {
                (WorkspaceName::unbounded(name), target.clone(), None)
            }
            HostAllocationPolicy::Managed => {
                let name_bytes = WorkspaceName::requested_bytes(name)
                    .map_err(HostAllocationError::from)
                    .and_then(|bytes| {
                        u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)
                    })
                    .and_then(|bytes| HostAllocationPlan::retained_exact_output_bytes(bytes, 1))
                    .map_err(|cause| workspace_admission_error(context, cause))?;
                let retained = name_bytes
                    .checked_add(
                        target
                            .additional_retained_bytes()
                            .map_err(|cause| workspace_admission_error(context, cause))?,
                    )
                    .ok_or_else(|| {
                        workspace_admission_error(context, HostAllocationError::Overflow)
                    })?;
                let receipt = HostAllocationReceipt::admit(
                    Some(context),
                    HostAllocationPlan::new(0, retained)
                        .map_err(|cause| workspace_admission_error(context, cause))?,
                )
                .map_err(|cause| workspace_admission_error(context, cause))?
                .expect("managed private workspace policy returns a receipt");
                let permit = receipt
                    .output_permit(context)
                    .map_err(|cause| workspace_admission_error(context, cause))?;
                let name = WorkspaceName::prepare(name, &permit)
                    .map_err(HostAllocationError::from)
                    .map_err(|cause| workspace_admission_error(context, cause))?;
                let target = target
                    .prepare_retained(&permit)
                    .map_err(HostAllocationError::from)
                    .map_err(|cause| workspace_admission_error(context, cause))?;
                (name, target, Some(receipt))
            }
        };
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let projection_prepared = projection_storage
            .prepare_global(Some(context), name, target)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(Some(context), &projection_prepared)
            .map_err(|cause| workspace_admission_error(context, cause))?;
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(context)
                .map_err(|cause| workspace_admission_error(context, cause))?;
        }
        self.install_private_transition(
            Some(context),
            operation.as_ref(),
            owner,
            owner_prepared,
            projection_prepared,
        )
    }

    pub(in crate::workspace::base) fn private_global_unbounded(
        &self,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.ensure_unbounded_compatibility()?;
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let projection_prepared = projection_storage
            .prepare_global(
                None,
                WorkspaceName::unbounded(name),
                GlobalBindingTarget::Named(WorkspaceName::unbounded(name)),
            )
            .map_err(private_storage_error)?;
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(None, &projection_prepared)
            .map_err(private_storage_error)?;
        self.install_private_transition(None, None, owner, owner_prepared, projection_prepared)
    }
}
