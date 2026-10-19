//! Single-owner private workspace mutations.
//!
//! Public projections retain their ordered delta/publication protocol. A
//! private executor projection instead updates one persistent namespace root
//! and one projection-only hot binding. Both physical roots are installed in
//! one synchronous section; displaced owners retire after every borrow ends.

#[path = "private_live/writes.rs"]
mod writes;
#[path = "private_live/detach.rs"]
mod detach;
#[path = "private_live/capture.rs"]
mod capture;
#[path = "private_live/completion.rs"]
mod completion;

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::workspace::{LogicalWorkspaceGeneration, WorkspaceName};
use crate::RuntimeError;

use super::super::storage::{PreparedPrivateTransition, PrivateLiveStorage};
use super::super::{
    base_workspace_error_to_runtime, BaseWorkspaceError, BaseWorkspaceOwner,
    BaseWorkspaceProjection,
};
use super::BaseWorkspaceProjectionService;

impl BaseWorkspaceProjectionService {
    pub fn private_from_generation(
        generation: &LogicalWorkspaceGeneration,
        context: Option<&RuntimeContext>,
    ) -> Result<(BaseWorkspaceOwner, Self), RuntimeError> {
        let storage = PrivateLiveStorage::prepare_from_generation(generation, context)
            .map_err(private_storage_error)?;
        Self::from_private_storage(storage, context)
    }

    pub fn private_empty(
        context: Option<&RuntimeContext>,
    ) -> Result<(BaseWorkspaceOwner, Self), RuntimeError> {
        let storage = PrivateLiveStorage::empty(context).map_err(private_storage_error)?;
        Self::from_private_storage(storage, context)
    }

    fn from_private_storage(
        storage: PrivateLiveStorage,
        context: Option<&RuntimeContext>,
    ) -> Result<(BaseWorkspaceOwner, Self), RuntimeError> {
        let owner = BaseWorkspaceOwner::from_private_storage(storage);
        let projection = BaseWorkspaceProjection::new_private(&owner)
            .map_err(base_workspace_error_to_runtime)?;
        let service = match context {
            Some(context) => Self::new_for_context(projection, context),
            None => Self::new(projection),
        };
        Ok((owner, service))
    }

    pub(super) fn is_private(&self) -> Result<bool, RuntimeError> {
        Ok(self.borrow()?.is_private())
    }

    fn private_snapshots(
        &self,
    ) -> Result<(BaseWorkspaceOwner, PrivateLiveStorage, PrivateLiveStorage), RuntimeError> {
        let projection = self.borrow()?;
        let owner = projection
            .canonical_owner()
            .map_err(base_workspace_error_to_runtime)?;
        let projection_storage = projection
            .private_storage()
            .ok_or_else(private_projection_required)?
            .clone();
        drop(projection);
        let owner_storage = owner
            .state
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?
            .private_live
            .as_ref()
            .ok_or_else(private_projection_required)?
            .clone();
        Ok((owner, owner_storage, projection_storage))
    }

    fn install_private_transition(
        &self,
        context: Option<&RuntimeContext>,
        operation: Option<&HostAllocationReceipt>,
        owner: BaseWorkspaceOwner,
        owner_prepared: PreparedPrivateTransition,
        projection_prepared: PreparedPrivateTransition,
    ) -> Result<(), RuntimeError> {
        match (context, operation) {
            (Some(context), Some(receipt)) => {
                receipt.checkpoint(context).map_err(private_storage_error)?;
            }
            _ => {
                HostAllocationReceipt::policy(context).map_err(private_storage_error)?;
            }
        }
        let next_revision = projection_prepared.next_revision();
        let mut service = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        match (service.runtime, context.map(RuntimeContext::local_identity)) {
            (None, None) => {}
            (Some(expected), Some(actual)) if expected == actual => {}
            _ => return Err(mutation_context_mismatch()),
        }
        service
            .projection
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        let mut canonical = owner
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        canonical
            .ensure_open()
            .map_err(base_workspace_error_to_runtime)?;
        let canonical_storage = canonical
            .private_live
            .as_mut()
            .ok_or_else(private_projection_required)?;
        let projection_storage = service
            .projection
            .private_storage_mut()
            .ok_or_else(private_projection_required)?;
        let installed = PrivateLiveStorage::install_pair_checked(
            canonical_storage,
            projection_storage,
            owner_prepared,
            projection_prepared,
        );
        let retired = match installed {
            Ok(retired) => retired,
            Err(prepared) => {
                drop(canonical);
                drop(service);
                drop(prepared);
                return Err(base_workspace_error_to_runtime(
                    BaseWorkspaceError::CompletionConflict,
                ));
            }
        };
        canonical.revision = next_revision;
        service.projection.set_private_revision(next_revision);
        drop(canonical);
        drop(service);
        drop(retired);
        Ok(())
    }
}

fn private_name_retained_bytes(name: &str) -> Result<u64, HostAllocationError> {
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(WorkspaceName::requested_bytes(name).map_err(HostAllocationError::from)?)
            .map_err(|_| HostAllocationError::Overflow)?,
        1,
    )
}

fn begin_private_operation(
    context: &RuntimeContext,
) -> Result<Option<HostAllocationReceipt>, RuntimeError> {
    HostAllocationReceipt::admit(
        Some(context),
        HostAllocationPlan::new(0, 0).map_err(private_storage_error)?,
    )
    .map_err(private_storage_error)
}

fn prepare_private_name(
    context: &RuntimeContext,
    name: &str,
) -> Result<WorkspaceName, RuntimeError> {
    if HostAllocationReceipt::policy(Some(context)).map_err(private_storage_error)?
        == HostAllocationPolicy::Unbounded
    {
        return Ok(WorkspaceName::unbounded(name));
    }
    let retained = WorkspaceName::requested_bytes(name)
        .map_err(HostAllocationError::from)
        .and_then(|bytes| u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow))
        .and_then(|bytes| HostAllocationPlan::retained_exact_output_bytes(bytes, 1))
        .map_err(private_storage_error)?;
    let receipt = HostAllocationReceipt::admit(
        Some(context),
        HostAllocationPlan::new(0, retained).map_err(private_storage_error)?,
    )
    .map_err(private_storage_error)?
    .expect("managed private workspace policy returns a receipt");
    let permit = receipt
        .output_permit(context)
        .map_err(private_storage_error)?;
    let name = WorkspaceName::prepare(name, &permit)
        .map_err(HostAllocationError::from)
        .map_err(private_storage_error)?;
    receipt.checkpoint(context).map_err(private_storage_error)?;
    Ok(name)
}

fn private_storage_error(error: HostAllocationError) -> RuntimeError {
    base_workspace_error_to_runtime(BaseWorkspaceError::HostAllocation(error))
}

fn private_projection_required() -> RuntimeError {
    base_workspace_error_to_runtime(BaseWorkspaceError::PrivateProjectionRequired)
}

fn mutation_context_mismatch() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:MutationContextMismatch",
        "workspace mutation context does not match the private projection invocation",
    )
}
