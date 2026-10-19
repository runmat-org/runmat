use crate::context::RuntimeContext;
use crate::workspace::LogicalWorkspaceGeneration;
use crate::RuntimeError;

use super::super::{workspace_admission_error, BaseWorkspaceProjectionService};
use super::{begin_private_operation, private_projection_required, private_storage_error};

impl BaseWorkspaceProjectionService {
    pub(in crate::workspace::base) fn private_snapshot(
        &self,
    ) -> Result<Vec<(String, runmat_value::Value)>, RuntimeError> {
        let projection = self.borrow()?;
        Ok(projection
            .iter()
            .map(|(name, value)| (name.to_owned(), value.clone()))
            .collect())
    }

    pub(in crate::workspace::base) fn private_global_names(
        &self,
    ) -> Result<Vec<String>, RuntimeError> {
        let projection = self.borrow()?;
        Ok(projection.global_names().map(str::to_owned).collect())
    }

    pub(in crate::workspace::base) fn private_generation(
        &self,
        context: &RuntimeContext,
    ) -> Result<LogicalWorkspaceGeneration, RuntimeError> {
        self.ensure_mutation_context(context)?;
        let operation = begin_private_operation(context)?;
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        if let Some(projection_prepared) = projection_storage
            .prepare_seal(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?
        {
            let owner_prepared = owner_storage
                .prepare_adopt_pruned(Some(context), &projection_prepared)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            self.install_private_transition(
                Some(context),
                operation.as_ref(),
                owner,
                owner_prepared,
                projection_prepared,
            )?;
        }
        let projection = self.borrow()?;
        let storage = projection
            .private_storage()
            .ok_or_else(private_projection_required)?
            .sealed_managed_storage()
            .map_err(private_storage_error)?;
        let snapshot = super::super::super::BaseWorkspaceSnapshot::from_managed_projection(
            projection.revision(),
            storage,
        );
        drop(projection);
        LogicalWorkspaceGeneration::from_base_snapshot(
            context,
            snapshot,
            &context.session_variable_authority(),
        )
    }
}
