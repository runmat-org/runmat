//! One logical detached result for the same projection/session generation
//! whose namespace effects are prepared by the Base projection.

use crate::context::{RuntimeContext, RuntimeWorkspaceService};
use crate::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection};
use crate::RuntimeError;

use super::{base_workspace_error_to_runtime, BaseWorkspaceProjectionService};

impl BaseWorkspaceProjectionService {
    pub(super) fn logical_detach_values(
        &self,
        context: Option<&RuntimeContext>,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        if let Some(context) = context {
            let generation = self
                .capture_workspace_generation_with_context(context)?
                .ok_or_else(|| {
                    crate::runtime_error::semantic_error(
                        "RunMat:Workspace:GenerationUnavailable",
                        "the live projection cannot retain a logical detach generation",
                    )
                })?;
            let values = generation
                .iter()
                .map(|(name, value)| (name.to_owned(), value.clone()))
                .collect();
            DetachedWorkspaceValues::new(selection.select(values))
                .map_err(base_workspace_error_to_runtime)
        } else {
            self.borrow()?
                .logical_detach_values_unbounded(selection)
                .map_err(base_workspace_error_to_runtime)
        }
    }
}
