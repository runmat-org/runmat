//! A compiled slot projection borrows the canonical immutable generation.
//! Names/slots remain compiler metadata; no detached language Value map is made.

use super::InterpreterVars;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::{LogicalWorkspaceGeneration, WorkspaceSlotReadLease};
use runmat_runtime::RuntimeError;

impl InterpreterVars {
    /// Prepare one owned VM frame from actual canonical workspace read leases.
    /// Unbound compiler-private slots start unassigned in the caller's separate
    /// admission metadata. This creates no workspace mutation authority.
    pub fn from_workspace_generation<'a>(
        context: &RuntimeContext,
        generation: &LogicalWorkspaceGeneration,
        bindings: impl IntoIterator<Item = (&'a str, usize)>,
        extent: usize,
    ) -> Result<Self, RuntimeError> {
        let mut frame = Self::prepare_captured_frame(context, extent).map_err(|cause| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(context), cause)
        })?;
        for (name, slot) in bindings {
            if slot >= extent {
                return Err(runmat_runtime::build_runtime_error(
                    "compiled workspace binding exceeds its prepared frame extent",
                )
                .with_identifier("RunMat:Interpreter:WorkspaceFrameExtent")
                .build());
            }
            if let Some(source) = generation.read_lease(name) {
                let displaced = frame.replace_retained(slot, WorkspaceSlotReadLease::from(source));
                drop(displaced);
            }
        }
        Ok(frame)
    }
}
