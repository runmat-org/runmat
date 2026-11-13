//! Atomic completion forwarding through an installed VM delegate chain.

use super::*;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::{WorkspaceCompletionIdentity, WorkspaceCompletionSource};

impl VmWorkspaceService {
    fn completion_delegate(
        &self,
        context: &RuntimeContext,
    ) -> Result<Option<Rc<dyn RuntimeWorkspaceService>>, RuntimeError> {
        self.ensure_context_owner(Some(context))?;
        self.ensure_frame_active()?;
        if super::super::workspace::frame_is_detaching(self.frame) {
            return Err(frame_error(
                "RunMat:Workspace:FrameUnavailable",
                "workspace completion conflicts with an active detach",
            ));
        }
        Ok(self.should_publish().then(|| self.delegate()).flatten())
    }

    pub(super) fn capture_completion_in_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<Option<WorkspaceCompletionIdentity>, RuntimeError> {
        self.completion_delegate(context)?
            .map_or(Ok(None), |delegate| {
                delegate.capture_workspace_completion_identity_with_context(context)
            })
    }

    pub(super) fn apply_completion_in_context(
        &self,
        context: &RuntimeContext,
        identity: WorkspaceCompletionIdentity,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<(), RuntimeError> {
        let delegate = self.completion_delegate(context)?.ok_or_else(|| {
            frame_error(
                "RunMat:Workspace:TargetState",
                "workspace frame has no atomic completion delegate",
            )
        })?;
        // The selected target's prepared token owns its frame swap. This
        // adapter forwards canonical publication only; it neither snapshots
        // another frame nor repeats the changes as individual mutations.
        delegate.apply_workspace_completion_with_context(context, identity, source)
    }

    pub(super) fn prepare_completion_in_context(
        &self,
        context: &RuntimeContext,
        identity: WorkspaceCompletionIdentity,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<runmat_runtime::workspace::PreparedWorkspaceCompletion, RuntimeError> {
        let delegate = self.completion_delegate(context)?.ok_or_else(|| {
            frame_error(
                "RunMat:Workspace:TargetState",
                "workspace frame has no atomic completion delegate",
            )
        })?;
        delegate.prepare_workspace_completion_with_context(context, identity, source)
    }
}
