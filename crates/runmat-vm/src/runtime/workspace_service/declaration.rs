//! One pre-effect declaration transaction for named and legacy globals.

use super::*;
use runmat_runtime::context::RuntimeContext;

#[derive(Clone, Copy)]
enum GlobalDeclarationKind {
    Named,
    LegacySlot,
}

impl VmWorkspaceService {
    pub(super) fn declare_global_in_context(
        &self,
        context: Option<&RuntimeContext>,
        slot: usize,
        name: &str,
    ) -> Result<(), RuntimeError> {
        let active = context
            .is_none()
            .then(runmat_runtime::context::legacy::active)
            .flatten();
        let context = context.or(active.as_ref()).ok_or_else(|| {
            frame_error(
                "RunMat:Workspace:SessionContextRequired",
                "global declaration requires its runtime context",
            )
        })?;
        self.declare_storage_global(context, slot, name, GlobalDeclarationKind::Named)
    }

    pub(super) fn declare_legacy_global_in_context(
        &self,
        context: &RuntimeContext,
        slot: usize,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.declare_storage_global(context, slot, name, GlobalDeclarationKind::LegacySlot)
    }

    fn declare_storage_global(
        &self,
        context: &RuntimeContext,
        slot: usize,
        name: &str,
        kind: GlobalDeclarationKind,
    ) -> Result<(), RuntimeError> {
        use super::super::workspace;

        self.ensure_mutation_in_context(Some(context))?;
        let session = match kind {
            GlobalDeclarationKind::Named => self
                .session_variables
                .prepare_declare_global_with_context(context, name)?,
            GlobalDeclarationKind::LegacySlot => self
                .session_variables
                .prepare_declare_global_slot_with_context(context, slot)?,
        };
        let frame = workspace::prepare_workspace_frame_assignment_at_slot(
            self.frame,
            Some(context),
            slot,
            name,
            session.value(),
            true,
        )?;
        let alias = match kind {
            GlobalDeclarationKind::Named => {
                workspace::prepare_workspace_frame_declare_global(self.frame, context, slot, name)?
            }
            GlobalDeclarationKind::LegacySlot => {
                workspace::prepare_workspace_frame_declare_legacy_global(self.frame, context, slot)?
            }
        }
        .ok_or_else(|| frame_error("RunMat:Workspace:FrameClosed", "workspace frame is closed"))?;
        let target = alias.target();
        session.checkpoint(context)?;
        alias
            .checkpoint()
            .map_err(workspace::frame_allocation_error)?;
        frame.checkpoint_before_publication()?;
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                delegate.bind_base_global_target_with_context(context, name, &target)?;
            }
        }
        session.commit();
        alias.commit();
        frame.commit();
        Ok(())
    }
}
