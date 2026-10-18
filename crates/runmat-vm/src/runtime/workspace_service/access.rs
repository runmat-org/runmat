//! Context-bearing synchronization and reads of the exact current frame.

use super::*;
use runmat_runtime::context::RuntimeContext;

impl VmWorkspaceService {
    pub(super) fn ensure_effect_in_context(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        self.ensure_context_owner(context)?;
        self.ensure_frame_active()?;
        if super::super::workspace::frame_is_detaching(self.frame) {
            return Err(frame_error(
                "RunMat:Workspace:FrameUnavailable",
                "workspace frame has an active detach transaction",
            ));
        }
        if let Some(delegate) = self.delegate() {
            match context {
                Some(context) => delegate.ensure_effect_authority_with_context(context)?,
                None => delegate.ensure_effect_authority()?,
            }
            if self.should_publish() {
                let revision = match context {
                    Some(context) => delegate.workspace_revision_with_context(context)?,
                    None => delegate.workspace_revision()?,
                };
                if revision.is_some() && revision == self.delegate_revision.get() {
                    return Ok(());
                }
                let canonical = match context {
                    Some(context) => delegate.try_snapshot_with_context(context)?,
                    None => delegate.try_snapshot()?,
                };
                reconcile_workspace_frame_with_context(self.frame, context, canonical)?;
                self.delegate_revision.set(revision);
            }
        }
        Ok(())
    }

    pub(super) fn lookup_in_context(
        &self,
        context: Option<&RuntimeContext>,
        name: &str,
    ) -> Result<Option<Value>, RuntimeError> {
        self.ensure_context_owner(context)?;
        if !self.is_pending() {
            self.ensure_effect_in_context(context)?;
            return Ok(workspace_frame_lookup(self.frame, name));
        }
        self.delegate().map_or(Ok(None), |delegate| match context {
            Some(context) => delegate.try_lookup_with_context(context, name),
            None => delegate.try_lookup(name),
        })
    }

    pub(super) fn snapshot_in_context(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<Vec<(String, Value)>, RuntimeError> {
        self.ensure_context_owner(context)?;
        if self.is_pending() {
            return self.delegate().map_or_else(
                || Ok(Vec::new()),
                |delegate| match context {
                    Some(context) => delegate.try_snapshot_with_context(context),
                    None => delegate.try_snapshot(),
                },
            );
        }
        self.ensure_effect_in_context(context)?;
        workspace_frame_snapshot(self.frame)
            .ok_or_else(|| frame_error("RunMat:Workspace:FrameClosed", "workspace frame is closed"))
    }

    pub(super) fn global_names_in_context(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<Vec<String>, RuntimeError> {
        self.ensure_context_owner(context)?;
        if !self.is_pending() {
            self.ensure_effect_in_context(context)?;
        }
        self.session_variables.global_names()
    }

    pub(super) fn detach_in_context(
        &self,
        context: Option<&RuntimeContext>,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        self.ensure_effect_in_context(context)?;
        let delegate = self.should_publish().then(|| self.delegate()).flatten();
        workspace_frame_detach_selected(self.frame, selection, |detached| {
            let _retired_delegate = if let Some(delegate) = delegate {
                let delegated = match selection {
                    WorkspaceDetachSelection::All => WorkspaceDetachSelection::All,
                    WorkspaceDetachSelection::Names(names) => {
                        WorkspaceDetachSelection::Names(names.clone())
                    }
                    WorkspaceDetachSelection::Except(_) => WorkspaceDetachSelection::Names(
                        detached
                            .entries()
                            .iter()
                            .map(|(name, _)| name.clone())
                            .collect(),
                    ),
                };
                Some(match context {
                    Some(context) => delegate.detach_selected_with_context(context, &delegated)?,
                    None => delegate.detach_selected(&delegated)?,
                })
            } else {
                None
            };
            Ok(())
        })
    }
}
