//! Frame mutation order shared by legacy hosts and operation-bearing calls.
use super::*;
use crate::bytecode::program::LexicalBindingKind;
use runmat_runtime::call::lexical::{
    LexicalCompletionChange, LexicalCompletionScope, PreparedLexicalCompletionBatch,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::{GlobalBindingTarget, WorkspaceSlotReadLease};

/// Inputs whose owners must remain live across one frame/delegate publication.
///
/// Keeping them together prevents the frame preparation and delegate
/// publication paths from accidentally selecting different values, slots, or
/// operation authorities.
pub(super) struct AssignmentPublication<'a> {
    pub(super) context: Option<&'a RuntimeContext>,
    pub(super) executor_slot: Option<usize>,
    pub(super) name: &'a str,
    pub(super) value: &'a Value,
    pub(super) retained: Option<WorkspaceSlotReadLease>,
    pub(super) reconcile_var: bool,
}

impl VmWorkspaceService {
    pub(super) fn ensure_mutation_in_context(
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
                Some(context) => delegate.ensure_mutation_authority_with_context(context)?,
                None => delegate.ensure_mutation_authority()?,
            }
        }
        Ok(())
    }

    pub(super) fn ensure_context_owner(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        if context.is_some_and(|context| context.local_identity() != self.runtime) {
            return Err(frame_error(
                "RunMat:Workspace:ContextMismatch",
                "workspace mutation belongs to a different runtime context",
            ));
        }
        Ok(())
    }

    pub(super) fn assign_in_context(
        &self,
        context: Option<&RuntimeContext>,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        self.publish_assignment(
            AssignmentPublication {
                context,
                executor_slot: None,
                name,
                value: &value,
                retained: None,
                reconcile_var: true,
            },
            |delegate, context| match context {
                Some(context) => delegate.assign_ref_with_context(context, name, &value),
                None => delegate.assign(name, value.clone()),
            },
        )
    }

    pub(super) fn assign_borrowed_in_context(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        self.publish_assignment(
            AssignmentPublication {
                context: Some(context),
                executor_slot: None,
                name,
                value,
                retained: None,
                reconcile_var: true,
            },
            |delegate, context| {
                delegate.assign_ref_with_context(
                    context.expect("operation-bearing assignment retains its context"),
                    name,
                    value,
                )
            },
        )
    }

    pub(super) fn publish_assignment(
        &self,
        assignment: AssignmentPublication<'_>,
        publish: impl FnOnce(
            &dyn RuntimeWorkspaceService,
            Option<&RuntimeContext>,
        ) -> Result<(), RuntimeError>,
    ) -> Result<(), RuntimeError> {
        let AssignmentPublication {
            context,
            executor_slot,
            name,
            value,
            retained,
            reconcile_var,
        } = assignment;
        let active = context
            .is_none()
            .then(runmat_runtime::context::legacy::active)
            .flatten();
        let context = context.or(active.as_ref());
        self.ensure_mutation_in_context(context)?;
        runmat_runtime::execution::validate_storable_value(value)?;
        let global = super::super::workspace::workspace_frame_global_binding(self.frame, name);
        let lexical = super::super::workspace::workspace_frame_lexical_binding(self.frame, name)
            .filter(|(_, _, binding)| binding.kind != LexicalBindingKind::Transit);
        if global.is_some() && lexical.is_some() {
            return Err(frame_error(
                "RunMat:Workspace:BindingConflict",
                "workspace binding cannot be both session-global and lexical",
            ));
        }
        let retained = match (retained, lexical.as_ref(), context) {
            (Some(retained), _, _) => Some(retained),
            (None, Some(_), Some(context)) => Some(WorkspaceSlotReadLease::prepare_owned(
                value.clone(),
                context,
            )?),
            (None, Some(_), None) => {
                return Err(frame_error(
                    "RunMat:Workspace:SessionContextRequired",
                    "lexical workspace assignment requires its runtime context",
                ));
            }
            (None, None, _) => None,
        };
        let lexical_source = lexical.as_ref().and_then(|_| retained.clone());
        let mut frame = match (executor_slot, retained) {
            (Some(slot), Some(retained)) => {
                let context = context.ok_or_else(|| {
                    frame_error(
                        "RunMat:Workspace:SessionContextRequired",
                        "retained executor assignment requires its runtime context",
                    )
                })?;
                super::super::workspace::prepare_workspace_frame_assignment_lease_at_slot(
                    self.frame,
                    context,
                    slot,
                    name,
                    retained,
                    reconcile_var,
                )?
            }
            (Some(slot), None) => {
                super::super::workspace::prepare_workspace_frame_assignment_at_slot(
                    self.frame,
                    context,
                    slot,
                    name,
                    value,
                    reconcile_var,
                )?
            }
            (None, Some(retained)) => {
                let context = context.expect("lexical retained assignment has a context");
                super::super::workspace::prepare_workspace_frame_assignment_lease(
                    self.frame,
                    context,
                    name,
                    retained,
                    reconcile_var,
                )?
            }
            (None, None) => super::super::workspace::prepare_workspace_frame_assignment(
                self.frame,
                context,
                name,
                value,
                reconcile_var,
            )?,
        };
        let mut lexical = match (lexical, lexical_source) {
            (Some((slot, environment, binding)), Some(source)) => {
                if frame.slot() != slot {
                    return Err(frame_error(
                        "RunMat:Workspace:LexicalSlot",
                        "lexical workspace binding resolved to another frame slot",
                    ));
                }
                let context = context.expect("lexical retained assignment has a context");
                let scope = match binding.kind {
                    LexicalBindingKind::Owned => LexicalCompletionScope::Owned,
                    LexicalBindingKind::Inherited => LexicalCompletionScope::Inherited,
                    LexicalBindingKind::Transit => {
                        unreachable!("transit lexical bindings were filtered before preparation")
                    }
                };
                Some(PreparedLexicalCompletionBatch::prepare(
                    context,
                    std::iter::once(LexicalCompletionChange::Assign {
                        environment,
                        binding: binding.binding,
                        scope,
                        source,
                    }),
                )?)
            }
            (None, None) => None,
            _ => unreachable!("lexical assignment retains exactly one source generation"),
        };
        let session = if let Some((_, binding)) = &global {
            let context = context.ok_or_else(|| {
                frame_error(
                    "RunMat:Workspace:SessionContextRequired",
                    "session-global assignment requires its runtime context",
                )
            })?;
            Some(match binding {
                GlobalBindingTarget::Named(bound) => self
                    .session_variables
                    .prepare_store_global_with_context(context, bound.as_str(), value.clone())?,
                GlobalBindingTarget::LegacySlot(bound) => self
                    .session_variables
                    .prepare_store_global_slot_with_context(context, *bound, value.clone())?,
            })
        } else {
            None
        };
        if lexical.is_some() && self.should_publish() && self.delegate().is_some() {
            return Err(frame_error(
                "RunMat:Workspace:LexicalPublication",
                "lexical frame cannot publish through an external workspace delegate",
            ));
        }
        if let Some(session) = &session {
            session.checkpoint(context.expect("prepared session has a context"))?;
        }
        if let Some(lexical) = lexical.as_mut() {
            let context = context.expect("prepared lexical assignment has a context");
            let retired = lexical.commit_with_participant(context, &mut frame)?;
            // Release the frame mutation phase before a displaced lexical or
            // frame value can notify a Foreign/GC retirement observer.
            drop(frame);
            drop(retired);
            return Ok(());
        }
        frame.checkpoint_before_publication()?;
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                match (global.is_some(), context) {
                    (true, Some(context)) => delegate.remove_with_context(context, name)?,
                    (true, None) => delegate.remove(name)?,
                    (false, _) => publish(delegate.as_ref(), context)?,
                }
            }
        }
        if let Some(session) = session {
            session.commit();
        }
        frame.commit();
        Ok(())
    }

    pub(super) fn publish_executor_assignment_lease(
        &self,
        context: &RuntimeContext,
        slot: usize,
        name: &str,
        value: runmat_runtime::workspace::WorkspaceSlotReadLease,
    ) -> Result<(), RuntimeError> {
        let delegate_value = value.clone();
        let frame_value = value.clone();
        self.publish_assignment(
            AssignmentPublication {
                context: Some(context),
                executor_slot: Some(slot),
                name,
                value: value.value(),
                retained: Some(frame_value),
                reconcile_var: false,
            },
            |delegate, _| {
                delegate.assign_executor_slot_lease_with_context(
                    context,
                    slot,
                    name,
                    delegate_value,
                )
            },
        )
    }

    pub(super) fn clear_in_context(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_in_context(context)?;
        let frame = super::super::workspace::prepare_workspace_frame_clear(self.frame, context)?;
        frame.checkpoint_before_publication()?;
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                match context {
                    Some(context) => delegate.clear_with_context(context)?,
                    None => delegate.clear()?,
                }
            }
        }
        frame.commit();
        Ok(())
    }

    pub(super) fn remove_in_context(
        &self,
        context: Option<&RuntimeContext>,
        name: &str,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_in_context(context)?;
        let frame =
            super::super::workspace::prepare_workspace_frame_remove(self.frame, context, name)?;
        frame.checkpoint_before_publication()?;
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                match context {
                    Some(context) => delegate.unbind_base_global_with_context(context, name)?,
                    None => delegate.unbind_base_global(name)?,
                }
            }
        }
        frame.commit();
        Ok(())
    }
}
