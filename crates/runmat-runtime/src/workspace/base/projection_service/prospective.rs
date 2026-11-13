//! Short-lived publication preparation around one authentic projection install.

use super::*;
use crate::workspace::base::projection::{
    PreparedProjectionCompletion, PreparedProjectionMutation, PreparedUnboundedProjectionMutation,
    UnboundedProjectionAction,
};

impl BaseWorkspaceProjectionService {
    pub(super) fn install_observed_completion(
        &self,
        context: &RuntimeContext,
        completion: PreparedProjectionCompletion,
    ) -> Result<(), RuntimeError> {
        if self.publication_observer.is_some() {
            let mut state = self
                .state
                .try_borrow_mut()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            let ProjectionServiceState {
                projection,
                published_effect,
                ..
            } = &mut *state;
            let view = completion
                .publication_view(projection)
                .map_err(base_workspace_error_to_runtime)?;
            published_effect.prepare_prospective_indices(&view);
        }
        let state = self
            .state
            .try_borrow()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let view = completion
            .publication_view(&state.projection)
            .map_err(base_workspace_error_to_runtime)?;
        let stamp = WorkspacePublicationStamp::issued(
            Rc::as_ptr(&self.state) as usize,
            view.revision(),
            view.before_generation(),
            view.generation(),
        );
        let observation = self
            .publication_observer
            .as_ref()
            .map(|observer| {
                observer.prepare_completion(
                    Some(context),
                    WorkspacePublicationBatch::prospective(
                        &view,
                        state.published_effect.publication_indices(),
                    )
                    .with_stamp(stamp),
                )
            })
            .transpose()?;
        drop(state);

        HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;

        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let retired = state
            .projection
            .ordered_view_mut()
            .map_err(base_workspace_error_to_runtime)?
            .install_completion(completion);
        state.last_publication_stamp = Some(stamp);
        drop(state);
        if let Some(observation) = observation {
            observation.commit();
        }
        drop(retired);
        Ok(())
    }

    pub(super) fn install_unbounded_observed_mutation(
        &self,
        context: &RuntimeContext,
        prepared: PreparedUnboundedProjectionMutation,
    ) -> Result<(), RuntimeError> {
        self.borrow_mut()?
            .reserve_unbounded_mutation_capacity(&prepared)
            .map_err(base_workspace_error_to_runtime)?;
        let projection = self.borrow()?;
        assert_eq!(projection.revision(), prepared.expected_revision);
        assert_eq!(
            projection.pending_generation(),
            prepared.before_generation()
        );
        let origin = match &prepared.action {
            UnboundedProjectionAction::Assignment {
                name, slot, local, ..
            } => {
                let global = (!local).then(|| projection.global_target(name)).flatten();
                match slot {
                    Some(slot) => WorkspacePublicationOrigin::ExecutorSlot {
                        name,
                        slot: *slot,
                        global,
                    },
                    None => WorkspacePublicationOrigin::DynamicEffect { name, global },
                }
            }
            UnboundedProjectionAction::Namespace {
                name,
                action,
                target,
            } => match action {
                super::super::storage::WorkspaceNamespaceAction::Remove => {
                    match projection.global_target(name) {
                        Some(target) => WorkspacePublicationOrigin::BindGlobal { name, target },
                        None => WorkspacePublicationOrigin::Retired { name },
                    }
                }
                super::super::storage::WorkspaceNamespaceAction::BindGlobal => {
                    WorkspacePublicationOrigin::BindGlobal {
                        name,
                        target: target.as_ref().expect("global target"),
                    }
                }
                super::super::storage::WorkspaceNamespaceAction::UnbindGlobal => {
                    WorkspacePublicationOrigin::UnbindGlobal { name }
                }
            },
            UnboundedProjectionAction::Clear => WorkspacePublicationOrigin::Cleared,
        };
        let stamp = WorkspacePublicationStamp::issued(
            Rc::as_ptr(&self.state) as usize,
            projection.revision(),
            prepared.before_generation(),
            prepared.after_generation(),
        );
        let observation = self
            .publication_observer
            .as_ref()
            .map(|observer| {
                observer.prepare_mutation(
                    Some(context),
                    WorkspacePublicationMutation::issued(stamp, origin),
                )
            })
            .transpose()?;
        drop(projection);
        HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        state
            .projection
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)?;
        let retired = state
            .projection
            .ordered_view_mut()
            .map_err(base_workspace_error_to_runtime)?
            .install_unbounded_mutation(prepared);
        state.last_publication_stamp = Some(stamp);
        drop(state);
        if let Some(observation) = observation {
            observation.commit();
        }
        drop(retired);
        Ok(())
    }

    pub(super) fn install_observed_mutation(
        &self,
        context: &RuntimeContext,
        prepared: PreparedProjectionMutation,
    ) -> Result<(), RuntimeError> {
        // The immutable projection borrow authenticates the candidate. The
        // protocol callback is synchronous metadata preparation, never user
        // execution; reentrant mutation fails with BorrowConflict.
        let projection = self.borrow()?;
        let view = prepared
            .publication_view(&projection)
            .map_err(base_workspace_error_to_runtime)?;
        let mutation =
            WorkspacePublicationMutation::prospective(&view, Rc::as_ptr(&self.state) as usize);
        let observation = self
            .publication_observer
            .as_ref()
            .map(|observer| observer.prepare_mutation(Some(context), mutation))
            .transpose()?;
        let stamp = mutation.stamp;
        drop(projection);

        HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;

        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        let retired = state
            .projection
            .install_prepared_mutation(prepared)
            .map_err(base_workspace_error_to_runtime)?;
        state.last_publication_stamp = Some(stamp);
        drop(state);
        if let Some(observation) = observation {
            observation.commit();
        }
        drop(retired);
        Ok(())
    }
}
