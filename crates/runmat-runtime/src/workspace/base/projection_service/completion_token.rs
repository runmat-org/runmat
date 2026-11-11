//! Prepared whole-workspace completion shared by Base and Native callers.
//!
//! All collection growth, value ownership and observer preparation precede
//! the short commit gate. A second physical participant may install while
//! both Base borrows are held; old owners retire only after both installs and
//! observer finalization.

use std::cell::RefMut;
use std::rc::Rc;

use crate::call::lexical::PreparedLexicalCompletionBatch;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::workspace::{
    WorkspaceCompletionIdentity, WorkspaceCompletionParticipant, WorkspaceCompletionSource,
};
use crate::RuntimeError;

use super::super::commit::PreparedDirectCommit;
use super::super::owner::{BaseWorkspaceOwner, BaseWorkspaceState};
use super::super::projection::PreparedOrderedProjection;
use super::super::storage::{PreparedPrivateTransition, PrivateLiveStorage};
use super::super::{base_workspace_error_to_runtime, BaseWorkspaceError};
use super::publication::{
    PreparedWorkspacePublication, WorkspacePublicationBatch, WorkspacePublicationStamp,
};
use super::{
    publication_preparation_error, workspace_admission_error, BaseWorkspaceProjectionService,
    ProjectionServiceState,
};

/// One entirely prepared Base completion. The candidate and its authentic
/// pending delta are never installed before the short final commit gate.
pub struct PreparedWorkspaceCompletion {
    owner: BaseWorkspaceOwner,
    service: Option<BaseWorkspaceProjectionService>,
    identity: WorkspaceCompletionIdentity,
    body: PreparedCompletionBody,
    private_receipt: Option<HostAllocationReceipt>,
}

enum PreparedCompletionBody {
    Empty,
    Ordered {
        candidate: PreparedOrderedProjection,
        canonical: PreparedDirectCommit,
        observation: Option<PreparedWorkspacePublication>,
    },
    Private {
        canonical: PreparedPrivateTransition,
        projection: PreparedPrivateTransition,
    },
}

impl PreparedWorkspaceCompletion {
    pub(in crate::workspace::base) fn for_direct_owner(
        owner: &BaseWorkspaceOwner,
        context: &RuntimeContext,
        identity: WorkspaceCompletionIdentity,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<Self, RuntimeError> {
        let validate = || -> Result<(), RuntimeError> {
            let state = owner
                .state
                .try_borrow()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            state
                .ensure_open()
                .map_err(base_workspace_error_to_runtime)?;
            if identity.issuer() != Rc::as_ptr(&owner.state) as usize
                || identity.revision() != state.revision
                || identity.pending_generation() != 0
            {
                return Err(base_workspace_error_to_runtime(
                    BaseWorkspaceError::CompletionConflict,
                ));
            }
            Ok(())
        };
        validate()?;
        if source.is_empty() {
            HostAllocationReceipt::policy(Some(context))
                .map_err(|cause| workspace_admission_error(context, cause))?;
            return Ok(Self {
                owner: owner.clone(),
                service: None,
                identity,
                body: PreparedCompletionBody::Empty,
                private_receipt: None,
            });
        }
        owner
            .capture_stable_with_context(context)
            .map_err(base_workspace_error_to_runtime)?;
        let policy = HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let projection = if policy == HostAllocationPolicy::Managed {
            owner
                .begin_managed_completion_projection()
                .map_err(base_workspace_error_to_runtime)?
        } else {
            owner
                .begin_projection()
                .map_err(base_workspace_error_to_runtime)?
        };
        let (completion, effect) = if policy == HostAllocationPolicy::Managed {
            let retained = projection
                .completion_retained_bytes(source)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            let plan = HostAllocationPlan::new(0, retained)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            let receipt = HostAllocationReceipt::admit(Some(context), plan)
                .map_err(|cause| workspace_admission_error(context, cause))?
                .ok_or_else(|| managed_receipt_unavailable(context))?;
            let permit = receipt
                .output_permit(context)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            let completion = projection
                .prepare_completion(context, source, &permit)
                .map_err(base_workspace_error_to_runtime)?;
            let effect = completion
                .prepare_effect(&permit)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            receipt
                .checkpoint(context)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            (completion, effect)
        } else {
            let completion = projection
                .prepare_completion_unbounded(source)
                .map_err(base_workspace_error_to_runtime)?;
            let effect = completion.prepare_effect_unbounded();
            (completion, effect)
        };
        validate()?;
        let mut candidate = projection
            .candidate_from_completion(completion)
            .map_err(base_workspace_error_to_runtime)?;
        let canonical = candidate
            .prepare_terminal_commit(effect, Some(context))
            .map_err(|(error, _effect)| base_workspace_error_to_runtime(error))?;
        Ok(Self {
            owner: owner.clone(),
            service: None,
            identity,
            body: PreparedCompletionBody::Ordered {
                candidate,
                canonical,
                observation: None,
            },
            private_receipt: None,
        })
    }

    pub(in crate::workspace::base) fn for_projection(
        service: &BaseWorkspaceProjectionService,
        context: &RuntimeContext,
        identity: WorkspaceCompletionIdentity,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<Self, RuntimeError> {
        service.ensure_mutation_context(context)?;
        let validate = || -> Result<(), RuntimeError> {
            let state = service.borrow()?;
            state
                .ordered_view()
                .map_err(base_workspace_error_to_runtime)?;
            if identity.issuer() != Rc::as_ptr(&service.state) as usize
                || identity.revision() != state.revision()
                || identity.pending_generation() != state.pending_generation()
            {
                return Err(base_workspace_error_to_runtime(
                    BaseWorkspaceError::CompletionConflict,
                ));
            }
            state
                .ensure_effect_authority()
                .map_err(base_workspace_error_to_runtime)
        };
        validate()?;
        let owner = service
            .borrow()?
            .canonical_owner()
            .map_err(base_workspace_error_to_runtime)?;
        if source.is_empty() {
            HostAllocationReceipt::policy(Some(context))
                .map_err(|cause| workspace_admission_error(context, cause))?;
            return Ok(Self {
                owner,
                service: Some(service.clone()),
                identity,
                body: PreparedCompletionBody::Empty,
                private_receipt: None,
            });
        }
        service.capture_canonical_prefix(Some(context))?;
        validate()?;
        let state = service.borrow()?;
        let upper = state
            .publication_upper_bound()
            .map_err(base_workspace_error_to_runtime)?
            .checked_add(source.len())
            .ok_or_else(publication_preparation_error)?;
        drop(state);
        let policy = HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;
        let (completion, effect) = if policy == HostAllocationPolicy::Managed {
            let retained = service
                .borrow()?
                .completion_retained_bytes(source)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            let publication = service.published_effect_retained_bytes(context, upper)?;
            let plan = HostAllocationPlan::new(
                0,
                retained
                    .checked_add(publication)
                    .ok_or_else(publication_preparation_error)?,
            )
            .map_err(|cause| workspace_admission_error(context, cause))?;
            let receipt = HostAllocationReceipt::admit(Some(context), plan)
                .map_err(|cause| workspace_admission_error(context, cause))?
                .ok_or_else(|| managed_receipt_unavailable(context))?;
            let permit = receipt
                .output_permit(context)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            let state = service.borrow()?;
            validate()?;
            let completion = state
                .prepare_completion(context, source, &permit)
                .map_err(base_workspace_error_to_runtime)?;
            let effect = completion
                .prepare_effect(&permit)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            drop(state);
            let capacity = service.prepare_published_effect(context, upper, &permit)?;
            receipt
                .checkpoint(context)
                .map_err(|cause| workspace_admission_error(context, cause))?;
            validate()?;
            service.install_published_effect(capacity)?;
            (completion, effect)
        } else {
            let state = service.borrow()?;
            let completion = state
                .prepare_completion_unbounded(source)
                .map_err(base_workspace_error_to_runtime)?;
            let effect = completion.prepare_effect_unbounded();
            drop(state);
            service
                .state
                .try_borrow_mut()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?
                .published_effect
                .prepare_unbounded_for_upper_bound(upper)?;
            (completion, effect)
        };
        let observation = if let Some(observer) = &service.publication_observer {
            {
                let mut state = service.state.try_borrow_mut().map_err(|_| {
                    base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict)
                })?;
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
            // The protocol may synchronously reenter workspace reads. Hold
            // only an immutable projection view while it prepares metadata.
            let state = service
                .state
                .try_borrow()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            let view = completion
                .publication_view(&state.projection)
                .map_err(base_workspace_error_to_runtime)?;
            let stamp = WorkspacePublicationStamp::issued(
                Rc::as_ptr(&service.state) as usize,
                view.revision(),
                view.before_generation(),
                view.generation(),
            );
            let prepared = observer.prepare_completion_publication(
                Some(context),
                WorkspacePublicationBatch::prospective(
                    &view,
                    state.published_effect.publication_indices(),
                )
                .with_stamp(stamp),
            );
            drop(state);
            match prepared {
                Ok(prepared) => Some(prepared),
                Err(error) => {
                    service.finish_publication_indices()?;
                    return Err(error);
                }
            }
        } else {
            None
        };
        let prepared = (|| {
            validate()?;
            let projection = service.borrow()?;
            let mut candidate = projection
                .candidate_from_completion(completion)
                .map_err(base_workspace_error_to_runtime)?;
            drop(projection);
            let canonical = candidate
                .prepare_terminal_commit(effect, Some(context))
                .map_err(|(error, _effect)| base_workspace_error_to_runtime(error))?;
            Ok::<_, RuntimeError>((candidate, canonical))
        })();
        let (candidate, canonical) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                // The observer's staged token aborts when this function
                // returns. Release its prospective index scratch first so
                // the unchanged pending publication remains retryable.
                service.finish_publication_indices()?;
                return Err(error);
            }
        };
        Ok(Self {
            owner,
            service: Some(service.clone()),
            identity,
            body: PreparedCompletionBody::Ordered {
                candidate,
                canonical,
                observation,
            },
            private_receipt: None,
        })
    }

    pub(in crate::workspace::base) fn for_private_projection(
        service: &BaseWorkspaceProjectionService,
        context: &RuntimeContext,
        identity: WorkspaceCompletionIdentity,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<Self, RuntimeError> {
        if source.is_empty() {
            service.ensure_mutation_context(context)?;
            HostAllocationReceipt::policy(Some(context))
                .map_err(|cause| workspace_admission_error(context, cause))?;
            let projection = service.borrow()?;
            let private = projection.private_storage().ok_or_else(|| {
                base_workspace_error_to_runtime(BaseWorkspaceError::PrivateProjectionRequired)
            })?;
            if identity.issuer() != Rc::as_ptr(&service.state) as usize
                || identity.revision() != projection.revision()
                || identity.pending_generation() != private.epoch()
            {
                return Err(base_workspace_error_to_runtime(
                    BaseWorkspaceError::CompletionConflict,
                ));
            }
            projection
                .ensure_effect_authority()
                .map_err(base_workspace_error_to_runtime)?;
            let owner = projection
                .canonical_owner()
                .map_err(base_workspace_error_to_runtime)?;
            return Ok(Self {
                owner,
                service: Some(service.clone()),
                identity,
                body: PreparedCompletionBody::Empty,
                private_receipt: None,
            });
        }
        let (owner, canonical, projection, receipt) =
            service.prepare_private_completion_token(context, identity, source)?;
        Ok(Self {
            owner,
            service: Some(service.clone()),
            identity,
            body: PreparedCompletionBody::Private {
                canonical,
                projection,
            },
            private_receipt: receipt,
        })
    }

    /// Install both prepared physical participants as one unobservable
    /// completion. Observer checkpoint precedes both mutable guards; its
    /// final callback and every retired value wait until both guards end.
    pub fn commit_with_lexical(
        mut self,
        context: &RuntimeContext,
        lexical: &mut PreparedLexicalCompletionBatch,
    ) -> Result<(), RuntimeError> {
        self.preflight_observation(context)?;
        let lexical_ready = lexical.checkpoint(context)?;
        let base_ready = self.lock_base()?;
        let lexical_retired = base_ready.commit_with(|| lexical_ready.commit());
        drop(lexical_retired);
        Ok(())
    }

    /// Publish a prevalidated source frame in the same short Base+lexical
    /// transaction. The participant's displaced frame is returned only after
    /// both mutation guards and observer finalization have retired.
    pub fn commit_with_lexical_and_participant<P>(
        mut self,
        context: &RuntimeContext,
        lexical: &mut PreparedLexicalCompletionBatch,
        participant: &mut P,
    ) -> Result<P::Retirement, RuntimeError>
    where
        P: WorkspaceCompletionParticipant,
    {
        self.preflight_observation(context)?;
        participant.checkpoint_before_install(context)?;
        let lexical_ready = lexical.checkpoint(context)?;
        let base_ready = self.lock_base()?;
        let (lexical_retired, frame_retired) =
            base_ready.commit_with(|| lexical_ready.commit_with_participant(participant));
        drop(lexical_retired);
        Ok(frame_retired)
    }

    fn preflight_observation(&mut self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        if let Some(service) = &self.service {
            service.ensure_mutation_context(context)?;
        }
        HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| workspace_admission_error(context, cause))?;
        if let Some(receipt) = &self.private_receipt {
            receipt
                .checkpoint(context)
                .map_err(|cause| workspace_admission_error(context, cause))?;
        }
        if let PreparedCompletionBody::Ordered {
            observation: Some(observation),
            ..
        } = &mut self.body
        {
            observation.checkpoint(Some(context))?;
        }
        Ok(())
    }

    /// Final Base authority fence. No language execution or external
    /// observer callback may occur between this gate and composite install.
    fn lock_base(&mut self) -> Result<ReadyWorkspaceCompletion<'_>, RuntimeError> {
        let service_state = if let Some(service) = &self.service {
            let state = service
                .state
                .try_borrow_mut()
                .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
            let pending_generation = match (&self.body, state.projection.private_storage()) {
                (
                    PreparedCompletionBody::Private { .. } | PreparedCompletionBody::Empty,
                    Some(private),
                ) => private.epoch(),
                (PreparedCompletionBody::Ordered { .. } | PreparedCompletionBody::Empty, None) => {
                    state
                        .projection
                        .ordered_view()
                        .map_err(base_workspace_error_to_runtime)?
                        .pending()
                        .generation()
                }
                _ => {
                    return Err(base_workspace_error_to_runtime(
                        BaseWorkspaceError::PrivateProjectionRequired,
                    ))
                }
            };
            if self.identity.issuer() != Rc::as_ptr(&service.state) as usize
                || self.identity.revision() != state.projection.revision()
                || self.identity.pending_generation() != pending_generation
            {
                return Err(base_workspace_error_to_runtime(
                    BaseWorkspaceError::CompletionConflict,
                ));
            }
            state
                .projection
                .ensure_effect_authority()
                .map_err(base_workspace_error_to_runtime)?;
            Some(state)
        } else {
            None
        };
        let canonical_state = self
            .owner
            .state
            .try_borrow_mut()
            .map_err(|_| base_workspace_error_to_runtime(BaseWorkspaceError::BorrowConflict))?;
        canonical_state
            .ensure_open()
            .map_err(base_workspace_error_to_runtime)?;
        if matches!(
            &self.body,
            PreparedCompletionBody::Ordered { canonical, .. }
                if canonical.expected_revision() != canonical_state.revision
        ) || (matches!(&self.body, PreparedCompletionBody::Empty)
            && self.identity.revision() != canonical_state.revision)
        {
            return Err(base_workspace_error_to_runtime(
                BaseWorkspaceError::CompletionConflict,
            ));
        }
        if let PreparedCompletionBody::Private {
            canonical: prepared_canonical,
            projection: prepared_projection,
        } = &self.body
        {
            let service = service_state.as_ref().ok_or_else(|| {
                base_workspace_error_to_runtime(BaseWorkspaceError::PrivateProjectionRequired)
            })?;
            let canonical = canonical_state.private_live.as_ref().ok_or_else(|| {
                base_workspace_error_to_runtime(BaseWorkspaceError::PrivateProjectionRequired)
            })?;
            let projection = service.projection.private_storage().ok_or_else(|| {
                base_workspace_error_to_runtime(BaseWorkspaceError::PrivateProjectionRequired)
            })?;
            if !PrivateLiveStorage::pair_matches(
                canonical,
                projection,
                prepared_canonical,
                prepared_projection,
            ) {
                return Err(base_workspace_error_to_runtime(
                    BaseWorkspaceError::CompletionConflict,
                ));
            }
        }
        Ok(ReadyWorkspaceCompletion {
            body: &mut self.body,
            service_state,
            canonical_state,
        })
    }
}

/// The exact short mutation gate for a composite Base plus lexical install.
/// Its typed coordinator calls `commit_with` only with a prepared, infallible
/// lexical batch returning its old owners instead of dropping them.
pub(crate) struct ReadyWorkspaceCompletion<'a> {
    body: &'a mut PreparedCompletionBody,
    service_state: Option<RefMut<'a, ProjectionServiceState>>,
    canonical_state: RefMut<'a, BaseWorkspaceState>,
}

impl ReadyWorkspaceCompletion<'_> {
    fn commit_with<R>(mut self, lexical_install: impl FnOnce() -> R) -> R {
        let mut prior_projection = None;
        let mut direct_candidate = None;
        let mut projection_retired = None;
        let mut publication_retired = None;
        let mut canonical_retired = None;
        let mut receipt = None;
        let mut private_retired = None;
        let body = std::mem::replace(self.body, PreparedCompletionBody::Empty);
        let observation = match body {
            PreparedCompletionBody::Empty => None,
            PreparedCompletionBody::Ordered {
                candidate,
                canonical,
                observation,
            } => {
                let (commit, old_storage) = self.canonical_state.install_direct_commit(canonical);
                canonical_retired = Some(old_storage);
                let (candidate, retired) = candidate.finish_terminal_commit(
                    self.canonical_state.managed_storage.clone(),
                    self.canonical_state.revision,
                );
                projection_retired = Some(retired);
                if let Some(service) = &mut self.service_state {
                    prior_projection = Some(std::mem::replace(&mut service.projection, candidate));
                    publication_retired = Some(service.published_effect.observe(commit.changes()));
                    service.published_effect.finish_publication();
                    service.last_publication_stamp = None;
                } else {
                    direct_candidate = Some(candidate);
                }
                receipt = Some(commit);
                observation
            }
            PreparedCompletionBody::Private {
                canonical,
                projection,
            } => {
                let revision = projection.next_revision();
                let service = self
                    .service_state
                    .as_mut()
                    .expect("private completion has a service guard");
                let canonical_storage = self
                    .canonical_state
                    .private_live
                    .as_mut()
                    .expect("private completion has canonical storage");
                let projection_storage = service
                    .projection
                    .private_storage_mut()
                    .expect("private completion has projection storage");
                let retired = PrivateLiveStorage::install_pair_checked(
                    canonical_storage,
                    projection_storage,
                    canonical,
                    projection,
                )
                .expect("private completion pair was validated under both guards");
                self.canonical_state.revision = revision;
                service.projection.set_private_revision(revision);
                private_retired = Some(retired);
                None
            }
        };
        // Both Base borrows exclude readers. The lexical batch performs only
        // preprepared owner swaps and returns retirement without releasing it.
        let lexical_retired = lexical_install();
        drop(self.service_state);
        drop(self.canonical_state);
        if let Some(observation) = observation {
            observation.commit();
        }
        drop(publication_retired);
        drop(prior_projection);
        drop(direct_candidate);
        drop(projection_retired);
        drop(canonical_retired);
        drop(receipt);
        drop(private_retired);
        lexical_retired
    }
}

fn managed_receipt_unavailable(context: &RuntimeContext) -> RuntimeError {
    workspace_admission_error(
        context,
        HostAllocationError::Resource(
            crate::context::host_effect::HostEffectAllocationError::Unavailable,
        ),
    )
}
