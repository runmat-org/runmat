//! Incremental pre-effect reconciliation of compiler workspace bindings.
//!
//! Runtime owns the canonical projection. This observer retains one admitted,
//! coalesced metadata patch across prospective mutations and applies it only
//! after Runtime publishes the matching projection generation.

#[path = "publication/completion.rs"]
mod completion;
#[path = "publication/patch.rs"]
mod patch;
#[path = "publication/pending.rs"]
mod pending;
#[path = "publication/preparation.rs"]
mod preparation;
#[path = "publication/semantic.rs"]
mod semantic;

use super::binding_control::BindingControl;
use super::binding_key::RetainedPublicationSource;
use crate::session::RunMatSession;
use pending::{PendingBindingParticipant, PendingUpdateToken};
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::{
    WorkspacePublicationBatch, WorkspacePublicationMutation, WorkspacePublicationObserver,
    WorkspacePublicationProtocol, WorkspacePublicationToken,
};
use runmat_runtime::RuntimeError;
use std::cell::{Cell, RefCell};

#[derive(Debug)]
struct BindingPublication {
    bindings: BindingControl,
    origin: RetainedPublicationSource,
    session: uuid::Uuid,
    state: RefCell<PublicationState>,
    generation: Cell<u64>,
}

struct PublicationState {
    pending: Option<PendingBindingParticipant>,
    prospective: Option<(WorkspacePublicationToken, PendingUpdateToken)>,
    final_token: Option<WorkspacePublicationToken>,
}

impl std::fmt::Debug for PublicationState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PublicationState")
            .field("pending", &self.pending.is_some())
            .field("prospective", &self.prospective.is_some())
            .field("final_token", &self.final_token)
            .finish()
    }
}

fn publication_error(message: &str) -> RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:WorkspaceBindingPublication")
        .build()
}

fn exhausted_slot_error() -> RuntimeError {
    runmat_runtime::build_runtime_error("workspace binding slot space is exhausted")
        .with_identifier("RunMat:WorkspaceBindingSlots")
        .build()
}

fn allocation_error(context: Option<&RuntimeContext>, cause: HostAllocationError) -> RuntimeError {
    let namespace = context.map_or_else(|| "RunMat".to_owned(), RuntimeContext::error_namespace);
    runmat_runtime::build_runtime_error(format!("workspace binding admission failed: {cause}"))
        .with_identifier(format!("{namespace}:{}", cause.runtime_identifier_suffix()))
        .with_source(cause)
        .build()
}

fn retained_exact_bytes(
    context: Option<&RuntimeContext>,
    bytes: usize,
) -> Result<u64, RuntimeError> {
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes)
            .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?,
        1,
    )
    .map_err(|cause| allocation_error(context, cause))
}

fn checked_add(
    context: Option<&RuntimeContext>,
    total: &mut u64,
    bytes: u64,
) -> Result<(), RuntimeError> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| allocation_error(context, HostAllocationError::Overflow))?;
    Ok(())
}

impl BindingPublication {
    fn next_token(&self) -> Result<WorkspacePublicationToken, RuntimeError> {
        let generation =
            self.generation.get().checked_add(1).ok_or_else(|| {
                publication_error("workspace publication token space is exhausted")
            })?;
        self.generation.set(generation);
        Ok(WorkspacePublicationToken::from_generation(generation))
    }
}

impl WorkspacePublicationProtocol for BindingPublication {
    fn prepare_mutation(
        &self,
        context: Option<&RuntimeContext>,
        mutation: WorkspacePublicationMutation<'_>,
    ) -> Result<Option<WorkspacePublicationToken>, RuntimeError> {
        let context = context.ok_or_else(|| {
            publication_error("prospective workspace publication requires its runtime context")
        })?;
        self.prepare_incremental(context, mutation).map(Some)
    }

    fn commit_mutation(&self, token: WorkspacePublicationToken) {
        let mut state = self.state.borrow_mut();
        let (_, internal) = state
            .prospective
            .take()
            .filter(|(prepared, _)| *prepared == token)
            .expect("workspace mutation token must match its preparation");
        state
            .pending
            .as_mut()
            .expect("pending publication participant exists")
            .commit_update(internal);
    }

    fn abort_mutation(&self, token: WorkspacePublicationToken) {
        let mut state = self.state.borrow_mut();
        if !state
            .prospective
            .is_some_and(|(prepared, _)| prepared == token)
        {
            return;
        }
        let (_, internal) = state.prospective.take().expect("matching update exists");
        if let Some(pending) = state.pending.as_mut() {
            pending.abort_update(internal);
        }
    }

    fn prepare_completion(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<Option<WorkspacePublicationToken>, RuntimeError> {
        self.prepare_completion_patch(context, changes)
    }

    fn prepare_completion_publication(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<WorkspacePublicationToken, RuntimeError> {
        let runtime = context.ok_or_else(|| {
            publication_error("workspace completion requires its runtime context")
        })?;
        let stamp = changes.publication_stamp().ok_or_else(|| {
            publication_error("workspace completion is missing its canonical identity")
        })?;
        let token = self
            .prepare_completion_patch(context, changes)?
            .ok_or_else(|| {
                publication_error("workspace completion did not prepare its participant")
            })?;
        let prepared = (|| {
            let mut state = self
                .state
                .try_borrow_mut()
                .map_err(|_| publication_error("workspace completion finalization is reentrant"))?;
            let (_, internal) = state
                .prospective
                .filter(|(prepared, _)| *prepared == token)
                .ok_or_else(|| publication_error("workspace completion token is stale"))?;
            state
                .pending
                .as_mut()
                .ok_or_else(|| publication_error("workspace completion participant is missing"))?
                .checkpoint_completion_final(runtime, internal, stamp)
                .map_err(|cause| allocation_error(context, cause))?;
            state.final_token = Some(token);
            Ok(token)
        })();
        if prepared.is_err() {
            self.abort_mutation(token);
        }
        prepared
    }

    fn prepare(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<WorkspacePublicationToken, RuntimeError> {
        if changes.is_empty() && self.state.borrow().pending.is_none() {
            return Ok(WorkspacePublicationToken::from_generation(0));
        }
        let context = context.ok_or_else(|| {
            publication_error("workspace publication requires its runtime context")
        })?;
        let stamp = changes.publication_stamp().ok_or_else(|| {
            publication_error("workspace publication is missing its canonical identity")
        })?;
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| publication_error("workspace publication preparation is reentrant"))?;
        if state.prospective.is_some() || state.final_token.is_some() {
            return Err(publication_error(
                "workspace publication already has a prepared phase",
            ));
        }
        let token = self.next_token()?;
        state
            .pending
            .as_mut()
            .ok_or_else(|| publication_error("workspace publication has no prepared mutations"))?
            .checkpoint_final(context, stamp)
            .map_err(|cause| allocation_error(Some(context), cause))?;
        state.final_token = Some(token);
        Ok(token)
    }

    fn checkpoint(
        &self,
        token: WorkspacePublicationToken,
        _context: Option<&RuntimeContext>,
    ) -> Result<(), RuntimeError> {
        if token.generation() == 0 {
            return Ok(());
        }
        if self.state.borrow().final_token != Some(token) {
            return Err(publication_error("workspace publication token is stale"));
        }
        Ok(())
    }

    fn commit(&self, token: WorkspacePublicationToken) {
        if token.generation() == 0 {
            return;
        }
        let mut state = self.state.borrow_mut();
        assert_eq!(state.final_token.take(), Some(token));
        if let Some((prepared, internal)) = state.prospective.take() {
            assert_eq!(prepared, token);
            state
                .pending
                .as_mut()
                .expect("prepared participant exists")
                .commit_update(internal);
        }
        let pending = state.pending.take().expect("prepared participant exists");
        drop(state);
        pending.commit_final();
    }

    fn abort(&self, token: WorkspacePublicationToken) {
        if token.generation() == 0 {
            return;
        }
        let mut state = self.state.borrow_mut();
        if state.final_token == Some(token) {
            state.final_token = None;
            if let Some(pending) = state.pending.as_mut() {
                pending.abort_final();
            }
            if let Some((prepared, internal)) = state.prospective.take() {
                assert_eq!(prepared, token);
                if let Some(pending) = state.pending.as_mut() {
                    pending.abort_update(internal);
                }
            }
        }
    }
}

impl RunMatSession {
    pub(crate) fn workspace_publication_observer_with_context(
        &self,
        context: &RuntimeContext,
    ) -> Result<WorkspacePublicationObserver, RuntimeError> {
        self.prepare_workspace_publication_observer(context)
    }

    fn prepare_workspace_publication_observer(
        &self,
        context: &RuntimeContext,
    ) -> Result<WorkspacePublicationObserver, RuntimeError> {
        let bytes = RetainedPublicationSource::requested_bytes(
            self.active_source_identity.as_ref(),
            self.current_source_name(),
        )
        .map_err(|cause| allocation_error(Some(context), HostAllocationError::Numeric(cause)))?;
        let retained = retained_exact_bytes(Some(context), bytes)?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, retained)
                .map_err(|cause| allocation_error(Some(context), cause))?,
        )
        .map_err(|cause| allocation_error(Some(context), cause))?;
        let origin = match receipt.as_ref() {
            Some(receipt) => receipt
                .with_output_allocation(context, |permit| {
                    RetainedPublicationSource::prepare(
                        self.active_source_identity.as_ref(),
                        self.current_source_name(),
                        permit,
                    )
                })
                .map_err(|cause| allocation_error(Some(context), cause))?,
            None => RetainedPublicationSource::unbounded(
                self.active_source_identity.as_ref(),
                self.current_source_name(),
            )
            .map_err(|cause| {
                allocation_error(Some(context), HostAllocationError::Numeric(cause))
            })?,
        };
        let publication = BindingPublication {
            bindings: self.workspace_bindings.clone(),
            origin,
            session: self.abi_workspace_handle.0,
            state: RefCell::new(PublicationState {
                pending: None,
                prospective: None,
                final_token: None,
            }),
            generation: Cell::new(0),
        };
        WorkspacePublicationObserver::new_with_context(context, publication)
    }
}
