//! Admission and materialization of one prospective binding mutation.

use crate::abi::WorkspaceBindingKey;
use crate::session::workspace::binding_key::RetainedBindingKey;
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::{
    WorkspaceName, WorkspacePublicationMutation, WorkspacePublicationToken,
};
use runmat_runtime::RuntimeError;

use super::patch::{
    PendingBindingAction, PendingBindingKey, PreparedPatchAction, PreparedPatchUpdate,
};
use super::pending::{PendingBindingParticipant, PendingBindingStart};
use super::semantic::{PlannedKey, PlannedMutation};
use super::BindingPublication;
use super::{allocation_error, exhausted_slot_error, publication_error, retained_exact_bytes};

impl BindingPublication {
    pub(super) fn source_key_matches(&self, key: &WorkspaceBindingKey, name: &str) -> bool {
        match (&self.origin.source, key) {
            (
                Some(source),
                WorkspaceBindingKey::SourceBinding {
                    source: installed_source,
                    def_path,
                    binding,
                },
            ) => {
                source == installed_source && &self.origin.def_path == def_path && binding.0 == name
            }
            (
                None,
                WorkspaceBindingKey::Interactive {
                    session,
                    name: installed,
                },
            ) => *session == self.session && installed.0 == name,
            _ => false,
        }
    }

    pub(super) fn interactive_key_matches(&self, key: &WorkspaceBindingKey, name: &str) -> bool {
        matches!(
            key,
            WorkspaceBindingKey::Interactive { session, name: installed }
                if *session == self.session && installed.0 == name
        )
    }

    pub(super) fn key_retained(
        &self,
        context: Option<&RuntimeContext>,
        key: PlannedKey<'_>,
    ) -> Result<u64, RuntimeError> {
        match key {
            PlannedKey::Preserve => Ok(0),
            PlannedKey::Source(name) => retained_exact_bytes(
                context,
                RetainedBindingKey::source_requested_bytes(
                    self.origin.source.as_ref(),
                    &self.origin.def_path,
                    name,
                )
                .map_err(|cause| allocation_error(context, HostAllocationError::Numeric(cause)))?,
            ),
            PlannedKey::Interactive(name) => retained_exact_bytes(
                context,
                RetainedBindingKey::interactive_requested_bytes(name).map_err(|cause| {
                    allocation_error(context, HostAllocationError::Numeric(cause))
                })?,
            ),
            PlannedKey::Global(target) => RetainedBindingKey::session_global_retained_bytes(target)
                .map_err(|cause| allocation_error(context, cause)),
        }
    }

    fn update_retained(
        &self,
        context: &RuntimeContext,
        planned: &PlannedMutation<'_>,
        patch_growth: u64,
        table_growth: u64,
    ) -> Result<u64, RuntimeError> {
        let mut retained = patch_growth;
        if planned.noop || planned.clear {
            return Ok(retained);
        }
        retained = retained
            .checked_add(table_growth)
            .ok_or_else(|| allocation_error(Some(context), HostAllocationError::Overflow))?;
        if planned.retained_name.is_none() {
            let bytes = WorkspaceName::requested_bytes(planned.name.expect("named binding action"))
                .map_err(|cause| {
                    allocation_error(Some(context), HostAllocationError::Numeric(cause))
                })?;
            retained = retained
                .checked_add(retained_exact_bytes(Some(context), bytes)?)
                .ok_or_else(|| allocation_error(Some(context), HostAllocationError::Overflow))?;
        }
        retained = retained
            .checked_add(self.key_retained(Some(context), planned.key)?)
            .ok_or_else(|| allocation_error(Some(context), HostAllocationError::Overflow))?;
        Ok(retained)
    }

    pub(super) fn prepare_name(
        &self,
        context: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
        planned: &PlannedMutation<'_>,
    ) -> Result<WorkspaceName, RuntimeError> {
        if let Some(name) = planned.retained_name.clone() {
            return Ok(name);
        }
        let name = planned.name.expect("named binding action");
        match receipt {
            Some(receipt) => receipt
                .with_output_allocation(context, |permit| WorkspaceName::prepare(name, permit))
                .map_err(|cause| allocation_error(Some(context), cause)),
            None => Ok(WorkspaceName::unbounded(name)),
        }
    }

    pub(super) fn prepare_key(
        &self,
        context: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
        key: PlannedKey<'_>,
    ) -> Result<PendingBindingKey, RuntimeError> {
        if matches!(key, PlannedKey::Preserve) {
            return Ok(PendingBindingKey::Preserve);
        }
        let retained = match receipt {
            Some(receipt) => receipt
                .with_output_allocation(context, |permit| match key {
                    PlannedKey::Source(name) => RetainedBindingKey::prepare_source(
                        self.origin.source.as_ref(),
                        &self.origin.def_path,
                        self.session,
                        name,
                        permit,
                    ),
                    PlannedKey::Interactive(name) => {
                        RetainedBindingKey::prepare_interactive(self.session, name, permit)
                    }
                    PlannedKey::Global(target) => {
                        RetainedBindingKey::prepare_session_global_target(
                            self.session,
                            target,
                            permit,
                        )
                    }
                    PlannedKey::Preserve => unreachable!(),
                })
                .map_err(|cause| allocation_error(Some(context), cause))?,
            None => match key {
                PlannedKey::Source(name) => RetainedBindingKey::unbounded(self.source_key(name)),
                PlannedKey::Interactive(name) => {
                    RetainedBindingKey::unbounded(self.interactive_key(name))
                }
                PlannedKey::Global(target) => RetainedBindingKey::unbounded(
                    super::super::global_key::session_global_key(self.session, target),
                ),
                PlannedKey::Preserve => unreachable!(),
            },
        };
        Ok(PendingBindingKey::Replace(retained))
    }

    fn source_key(&self, name: &str) -> WorkspaceBindingKey {
        match &self.origin.source {
            Some(source) => WorkspaceBindingKey::SourceBinding {
                source: source.clone(),
                def_path: self.origin.def_path.clone(),
                binding: runmat_hir::BindingName(name.to_owned()),
            },
            None => self.interactive_key(name),
        }
    }

    fn interactive_key(&self, name: &str) -> WorkspaceBindingKey {
        WorkspaceBindingKey::Interactive {
            session: self.session,
            name: runmat_hir::BindingName(name.to_owned()),
        }
    }

    fn materialize_update(
        &self,
        participant: &PendingBindingParticipant,
        context: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
        planned: PlannedMutation<'_>,
    ) -> Result<PreparedPatchUpdate, RuntimeError> {
        if planned.clear {
            return participant
                .patch()
                .next_clear_update()
                .map_err(|cause| allocation_error(Some(context), cause));
        }
        if planned.noop {
            return Ok(PreparedPatchUpdate::Noop);
        }
        let name = self.prepare_name(context, receipt, &planned)?;
        let action = if planned.retire {
            PendingBindingAction::Retire
        } else {
            PendingBindingAction::Upsert {
                slot: planned.slot,
                key: self.prepare_key(context, receipt, planned.key)?,
                introduces_binding: planned.introduces_binding,
            }
        };
        Ok(PreparedPatchUpdate::Action(PreparedPatchAction {
            name,
            action,
        }))
    }

    pub(super) fn prepare_incremental(
        &self,
        context: &RuntimeContext,
        mutation: WorkspacePublicationMutation<'_>,
    ) -> Result<WorkspacePublicationToken, RuntimeError> {
        if matches!(
            mutation.origin,
            runmat_runtime::workspace::WorkspacePublicationOrigin::ExecutorSlot {
                slot: usize::MAX,
                ..
            }
        ) {
            return Err(exhausted_slot_error());
        }
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| publication_error("workspace publication preparation is reentrant"))?;
        if state.prospective.is_some() || state.final_token.is_some() {
            return Err(publication_error(
                "workspace publication already has a prepared phase",
            ));
        }
        if state.pending.is_none() {
            let start = PendingBindingStart::reserve(&self.bindings, context, mutation.stamp)
                .map_err(|cause| allocation_error(Some(context), cause))?;
            let planned = match mutation.origin.name() {
                Some(name) => start
                    .with_installed_binding(name, |installed| {
                        super::semantic::plan(self, None, installed, mutation.origin)
                    })
                    .map_err(|cause| allocation_error(Some(context), cause))?,
                None => super::semantic::plan(self, None, None, mutation.origin),
            };
            let patch_growth = if planned.noop || planned.clear {
                PendingBindingParticipant::owner_retained_bytes()
            } else {
                PendingBindingParticipant::first_action_retained_bytes()
            }
            .map_err(|cause| allocation_error(Some(context), cause))?;
            let table_growth = start
                .table_growth_retained_bytes(usize::from(planned.introduces_binding))
                .map_err(|cause| allocation_error(Some(context), cause))?;
            let retained = self.update_retained(context, &planned, patch_growth, table_growth)?;
            let receipt = admit(context, retained)?;
            let mut participant = start
                .finish(context, receipt.as_ref())
                .map_err(|cause| allocation_error(Some(context), cause))?;
            let update =
                self.materialize_update(&participant, context, receipt.as_ref(), planned)?;
            validate_slots(&participant, &update, context)?;
            let external = self.next_token()?;
            let internal = participant
                .prepare_update(context, receipt.as_ref(), update, mutation.stamp)
                .map_err(|cause| allocation_error(Some(context), cause))?;
            state.pending = Some(participant);
            state.prospective = Some((external, internal));
            return Ok(external);
        }
        let participant = state.pending.as_mut().expect("pending participant exists");
        if !participant.matches(context, mutation.stamp) {
            return Err(publication_error(
                "workspace publication mutation is not contiguous",
            ));
        }
        let planned = match mutation.origin.name() {
            Some(name) => participant
                .with_installed_binding(name, |installed| {
                    super::semantic::plan(
                        self,
                        Some(participant.patch()),
                        installed,
                        mutation.origin,
                    )
                })
                .map_err(|cause| allocation_error(Some(context), cause))?,
            None => super::semantic::plan(self, Some(participant.patch()), None, mutation.origin),
        };
        let patch_growth = if planned.noop || planned.clear {
            0
        } else {
            participant
                .patch_growth_retained_bytes(mutation.origin.name().expect("named action"))
                .map_err(|cause| allocation_error(Some(context), cause))?
        };
        let table_growth = match mutation.origin.name() {
            Some(name) => participant
                .table_growth_for_action(name, planned.introduces_binding)
                .map_err(|cause| allocation_error(Some(context), cause))?,
            None => 0,
        };
        let retained = self.update_retained(context, &planned, patch_growth, table_growth)?;
        let receipt = admit(context, retained)?;
        let update = self.materialize_update(participant, context, receipt.as_ref(), planned)?;
        validate_slots(participant, &update, context)?;
        let external = self.next_token()?;
        let internal = participant
            .prepare_update(context, receipt.as_ref(), update, mutation.stamp)
            .map_err(|cause| allocation_error(Some(context), cause))?;
        state.prospective = Some((external, internal));
        Ok(external)
    }
}

fn validate_slots(
    participant: &PendingBindingParticipant,
    update: &PreparedPatchUpdate,
    context: &RuntimeContext,
) -> Result<(), RuntimeError> {
    participant.validate_update_slots(update).map_err(|cause| {
        if cause == HostAllocationError::Overflow {
            exhausted_slot_error()
        } else {
            allocation_error(Some(context), cause)
        }
    })
}

fn admit(
    context: &RuntimeContext,
    retained: u64,
) -> Result<Option<HostAllocationReceipt>, RuntimeError> {
    let plan = HostAllocationPlan::new(0, retained)
        .map_err(|cause| allocation_error(Some(context), cause))?;
    HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|cause| allocation_error(Some(context), cause))
}
