//! Atomic preparation of a coalesced dynamic workspace completion.

use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::{
    WorkspaceName, WorkspacePublicationBatch, WorkspacePublicationOrigin, WorkspacePublicationToken,
};
use runmat_runtime::RuntimeError;

use super::patch::{PendingBindingAction, PendingBindingPatch, PreparedPatchAction};
use super::pending::{PendingBindingParticipant, PendingBindingStart};
use super::semantic::PlannedKey;
use super::{
    allocation_error, checked_add, exhausted_slot_error, publication_error, retained_exact_bytes,
    BindingPublication,
};

impl BindingPublication {
    pub(super) fn prepare_completion_patch(
        &self,
        context: Option<&RuntimeContext>,
        changes: WorkspacePublicationBatch<'_>,
    ) -> Result<Option<WorkspacePublicationToken>, RuntimeError> {
        let context = context.ok_or_else(|| {
            publication_error("workspace completion requires its runtime context")
        })?;
        let stamp = changes.publication_stamp().ok_or_else(|| {
            publication_error("workspace completion is missing its canonical identity")
        })?;
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| publication_error("workspace completion preparation is reentrant"))?;
        if state.prospective.is_some() || state.final_token.is_some() {
            return Err(publication_error(
                "workspace publication already has a prepared phase",
            ));
        }
        let action_upper = changes.len();
        let mut retained = PendingBindingPatch::replacement_retained_bytes(action_upper)
            .map_err(|cause| allocation_error(Some(context), cause))?;
        // A first completion temporarily owns both the empty participant and
        // its detached replacement until Runtime commits the replacement.
        let mut start = None;
        if state.pending.is_none() {
            let reserved = PendingBindingStart::reserve(&self.bindings, context, stamp)
                .map_err(|cause| allocation_error(Some(context), cause))?;
            checked_add(
                Some(context),
                &mut retained,
                PendingBindingParticipant::owner_retained_bytes()
                    .map_err(|cause| allocation_error(Some(context), cause))?,
            )?;
            checked_add(
                Some(context),
                &mut retained,
                reserved
                    .table_growth_retained_bytes(action_upper)
                    .map_err(|cause| allocation_error(Some(context), cause))?,
            )?;
            start = Some(reserved);
        } else {
            let pending = state.pending.as_ref().expect("pending participant exists");
            if !pending.matches_completion(context, stamp) {
                return Err(publication_error("workspace completion is not contiguous"));
            }
            checked_add(
                Some(context),
                &mut retained,
                pending
                    .table_growth_retained_bytes_for_insertions(action_upper)
                    .map_err(|cause| allocation_error(Some(context), cause))?,
            )?;
        }
        for origin in changes.iter() {
            let Some(name) = origin.name() else {
                continue;
            };
            let name_bytes = WorkspaceName::requested_bytes(name).map_err(|cause| {
                allocation_error(Some(context), HostAllocationError::Numeric(cause))
            })?;
            checked_add(
                Some(context),
                &mut retained,
                retained_exact_bytes(Some(context), name_bytes)?,
            )?;
            checked_add(
                Some(context),
                &mut retained,
                self.key_retained(Some(context), planned_key(origin, name))?,
            )?;
        }
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, retained)
                .map_err(|cause| allocation_error(Some(context), cause))?,
        )
        .map_err(|cause| allocation_error(Some(context), cause))?;
        // A first participant remains local until every fallible replacement
        // preparation succeeds. Failure must release its lease rather than
        // strand an empty participant without an external rollback token.
        let mut fresh = start
            .map(|start| start.finish(context, receipt.as_ref()))
            .transpose()
            .map_err(|cause| allocation_error(Some(context), cause))?;
        let participant = match fresh.as_mut() {
            Some(participant) => participant,
            None => state.pending.as_mut().expect("pending participant exists"),
        };
        let mut replacement =
            PendingBindingPatch::prepare_replacement(Some(context), receipt.as_ref(), action_upper)
                .map_err(|cause| allocation_error(Some(context), cause))?;
        for origin in changes.iter() {
            let planned = match origin.name() {
                Some(name) => participant
                    .with_installed_binding(name, |installed| {
                        super::semantic::plan(self, Some(&replacement), installed, origin)
                    })
                    .map_err(|cause| allocation_error(Some(context), cause))?,
                None => super::semantic::plan(self, Some(&replacement), None, origin),
            };
            if planned.clear {
                replacement.push_replacement_clear();
            } else if !planned.noop {
                let name = self.prepare_name(context, receipt.as_ref(), &planned)?;
                let action = if planned.retire {
                    PendingBindingAction::Retire
                } else {
                    PendingBindingAction::Upsert {
                        slot: planned.slot,
                        key: self.prepare_key(context, receipt.as_ref(), planned.key)?,
                        introduces_binding: planned.introduces_binding,
                    }
                };
                replacement.push_replacement_action(PreparedPatchAction { name, action });
            }
        }
        participant
            .validate_replacement_slots(&replacement)
            .map_err(|cause| {
                if cause == HostAllocationError::Overflow {
                    exhausted_slot_error()
                } else {
                    allocation_error(Some(context), cause)
                }
            })?;
        let external = self.next_token()?;
        let internal = participant
            .prepare_replacement(context, receipt.as_ref(), replacement, stamp)
            .map_err(|cause| allocation_error(Some(context), cause))?;
        if let Some(participant) = fresh {
            state.pending = Some(participant);
        }
        state.prospective = Some((external, internal));
        Ok(Some(external))
    }
}

fn planned_key<'a>(origin: WorkspacePublicationOrigin<'a>, name: &'a str) -> PlannedKey<'a> {
    match origin {
        WorkspacePublicationOrigin::ExecutorSlot {
            global: Some(target),
            ..
        }
        | WorkspacePublicationOrigin::DynamicEffect {
            global: Some(target),
            ..
        }
        | WorkspacePublicationOrigin::BindGlobal { target, .. } => PlannedKey::Global(target),
        WorkspacePublicationOrigin::ExecutorSlot { .. } => PlannedKey::Source(name),
        WorkspacePublicationOrigin::DynamicEffect { .. } => PlannedKey::Interactive(name),
        WorkspacePublicationOrigin::UnbindGlobal { .. }
        | WorkspacePublicationOrigin::Retired { .. }
        | WorkspacePublicationOrigin::Cleared => PlannedKey::Preserve,
    }
}
