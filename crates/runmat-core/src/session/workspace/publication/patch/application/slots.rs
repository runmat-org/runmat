//! Pre-effect validation of explicit and deterministic dynamic slots.

use runmat_runtime::context::host_allocation::HostAllocationError;

use super::super::{
    PendingBindingAction, PendingBindingPatch, PendingBindingSlot, PreparedPatchAction,
    PreparedPatchUpdate,
};
use crate::session::workspace::binding_table::BindingTable;

impl PendingBindingPatch {
    pub(in crate::session::workspace::publication) fn validate_slots_after(
        &self,
        update: &PreparedPatchUpdate,
        table: &BindingTable,
    ) -> Result<(), HostAllocationError> {
        // The committed prefix has already passed this check. A no-op,
        // deletion, clear, or key-only replacement cannot increase its slot
        // tail; keep ordinary repeated scalar stores independent of the
        // number of workspace bindings.
        if matches!(
            update,
            PreparedPatchUpdate::Noop
                | PreparedPatchUpdate::Clear { .. }
                | PreparedPatchUpdate::Action(PreparedPatchAction {
                    action: PendingBindingAction::Retire,
                    ..
                })
                | PreparedPatchUpdate::Action(PreparedPatchAction {
                    action: PendingBindingAction::Upsert {
                        slot: PendingBindingSlot::Preserve,
                        ..
                    },
                    ..
                })
        ) {
            return Ok(());
        }
        let replaced_name = match update {
            PreparedPatchUpdate::Action(action) => Some(action.name.as_str()),
            PreparedPatchUpdate::Noop | PreparedPatchUpdate::Clear { .. } => None,
        };
        let mut largest = if self.cleared {
            None
        } else {
            table.values().map(|binding| binding.slot).max()
        };
        let mut dynamic = 0usize;
        for (name, entry) in self
            .entries
            .iter()
            .filter(|(_, entry)| entry.epoch == self.epoch)
        {
            if replaced_name != Some(name.as_str()) {
                accumulate(&entry.action, &mut largest, &mut dynamic)?;
            }
        }
        if let PreparedPatchUpdate::Action(action) = update {
            accumulate(&action.action, &mut largest, &mut dynamic)?;
        }
        validate_tail(largest, dynamic)
    }
}

fn accumulate(
    action: &PendingBindingAction,
    largest: &mut Option<usize>,
    dynamic: &mut usize,
) -> Result<(), HostAllocationError> {
    let PendingBindingAction::Upsert { slot, .. } = action else {
        return Ok(());
    };
    match slot {
        PendingBindingSlot::Preserve => {}
        PendingBindingSlot::Explicit(slot) => {
            if *slot == usize::MAX {
                return Err(HostAllocationError::Overflow);
            }
            *largest = Some((*largest).map_or(*slot, |old| old.max(*slot)));
        }
        PendingBindingSlot::Dynamic => {
            *dynamic = dynamic
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?;
        }
    }
    Ok(())
}

fn validate_tail(largest: Option<usize>, dynamic: usize) -> Result<(), HostAllocationError> {
    if dynamic == 0 {
        return Ok(());
    }
    let start = largest.map_or(Ok(0), |slot| {
        slot.checked_add(1).ok_or(HostAllocationError::Overflow)
    })?;
    start
        .checked_add(dynamic - 1)
        .ok_or(HostAllocationError::Overflow)?;
    Ok(())
}
