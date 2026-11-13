//! Allocation-free validation and application of one coalesced patch.

#[path = "application/slots.rs"]
mod slots;

use runmat_runtime::context::host_allocation::HostAllocationError;
use runmat_runtime::workspace::WorkspaceName;

use super::{PendingBindingAction, PendingBindingKey, PendingBindingPatch, PendingBindingSlot};
use crate::session::workspace::binding_table::BindingTable;
use crate::session::SessionWorkspaceBinding;

impl PendingBindingPatch {
    pub(in crate::session::workspace::publication) fn validate_for_table(
        &self,
        table: &BindingTable,
    ) -> Result<(), HostAllocationError> {
        let mut largest = if self.cleared {
            None
        } else {
            table.values().map(|binding| binding.slot).max()
        };
        let mut dynamic = 0usize;
        for entry in self
            .entries
            .values()
            .filter(|entry| entry.epoch == self.epoch)
        {
            let PendingBindingAction::Upsert { slot, key, .. } = &entry.action else {
                continue;
            };
            match slot {
                PendingBindingSlot::Preserve => {}
                PendingBindingSlot::Explicit(slot) => {
                    if *slot == usize::MAX {
                        return Err(HostAllocationError::Overflow);
                    }
                    largest = Some(largest.map_or(*slot, |old| old.max(*slot)));
                }
                PendingBindingSlot::Dynamic => {
                    dynamic = dynamic
                        .checked_add(1)
                        .ok_or(HostAllocationError::Overflow)?;
                }
            }
            if self.cleared && matches!(key, PendingBindingKey::Preserve) {
                return Err(HostAllocationError::StaleAuthority);
            }
        }
        if dynamic != 0 {
            let start = largest.map_or(Ok(0), |slot| {
                slot.checked_add(1).ok_or(HostAllocationError::Overflow)
            })?;
            start
                .checked_add(dynamic - 1)
                .ok_or(HostAllocationError::Overflow)?;
        }
        for name in self.lexical_names.iter() {
            let Some(entry) = self.entries.get(name.as_str()) else {
                return Err(HostAllocationError::StaleAuthority);
            };
            if entry.epoch != self.epoch {
                continue;
            }
            if let PendingBindingAction::Upsert { slot, key, .. } = &entry.action {
                let installed = !self.cleared && table.contains_key(name.as_str());
                if matches!(slot, PendingBindingSlot::Preserve) && !installed {
                    return Err(HostAllocationError::StaleAuthority);
                }
                if !installed && matches!(key, PendingBindingKey::Preserve) {
                    return Err(HostAllocationError::StaleAuthority);
                }
            }
        }
        Ok(())
    }

    pub(in crate::session::workspace::publication) fn apply_to_table(
        mut self,
        table: &mut BindingTable,
    ) {
        self.lexical_names.sort_unstable();
        if self.cleared {
            table.clear_prepared();
        }
        let mut largest = if self.cleared {
            None
        } else {
            table.values().map(|binding| binding.slot).max()
        };
        for name in self.lexical_names.iter() {
            let Some(entry) = self.entries.get(name.as_str()) else {
                continue;
            };
            if entry.epoch != self.epoch {
                continue;
            }
            if let PendingBindingAction::Upsert {
                slot: PendingBindingSlot::Explicit(slot),
                ..
            } = &entry.action
            {
                largest = Some(largest.map_or(*slot, |old| old.max(*slot)));
            }
        }
        let mut next_dynamic = largest.map_or(Some(0), |slot| slot.checked_add(1));
        for name in self.lexical_names.drain(..) {
            let Some(entry) = self.entries.remove(name.as_str()) else {
                continue;
            };
            if entry.epoch != self.epoch {
                continue;
            }
            match entry.action {
                PendingBindingAction::Retire => table.remove_prepared(name.as_str()),
                PendingBindingAction::Upsert { slot, key, .. } => {
                    let slot = match slot {
                        PendingBindingSlot::Preserve => None,
                        PendingBindingSlot::Explicit(slot) => Some(slot),
                        PendingBindingSlot::Dynamic => {
                            let slot = next_dynamic
                                .expect("prepared dynamic binding slots remain representable");
                            next_dynamic = slot.checked_add(1);
                            Some(slot)
                        }
                    };
                    apply_upsert(table, name, slot, key);
                }
            }
        }
    }
}

fn apply_upsert(
    table: &mut BindingTable,
    name: WorkspaceName,
    slot: Option<usize>,
    key: PendingBindingKey,
) {
    if let Some(binding) = table.get_mut(name.as_str()) {
        if let Some(slot) = slot {
            binding.slot = slot;
        }
        if let PendingBindingKey::Replace(key) = key {
            binding.replace_key(key);
        }
        return;
    }
    let slot = slot.expect("new compiler binding has a prepared slot");
    let PendingBindingKey::Replace(key) = key else {
        panic!("new compiler binding has a prepared key")
    };
    table.insert_prepared(name, SessionWorkspaceBinding::admitted(key, slot));
}
