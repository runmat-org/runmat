//! Admitted, coalesced compiler-binding changes for one workspace publication.
//!
//! The hash table provides O(1) replacement by canonical workspace name. A
//! separately admitted name index preserves deterministic lexical slot
//! assignment without depending on hash iteration order. Neither collection
//! retains a permit: each physical block owns its exact claim after the
//! operation which grew it has retired.

use allocator_api2::vec::Vec as AllocatedVec;
use hashbrown::HashMap as AllocatedMap;
use runmat_runtime::context::host_allocation::{
    collection_block_retained_bytes, frame_hash_table_retained_bytes, FrameCollectionGate,
    HostAllocationError, HostAllocationReceipt, OperationCollectionAllocator,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::WorkspaceName;
use std::alloc::Layout;
use std::collections::hash_map::RandomState;

#[path = "patch/application.rs"]
mod application;
#[path = "patch/types.rs"]
mod types;
pub(in crate::session::workspace::publication) use types::{
    PendingBindingAction, PendingBindingKey, PendingBindingSlot, PreparedPatchAction,
    PreparedPatchUpdate,
};

type PatchMap = AllocatedMap<WorkspaceName, PatchEntry, RandomState, OperationCollectionAllocator>;
type NameIndex = AllocatedVec<WorkspaceName, OperationCollectionAllocator>;

#[derive(Debug)]
struct PatchEntry {
    epoch: u64,
    action: PendingBindingAction,
}

/// Logical binding patch retained from the first staged workspace write until
/// its canonical publication. Capacity growth happens during preparation;
/// `apply_update` and `apply_to_table` cannot allocate.
pub(in crate::session::workspace::publication) struct PendingBindingPatch {
    entries: PatchMap,
    lexical_names: NameIndex,
    gate: FrameCollectionGate,
    epoch: u64,
    cleared: bool,
    possible_insertions: usize,
}

impl PendingBindingPatch {
    pub(in crate::session::workspace::publication) fn owner_retained_bytes(
    ) -> Result<u64, HostAllocationError> {
        FrameCollectionGate::retained_bytes()
    }

    pub(in crate::session::workspace::publication) fn first_action_retained_bytes(
    ) -> Result<u64, HostAllocationError> {
        let map = frame_hash_table_retained_bytes::<WorkspaceName, PatchEntry>(1)?;
        let index = collection_block_retained_bytes(
            Layout::array::<WorkspaceName>(1).map_err(|_| HostAllocationError::Overflow)?,
        )?;
        Self::owner_retained_bytes()?
            .checked_add(map)
            .and_then(|bytes| bytes.checked_add(index))
            .ok_or(HostAllocationError::Overflow)
    }

    pub(in crate::session::workspace::publication) fn prepare_empty(
        context: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> Result<Self, HostAllocationError> {
        let gate = match (context, receipt) {
            (Some(context), Some(receipt)) => FrameCollectionGate::prepare(context, receipt)?,
            _ => FrameCollectionGate::unbounded(),
        };
        let allocator = gate.allocator();
        Ok(Self {
            entries: PatchMap::with_hasher_in(RandomState::new(), allocator.clone()),
            lexical_names: NameIndex::new_in(allocator),
            gate,
            epoch: 0,
            cleared: false,
            possible_insertions: 0,
        })
    }

    pub(in crate::session::workspace::publication) fn replacement_retained_bytes(
        action_upper_bound: usize,
    ) -> Result<u64, HostAllocationError> {
        let mut retained = Self::owner_retained_bytes()?;
        retained = retained
            .checked_add(
                frame_hash_table_retained_bytes::<WorkspaceName, PatchEntry>(action_upper_bound)?,
            )
            .ok_or(HostAllocationError::Overflow)?;
        if action_upper_bound != 0 {
            let layout = Layout::array::<WorkspaceName>(action_upper_bound)
                .map_err(|_| HostAllocationError::Overflow)?;
            retained = retained
                .checked_add(collection_block_retained_bytes(layout)?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        Ok(retained)
    }

    pub(in crate::session::workspace::publication) fn prepare_replacement(
        context: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
        action_upper_bound: usize,
    ) -> Result<Self, HostAllocationError> {
        let mut patch = Self::prepare_empty(context, receipt)?;
        if action_upper_bound == 0 {
            return Ok(patch);
        }
        let _installed = match (context, receipt) {
            (Some(context), Some(receipt)) => patch.gate.install(context, receipt)?,
            _ => patch.gate.install_unbounded()?,
        };
        patch
            .entries
            .try_reserve(action_upper_bound)
            .map_err(|_| allocation_failed())?;
        patch
            .lexical_names
            .try_reserve_exact(action_upper_bound)
            .map_err(|_| allocation_failed())?;
        Ok(patch)
    }

    /// Completion preparation owns a detached replacement, so it may install
    /// its coalesced actions directly before the replacement becomes pending.
    pub(in crate::session::workspace::publication) fn push_replacement_action(
        &mut self,
        action: PreparedPatchAction,
    ) {
        if !self.entries.contains_key(action.name.as_str()) {
            debug_assert!(self.entries.len() < self.entries.capacity());
            debug_assert!(self.lexical_names.len() < self.lexical_names.capacity());
        }
        self.apply_update(PreparedPatchUpdate::Action(action));
    }

    pub(in crate::session::workspace::publication) fn push_replacement_clear(&mut self) {
        let update = self
            .next_clear_update()
            .expect("fresh replacement epoch is representable");
        self.apply_update(update);
    }

    pub(in crate::session::workspace::publication) fn current_action(
        &self,
        name: &str,
    ) -> Option<&PendingBindingAction> {
        self.entries
            .get(name)
            .filter(|entry| entry.epoch == self.epoch)
            .map(|entry| &entry.action)
    }

    pub(in crate::session::workspace::publication) fn retained_name(
        &self,
        name: &str,
    ) -> Option<WorkspaceName> {
        self.entries
            .get_key_value(name)
            .and_then(|(name, entry)| (entry.epoch == self.epoch).then(|| name.clone()))
    }

    pub(in crate::session::workspace::publication) const fn was_cleared(&self) -> bool {
        self.cleared
    }

    pub(in crate::session::workspace::publication) fn possible_insertions(&self) -> usize {
        self.possible_insertions
    }

    pub(in crate::session::workspace::publication) fn patch_insertions_after(
        &self,
        name: &str,
        introduces_binding: bool,
    ) -> Result<usize, HostAllocationError> {
        let old = self
            .entries
            .get(name)
            .is_some_and(|entry| entry.epoch == self.epoch && introduces(&entry.action));
        self.possible_insertions
            .checked_sub(usize::from(old))
            .and_then(|count| count.checked_add(usize::from(introduces_binding)))
            .ok_or(HostAllocationError::Overflow)
    }

    pub(in crate::session::workspace::publication) fn insertions_after(
        &self,
        update: &PreparedPatchUpdate,
    ) -> Result<usize, HostAllocationError> {
        match update {
            PreparedPatchUpdate::Noop => Ok(self.possible_insertions),
            PreparedPatchUpdate::Clear { .. } => Ok(0),
            PreparedPatchUpdate::Action(prepared) => {
                let old = self
                    .entries
                    .get(prepared.name.as_str())
                    .is_some_and(|entry| entry.epoch == self.epoch && introduces(&entry.action));
                self.possible_insertions
                    .checked_sub(usize::from(old))
                    .and_then(|count| count.checked_add(usize::from(introduces(&prepared.action))))
                    .ok_or(HostAllocationError::Overflow)
            }
        }
    }

    pub(in crate::session::workspace::publication) fn update_name(
        update: &PreparedPatchUpdate,
    ) -> Option<&str> {
        match update {
            PreparedPatchUpdate::Action(prepared) => Some(prepared.name.as_str()),
            PreparedPatchUpdate::Noop | PreparedPatchUpdate::Clear { .. } => None,
        }
    }

    /// Physical growth required before installing one action for `name`.
    /// Existing names replace in place and require no collection allocation.
    pub(in crate::session::workspace::publication) fn growth_retained_bytes(
        &self,
        name: &str,
    ) -> Result<u64, HostAllocationError> {
        if self.entries.contains_key(name) {
            return Ok(0);
        }
        let target = self
            .entries
            .len()
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let mut retained = if target > self.entries.capacity() {
            frame_hash_table_retained_bytes::<WorkspaceName, PatchEntry>(target)?
        } else {
            0
        };
        if target > self.lexical_names.capacity() {
            let layout = Layout::array::<WorkspaceName>(target)
                .map_err(|_| HostAllocationError::Overflow)?;
            retained = retained
                .checked_add(collection_block_retained_bytes(layout)?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        Ok(retained)
    }

    /// Reserve both collection blocks under the current operation. The
    /// operation guard is dropped before the workspace mutation is installed;
    /// successful blocks retain their own claims.
    pub(in crate::session::workspace::publication) fn reserve_action(
        &mut self,
        context: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
        name: &str,
    ) -> Result<(), HostAllocationError> {
        if self.entries.contains_key(name) {
            return Ok(());
        }
        let _installed = match (context, receipt) {
            (Some(context), Some(receipt)) => self.gate.install(context, receipt)?,
            _ => self.gate.install_unbounded()?,
        };
        self.entries
            .try_reserve(1)
            .map_err(|_| allocation_failed())?;
        self.lexical_names
            .try_reserve_exact(1)
            .map_err(|_| allocation_failed())?;
        Ok(())
    }

    pub(in crate::session::workspace::publication) fn next_clear_update(
        &self,
    ) -> Result<PreparedPatchUpdate, HostAllocationError> {
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(PreparedPatchUpdate::Clear { next_epoch })
    }

    /// Install an already checked update. No allocator is available here.
    pub(in crate::session::workspace::publication) fn apply_update(
        &mut self,
        update: PreparedPatchUpdate,
    ) {
        match update {
            PreparedPatchUpdate::Noop => {}
            PreparedPatchUpdate::Clear { next_epoch } => {
                debug_assert_eq!(next_epoch, self.epoch + 1);
                // Clear is a language barrier, not a retained history event.
                // Release all superseded names and keys now, outside the
                // BindingControl table borrow, while retaining reusable block
                // capacity for the post-clear suffix.
                self.entries.clear();
                self.lexical_names.clear();
                self.epoch = next_epoch;
                self.cleared = true;
                self.possible_insertions = 0;
            }
            PreparedPatchUpdate::Action(prepared) => {
                let lookup_name = prepared.name.clone();
                let mut action = prepared.action;
                if let Some(entry) = self.entries.get_mut(prepared.name.as_str()) {
                    if entry.epoch == self.epoch && introduces(&entry.action) {
                        self.possible_insertions -= 1;
                    }
                    if entry.epoch == self.epoch {
                        inherit_effective_fields(&mut action, &mut entry.action);
                    }
                    entry.epoch = self.epoch;
                    entry.action = action;
                } else {
                    debug_assert!(self.entries.len() < self.entries.capacity());
                    debug_assert!(self.lexical_names.len() < self.lexical_names.capacity());
                    self.lexical_names.push(prepared.name.clone());
                    self.entries.insert(
                        prepared.name,
                        PatchEntry {
                            epoch: self.epoch,
                            action,
                        },
                    );
                }
                let introduced = self
                    .entries
                    .get(lookup_name.as_str())
                    .is_some_and(|entry| entry.epoch == self.epoch && introduces(&entry.action));
                if introduced {
                    self.possible_insertions += 1;
                }
            }
        }
    }
}

fn inherit_effective_fields(
    replacement: &mut PendingBindingAction,
    previous: &mut PendingBindingAction,
) {
    let PendingBindingAction::Upsert {
        slot: replacement_slot,
        key: replacement_key,
        ..
    } = replacement
    else {
        return;
    };
    let PendingBindingAction::Upsert {
        slot: previous_slot,
        key: previous_key,
        ..
    } = previous
    else {
        return;
    };
    if matches!(replacement_slot, PendingBindingSlot::Preserve) {
        *replacement_slot = *previous_slot;
    }
    if matches!(replacement_key, PendingBindingKey::Preserve) {
        *replacement_key = std::mem::replace(previous_key, PendingBindingKey::Preserve);
    }
}

fn introduces(action: &PendingBindingAction) -> bool {
    matches!(
        action,
        PendingBindingAction::Upsert {
            introduces_binding: true,
            ..
        }
    )
}

fn allocation_failed() -> HostAllocationError {
    HostAllocationError::Numeric(runmat_value::ExactHostNumericAllocationError::AllocationFailed)
}
