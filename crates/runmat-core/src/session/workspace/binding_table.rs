//! Canonical physical owner of compiler workspace bindings.
//!
//! Only the table's exact hashbrown layout is allocated through the Runtime
//! operation gate. Names and deep ABI keys have independent retained owners;
//! a table rehash moves those handles rather than copying their payloads.

use super::super::SessionWorkspaceBinding;
use hashbrown::HashMap;
use runmat_runtime::context::host_allocation::{
    frame_hash_table_retained_bytes, FrameCollectionGate, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt, OperationCollectionAllocator,
};
use runmat_runtime::context::{RuntimeContext, RuntimeContextLocalId};
use runmat_runtime::workspace::WorkspaceName;
use std::collections::hash_map::RandomState;
use std::ops::Deref;

type Entries =
    HashMap<WorkspaceName, SessionWorkspaceBinding, RandomState, OperationCollectionAllocator>;

#[derive(Debug)]
pub(in crate::session) struct BindingTable {
    entries: Entries,
    gate: FrameCollectionGate,
    runtime: RuntimeContextLocalId,
    full_capacity: usize,
}

impl BindingTable {
    pub(in crate::session) fn new_for_context(
        context: &RuntimeContext,
    ) -> Result<Self, HostAllocationError> {
        let plan = HostAllocationPlan::new(0, FrameCollectionGate::retained_bytes()?)?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)?;
        let gate = match receipt.as_ref() {
            Some(receipt) => FrameCollectionGate::prepare(context, receipt)?,
            None => FrameCollectionGate::unbounded(),
        };
        let entries = HashMap::with_hasher_in(RandomState::new(), gate.allocator());
        Ok(Self {
            entries,
            gate,
            runtime: context.local_identity(),
            full_capacity: 0,
        })
    }

    pub(super) fn growth_retained_bytes(
        &self,
        additional: usize,
    ) -> Result<u64, HostAllocationError> {
        let required = self
            .entries
            .len()
            .checked_add(additional)
            .ok_or(HostAllocationError::Overflow)?;
        if required <= self.entries.capacity() {
            return Ok(0);
        }
        let fresh_capacity = required.max(
            self.full_capacity
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?,
        );
        frame_hash_table_retained_bytes::<WorkspaceName, SessionWorkspaceBinding>(fresh_capacity)
    }

    pub(super) fn reserve_admitted(
        &mut self,
        context: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
        additional: usize,
    ) -> Result<(), HostAllocationError> {
        if self.runtime != context.local_identity() {
            return Err(HostAllocationError::StaleAuthority);
        }
        let _installed = match receipt {
            Some(receipt) => self.gate.install(context, receipt)?,
            None => self.gate.install_unbounded()?,
        };
        self.entries
            .try_reserve(additional)
            .map_err(|_| HostAllocationError::Overflow)?;
        self.full_capacity = self.full_capacity.max(self.entries.capacity());
        Ok(())
    }

    pub(super) fn reserve_unbounded(
        &mut self,
        additional: usize,
    ) -> Result<(), HostAllocationError> {
        let _installed = self.gate.install_unbounded()?;
        self.entries
            .try_reserve(additional)
            .map_err(|_| HostAllocationError::Overflow)?;
        self.full_capacity = self.full_capacity.max(self.entries.capacity());
        Ok(())
    }

    /// Insert only after a selected operation reserved the table's physical
    /// growth and prepared the independent name/key owners.
    pub(super) fn insert_prepared(
        &mut self,
        name: WorkspaceName,
        binding: SessionWorkspaceBinding,
    ) {
        // hashbrown may reserve before detecting an occupied key. Replacing
        // an admitted binding must not grow a full table during publication.
        if let Some(current) = self.entries.get_mut(name.as_str()) {
            *current = binding;
            return;
        }
        assert!(
            self.entries.len() < self.entries.capacity(),
            "binding insertion requires a prepared table slot"
        );
        self.entries.insert(name, binding);
    }

    pub(super) fn get_mut(&mut self, name: &str) -> Option<&mut SessionWorkspaceBinding> {
        self.entries.get_mut(name)
    }

    pub(super) fn remove_prepared(&mut self, name: &str) {
        self.entries.remove(name);
    }

    pub(super) fn clear_prepared(&mut self) {
        self.entries.clear();
    }
}

impl Deref for BindingTable {
    type Target = Entries;

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::WorkspaceBindingKey;
    use runmat_runtime::execution::RuntimeExecutionService;
    use std::rc::Rc;

    #[test]
    fn replacing_an_occupied_binding_does_not_grow_a_full_table() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let mut table = BindingTable::new_for_context(&context).unwrap();
        table.reserve_admitted(&context, None, 3).unwrap();
        for (slot, name) in ["a", "b", "c"].into_iter().enumerate() {
            table.insert_prepared(
                WorkspaceName::unbounded(name),
                SessionWorkspaceBinding::unbounded(
                    WorkspaceBindingKey::Interactive {
                        session: uuid::Uuid::nil(),
                        name: runmat_hir::BindingName(name.into()),
                    },
                    slot,
                ),
            );
        }
        assert_eq!(table.len(), table.capacity());
        let capacity = table.capacity();
        table.insert_prepared(
            WorkspaceName::unbounded("b"),
            SessionWorkspaceBinding::unbounded(
                WorkspaceBindingKey::Interactive {
                    session: uuid::Uuid::nil(),
                    name: runmat_hir::BindingName("b".into()),
                },
                7,
            ),
        );
        assert_eq!(table.len(), 3);
        assert_eq!(table.capacity(), capacity);
        assert_eq!(table["b"].slot, 7);
    }
}
