//! One shared frame-alias authority. Detached wire snapshots are explicit;
//! live VM/native adapters only clone the exact control handle.

use super::{FrameGlobalBinding, GlobalBindingTarget};
use crate::context::host_allocation::{
    FrameCollectionGate, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    OperationCollectionAllocator,
};
use crate::context::RuntimeContext;
use crate::workspace::base::WorkspaceName;
use hashbrown::HashMap as AllocatedMap;
use runmat_execution::resource::ExactShared;
use runmat_value::ExactHostAllocationClaim;
use std::cell::{Cell, RefCell};
use std::collections::{hash_map::RandomState, HashMap};

pub(super) type AliasBinding = GlobalBindingTarget;
pub(super) type AliasMap =
    AllocatedMap<usize, AliasBinding, RandomState, OperationCollectionAllocator>;

fn detached(binding: &GlobalBindingTarget) -> FrameGlobalBinding {
    match binding {
        GlobalBindingTarget::Named(name) => FrameGlobalBinding::Named(name.as_str().to_owned()),
        GlobalBindingTarget::LegacySlot(slot) => FrameGlobalBinding::LegacySlot(*slot),
    }
}

#[derive(Debug)]
pub(super) struct BindingOwner {
    pub(super) entries: RefCell<AliasMap>,
    pub(super) gate: RefCell<FrameCollectionGate>,
    pub(super) busy: Cell<bool>,
    // ExactShared deallocates its header before dropping the payload claim.
    pub(super) header_claim: RefCell<Option<ExactHostAllocationClaim>>,
    pub(super) issuer: Cell<Option<crate::context::RuntimeContextLocalId>>,
}

#[derive(Clone, Debug)]
pub struct FrameGlobalBindings(pub(super) ExactShared<BindingOwner>);

pub struct PreparedAliasReplacement {
    target: Option<FrameGlobalBindings>,
    source: Option<FrameGlobalBindings>,
    reserved: bool,
}

/// Displaced alias backing retained until the caller's enclosing publication
/// releases every Base, lexical, and frame mutation guard.
pub struct PreparedAliasReplacementRetirement {
    // Old entries retire before the gate funding their allocator. The shared
    // target/source owners remain live until all displaced backing is gone.
    _entries: Option<AliasMap>,
    _gate: Option<FrameCollectionGate>,
    _target: FrameGlobalBindings,
    _source: Option<FrameGlobalBindings>,
}

/// Alias backing displaced by a frame lifecycle mutation. Callers retain this
/// value until their enclosing frame borrow and mutation phase are released.
pub struct FrameGlobalBindingsRetirement {
    _binding: Option<AliasBinding>,
    _entries: Option<AliasMap>,
    _owner: FrameGlobalBindings,
}

impl PreparedAliasReplacement {
    pub fn checkpoint(&self) -> Result<(), HostAllocationError> {
        if !self.reserved
            || !self
                .target
                .as_ref()
                .is_some_and(|target| target.0.get().busy.get())
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        if self
            .source
            .as_ref()
            .is_none_or(|source| source.0.get().busy.get())
        {
            Ok(())
        } else {
            Err(HostAllocationError::StaleAuthority)
        }
    }

    pub fn commit_retaining(mut self) -> PreparedAliasReplacementRetirement {
        let target = self.target.take().expect("prepared alias target");
        if !self.reserved {
            return PreparedAliasReplacementRetirement {
                _entries: None,
                _gate: None,
                _target: target,
                _source: self.source.take(),
            };
        }
        let mut prior_entries = None;
        let mut prior_gate = None;
        if let Some(source) = self.source.take() {
            let source_gate = source.0.get().gate.borrow().clone();
            let moved = std::mem::replace(
                &mut *source.0.get().entries.borrow_mut(),
                empty_map(source_gate.allocator()),
            );
            prior_entries = Some(std::mem::replace(
                &mut *target.0.get().entries.borrow_mut(),
                moved,
            ));
            prior_gate = Some(std::mem::replace(
                &mut *target.0.get().gate.borrow_mut(),
                source_gate,
            ));
            target.0.get().issuer.set(source.0.get().issuer.get());
            source.0.get().busy.set(false);
            target.0.get().busy.set(false);
            self.reserved = false;
            return PreparedAliasReplacementRetirement {
                _entries: prior_entries,
                _gate: prior_gate,
                _target: target,
                _source: Some(source),
            };
        }
        target.0.get().busy.set(false);
        self.reserved = false;
        PreparedAliasReplacementRetirement {
            _entries: prior_entries,
            _gate: prior_gate,
            _target: target,
            _source: None,
        }
    }

    pub fn commit(self) {
        drop(self.commit_retaining());
    }
}

impl Drop for PreparedAliasReplacement {
    fn drop(&mut self) {
        if self.reserved {
            if let Some(target) = &self.target {
                target.0.get().busy.set(false);
            }
            if let Some(source) = &self.source {
                source.0.get().busy.set(false);
            }
        }
    }
}

mod prepared;
pub use prepared::PreparedAliasBinding;
mod ingress;

fn empty_map(allocator: OperationCollectionAllocator) -> AliasMap {
    AliasMap::with_hasher_in(RandomState::new(), allocator)
}

impl Default for FrameGlobalBindings {
    fn default() -> Self {
        let gate = FrameCollectionGate::unbounded();
        Self(ExactShared::new(BindingOwner {
            entries: RefCell::new(empty_map(gate.allocator())),
            gate: RefCell::new(gate),
            busy: Cell::new(false),
            header_claim: RefCell::new(None),
            issuer: Cell::new(None),
        }))
    }
}

impl FrameGlobalBindings {
    /// Construct an empty live alias owner under the selected operation.
    /// Unlike detached `Default`, even an alias-free finite frame retains
    /// claims for its exact shared header and collection gate.
    pub fn prepare_empty(context: &RuntimeContext) -> Result<Self, HostAllocationError> {
        let header = ExactShared::<BindingOwner>::requested_bytes();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(header).map_err(|_| HostAllocationError::Overflow)?,
            1,
        )?
        .checked_add(FrameCollectionGate::retained_bytes()?)
        .ok_or(HostAllocationError::Overflow)?;
        let plan = HostAllocationPlan::new(0, retained)?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), plan)? else {
            return Ok(Self::default());
        };
        let gate = FrameCollectionGate::prepare(context, &receipt)?;
        let claim =
            receipt.with_output_allocation(context, |permit| permit.reserve_exact_claim(header))?;
        let owner = ExactShared::try_new(BindingOwner {
            entries: RefCell::new(empty_map(gate.allocator())),
            gate: RefCell::new(gate),
            busy: Cell::new(false),
            header_claim: RefCell::new(Some(claim)),
            issuer: Cell::new(Some(context.local_identity())),
        })
        .map_err(|_| {
            HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            )
        })?;
        receipt.checkpoint(context)?;
        Ok(Self(owner))
    }

    pub fn from_snapshot(bindings: HashMap<usize, FrameGlobalBinding>) -> Self {
        let owner = Self::default();
        {
            let gate = owner.0.get().gate.borrow();
            let _installed = gate
                .install_unbounded()
                .expect("detached alias snapshot is not a reentrant operation");
            let mut entries = owner.0.get().entries.borrow_mut();
            entries
                .try_reserve(bindings.len())
                .expect("unbounded alias snapshot allocation failed");
            for (slot, binding) in bindings {
                entries.insert(
                    slot,
                    match binding {
                        FrameGlobalBinding::Named(name) => {
                            AliasBinding::Named(WorkspaceName::unbounded(&name))
                        }
                        FrameGlobalBinding::LegacySlot(target) => AliasBinding::LegacySlot(target),
                    },
                );
            }
        }
        owner
    }

    /// Detached transport only. Live finite frame paths use the shared alias
    /// owner itself and do not allocate this second mutable map.
    pub fn snapshot(&self) -> HashMap<usize, FrameGlobalBinding> {
        self.0
            .get()
            .entries
            .borrow()
            .iter()
            .map(|(slot, binding)| (*slot, detached(binding)))
            .collect()
    }

    pub fn replace_with_snapshot(&self, bindings: HashMap<usize, FrameGlobalBinding>) {
        let detached = Self::from_snapshot(bindings);
        self.replace_with_owned(detached);
    }

    /// Reserve both shared authorities before delegating an external
    /// publication. Commit then only moves already-admitted backing.
    pub fn prepare_replace_with_owned(
        &self,
        source: Self,
    ) -> Result<PreparedAliasReplacement, HostAllocationError> {
        if self.0.get().issuer.get() != source.0.get().issuer.get() {
            return Err(HostAllocationError::StaleAuthority);
        }
        if self.0.ptr_eq(&source.0) {
            if self.0.get().busy.replace(true) {
                return Err(HostAllocationError::StaleAuthority);
            }
            return Ok(PreparedAliasReplacement {
                target: Some(self.clone()),
                source: None,
                reserved: true,
            });
        }
        if self.0.get().busy.replace(true) {
            return Err(HostAllocationError::StaleAuthority);
        }
        if source.0.get().busy.replace(true) {
            self.0.get().busy.set(false);
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(PreparedAliasReplacement {
            target: Some(self.clone()),
            source: Some(source),
            reserved: true,
        })
    }

    pub fn replace_with_owned(&self, source: Self) {
        self.prepare_replace_with_owned(source)
            .expect("alias authority is not reserved")
            .commit();
    }

    pub fn named_bindings(&self) -> Vec<(usize, String)> {
        let mut bindings = self
            .0
            .get()
            .entries
            .borrow()
            .iter()
            .filter_map(|(slot, binding)| match binding {
                AliasBinding::Named(name) => Some((*slot, name.as_str().to_owned())),
                AliasBinding::LegacySlot(_) => None,
            })
            .collect::<Vec<_>>();
        bindings.sort_by_key(|(slot, _)| *slot);
        bindings
    }

    pub fn declare_named(&self, slot: usize, name: String) {
        assert!(!self.0.get().busy.get(), "alias authority is reserved");
        let gate = self.0.get().gate.borrow();
        let _installed = gate
            .install_unbounded()
            .expect("legacy alias declaration is not a reentrant operation");
        self.0
            .get()
            .entries
            .borrow_mut()
            .insert(slot, AliasBinding::Named(WorkspaceName::unbounded(&name)));
    }

    pub fn declare_legacy_slot(&self, slot: usize) {
        assert!(!self.0.get().busy.get(), "alias authority is reserved");
        let gate = self.0.get().gate.borrow();
        let _installed = gate
            .install_unbounded()
            .expect("legacy alias declaration is not a reentrant operation");
        self.0
            .get()
            .entries
            .borrow_mut()
            .insert(slot, AliasBinding::LegacySlot(slot));
    }

    pub fn binding_for_slot(&self, slot: usize) -> Option<FrameGlobalBinding> {
        self.0.get().entries.borrow().get(&slot).map(detached)
    }

    pub fn binding_handle_for_slot(&self, slot: usize) -> Option<GlobalBindingTarget> {
        self.0
            .get()
            .entries
            .borrow()
            .get(&slot)
            .map(|binding| match binding {
                AliasBinding::Named(name) => GlobalBindingTarget::Named(name.clone()),
                AliasBinding::LegacySlot(target) => GlobalBindingTarget::LegacySlot(*target),
            })
    }

    pub fn has_binding_for_slot(&self, slot: usize) -> bool {
        self.0.get().entries.borrow().contains_key(&slot)
    }

    pub fn name_for_slot(&self, slot: usize) -> Option<String> {
        self.0
            .get()
            .entries
            .borrow()
            .get(&slot)
            .and_then(|binding| match binding {
                AliasBinding::Named(name) => Some(name.as_str().to_owned()),
                AliasBinding::LegacySlot(_) => None,
            })
    }

    pub fn is_bound(&self, slot: usize, name: &str) -> bool {
        self.0.get().entries.borrow().get(&slot).is_some_and(
            |binding| matches!(binding, AliasBinding::Named(bound) if bound.as_str() == name),
        )
    }

    pub fn unbind_slot(&self, slot: usize) -> bool {
        let removed = self.unbind_slot_retaining(slot);
        let changed = removed._binding.is_some();
        drop(removed);
        changed
    }

    pub fn unbind_slot_retaining(&self, slot: usize) -> FrameGlobalBindingsRetirement {
        assert!(!self.0.get().busy.get(), "alias authority is reserved");
        let binding = self.0.get().entries.borrow_mut().remove(&slot);
        FrameGlobalBindingsRetirement {
            _binding: binding,
            _entries: None,
            _owner: self.clone(),
        }
    }

    pub fn unbind_name(&self, name: &str) {
        assert!(!self.0.get().busy.get(), "alias authority is reserved");
        self.0.get().entries.borrow_mut().retain(
            |_, binding| !matches!(binding, AliasBinding::Named(bound) if bound.as_str() == name),
        );
    }

    pub fn unbind_all(&self) {
        drop(self.unbind_all_retaining());
    }

    pub fn unbind_all_retaining(&self) -> FrameGlobalBindingsRetirement {
        assert!(!self.0.get().busy.get(), "alias authority is reserved");
        let gate = self.0.get().gate.borrow().clone();
        let entries = std::mem::replace(
            &mut *self.0.get().entries.borrow_mut(),
            empty_map(gate.allocator()),
        );
        FrameGlobalBindingsRetirement {
            _binding: None,
            _entries: Some(entries),
            _owner: self.clone(),
        }
    }
}
