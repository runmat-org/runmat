//! One selected global-alias mutation. The table and name are admitted before
//! an external workspace delegate is allowed to publish its value.

use super::{empty_map, AliasBinding, AliasMap, FrameGlobalBindings};
use crate::context::host_allocation::{
    frame_hash_table_retained_bytes, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::workspace::base::WorkspaceName;
use crate::workspace::GlobalBindingTarget;

pub struct PreparedAliasBinding {
    owner: FrameGlobalBindings,
    slot: usize,
    binding: Option<AliasBinding>,
    replacement: Option<AliasMap>,
    receipt: Option<HostAllocationReceipt>,
    reserved: bool,
}

impl PreparedAliasBinding {
    /// The admitted destination to publish through the canonical base owner.
    pub fn target(&self) -> GlobalBindingTarget {
        self.binding
            .as_ref()
            .expect("prepared alias binding")
            .clone()
    }

    pub fn checkpoint(&self) -> Result<(), HostAllocationError> {
        if !self.reserved || !self.owner.0.get().busy.get() {
            return Err(HostAllocationError::StaleAuthority);
        }
        if let Some(receipt) = &self.receipt {
            receipt.checkpoint_captured()?;
        }
        Ok(())
    }

    /// No allocation or admission is allowed after the external delegate has
    /// committed its visible value. A fresh table, when needed, was built in
    /// preparation from shared name handles.
    pub fn commit(mut self) {
        let payload = self.owner.0.get();
        if let Some(table) = self.replacement.take() {
            *payload.entries.borrow_mut() = table;
        }
        payload.entries.borrow_mut().insert(
            self.slot,
            self.binding.take().expect("prepared alias binding"),
        );
        payload.busy.set(false);
        self.reserved = false;
    }
}

impl Drop for PreparedAliasBinding {
    fn drop(&mut self) {
        if self.reserved {
            self.owner.0.get().busy.set(false);
        }
    }
}

impl FrameGlobalBindings {
    pub fn prepare_named(
        &self,
        context: &RuntimeContext,
        slot: usize,
        name: &str,
    ) -> Result<PreparedAliasBinding, HostAllocationError> {
        self.prepare_binding(context, slot, RequestedTarget::Named(name))
    }

    pub fn prepare_legacy_slot(
        &self,
        context: &RuntimeContext,
        slot: usize,
    ) -> Result<PreparedAliasBinding, HostAllocationError> {
        self.prepare_binding(context, slot, RequestedTarget::LegacySlot(slot))
    }

    /// Prepare a binding whose session destination is independent of its
    /// frame slot. A retained named target shares its existing name owner.
    pub fn prepare_target(
        &self,
        context: &RuntimeContext,
        slot: usize,
        target: &GlobalBindingTarget,
    ) -> Result<PreparedAliasBinding, HostAllocationError> {
        self.prepare_binding(context, slot, RequestedTarget::Existing(target))
    }

    fn prepare_binding(
        &self,
        context: &RuntimeContext,
        slot: usize,
        requested: RequestedTarget<'_>,
    ) -> Result<PreparedAliasBinding, HostAllocationError> {
        if self.0.get().busy.get() {
            return Err(HostAllocationError::StaleAuthority);
        }
        let issuer = self.0.get().issuer.get();
        if issuer.is_some_and(|issuer| issuer != context.local_identity()) {
            return Err(HostAllocationError::StaleAuthority);
        }
        if issuer.is_none() {
            let ingress = HostAllocationPlan::new(0, self.ingress_retained_bytes()?)?;
            if let Some(receipt) = HostAllocationReceipt::admit(Some(context), ingress)? {
                self.admit_ingress(context, &receipt)?;
            }
        }
        let payload = self.0.get();
        if payload.busy.replace(true) {
            return Err(HostAllocationError::StaleAuthority);
        }
        let mut prepared = PreparedAliasBinding {
            owner: self.clone(),
            slot,
            binding: None,
            replacement: None,
            receipt: None,
            reserved: true,
        };
        let entries = payload.entries.borrow();
        let existing = entries.get(&slot);
        let name = requested.name();
        let existing_name = match (existing, name) {
            (Some(AliasBinding::Named(existing)), Some(name)) if existing.as_str() == name => {
                Some(existing.clone())
            }
            _ => None,
        };
        let new_name = name.is_some()
            && existing_name.is_none()
            && !matches!(requested, RequestedTarget::Existing(GlobalBindingTarget::Named(name)) if name.has_retained_claim());
        let grow = existing.is_none() && entries.len() == entries.capacity();
        let target = entries
            .len()
            .checked_add(usize::from(existing.is_none()))
            .ok_or(HostAllocationError::Overflow)?;
        let mut retained = if grow {
            frame_hash_table_retained_bytes::<usize, AliasBinding>(target)?
        } else {
            0
        };
        if let Some(name) = name.filter(|_| new_name) {
            let bytes =
                WorkspaceName::requested_bytes(name).map_err(HostAllocationError::Numeric)?;
            let bytes = u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?;
            retained = retained
                .checked_add(HostAllocationPlan::retained_exact_output_bytes(bytes, 1)?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        let plan = HostAllocationPlan::new(0, retained)?;
        drop(entries);
        prepared.receipt = HostAllocationReceipt::admit(Some(context), plan)?;
        let gate = payload.gate.borrow().clone();
        let _installed = match &prepared.receipt {
            Some(receipt) => gate.install(context, receipt)?,
            None => gate.install_unbounded()?,
        };
        if grow {
            let entries = payload.entries.borrow();
            let mut replacement = empty_map(gate.allocator());
            replacement.try_reserve(target).map_err(|_| {
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                )
            })?;
            replacement.extend(
                entries
                    .iter()
                    .map(|(slot, binding)| (*slot, binding.clone())),
            );
            prepared.replacement = Some(replacement);
        }
        prepared.binding = Some(match requested {
            RequestedTarget::Named(name) => AliasBinding::Named(match existing_name {
                Some(existing) => existing,
                None => match &prepared.receipt {
                    Some(receipt) => receipt.with_output_allocation(context, |permit| {
                        WorkspaceName::prepare(name, permit)
                    })?,
                    None => WorkspaceName::unbounded(name),
                },
            }),
            RequestedTarget::LegacySlot(target) => AliasBinding::LegacySlot(target),
            RequestedTarget::Existing(target) => match target {
                GlobalBindingTarget::Named(name) => AliasBinding::Named(match existing_name {
                    Some(existing) => existing,
                    None if name.has_retained_claim() => name.clone(),
                    None => match &prepared.receipt {
                        Some(receipt) => receipt.with_output_allocation(context, |permit| {
                            WorkspaceName::prepare(name.as_str(), permit)
                        })?,
                        None => WorkspaceName::unbounded(name.as_str()),
                    },
                }),
                GlobalBindingTarget::LegacySlot(target) => AliasBinding::LegacySlot(*target),
            },
        });
        prepared.checkpoint()?;
        Ok(prepared)
    }
}

#[derive(Clone, Copy)]
enum RequestedTarget<'a> {
    Named(&'a str),
    LegacySlot(usize),
    Existing(&'a GlobalBindingTarget),
}

impl<'a> RequestedTarget<'a> {
    fn name(self) -> Option<&'a str> {
        match self {
            Self::Named(name) => Some(name),
            Self::LegacySlot(_) => None,
            Self::Existing(GlobalBindingTarget::Named(name)) => Some(name.as_str()),
            Self::Existing(GlobalBindingTarget::LegacySlot(_)) => None,
        }
    }
}
