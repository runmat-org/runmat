//! Rehome an existing detached alias snapshot into the selected frame's
//! retained collection and name owners before the frame becomes visible.

use super::{empty_map, AliasBinding, FrameGlobalBindings};
use crate::context::host_allocation::{
    frame_hash_table_retained_bytes, FrameCollectionGate, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::workspace::base::WorkspaceName;
use runmat_execution::resource::ExactShared;

impl FrameGlobalBindings {
    pub fn ingress_retained_bytes(&self) -> Result<u64, HostAllocationError> {
        let payload = self.0.get();
        if payload.header_claim.borrow().is_some() {
            return Ok(0);
        }
        let entries = payload.entries.borrow();
        let mut bytes = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(ExactShared::<super::BindingOwner>::requested_bytes())
                .map_err(|_| HostAllocationError::Overflow)?,
            1,
        )?
        .checked_add(FrameCollectionGate::retained_bytes()?)
        .ok_or(HostAllocationError::Overflow)?;
        bytes = bytes
            .checked_add(frame_hash_table_retained_bytes::<usize, AliasBinding>(
                entries.len(),
            )?)
            .ok_or(HostAllocationError::Overflow)?;
        for binding in entries.values() {
            if let AliasBinding::Named(name) = binding {
                let requested = WorkspaceName::requested_bytes(name.as_str())
                    .map_err(HostAllocationError::Numeric)?;
                let requested =
                    u64::try_from(requested).map_err(|_| HostAllocationError::Overflow)?;
                bytes = bytes
                    .checked_add(HostAllocationPlan::retained_exact_output_bytes(
                        requested, 1,
                    )?)
                    .ok_or(HostAllocationError::Overflow)?;
            }
        }
        Ok(bytes)
    }

    pub fn admit_ingress(
        &self,
        context: &RuntimeContext,
        receipt: &HostAllocationReceipt,
    ) -> Result<(), HostAllocationError> {
        let payload = self.0.get();
        if payload.header_claim.borrow().is_some() {
            if payload.issuer.get() != Some(context.local_identity()) {
                return Err(HostAllocationError::StaleAuthority);
            }
            return receipt.checkpoint(context);
        }
        if payload.busy.replace(true) {
            return Err(HostAllocationError::StaleAuthority);
        }
        let result = (|| {
            let gate = FrameCollectionGate::prepare(context, receipt)?;
            let _installed = gate.install(context, receipt)?;
            let entries = payload.entries.borrow();
            let mut rebuilt = empty_map(gate.allocator());
            rebuilt.try_reserve(entries.len()).map_err(|_| {
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                )
            })?;
            for (slot, binding) in entries.iter() {
                let binding = match binding {
                    AliasBinding::Named(name) => {
                        AliasBinding::Named(receipt.with_output_allocation(context, |permit| {
                            WorkspaceName::prepare(name.as_str(), permit)
                        })?)
                    }
                    AliasBinding::LegacySlot(slot) => AliasBinding::LegacySlot(*slot),
                };
                rebuilt.insert(*slot, binding);
            }
            drop(entries);
            let claim = receipt.with_output_allocation(context, |permit| {
                permit.reserve_exact_claim(ExactShared::<super::BindingOwner>::requested_bytes())
            })?;
            receipt.checkpoint(context)?;
            *payload.entries.borrow_mut() = rebuilt;
            *payload.gate.borrow_mut() = gate;
            *payload.header_claim.borrow_mut() = Some(claim);
            payload.issuer.set(Some(context.local_identity()));
            Ok(())
        })();
        payload.busy.set(false);
        result
    }
}
