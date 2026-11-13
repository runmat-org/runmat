//! Exact real whole-Tensor slots and source-ordered retirement for a selected
//! Managed VM host region. Legacy mixed carriers remain in the parent module.

use super::{
    collection_block_retained_bytes, invalid_output, unique_real_tensor, AllocatedVec,
    EntryLifecycle, HostAllocationError, HostAllocationPlan, HostAllocationReceipt, Layout,
    RecycledRealSlot, RuntimeContext, RuntimeError, VmFusionHostInstruction, VmFusionHostOperation,
    VmFusionHostOutputEntry, VmFusionHostOutputs,
};
use runmat_value::{NumericDType, Tensor, Value};

/// One already-admitted real whole-Tensor slot, described before readback.
pub struct VmFusionHostRealSlotSpec<'a> {
    pub dtype: NumericDType,
    pub shape: &'a [usize],
}

/// A source operand whose original Tensor owner returns to a physical slot
/// only after the operation has finished reading all its operands.
#[derive(Clone, Copy)]
pub struct VmFusionHostRetirement {
    pub operand_position: usize,
    pub slot: usize,
}

impl VmFusionHostOutputs {
    pub(in crate::context) fn selected_builtin_retirement_instruction(
        &self,
    ) -> Result<Option<VmFusionHostInstruction>, RuntimeError> {
        if self.real_slots.is_none() {
            return Ok(None);
        }
        let Some(entry) = self.entries.get(self.next) else {
            return Err(invalid_output());
        };
        if !matches!(entry.operation, VmFusionHostOperation::Builtin(_))
            || !matches!(&entry.lifecycle, EntryLifecycle::AwaitingRetirement { .. })
        {
            return Err(invalid_output());
        }
        Ok(Some(entry.instruction))
    }

    pub fn retained_bytes_with_slots(
        entry_count: usize,
        slots: &[VmFusionHostRealSlotSpec<'_>],
    ) -> Result<u64, HostAllocationError> {
        let mut total = Self::retained_bytes(entry_count)?;
        if !slots.is_empty() {
            let slot_layout = Layout::array::<RecycledRealSlot>(slots.len())
                .map_err(|_| HostAllocationError::Overflow)?;
            total = total
                .checked_add(collection_block_retained_bytes(slot_layout)?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        for slot in slots {
            if !matches!(slot.dtype, NumericDType::F32 | NumericDType::F64) {
                return Err(HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::UnsupportedClass,
                ));
            }
            total = total
                .checked_add(HostAllocationPlan::retained_tensor_bytes(
                    slot.dtype, slot.shape,
                )?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        Ok(total)
    }

    /// Materialize every reusable Tensor and each operation's independent
    /// scratch work before the selected provider readback begins.
    pub fn prepare_recycled_with(
        context: &RuntimeContext,
        slots: &[VmFusionHostRealSlotSpec<'_>],
        entry_count: usize,
        mut prepare_entry: impl FnMut(
            usize,
            &[VmFusionHostRealSlotSpec<'_>],
        ) -> Result<VmFusionHostOutputEntry, HostAllocationError>,
    ) -> Result<Self, HostAllocationError> {
        let retained = Self::retained_bytes_with_slots(entry_count, slots)?;
        let Some(receipt) =
            HostAllocationReceipt::admit(Some(context), HostAllocationPlan::new(0, retained)?)?
        else {
            return Err(HostAllocationError::MissingManagedOwner);
        };
        let mut real_slots = AllocatedVec::new_in(receipt.collection_allocator(context)?);
        real_slots
            .try_reserve_exact(slots.len())
            .map_err(|_| allocation_failed())?;
        let mut entries = AllocatedVec::new_in(receipt.collection_allocator(context)?);
        entries
            .try_reserve_exact(entry_count)
            .map_err(|_| allocation_failed())?;
        receipt.checkpoint(context)?;
        for slot in slots {
            receipt.checkpoint(context)?;
            let value = receipt
                .exact_output_tensor(context, slot.dtype, slot.shape)?
                .into_tensor()
                .map_err(HostAllocationError::Numeric)?;
            real_slots.push(RecycledRealSlot {
                shape: value.shared_shape(),
                dtype: slot.dtype,
                value: Some(value),
            });
        }
        for index in 0..entry_count {
            receipt.checkpoint(context)?;
            let entry = prepare_entry(index, slots)?;
            let EntryLifecycle::RecycledReady {
                output_slot,
                retirements,
            } = &entry.lifecycle
            else {
                return Err(invalid_prepared_entry());
            };
            let exact_intrinsic = matches!(
                (entry.operation, entry.instruction),
                (super::VmFusionHostOperation::Intrinsic(owner), VmFusionHostInstruction::Intrinsic(instruction))
                    if owner == instruction
            );
            let supported_form = match entry.operation {
                super::VmFusionHostOperation::Intrinsic(runmat_types::OperatorKind::UnaryPlus) => {
                    output_slot.is_none() && retirements.iter().all(Option::is_none)
                }
                super::VmFusionHostOperation::Intrinsic(
                    runmat_types::OperatorKind::UnaryMinus
                    | runmat_types::OperatorKind::MatrixMultiply,
                ) => output_slot.is_some(),
                VmFusionHostOperation::Builtin(_) => {
                    matches!(
                        entry.instruction,
                        VmFusionHostInstruction::ElementwiseCall
                            | VmFusionHostInstruction::ElementwiseBinary
                    ) && output_slot.is_some()
                }
                _ => false,
            };
            let exact_pair =
                exact_intrinsic || matches!(entry.operation, VmFusionHostOperation::Builtin(_));
            if !exact_pair
                || !supported_form
                || output_slot.is_some() != entry.work.is_some()
                || output_slot.is_some_and(|slot| slot >= slots.len())
                || retirements.iter().flatten().any(|retirement| {
                    retirement.operand_position >= 2
                        || retirement.slot >= slots.len()
                        || Some(retirement.slot) == *output_slot
                })
                || retirements[0]
                    .zip(retirements[1])
                    .is_some_and(|(left, right)| left.operand_position == right.operand_position)
            {
                return Err(invalid_prepared_entry());
            }
            entries.push(entry);
        }
        receipt.checkpoint(context)?;
        Ok(Self {
            entries,
            real_slots: Some(real_slots),
            next: 0,
        })
    }

    /// Consume every operand of the completed instruction together. For
    /// repeated operands, all source handles must be dropped before checking
    /// unique mutable backing on the one Tensor returned to its slot.
    pub(in crate::context) fn retire_numerical_operands(
        &mut self,
        instruction: VmFusionHostInstruction,
        mut operands: [Option<Value>; 2],
    ) -> Result<(), RuntimeError> {
        // Legacy mixed-operation carriers advance when their preallocated
        // output is taken. The VM always delivers the same post-operation
        // batch, but only the recycled carrier has slots to replenish.
        if self.real_slots.is_none() {
            return Ok(());
        }
        let Some(entry) = self.entries.get(self.next) else {
            return Err(invalid_output());
        };
        if entry.instruction != instruction {
            return Err(invalid_output());
        }
        let retirements = match &entry.lifecycle {
            EntryLifecycle::RecycledReady {
                output_slot: None,
                retirements,
            } if entry.work.is_none() => *retirements,
            EntryLifecycle::AwaitingRetirement { retirements } => *retirements,
            _ => return Err(invalid_output()),
        };
        let Some(slots) = self.real_slots.as_mut() else {
            return Err(invalid_output());
        };
        let mut returned: [Option<(usize, Tensor)>; 2] = [None, None];
        for retirement in retirements.into_iter().flatten() {
            let Some(value) = operands
                .get_mut(retirement.operand_position)
                .and_then(Option::take)
            else {
                return Err(invalid_output());
            };
            let Value::Tensor(tensor) = value else {
                return Err(invalid_output());
            };
            let Some(slot) = slots.get(retirement.slot) else {
                return Err(invalid_output());
            };
            if slot.value.is_some()
                || tensor.numeric_dtype() != slot.dtype
                || tensor.shape() != slot.shape.as_slice()
            {
                return Err(invalid_output());
            }
            if let Some((_, existing)) = returned
                .iter_mut()
                .flatten()
                .find(|(index, _)| *index == retirement.slot)
            {
                if !existing
                    .host_buffer()
                    .shares_allocation_with(tensor.host_buffer())
                {
                    return Err(invalid_output());
                }
                // Drop the duplicate before testing uniqueness below.
            } else if let Some(vacant) = returned.iter_mut().find(|value| value.is_none()) {
                *vacant = Some((retirement.slot, tensor));
            } else {
                return Err(invalid_output());
            }
        }
        drop(operands);
        for (_, tensor) in returned.iter_mut().flatten() {
            if !unique_real_tensor(tensor) {
                return Err(invalid_output());
            }
        }
        for (index, tensor) in returned.into_iter().flatten() {
            slots[index].value = Some(tensor);
        }
        self.next += 1;
        Ok(())
    }

    pub(in crate::context) fn retire_builtin_operands(
        &mut self,
        instruction: VmFusionHostInstruction,
        operands: Vec<Value>,
    ) -> Result<(), RuntimeError> {
        if self.real_slots.is_none() {
            return Ok(());
        }
        let Some(entry) = self.entries.get(self.next) else {
            return Err(invalid_output());
        };
        if entry.instruction != instruction
            || !matches!(entry.operation, VmFusionHostOperation::Builtin(_))
            || !matches!(&entry.lifecycle, EntryLifecycle::AwaitingRetirement { .. })
        {
            return Err(invalid_output());
        }
        if operands.is_empty() || operands.len() > 2 {
            return Err(invalid_output());
        }
        let mut operands = operands.into_iter();
        let batch = [operands.next(), operands.next()];
        // The Vec allocation and any unplanned trailing owners must retire
        // before the shared batch checks unique physical Tensor ownership.
        drop(operands);
        self.retire_numerical_operands(instruction, batch)
    }
}

fn allocation_failed() -> HostAllocationError {
    HostAllocationError::Numeric(runmat_value::ExactHostNumericAllocationError::AllocationFailed)
}

fn invalid_prepared_entry() -> HostAllocationError {
    HostAllocationError::Numeric(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
}

#[cfg(test)]
mod tests;
