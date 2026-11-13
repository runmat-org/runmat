//! Exact, source-ordered numerical outputs for one selected VM host region.

use super::host_allocation::{
    collection_block_retained_bytes, host_allocation_runtime_error, HostAllocationError,
    HostAllocationPlan, HostAllocationReceipt,
};
use super::{RuntimeContext, VmFusionHostInstruction, VmFusionHostOperation};
use crate::context::preparation::PreparedHostNumericalOutput;
use crate::RuntimeError;
use allocator_api2::vec::Vec as AllocatedVec;
use runmat_builtins::BuiltinBindingIdentity;
use runmat_value::{NumericDType, SharedShape, Tensor};
use std::alloc::Layout;

mod recycling;
pub use recycling::{VmFusionHostRealSlotSpec, VmFusionHostRetirement};

pub struct VmFusionHostOutputEntry {
    operation: VmFusionHostOperation,
    instruction: VmFusionHostInstruction,
    work: Option<PreparedHostNumericalOutput>,
    lifecycle: EntryLifecycle,
}

enum EntryLifecycle {
    Legacy,
    RecycledReady {
        output_slot: Option<usize>,
        retirements: [Option<VmFusionHostRetirement>; 2],
    },
    AwaitingRetirement {
        retirements: [Option<VmFusionHostRetirement>; 2],
    },
}

impl VmFusionHostOutputEntry {
    pub fn builtin(
        binding: BuiltinBindingIdentity,
        instruction: VmFusionHostInstruction,
        output: PreparedHostNumericalOutput,
    ) -> Self {
        Self {
            operation: VmFusionHostOperation::Builtin(binding),
            instruction,
            work: Some(output),
            lifecycle: EntryLifecycle::Legacy,
        }
    }

    pub fn intrinsic(
        operator: runmat_types::OperatorKind,
        instruction: VmFusionHostInstruction,
        output: PreparedHostNumericalOutput,
    ) -> Self {
        Self {
            operation: VmFusionHostOperation::Intrinsic(operator),
            instruction,
            work: Some(output),
            lifecycle: EntryLifecycle::Legacy,
        }
    }

    /// Entries include allocation-free aliases such as UPlus. Their empty
    /// retirement batch still advances source ordering after VM completion.
    pub fn recycled_intrinsic(
        operator: runmat_types::OperatorKind,
        instruction: VmFusionHostInstruction,
        output_slot: Option<usize>,
        retirements: [Option<VmFusionHostRetirement>; 2],
        work: Option<PreparedHostNumericalOutput>,
    ) -> Self {
        Self {
            operation: VmFusionHostOperation::Intrinsic(operator),
            instruction,
            work,
            lifecycle: EntryLifecycle::RecycledReady {
                output_slot,
                retirements,
            },
        }
    }

    pub fn recycled_builtin(
        binding: BuiltinBindingIdentity,
        instruction: VmFusionHostInstruction,
        output_slot: usize,
        retirements: [Option<VmFusionHostRetirement>; 2],
        work: PreparedHostNumericalOutput,
    ) -> Self {
        Self {
            operation: VmFusionHostOperation::Builtin(binding),
            instruction,
            work: Some(work),
            lifecycle: EntryLifecycle::RecycledReady {
                output_slot: Some(output_slot),
                retirements,
            },
        }
    }
}

struct RecycledRealSlot {
    value: Option<Tensor>,
    dtype: NumericDType,
    shape: SharedShape,
}

pub struct VmFusionHostOutputs {
    entries:
        AllocatedVec<VmFusionHostOutputEntry, super::host_allocation::AdmittedCollectionAllocator>,
    real_slots:
        Option<AllocatedVec<RecycledRealSlot, super::host_allocation::AdmittedCollectionAllocator>>,
    next: usize,
}

impl VmFusionHostOutputs {
    pub fn retained_bytes(count: usize) -> Result<u64, HostAllocationError> {
        let layout = Layout::array::<VmFusionHostOutputEntry>(count)
            .map_err(|_| HostAllocationError::Overflow)?;
        collection_block_retained_bytes(layout)
    }

    /// The callback runs only after the complete carrier-array backing is
    /// admitted. It must prepare each per-operation exact output before any
    /// provider readback or VM effect.
    pub fn prepare_with(
        context: &RuntimeContext,
        count: usize,
        mut prepare: impl FnMut(usize) -> Result<VmFusionHostOutputEntry, HostAllocationError>,
    ) -> Result<Option<Self>, HostAllocationError> {
        let retained = Self::retained_bytes(count)?;
        let Some(receipt) =
            HostAllocationReceipt::admit(Some(context), HostAllocationPlan::new(0, retained)?)?
        else {
            return Ok(None);
        };
        let allocator = receipt.collection_allocator(context)?;
        let mut entries = AllocatedVec::new_in(allocator);
        entries.try_reserve_exact(count).map_err(|_| {
            HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            )
        })?;
        receipt.checkpoint(context)?;
        for index in 0..count {
            receipt.checkpoint(context)?;
            entries.push(prepare(index)?);
        }
        receipt.checkpoint(context)?;
        Ok(Some(Self {
            entries,
            real_slots: None,
            next: 0,
        }))
    }

    pub(super) fn take_builtin(
        &mut self,
        context: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        binding: BuiltinBindingIdentity,
    ) -> Result<PreparedHostNumericalOutput, RuntimeError> {
        self.take(
            context,
            instruction,
            VmFusionHostOperation::Builtin(binding),
        )
    }

    pub(super) fn take_intrinsic(
        &mut self,
        context: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        operator: runmat_types::OperatorKind,
    ) -> Result<PreparedHostNumericalOutput, RuntimeError> {
        self.take(
            context,
            instruction,
            VmFusionHostOperation::Intrinsic(operator),
        )
    }

    fn take(
        &mut self,
        context: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        operation: VmFusionHostOperation,
    ) -> Result<PreparedHostNumericalOutput, RuntimeError> {
        let Some(entry) = self.entries.get_mut(self.next) else {
            return Err(missing_output(context));
        };
        if entry.instruction != instruction || entry.operation != operation {
            return Err(invalid_output());
        }
        let (slot_index, output, retirements) = match &entry.lifecycle {
            EntryLifecycle::Legacy => {
                let output = entry.work.take().ok_or_else(invalid_output)?;
                self.next += 1;
                return Ok(output);
            }
            EntryLifecycle::RecycledReady {
                output_slot: Some(slot_index),
                retirements,
            } => (
                *slot_index,
                entry.work.take().ok_or_else(invalid_output)?,
                *retirements,
            ),
            _ => return Err(invalid_output()),
        };
        let Some(slot) = self
            .real_slots
            .as_mut()
            .and_then(|slots| slots.get_mut(slot_index))
        else {
            return Err(invalid_output());
        };
        let Some(mut tensor) = slot.value.take() else {
            return Err(invalid_output());
        };
        if tensor.numeric_dtype() != slot.dtype
            || tensor.shape() != slot.shape.as_slice()
            || !unique_real_tensor(&mut tensor)
        {
            return Err(invalid_output());
        }
        let output = output
            .bind_recycled_tensor(tensor)
            .map_err(|_| invalid_output())?;
        entry.lifecycle = EntryLifecycle::AwaitingRetirement { retirements };
        Ok(output)
    }

    pub(super) fn complete(&self) -> bool {
        self.next == self.entries.len()
    }
}

fn unique_real_tensor(tensor: &mut Tensor) -> bool {
    match tensor.numeric_dtype() {
        NumericDType::F32 => tensor.try_as_f32_slice_mut_unique().is_some(),
        NumericDType::F64 => tensor.try_as_f64_slice_mut_unique().is_some(),
        _ => false,
    }
}

pub(super) fn missing_output(context: &RuntimeContext) -> RuntimeError {
    host_allocation_runtime_error(
        Some(context),
        "RunMat:Placement:HostNumericalOutput",
        "selected VM host numerical output",
        match HostAllocationReceipt::policy(Some(context)) {
            Ok(crate::context::host_allocation::HostAllocationPolicy::Managed) => {
                HostAllocationError::MissingManagedOwner
            }
            Ok(crate::context::host_allocation::HostAllocationPolicy::Unbounded) => {
                return invalid_output();
            }
            Err(error) => error,
        },
    )
}

pub(super) fn invalid_output() -> RuntimeError {
    crate::build_runtime_error("VM host numerical output disagrees with its selected instruction")
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
