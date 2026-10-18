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
use std::alloc::Layout;

pub struct VmFusionHostOutputEntry {
    operation: VmFusionHostOperation,
    instruction: VmFusionHostInstruction,
    output: Option<PreparedHostNumericalOutput>,
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
            output: Some(output),
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
            output: Some(output),
        }
    }
}

pub struct VmFusionHostOutputs {
    entries:
        AllocatedVec<VmFusionHostOutputEntry, super::host_allocation::AdmittedCollectionAllocator>,
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
        Ok(Some(Self { entries, next: 0 }))
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
        let Some(output) = entry.output.take() else {
            return Err(invalid_output());
        };
        self.next += 1;
        Ok(output)
    }

    pub(super) fn complete(&self) -> bool {
        self.next == self.entries.len()
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

fn invalid_output() -> RuntimeError {
    crate::build_runtime_error("VM host numerical output disagrees with its selected instruction")
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
