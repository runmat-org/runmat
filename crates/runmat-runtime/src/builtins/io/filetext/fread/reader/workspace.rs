//! Invocation-local streaming units. Every growth uses the canonical physical
//! collection allocator; no registry authority or file state is held here.

use super::*;
use crate::context::{
    host_allocation::{
        collection_block_retained_bytes, FrameCollectionGate, HostAllocationCheckpoint,
        HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
        OperationCollectionAllocator,
    },
    RuntimeContext,
};
use allocator_api2::vec::Vec as AllocatedVec;

pub(super) struct StreamValues<'a, T: Copy> {
    values: AllocatedVec<T, OperationCollectionAllocator>,
    gate: FrameCollectionGate,
    runtime: Option<&'a RuntimeContext>,
    checkpoint: Option<HostAllocationCheckpoint>,
}

pub(super) fn allocation(
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    let mut error = crate::context::host_allocation::host_allocation_runtime_error(
        runtime,
        FREAD_ERROR_IO
            .identifier
            .expect("fread I/O descriptor has an identifier"),
        format!("fread: input storage allocation failed: {source}"),
        source,
    );
    error.context.builtin = Some(BUILTIN_NAME.to_string());
    error
}

impl<'a, T: Copy> StreamValues<'a, T> {
    pub(super) fn new(runtime: Option<&'a RuntimeContext>) -> BuiltinResult<Self> {
        let allocation = |source| allocation(runtime, source);
        let checkpoint = HostAllocationReceipt::capture_owner(runtime).map_err(allocation)?;
        let gate = if checkpoint.is_some() {
            let runtime = runtime.expect("managed stream has runtime");
            let plan = HostAllocationPlan::new(
                0,
                FrameCollectionGate::retained_bytes().map_err(allocation)?,
            )
            .map_err(allocation)?;
            let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
                .map_err(allocation)?
                .ok_or_else(|| allocation(HostAllocationError::StaleAuthority))?;
            FrameCollectionGate::prepare(runtime, &receipt).map_err(allocation)?
        } else {
            FrameCollectionGate::unbounded()
        };
        Ok(Self {
            values: AllocatedVec::new_in(gate.allocator()),
            gate,
            runtime,
            checkpoint,
        })
    }

    pub(super) fn runtime(&self) -> Option<&'a RuntimeContext> {
        self.runtime
    }
    pub(super) fn values(&self) -> &[T] {
        &self.values
    }
    pub(super) fn check(&self) -> BuiltinResult<()> {
        let allocation = |source| allocation(self.runtime, source);
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint
                .check(self.runtime.expect("managed stream has runtime"))
                .map_err(allocation)?;
        } else if HostAllocationReceipt::policy(self.runtime).map_err(allocation)?
            != HostAllocationPolicy::Unbounded
        {
            return Err(allocation(HostAllocationError::StaleAuthority));
        }
        Ok(())
    }

    pub(super) fn reserve_next(&mut self) -> BuiltinResult<()> {
        let runtime = self.runtime;
        let allocation = |source| allocation(runtime, source);
        self.check()?;
        if self.values.len() == self.values.capacity() {
            let capacity = self
                .values
                .capacity()
                .checked_mul(2)
                .map(|n| n.max(32))
                .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
            let layout = std::alloc::Layout::array::<T>(capacity)
                .map_err(|_| allocation(HostAllocationError::Overflow))?;
            let failure = || {
                allocation(HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ))
            };
            if self.checkpoint.is_some() {
                let runtime = self.runtime.expect("managed stream has runtime");
                let bytes = collection_block_retained_bytes(layout).map_err(allocation)?;
                let receipt = HostAllocationReceipt::admit(
                    Some(runtime),
                    HostAllocationPlan::new(0, bytes).map_err(allocation)?,
                )
                .map_err(allocation)?
                .ok_or_else(|| allocation(HostAllocationError::StaleAuthority))?;
                let _permit = self.gate.install(runtime, &receipt).map_err(allocation)?;
                let result = self.values.try_reserve_exact(capacity - self.values.len());
                receipt.checkpoint(runtime).map_err(allocation)?;
                self.check()?;
                result.map_err(|_| failure())?;
            } else {
                let _permit = self.gate.install_unbounded().map_err(allocation)?;
                self.values
                    .try_reserve_exact(capacity - self.values.len())
                    .map_err(|_| failure())?;
            }
        }
        Ok(())
    }

    pub(super) fn push(&mut self, unit: T) -> BuiltinResult<()> {
        self.reserve_next()?;
        self.values.push(unit);
        Ok(())
    }
}
