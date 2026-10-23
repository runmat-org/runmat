//! Mutable destination blocks reuse the canonical operation allocator.
use crate::context::{
    host_allocation::{
        collection_block_retained_bytes, FrameCollectionGate, HostAllocationCheckpoint,
        HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
        OperationCollectionAllocator,
    },
    RuntimeContext,
};
use crate::{object::indexing::allocation_error, RuntimeError};
use allocator_api2::{boxed::Box as AllocatedBox, vec::Vec as AllocatedVec};
use std::alloc::Layout;

#[inline(never)]
pub(super) fn admit_future<'a, F>(
    context: Option<&RuntimeContext>,
    make_future: impl FnOnce() -> F,
) -> crate::execution::RuntimeContinuationFuture<'a, crate::execution::RootedValue>
where
    F: std::future::Future<Output = Result<crate::execution::RootedValue, RuntimeError>> + 'a,
{
    // Recursive callers pass their captures, not a fully constructed child
    // future. Keep its existing single allocation and exact admission owner.
    crate::execution::rooted_continuation::admit_rooted_future_with(
        context.cloned(),
        "sequence assignment",
        crate::builtin::builtin_control_error,
        move |_| make_future(),
    )
}

pub(super) type ControlVec<T> = AllocatedVec<T, OperationCollectionAllocator>;
pub(super) type ControlIter<T> = allocator_api2::vec::IntoIter<T, OperationCollectionAllocator>;
pub(super) type ControlBox<T> = AllocatedBox<T, OperationCollectionAllocator>;
pub(super) struct DestinationControl {
    gate: FrameCollectionGate,
    checkpoint: Option<HostAllocationCheckpoint>,
}
impl std::fmt::Debug for DestinationControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DestinationControl").finish_non_exhaustive()
    }
}
impl DestinationControl {
    pub(super) fn new(context: Option<&RuntimeContext>) -> Result<Self, RuntimeError> {
        let checkpoint = HostAllocationReceipt::capture_owner(context)
            .map_err(|e| allocation_error(context, e))?;
        let gate = if checkpoint.is_some() {
            let runtime = context.expect("managed owner has context");
            let bytes =
                FrameCollectionGate::retained_bytes().map_err(|e| allocation_error(context, e))?;
            let receipt = HostAllocationReceipt::admit(
                context,
                HostAllocationPlan::new(0, bytes).map_err(|e| allocation_error(context, e))?,
            )
            .map_err(|e| allocation_error(context, e))?
            .ok_or_else(|| allocation_error(context, HostAllocationError::StaleAuthority))?;
            FrameCollectionGate::prepare(runtime, &receipt)
                .map_err(|e| allocation_error(context, e))?
        } else {
            FrameCollectionGate::unbounded()
        };
        Ok(Self { gate, checkpoint })
    }
    pub(super) fn check_owner(&self, context: Option<&RuntimeContext>) -> Result<(), RuntimeError> {
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint
                .check(context.ok_or_else(|| {
                    allocation_error(context, HostAllocationError::StaleAuthority)
                })?)
                .map_err(|e| allocation_error(context, e))?;
        } else if HostAllocationReceipt::policy(context)
            .map_err(|e| allocation_error(context, e))?
            != HostAllocationPolicy::Unbounded
        {
            return Err(allocation_error(
                context,
                HostAllocationError::StaleAuthority,
            ));
        }
        Ok(())
    }
    fn allocate<T>(
        &self,
        context: Option<&RuntimeContext>,
        layout: Layout,
        construct: impl FnOnce() -> Result<T, ()>,
    ) -> Result<T, RuntimeError> {
        self.check_owner(context)?;
        let failure = || {
            allocation_error(
                context,
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        };
        if let Some(runtime) = context.filter(|_| self.checkpoint.is_some()) {
            let bytes = collection_block_retained_bytes(layout)
                .map_err(|e| allocation_error(context, e))?;
            let receipt = HostAllocationReceipt::admit(
                context,
                HostAllocationPlan::new(0, bytes).map_err(|e| allocation_error(context, e))?,
            )
            .map_err(|e| allocation_error(context, e))?
            .ok_or_else(|| allocation_error(context, HostAllocationError::StaleAuthority))?;
            let _permit = self
                .gate
                .install(runtime, &receipt)
                .map_err(|e| allocation_error(context, e))?;
            let result = construct();
            receipt
                .checkpoint(runtime)
                .map_err(|e| allocation_error(context, e))?;
            result.map_err(|()| failure())
        } else {
            let _permit = self
                .gate
                .install_unbounded()
                .map_err(|e| allocation_error(context, e))?;
            construct().map_err(|()| failure())
        }
    }
    pub(super) fn vector<T>(&self) -> ControlVec<T> {
        ControlVec::new_in(self.gate.allocator())
    }
    pub(super) fn reserve<T>(
        &self,
        context: Option<&RuntimeContext>,
        values: &mut ControlVec<T>,
        additional: usize,
    ) -> Result<(), RuntimeError> {
        self.check_owner(context)?;
        let count = values
            .len()
            .checked_add(additional)
            .ok_or_else(|| allocation_error(context, HostAllocationError::Overflow))?;
        if count <= values.capacity() {
            return Ok(());
        }
        let capacity = count
            .checked_next_power_of_two()
            .ok_or_else(|| allocation_error(context, HostAllocationError::Overflow))?;
        let layout = Layout::array::<T>(capacity)
            .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?;
        self.allocate(context, layout, || {
            values
                .try_reserve_exact(capacity - values.len())
                .map_err(|_| ())
        })
    }
    pub(super) fn push<T>(
        &self,
        context: Option<&RuntimeContext>,
        values: &mut ControlVec<T>,
        value: T,
    ) -> Result<(), RuntimeError> {
        self.reserve(context, values, 1)?;
        values.push(value);
        Ok(())
    }
    pub(super) fn boxed<T>(
        &self,
        context: Option<&RuntimeContext>,
        value: T,
    ) -> Result<ControlBox<T>, RuntimeError> {
        self.allocate(context, Layout::new::<T>(), || {
            ControlBox::try_new_in(value, self.gate.allocator()).map_err(|_| ())
        })
    }
}
