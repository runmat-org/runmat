//! Quoted storage for one moved optimization control phase. Numerical payloads and
//! root leases move unchanged; no population or model is mirrored.
use super::ScalarCooperation;
use crate::{
    context::host_allocation::{
        collection_block_retained_bytes, host_allocation_runtime_error,
        AdmittedCollectionAllocator, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    execution::ExecutionServiceError,
    BuiltinResult,
};
use allocator_api2::boxed::Box as AllocatedBox;
use std::{
    alloc::Layout,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Copy)]
pub(crate) struct Policy {
    pub identifier: &'static str,
    pub operation: &'static str,
}

pub(crate) enum Retained<T> {
    Ordinary(AllocatedBox<T>),
    Managed(AllocatedBox<T, AdmittedCollectionAllocator>),
}
impl<T> Retained<T> {
    pub fn into_inner(self) -> T {
        match self {
            Self::Ordinary(value) => AllocatedBox::into_inner(value),
            Self::Managed(value) => AllocatedBox::into_inner(value),
        }
    }

    pub fn new(search: T, cooperation: &ScalarCooperation, policy: Policy) -> BuiltinResult<Self> {
        check_original(cooperation)?;
        let runtime = cooperation.runtime();
        let map = |source| {
            host_allocation_runtime_error(runtime, policy.identifier, policy.operation, source)
        };
        let bytes = collection_block_retained_bytes(Layout::new::<T>()).map_err(map)?;
        let plan = HostAllocationPlan::new(0, bytes).map_err(map)?;
        let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(map)?;
        let result = if let Some(receipt) = receipt {
            let runtime = runtime.expect("managed state allocation has its original runtime");
            let allocator = receipt.collection_allocator(runtime).map_err(map)?;
            let result = AllocatedBox::try_new_in(search, allocator).map(Self::Managed);
            // A rejected stale permit must retain its original control cause,
            // rather than becoming a generic heap allocation failure.
            receipt.checkpoint(runtime).map_err(map)?;
            result
        } else {
            AllocatedBox::try_new(search).map(Self::Ordinary)
        };
        check_original(cooperation)?;
        result.map_err(|_| {
            map(HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            ))
        })
    }
}
fn check_original(cooperation: &ScalarCooperation) -> BuiltinResult<()> {
    if cooperation
        .runtime()
        .is_some_and(|runtime| runtime.cancellation().is_cancelled())
    {
        Err(ExecutionServiceError::Cancelled.into_runtime_error_in_context(cooperation.runtime()))
    } else {
        Ok(())
    }
}
impl<T> Deref for Retained<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Ordinary(search) => search,
            Self::Managed(search) => search,
        }
    }
}
impl<T> DerefMut for Retained<T> {
    fn deref_mut(&mut self) -> &mut T {
        match self {
            Self::Ordinary(search) => search,
            Self::Managed(search) => search,
        }
    }
}
