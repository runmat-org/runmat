//! Fixed native completion blocks, using the canonical physical allocator.
use std::{alloc::Layout, ops::Deref};

use allocator_api2::boxed::Box as AllocatedBox;
use runmat_runtime::context::{
    host_allocation::{
        collection_block_retained_bytes, AdmittedCollectionAllocator, HostAllocationError,
        HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};

use crate::{NativeExecutorError, NativeExecutorResult};

pub(in crate::execute) enum PayloadBox<T> {
    Ordinary(Box<T>),
    Admitted(AllocatedBox<T, AdmittedCollectionAllocator>),
}

impl<T> Deref for PayloadBox<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Ordinary(value) => value,
            Self::Admitted(value) => value,
        }
    }
}

impl<T> PayloadBox<T> {
    /// Both branches physically retire the old backing before returning T.
    pub(super) fn into_inner(self) -> T {
        match self {
            Self::Ordinary(value) => *value,
            Self::Admitted(value) => AllocatedBox::into_inner(value),
        }
    }
}

pub(in crate::execute) struct PayloadReservation {
    receipt: Option<HostAllocationReceipt>,
}

pub(super) enum PayloadAllocator {
    Ordinary,
    Admitted(AdmittedCollectionAllocator),
}

impl PayloadReservation {
    pub(super) fn prepare<T>(runtime: &RuntimeContext) -> NativeExecutorResult<Self> {
        Self::prepare_sequence::<T, T>(runtime)
    }

    /// These two physical blocks are sequential, never simultaneously live.
    /// The first block's layout includes any inline reservation it retains.
    pub(super) fn prepare_sequence<A, B>(runtime: &RuntimeContext) -> NativeExecutorResult<Self> {
        let first = collection_block_retained_bytes(Layout::new::<A>())
            .map_err(|cause| error(runtime, cause))?;
        let second = collection_block_retained_bytes(Layout::new::<B>())
            .map_err(|cause| error(runtime, cause))?;
        let plan =
            HostAllocationPlan::new(0, first.max(second)).map_err(|cause| error(runtime, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
            .map_err(|cause| error(runtime, cause))?;
        Ok(Self { receipt })
    }

    pub(super) fn check(&self, runtime: &RuntimeContext) -> NativeExecutorResult<()> {
        match &self.receipt {
            Some(receipt) => receipt.checkpoint(runtime),
            None => match HostAllocationReceipt::policy(Some(runtime))
                .map_err(|cause| error(runtime, cause))?
            {
                HostAllocationPolicy::Unbounded => Ok(()),
                _ => Err(HostAllocationError::StaleAuthority),
            },
        }
        .map_err(|cause| error(runtime, cause))
    }

    pub(super) fn allocator(
        &self,
        runtime: &RuntimeContext,
    ) -> NativeExecutorResult<PayloadAllocator> {
        self.check(runtime)?;
        match &self.receipt {
            Some(receipt) => receipt
                .collection_allocator(runtime)
                .map(PayloadAllocator::Admitted)
                .map_err(|cause| error(runtime, cause)),
            None => Ok(PayloadAllocator::Ordinary),
        }
    }

    pub(super) fn allocate<T>(
        &self,
        runtime: &RuntimeContext,
        value: T,
    ) -> NativeExecutorResult<PayloadBox<T>> {
        let result = self.allocator(runtime)?.allocate(value);
        // A changed authority/cancellation wins over the narrow allocator error.
        self.check(runtime)?;
        result.map_err(|cause| error(runtime, cause))
    }
}

impl PayloadAllocator {
    pub(super) fn allocate<T>(self, value: T) -> Result<PayloadBox<T>, HostAllocationError> {
        match self {
            Self::Ordinary => Ok(PayloadBox::Ordinary(Box::new(value))),
            Self::Admitted(allocator) => AllocatedBox::try_new_in(value, allocator)
                .map(PayloadBox::Admitted)
                .map_err(|_| {
                    HostAllocationError::Numeric(
                        runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                    )
                }),
        }
    }
}

pub(super) fn error(runtime: &RuntimeContext, cause: HostAllocationError) -> NativeExecutorError {
    crate::error::allocation_error(
        Some(runtime),
        format!("native completion payload admission failed: {cause}"),
        cause,
    )
    .into()
}
