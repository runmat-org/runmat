//! The file-owned decoder is one exact, Send control allocation. Unlike a
//! growing local collection it needs no mutable permit gate or RuntimeContext.

use super::EncodedInput;
use crate::context::{
    host_allocation::{
        collection_block_retained_bytes, AdmittedCollectionAllocator, HostAllocationError,
        HostAllocationPlan, HostAllocationReceipt,
    },
    RuntimeContext,
};
use allocator_api2::boxed::Box as AllocatedBox;
use std::ops::{Deref, DerefMut};

pub(crate) enum TextInputBox {
    Ordinary(Box<EncodedInput>),
    Managed(AllocatedBox<EncodedInput, AdmittedCollectionAllocator>),
}

impl Deref for TextInputBox {
    type Target = EncodedInput;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ordinary(value) => value,
            Self::Managed(value) => value,
        }
    }
}

impl DerefMut for TextInputBox {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Ordinary(value) => value,
            Self::Managed(value) => value,
        }
    }
}

pub(super) fn allocate(
    runtime: Option<&RuntimeContext>,
    input: EncodedInput,
) -> Result<TextInputBox, HostAllocationError> {
    let bytes = collection_block_retained_bytes(std::alloc::Layout::new::<EncodedInput>())?;
    let receipt = HostAllocationReceipt::admit(runtime, HostAllocationPlan::new(0, bytes)?)?;
    let Some(receipt) = receipt else {
        return Ok(TextInputBox::Ordinary(Box::new(input)));
    };
    let runtime = runtime.expect("managed decoder has runtime");
    let allocator = AdmittedCollectionAllocator::new(receipt.output_permit(runtime)?);
    let result = AllocatedBox::try_new_in(input, allocator).map_err(|_| {
        HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
        )
    });
    // Preserve cancellation/revocation if the allocator rejected a stale permit.
    receipt.checkpoint(runtime)?;
    result.map(TextInputBox::Managed)
}
