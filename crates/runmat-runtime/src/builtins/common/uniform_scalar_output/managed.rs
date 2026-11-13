//! Fixed-count callback scalars. Scratch contains only native scalar bits,
//! never GC children. Final payload is allocated once after the common class
//! join is known. Checkpoints fence authority; these loops do not async-yield.
use super::{classify::ClassifiedValue, promotion::Class, UniformScalarError};
use crate::context::{
    host_allocation::{
        collection_block_retained_bytes, host_allocation_runtime_error,
        AdmittedCollectionAllocator, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use allocator_api2::boxed::Box as AllocatedBox;
use runmat_value::{ExactArrayStorage, ExactArrayStorageBuilder, ExactHostNumericAllocationError};
use std::alloc::Layout;

mod output;
#[cfg(test)]
mod tests;
pub(crate) type ManagedOwner<'a> = AllocatedBox<Managed<'a>, AdmittedCollectionAllocator>;
pub(crate) struct Managed<'a> {
    // Scalar backing and its control allocation retire before receipt authority.
    values: ExactArrayStorageBuilder<ClassifiedValue>,
    class: Option<Class>,
    receipt: HostAllocationReceipt,
    runtime: &'a RuntimeContext,
}
impl<'a> Managed<'a> {
    pub(super) fn new(
        runtime: &'a RuntimeContext,
        count: usize,
    ) -> Result<Option<ManagedOwner<'a>>, UniformScalarError> {
        if HostAllocationReceipt::policy(Some(runtime)).map_err(|e| allocation(runtime, e))?
            == HostAllocationPolicy::Unbounded
        {
            return Ok(None);
        }
        let raw = ExactArrayStorage::<ClassifiedValue>::requested_bytes(count)
            .map_err(|e| allocation(runtime, e.into()))?;
        let bytes = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
            usize::from(count != 0),
        )
        .and_then(|bytes| {
            bytes
                .checked_add(collection_block_retained_bytes(Layout::new::<Self>())?)
                .ok_or(HostAllocationError::Overflow)
        })
        .map_err(|e| allocation(runtime, e))?;
        let plan = HostAllocationPlan::new(0, bytes).map_err(|e| allocation(runtime, e))?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
            .map_err(|e| allocation(runtime, e))?
            .ok_or_else(|| allocation(runtime, HostAllocationError::StaleAuthority))?;
        let permit = receipt
            .output_permit(runtime)
            .map_err(|e| allocation(runtime, e))?;
        let values = ExactArrayStorage::try_begin(count, &permit);
        receipt
            .checkpoint(runtime)
            .map_err(|e| allocation(runtime, e))?;
        let values = values.map_err(|e| allocation(runtime, e.into()))?;
        let allocator = receipt
            .collection_allocator(runtime)
            .map_err(|e| allocation(runtime, e))?;
        let owner = AllocatedBox::<Self, _>::try_new_uninit_in(allocator);
        receipt
            .checkpoint(runtime)
            .map_err(|e| allocation(runtime, e))?;
        let owner = owner.map_err(|_| {
            allocation(
                runtime,
                ExactHostNumericAllocationError::AllocationFailed.into(),
            )
        })?;
        Ok(Some(AllocatedBox::write(
            owner,
            Self {
                values,
                class: None,
                receipt,
                runtime,
            },
        )))
    }
    pub(super) fn class(&self) -> Option<Class> {
        self.class
    }
    pub(super) fn checkpoint(&self) -> Result<(), UniformScalarError> {
        self.receipt
            .checkpoint(self.runtime)
            .map_err(|e| allocation(self.runtime, e))
    }
    pub(super) fn push(&mut self, value: ClassifiedValue) -> Result<(), UniformScalarError> {
        self.checkpoint()?;
        let next = Class::of(&value)?;
        let class = self.class.map_or(Ok(next), |current| current.join(next))?;
        self.values
            .try_push(value)
            .map_err(|_| UniformScalarError::invalid_storage("callback count"))?;
        self.class = Some(class);
        Ok(())
    }
    pub(super) fn finish(
        &self,
        shape: &[usize],
    ) -> Result<runmat_value::Value, UniformScalarError> {
        self.checkpoint()?;
        if self.values.initialized().len() != self.values.expected_len() {
            return Err(UniformScalarError::invalid_storage("callback count"));
        }
        output::finish(self, shape)
    }
}
fn allocation(runtime: &RuntimeContext, source: HostAllocationError) -> UniformScalarError {
    UniformScalarError::execution(host_allocation_runtime_error(
        Some(runtime),
        "RunMat:UniformOutput:Allocation",
        "uniform callback output admission failed",
        source,
    ))
}
