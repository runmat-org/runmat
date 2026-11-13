//! Sparse numerical fill; each replacement owns its actual layout and claim.
use super::{admit, Fence, SparseWorkError, Work};
use crate::context::host_allocation::{
    collection_block_retained_bytes, AdmittedCollectionAllocator, HostAllocationError,
};
use allocator_api2::vec::Vec as AdmittedVec;
use runmat_value::ExactHostNumericAllocationError;
use std::alloc::Layout;
mod transfer;

enum Storage<T> {
    Managed(AdmittedVec<T, AdmittedCollectionAllocator>),
    Unbounded(Vec<T>),
}
impl<T> Storage<T> {
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Managed(v) => v,
            Self::Unbounded(v) => v,
        }
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Managed(v) => v,
            Self::Unbounded(v) => v,
        }
    }
    fn capacity(&self) -> usize {
        match self {
            Self::Managed(v) => v.capacity(),
            Self::Unbounded(v) => v.capacity(),
        }
    }
    fn push(&mut self, value: T) {
        match self {
            Self::Managed(v) => v.push(value),
            Self::Unbounded(v) => v.push(value),
        }
    }
    fn pop(&mut self) -> Option<T> {
        match self {
            Self::Managed(v) => v.pop(),
            Self::Unbounded(v) => v.pop(),
        }
    }
}
pub(in crate::builtins::math::linalg::ops::sparse_factor) struct ClaimedVec<T: Copy> {
    values: Option<Storage<T>>,
}
impl<T: Copy> ClaimedVec<T> {
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn new() -> Self {
        Self { values: None }
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn as_slice(&self) -> &[T] {
        self.values.as_ref().map_or(&[], Storage::as_slice)
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn as_mut_slice(
        &mut self,
    ) -> &mut [T] {
        self.values.as_mut().map_or(&mut [], Storage::as_mut_slice)
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn len(&self) -> usize {
        self.as_slice().len()
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn ensure(
        &mut self,
        additional: usize,
        work: &mut Work<'_>,
    ) -> Result<(), SparseWorkError> {
        work.fence.checkpoint_optional(work.runtime)?;
        let needed = self
            .len()
            .checked_add(additional)
            .ok_or(HostAllocationError::Overflow)?;
        let capacity = self.values.as_ref().map_or(0, Storage::capacity);
        if needed <= capacity {
            return Ok(());
        }
        let next = capacity
            .checked_mul(2)
            .ok_or(HostAllocationError::Overflow)?
            .max(needed)
            .max(4);
        let layout = Layout::array::<T>(next).map_err(|_| HostAllocationError::Overflow)?;
        let receipt = match &work.fence {
            Fence::Managed(_) => Some(admit(
                work.required_runtime()?,
                0,
                collection_block_retained_bytes(layout)?,
            )?),
            Fence::Unbounded => None,
        };
        work.fence.checkpoint_optional(work.runtime)?;
        let mut replacement = match receipt.as_ref() {
            Some(receipt) => {
                let mut values =
                    AdmittedVec::new_in(receipt.collection_allocator(work.required_runtime()?)?);
                let result = values.try_reserve_exact(next);
                receipt.checkpoint(work.required_runtime()?)?;
                result.map_err(|_| allocation_failed())?;
                Storage::Managed(values)
            }
            None => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(next)
                    .map_err(|_| allocation_failed())?;
                work.fence.checkpoint_optional(work.runtime)?;
                Storage::Unbounded(values)
            }
        };
        work.fence.checkpoint_optional(work.runtime)?;
        for &value in self.as_slice() {
            replacement.push(value);
            work.tick().await?;
        }
        work.flush().await?;
        if let Some(receipt) = &receipt {
            receipt.checkpoint(work.required_runtime()?)?;
        }
        self.values = Some(replacement);
        Ok(())
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn push_reserved(
        &mut self,
        value: T,
    ) {
        let values = self.values.as_mut().expect("ensure precedes append");
        assert!(
            values.as_slice().len() < values.capacity(),
            "implicit sparse growth is forbidden"
        );
        values.push(value);
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn pop(&mut self) -> Option<T> {
        self.values.as_mut().and_then(Storage::pop)
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn zeroed(
        len: usize,
        work: &mut Work<'_>,
    ) -> Result<Self, SparseWorkError>
    where
        T: Default,
    {
        let mut result = Self::new();
        result.ensure(len, work).await?;
        for _ in 0..len {
            result.push_reserved(T::default());
            work.tick().await?;
        }
        Ok(result)
    }
}
fn allocation_failed() -> SparseWorkError {
    HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed).into()
}
