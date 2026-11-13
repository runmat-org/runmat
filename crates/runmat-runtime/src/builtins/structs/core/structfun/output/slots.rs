//! Known-count collector control storage. The selected managed path reserves
//! the actual final slots before constructing any collector; no temporary Vec.
use super::{allocation, ordinary_policy};
use crate::context::host_allocation::HostAllocationPolicy;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{ExactArrayStorage, ExactArrayStorageBuilder};

enum Storage<T> {
    Exact(ExactArrayStorageBuilder<Option<T>>),
    Ordinary(Vec<Option<T>>),
}
pub(crate) struct Slots<'a, T> {
    storage: Storage<T>,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
impl<'a, T> Slots<'a, T> {
    pub(crate) fn new(
        count: usize,
        runtime: Option<&'a RuntimeContext>,
        mut initialize: impl FnMut() -> crate::BuiltinResult<T>,
    ) -> crate::BuiltinResult<Self> {
        let raw = ExactArrayStorage::<Option<T>>::requested_bytes(count)
            .map_err(|e| allocation(runtime, e.into()))?;
        let raw =
            u64::try_from(raw).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?;
        let bytes = HostAllocationPlan::retained_exact_output_bytes(raw, usize::from(count != 0))
            .map_err(|e| allocation(runtime, e))?;
        let plan = HostAllocationPlan::new(0, bytes).map_err(|e| allocation(runtime, e))?;
        let receipt =
            HostAllocationReceipt::admit(runtime, plan).map_err(|e| allocation(runtime, e))?;
        let mut storage = if let Some(receipt) = &receipt {
            let runtime = runtime.expect("managed receipt has selected context");
            let permit = receipt
                .output_permit(runtime)
                .map_err(|e| allocation(Some(runtime), e))?;
            let slots = ExactArrayStorage::try_begin(count, &permit);
            receipt
                .checkpoint(runtime)
                .map_err(|e| allocation(Some(runtime), e))?;
            Storage::Exact(slots.map_err(|e| allocation(Some(runtime), e.into()))?)
        } else {
            check_ordinary(runtime)?;
            Storage::Ordinary(Vec::with_capacity(count))
        };
        for _ in 0..count {
            if let Some(receipt) = &receipt {
                receipt
                    .checkpoint(runtime.expect("managed runtime"))
                    .map_err(|e| allocation(runtime, e))?;
            }
            let value = Some(initialize()?);
            match &mut storage {
                Storage::Exact(slots) => slots
                    .try_push(value)
                    .map_err(|_| super::error::internal("structfun: collector count changed"))?,
                Storage::Ordinary(slots) => slots.push(value),
            }
        }
        Ok(Self {
            storage,
            receipt,
            runtime,
        })
    }
    pub(crate) fn values_mut(&mut self) -> crate::BuiltinResult<&mut [Option<T>]> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed runtime"))
                .map_err(|e| allocation(self.runtime, e))?;
        } else {
            check_ordinary(self.runtime)?;
        }
        Ok(match &mut self.storage {
            Storage::Exact(slots) => slots.initialized_mut(),
            Storage::Ordinary(slots) => slots,
        })
    }
}
fn check_ordinary(runtime: Option<&RuntimeContext>) -> crate::BuiltinResult<()> {
    if runtime.is_none() {
        return ordinary_policy();
    }
    match HostAllocationReceipt::policy(runtime).map_err(|e| allocation(runtime, e))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => {
            Err(allocation(runtime, HostAllocationError::StaleAuthority))
        }
    }
}
