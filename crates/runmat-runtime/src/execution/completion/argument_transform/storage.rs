//! Shared admission, final slot owner, and retained registration publication.
use super::super::{RootedValue, RootedValueChildren, RootedValues};
use crate::call::RootedFunctionArguments;
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::RuntimeError;
use runmat_value::{exact_capacity_vec, ExactArrayStorageBuilder, OutputListStorage, Value};

enum Slots {
    Exact(ExactArrayStorageBuilder<Value>),
    Ordinary(Vec<Value>),
}
pub(super) struct ArgumentTransform<'a> {
    slots: Slots,
    children: RootedValueChildren<'a>,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
pub(super) fn allocation(
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:Call:Arguments",
        "argument transformation failed",
        source,
    )
}
impl<'a> ArgumentTransform<'a> {
    pub(super) fn new(
        source: &RootedFunctionArguments,
        count: usize,
        inserted: Option<&RootedValue>,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, RuntimeError> {
        Self::with_roots(count, runtime, |children| {
            source.retain_existing_roots(children)?;
            if let Some(inserted) = inserted {
                children.try_retain_handles(inserted._roots.clone())?;
            }
            Ok(())
        })
    }
    pub(super) fn from_source_lease(
        count: usize,
        source: &super::super::RootedValueLease,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, RuntimeError> {
        Self::with_roots(count, runtime, |children| {
            children.try_retain_handles(source.shared_handles())
        })
    }
    fn with_roots(
        count: usize,
        runtime: Option<&'a RuntimeContext>,
        retain: impl FnOnce(&mut RootedValueChildren<'a>) -> Result<(), HostAllocationError>,
    ) -> Result<Self, RuntimeError> {
        let error = |source| allocation(runtime, source);
        let bytes = OutputListStorage::requested_bytes(count).map_err(|e| error(e.into()))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| error(HostAllocationError::Overflow))?,
            usize::from(count != 0),
        )
        .map_err(error)?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained).map_err(error)?,
        )
        .map_err(error)?;
        let mut children = match runtime {
            Some(runtime) => RootedValueChildren::in_context(runtime).map_err(error)?,
            None => RootedValueChildren::new(), // admit(None) already verified ordinary policy.
        };
        retain(&mut children).map_err(error)?;
        let slots = if let Some(receipt) = &receipt {
            let permit = receipt
                .output_permit(runtime.expect("managed transform context"))
                .map_err(error)?;
            Slots::Exact(OutputListStorage::try_begin(count, &permit).map_err(|e| error(e.into()))?)
        } else {
            Slots::Ordinary(exact_capacity_vec(count).map_err(|e| error(e.into()))?)
        };
        let result = Self {
            slots,
            children,
            receipt,
            runtime,
        };
        result.checkpoint()?;
        Ok(result)
    }
    pub(super) fn checkpoint(&self) -> Result<(), RuntimeError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed transform context"))
                .map_err(|e| allocation(self.runtime, e))
        } else {
            match HostAllocationReceipt::policy(self.runtime)
                .map_err(|e| allocation(self.runtime, e))?
            {
                HostAllocationPolicy::Unbounded => Ok(()),
                HostAllocationPolicy::Managed => Err(allocation(
                    self.runtime,
                    HostAllocationError::StaleAuthority,
                )),
            }
        }
    }
    pub(super) fn push(&mut self, value: Value) -> Result<(), RuntimeError> {
        self.checkpoint()?;
        match &mut self.slots {
            Slots::Exact(slots) => slots.try_push(value).expect("quoted argument count"),
            Slots::Ordinary(slots) => slots.push(value),
        }
        self.checkpoint()
    }
    pub(super) fn push_owned(&mut self, rooted: RootedValue) -> Result<(), RuntimeError> {
        let super::super::RootedResult { value, _roots } = rooted;
        // Preserve the returned owner's independent registration until both
        // registration metadata and physical final slot have accepted it.
        if let Err(error) = self.children.try_retain_handles(_roots.clone()) {
            drop(value);
            drop(_roots);
            return Err(allocation(self.runtime, error));
        }
        let result = self.push(value);
        drop(_roots);
        result
    }
    pub(super) fn finish(self) -> Result<RootedValues, RuntimeError> {
        self.checkpoint()?;
        let values = match self.slots {
            Slots::Exact(slots) => slots.try_finish().expect("all argument slots filled"),
            Slots::Ordinary(slots) => {
                if slots.capacity() == 0 {
                    OutputListStorage::empty()
                } else {
                    OutputListStorage::shared_unbounded(slots)
                }
            }
        };
        self.children
            .try_publish_values(values)
            .map_err(|e| allocation(self.runtime, e))
    }
}
