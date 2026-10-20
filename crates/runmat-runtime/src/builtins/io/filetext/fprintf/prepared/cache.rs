//! Exact, operation-local storage for already gathered format/fid candidates.
//! Entries contain only the root-preserving results of the gather operation;
//! the original argument owner remains live until final data-slot publication.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use runmat_value::{ExactArrayStorage, Value};

pub(super) struct Cache<'a> {
    values: ExactArrayStorage<Option<Value>>,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
impl<'a> Cache<'a> {
    pub(super) fn new(
        count: usize,
        runtime: Option<&'a RuntimeContext>,
    ) -> crate::BuiltinResult<Self> {
        let error = |source| allocation(runtime, source);
        let bytes = ExactArrayStorage::<Option<Value>>::requested_bytes(count)
            .map_err(|e| error(e.into()))?;
        let bytes = u64::try_from(bytes).map_err(|_| error(HostAllocationError::Overflow))?;
        let retained =
            HostAllocationPlan::retained_exact_output_bytes(bytes, usize::from(count != 0))
                .map_err(error)?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained).map_err(error)?,
        )
        .map_err(error)?;
        let values = if let Some(receipt) = &receipt {
            let runtime = runtime.expect("managed cache context");
            let permit = receipt.output_permit(runtime).map_err(error)?;
            let mut values =
                ExactArrayStorage::try_begin(count, &permit).map_err(|e| error(e.into()))?;
            for _ in 0..count {
                receipt.checkpoint(runtime).map_err(error)?;
                values.try_push(None).expect("quoted cache slots");
            }
            values.try_finish().expect("initialized cache slots")
        } else {
            // Admission above checked the explicitly selected ordinary policy.
            let mut values =
                runmat_value::exact_capacity_vec(count).map_err(|e| error(e.into()))?;
            values.resize_with(count, || None);
            ExactArrayStorage::unbounded(values)
        };
        Ok(Self {
            values,
            receipt,
            runtime,
        })
    }
    pub(super) fn checkpoint(&self) -> crate::BuiltinResult<()> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed cache context"))
                .map_err(|e| allocation(self.runtime, e))?;
        } else if HostAllocationReceipt::policy(self.runtime)
            .map_err(|e| allocation(self.runtime, e))?
            != HostAllocationPolicy::Unbounded
        {
            return Err(allocation(
                self.runtime,
                HostAllocationError::StaleAuthority,
            ));
        }
        Ok(())
    }
    pub(super) fn get(&self, index: usize) -> Option<&Value> {
        self.values[index].as_ref()
    }
    pub(super) fn put(&mut self, index: usize, value: Value) -> crate::BuiltinResult<()> {
        self.checkpoint()?;
        self.values
            .try_as_mut_slice_unique()
            .expect("unpublished cache has one owner")[index] = Some(value);
        self.checkpoint()
    }
    pub(super) fn take(&mut self, index: usize) -> Option<Value> {
        self.values
            .try_as_mut_slice_unique()
            .expect("unpublished cache has one owner")[index]
            .take()
    }
}
fn allocation(runtime: Option<&RuntimeContext>, cause: HostAllocationError) -> crate::RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:fprintf:Allocation",
        "fprintf argument staging allocation failed",
        cause,
    )
}
