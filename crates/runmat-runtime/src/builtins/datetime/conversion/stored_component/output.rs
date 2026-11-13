use super::{allocation_error, Work};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_value::{ExactHostTensorAllocation, NumericDType, Tensor, Value};

/// Exactly one final backing exists. The Managed state also retains its real
/// context and output headroom; the Unbounded state has neither finite claim.
pub(super) enum Output<'runtime, 'receipt> {
    Managed {
        runtime: &'runtime RuntimeContext,
        receipt: &'receipt HostAllocationReceipt,
        allocation: ExactHostTensorAllocation,
    },
    Unbounded(Tensor),
}

pub(super) fn admit(
    work: &Work<'_>,
    shape: &[usize],
    scratch: u64,
) -> BuiltinResult<Option<HostAllocationReceipt>> {
    work.check()?;
    let bytes = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, shape)
        .map_err(|source| allocation_error(work.runtime, work.label, source))?;
    let plan = HostAllocationPlan::new(scratch, bytes)
        .map_err(|source| allocation_error(work.runtime, work.label, source))?;
    let receipt = HostAllocationReceipt::admit(work.runtime, plan)
        .map_err(|source| allocation_error(work.runtime, work.label, source))?;
    // Admission cannot switch an initially Unbounded operation to Managed,
    // or replace the original Managed issuer, before physical allocation.
    work.check()?;
    Ok(receipt)
}

impl<'runtime, 'receipt> Output<'runtime, 'receipt> {
    pub(super) fn new(
        work: &Work<'runtime>,
        shape: &[usize],
        len: usize,
        receipt: Option<&'receipt HostAllocationReceipt>,
    ) -> BuiltinResult<Self> {
        // None cannot bypass a captured Managed issuer or an Unbounded policy
        // upgrade: check the original owner again before physical allocation.
        work.check()?;
        match (receipt, work.owner.as_ref()) {
            (Some(_), Some(_)) | (None, None) => {}
            _ => {
                return Err(allocation_error(
                    work.runtime,
                    work.label,
                    HostAllocationError::StaleAuthority,
                ))
            }
        }
        let result = if let Some(receipt) = receipt {
            let runtime = work.runtime.ok_or_else(|| {
                allocation_error(
                    work.runtime,
                    work.label,
                    HostAllocationError::StaleAuthority,
                )
            })?;
            let allocation = receipt
                .exact_output_tensor(runtime, NumericDType::F64, shape)
                .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
            Self::Managed {
                runtime,
                receipt,
                allocation,
            }
        } else {
            let mut values = Vec::new();
            values.try_reserve_exact(len).map_err(|_| {
                allocation_error(
                    work.runtime,
                    work.label,
                    HostAllocationError::Numeric(
                        runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                    ),
                )
            })?;
            values.resize(len, 0.0);
            let tensor = Tensor::new(values, shape.to_vec()).map_err(|error| {
                super::never_retry(super::super::super::datetime_error(format!(
                    "{}: {error}",
                    work.label
                )))
            })?;
            Self::Unbounded(tensor)
        };
        result.check(work)?;
        Ok(result)
    }

    pub(super) fn check(&self, work: &Work<'_>) -> BuiltinResult<()> {
        work.check()?;
        if let Self::Managed {
            runtime, receipt, ..
        } = self
        {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
        }
        Ok(())
    }

    pub(super) fn set(&mut self, index: usize, value: f64) {
        let target = match self {
            Self::Managed { allocation, .. } => allocation
                .as_f64_slice_mut()
                .expect("exact component output is double"),
            Self::Unbounded(tensor) => tensor
                .try_as_f64_slice_mut_unique()
                .expect("fresh component output is unique double"),
        };
        target[index] = value;
    }

    pub(super) fn finish(self, work: &Work<'_>) -> BuiltinResult<Value> {
        self.check(work)?;
        let tensor = match self {
            Self::Managed {
                runtime,
                receipt,
                allocation,
            } => {
                let tensor = allocation
                    .into_tensor()
                    .map_err(|source| allocation_error(Some(runtime), work.label, source.into()))?;
                // Receipt stays alive through conversion and the final fence;
                // returned payload and shape claims then outlive the stage.
                work.check()?;
                receipt
                    .checkpoint(runtime)
                    .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
                tensor
            }
            Self::Unbounded(tensor) => {
                work.check()?;
                tensor
            }
        };
        Ok(Value::Tensor(tensor))
    }
}
