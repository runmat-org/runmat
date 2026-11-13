//! Canonical requested-output adaptation over independently owned results.
#[cfg(test)]
mod tests;
use crate::{
    context::{
        host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
        RuntimeContext,
    },
    BuiltinResult,
};
use runmat_builtins::CumulativeKind;
use runmat_value::{OutputListInitializationError, OutputListStorage, Value};
pub(in crate::builtins::math::reduction) struct Publication<'a> {
    kind: CumulativeKind,
    runtime: Option<&'a RuntimeContext>,
    receipt: Option<HostAllocationReceipt>,
    count: Option<usize>,
}
impl<'a> Publication<'a> {
    pub fn admit(kind: CumulativeKind, runtime: Option<&'a RuntimeContext>) -> BuiltinResult<Self> {
        let entry = kind.catalog_entry();
        crate::output_count::validate_bounded_outputs(
            entry.descriptor,
            &entry.descriptor.errors[0],
            entry.identity.name,
        )?;
        let count = crate::output_count::current_output_count();
        let receipt = match count {
            Some(count) if count > 0 => {
                let bytes = OutputListStorage::requested_bytes(count)
                    .map_err(HostAllocationError::Numeric)
                    .map_err(|e| super::admission::control(kind, runtime, e))?;
                let bytes = u64::try_from(bytes).map_err(|_| {
                    super::admission::control(kind, runtime, HostAllocationError::Overflow)
                })?;
                let plan = HostAllocationPlan::retained_exact_output_bytes(bytes, 1)
                    .and_then(|bytes| HostAllocationPlan::new(0, bytes))
                    .map_err(|e| super::admission::control(kind, runtime, e))?;
                HostAllocationReceipt::admit(runtime, plan)
                    .map_err(|e| super::admission::control(kind, runtime, e))?
            }
            _ => None,
        };
        Ok(Self {
            kind,
            runtime,
            receipt,
            count,
        })
    }
    pub fn demand(&self) -> runmat_accelerate_api::ProviderExtremaDemand {
        if self.count.is_some_and(|count| count > 1) {
            runmat_accelerate_api::ProviderExtremaDemand::ValuesAndIndices
        } else {
            runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly
        }
    }
    pub fn finish(self, values: Value, indices: Value) -> BuiltinResult<Value> {
        let mut values = [Some(values), Some(indices)];
        let retained = self.count.unwrap_or(1);
        for index in retained..values.len() {
            let Some(value) = values[index].take() else {
                continue;
            };
            if let Err(mut error) = retire(self.kind, value) {
                for value in values.into_iter().flatten() {
                    if let Value::GpuTensor(handle) = value {
                        error =
                            crate::builtins::common::gpu_helpers::reject_gpu_output(handle, error);
                    }
                }
                return Err(error);
            }
        }
        if self.count.is_none() {
            return Ok(values[0].take().expect("one value"));
        }
        if retained == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if let (Some(runtime), Some(receipt)) = (self.runtime, self.receipt.as_ref()) {
            let result = receipt.with_output_allocation(runtime, |permit| {
                OutputListStorage::try_initialize(retained, permit, |index| {
                    values[index]
                        .take()
                        .ok_or(runmat_value::ExactHostNumericAllocationError::UnsupportedClass)
                })
                .map(Value::OutputList)
                .map_err(|e| match e {
                    OutputListInitializationError::Allocation(e)
                    | OutputListInitializationError::Element(e) => e,
                })
            });
            return result.map_err(|e| {
                let mut error = super::admission::control(self.kind, self.runtime, e);
                for value in values.into_iter().flatten() {
                    if let Value::GpuTensor(handle) = value {
                        error =
                            crate::builtins::common::gpu_helpers::reject_gpu_output(handle, error);
                    }
                }
                error
            });
        }
        Ok(Value::output_list(
            values
                .into_iter()
                .take(retained)
                .flatten()
                .collect::<Vec<_>>(),
        ))
    }
}
pub(in crate::builtins::math::reduction) fn retire(
    kind: CumulativeKind,
    value: Value,
) -> BuiltinResult<()> {
    if let Value::GpuTensor(handle) = value {
        handle.release().map_err(|source| {
            let mut error = super::failure(kind, "discarded provider result could not retire");
            error.source = Some(Box::new(source));
            error
        })?;
    }
    Ok(())
}
