//! One claim-preserving result-list transport shared by iterative solvers.

use runmat_value::{
    ExactHostAllocationPermit, ExactHostNumericAllocationError, OutputListInitializationError,
    OutputListStorage, Value,
};

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};

/// History is real in the selected precision, independent of a complex x.
pub(in crate::builtins::math::linalg::factor) async fn admitted_history(
    receipt: &crate::context::host_allocation::HostAllocationReceipt,
    runtime: &crate::context::RuntimeContext,
    dtype: runmat_value::NumericDType,
    values: &[f64],
    builtin: &'static str,
) -> Result<Value, crate::RuntimeError> {
    let fail = |e| super::copy::admission(Some(runtime), builtin, e);
    let mut output = receipt
        .exact_output_tensor(runtime, dtype, &[values.len(), 1])
        .map_err(fail)?;
    let mut budget = runtime.cpu_cooperation();
    for (index, chunk) in values.chunks(1024).enumerate() {
        receipt.checkpoint(runtime).map_err(fail)?;
        budget
            .checkpoint(chunk.len() as u64)
            .await
            .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?;
        receipt.checkpoint(runtime).map_err(fail)?;
        let start = index * 1024;
        match dtype {
            runmat_value::NumericDType::F32 => {
                let target = output.as_f32_slice_mut().ok_or_else(|| {
                    fail(HostAllocationError::Numeric(
                        ExactHostNumericAllocationError::UnsupportedClass,
                    ))
                })?;
                for (target, source) in target[start..start + chunk.len()].iter_mut().zip(chunk) {
                    *target = *source as f32;
                }
            }
            runmat_value::NumericDType::F64 => {
                let target = output.as_f64_slice_mut().ok_or_else(|| {
                    fail(HostAllocationError::Numeric(
                        ExactHostNumericAllocationError::UnsupportedClass,
                    ))
                })?;
                target[start..start + chunk.len()].copy_from_slice(chunk);
            }
            _ => {
                return Err(fail(HostAllocationError::Numeric(
                    ExactHostNumericAllocationError::UnsupportedClass,
                )))
            }
        }
    }
    let value = output
        .into_tensor()
        .map(Value::Tensor)
        .map_err(|e| fail(HostAllocationError::Numeric(e)))?;
    receipt.checkpoint(runtime).map_err(fail)?;
    Ok(value)
}

pub(in crate::builtins::math::linalg::factor) async fn unbounded_history(
    runtime: Option<&crate::context::RuntimeContext>,
    dtype: runmat_value::NumericDType,
    values: &[f64],
    builtin: &'static str,
) -> Result<Value, crate::RuntimeError> {
    use runmat_value::{NumericDType, Tensor};
    if dtype == NumericDType::F64 {
        return super::copy::unbounded::<f64>(runtime, values, builtin).await;
    }
    let fail = |e| super::copy::admission(runtime, builtin, HostAllocationError::Numeric(e));
    if dtype != NumericDType::F32 {
        return Err(fail(ExactHostNumericAllocationError::UnsupportedClass));
    }
    super::copy::unbounded_checkpoint(runtime, builtin)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| fail(ExactHostNumericAllocationError::AllocationFailed))?;
    let mut budget = runtime.map(crate::context::RuntimeContext::cpu_cooperation);
    for chunk in values.chunks(1024) {
        if let Some(budget) = budget.as_mut() {
            budget
                .checkpoint(chunk.len() as u64)
                .await
                .map_err(|e| e.into_runtime_error_in_context(runtime))?;
        }
        super::copy::unbounded_checkpoint(runtime, builtin)?;
        output.extend(chunk.iter().map(|value| *value as f32));
    }
    let result = Tensor::from_f32(output, vec![values.len(), 1])
        .map(Value::Tensor)
        .map_err(|e| {
            crate::build_runtime_error(e.to_string())
                .with_builtin(builtin)
                .build()
        })?;
    super::copy::unbounded_checkpoint(runtime, builtin)?;
    Ok(result)
}

pub(in crate::builtins::math::linalg::factor) fn retained_list_bytes(
    count: usize,
) -> Result<u64, HostAllocationError> {
    let raw = OutputListStorage::requested_bytes(count).map_err(HostAllocationError::Numeric)?;
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
        1,
    )
}

pub(in crate::builtins::math::linalg::factor) fn initialize_list(
    count: usize,
    mut values: [Option<Value>; 5],
    permit: &dyn ExactHostAllocationPermit,
) -> Result<Value, ExactHostNumericAllocationError> {
    let list = OutputListStorage::try_initialize(count, permit, |index| {
        values
            .get_mut(index)
            .and_then(Option::take)
            .ok_or(ExactHostNumericAllocationError::UnsupportedClass)
    })
    .map_err(|error| match error {
        OutputListInitializationError::Allocation(error)
        | OutputListInitializationError::Element(error) => error,
    })?;
    Ok(Value::OutputList(list))
}
