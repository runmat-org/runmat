use runmat_value::{NumericDType, Tensor, Value};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::BuiltinResult;

pub(super) async fn output_f64(
    shape: &[usize],
    len: usize,
    runtime: Option<&RuntimeContext>,
    fill: impl FnMut(usize, &mut [f64]) -> usize,
) -> BuiltinResult<Value> {
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    if let Some(budget) = budget.as_mut() {
        budget
            .checkpoint(0)
            .await
            .map_err(|error| cooperation_error(runtime, error))?;
    }
    let receipt = output_receipt(shape, len, NumericDType::F64, runtime)?;
    if let (Some(runtime), Some(receipt)) = (runtime, receipt.as_ref()) {
        let mut exact = receipt
            .exact_output_tensor(runtime, NumericDType::F64, shape)
            .map_err(|error| allocation_error(Some(runtime), error))?;
        fill_chunks(
            exact.as_f64_slice_mut().expect("admitted double output"),
            fill,
            &mut budget,
            Some(runtime),
        )
        .await?;
        receipt
            .checkpoint(runtime)
            .map_err(|error| allocation_error(Some(runtime), error))?;
        let tensor = exact.into_tensor().map_err(|error| {
            allocation_error(Some(runtime), HostAllocationError::Numeric(error))
        })?;
        receipt
            .checkpoint(runtime)
            .map_err(|error| allocation_error(Some(runtime), error))?;
        return Ok(Value::Tensor(tensor));
    }
    let shape = copy_exact(shape)?;
    let mut values = filled(len, 0.0f64)?;
    fill_chunks(&mut values, fill, &mut budget, runtime).await?;
    Tensor::new(values, shape)
        .map(Value::Tensor)
        .map_err(|error| super::super::errors::internal("eps", error))
}

pub(super) async fn output_f32(
    shape: &[usize],
    len: usize,
    runtime: Option<&RuntimeContext>,
    fill: impl FnMut(usize, &mut [f32]) -> usize,
) -> BuiltinResult<Value> {
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    if let Some(budget) = budget.as_mut() {
        budget
            .checkpoint(0)
            .await
            .map_err(|error| cooperation_error(runtime, error))?;
    }
    let receipt = output_receipt(shape, len, NumericDType::F32, runtime)?;
    if let (Some(runtime), Some(receipt)) = (runtime, receipt.as_ref()) {
        let mut exact = receipt
            .exact_output_tensor(runtime, NumericDType::F32, shape)
            .map_err(|error| allocation_error(Some(runtime), error))?;
        fill_chunks(
            exact.as_f32_slice_mut().expect("admitted single output"),
            fill,
            &mut budget,
            Some(runtime),
        )
        .await?;
        receipt
            .checkpoint(runtime)
            .map_err(|error| allocation_error(Some(runtime), error))?;
        let tensor = exact.into_tensor().map_err(|error| {
            allocation_error(Some(runtime), HostAllocationError::Numeric(error))
        })?;
        receipt
            .checkpoint(runtime)
            .map_err(|error| allocation_error(Some(runtime), error))?;
        return Ok(Value::Tensor(tensor));
    }
    let shape = copy_exact(shape)?;
    let mut values = filled(len, 0.0f32)?;
    fill_chunks(&mut values, fill, &mut budget, runtime).await?;
    Tensor::from_f32(values, shape)
        .map(Value::Tensor)
        .map_err(|error| super::super::errors::internal("eps", error))
}

fn output_receipt(
    shape: &[usize],
    len: usize,
    dtype: NumericDType,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Option<HostAllocationReceipt>> {
    let bytes = HostAllocationPlan::retained_tensor_bytes_for_dimensions(dtype, len, shape.len())
        .map_err(|error| allocation_error(runtime, error))?;
    let plan =
        HostAllocationPlan::new(0, bytes).map_err(|error| allocation_error(runtime, error))?;
    HostAllocationReceipt::admit(runtime, plan).map_err(|error| allocation_error(runtime, error))
}

fn allocation_error(
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> crate::RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), "eps", source)
}

const OUTPUT_WORK_CHUNK: usize = 4096;

async fn fill_chunks<T>(
    output: &mut [T],
    mut fill: impl FnMut(usize, &mut [T]) -> usize,
    budget: &mut Option<CpuCooperationBudget>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for (chunk_index, chunk) in output.chunks_mut(OUTPUT_WORK_CHUNK).enumerate() {
        let work = fill(chunk_index * OUTPUT_WORK_CHUNK, chunk);
        if let Some(budget) = budget.as_mut() {
            budget
                .checkpoint(work as u64)
                .await
                .map_err(|error| cooperation_error(runtime, error))?;
        }
    }
    Ok(())
}

fn cooperation_error(
    runtime: Option<&RuntimeContext>,
    source: crate::execution::ExecutionServiceError,
) -> crate::RuntimeError {
    let mut error = source.into_runtime_error_in_context(runtime);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    crate::builtins::common::map_control_flow_with_builtin(error, "eps")
}

fn filled<T: Copy>(len: usize, value: T) -> BuiltinResult<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|error| super::super::errors::internal("eps", error.to_string()))?;
    result.resize(len, value);
    Ok(result)
}

fn copy_exact(input: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(input.len())
        .map_err(|error| super::super::errors::internal("eps", error.to_string()))?;
    result.extend_from_slice(input);
    Ok(result)
}

#[cfg(test)]
#[path = "output/tests.rs"]
mod tests;
