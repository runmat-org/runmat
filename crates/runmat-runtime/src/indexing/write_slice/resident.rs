//! Device slice mutation with request-local owning intermediates.
use super::*;
use crate::builtins::common::gpu_helpers;
use crate::indexing::resident as ownership;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ProviderExecutionError, ProviderPrecision,
};

#[cfg(test)]
mod tests;

fn optional<T>(result: anyhow::Result<T>, context: &str) -> Result<Option<T>, RuntimeError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(cause) if gpu_helpers::provider_hook_is_unsupported(&cause) => Ok(None),
        Err(cause) => Err(ownership::failure(context, cause.into())),
    }
}

fn same_owner(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    match (
        gpu_helpers::exact_provider_for_handle(left),
        gpu_helpers::exact_provider_for_handle(right),
    ) {
        (Some(left), Some(right)) => std::ptr::eq(left, right),
        _ => false,
    }
}

fn functional_result(
    output: GpuTensorHandle,
    target: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> Result<Value, RuntimeError> {
    if output.shares_allocation_with(rhs) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            mex(
                "AccelerationOperationFailed",
                "Functional slice assignment aliased its RHS",
            ),
        ));
    }
    ownership::admit_read(output, target, target.shape())
}

pub async fn assign_gpu_slice_with_plan(
    target: &GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        return Ok(Value::GpuTensor(target.clone()));
    }
    let mut indices = plan.indices.clone();
    indices.sort_unstable();
    if indices.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(mex(
            "RepeatedGpuAssignmentIndex",
            "gpuArray indexed assignment does not support repeated target subscripts",
        ));
    }
    let provider = ownership::owner(target)?;
    let protected: Vec<_> = match rhs {
        Value::GpuTensor(handle) => vec![handle],
        _ => Vec::new(),
    };
    let _target_residency = gpu_helpers::ResidencyRestoreGuard::new(target);
    let _rhs_residency: Vec<_> = protected
        .iter()
        .map(|input| gpu_helpers::ResidencyRestoreGuard::new(input))
        .collect();
    let integer = runmat_accelerate_api::handle_integer_type(target).is_some();
    let real = runmat_accelerate_api::handle_storage(target) == GpuTensorStorage::Real;
    if let Value::GpuTensor(rhs_handle) = rhs {
        if same_owner(target, rhs_handle)
            && real
            && runmat_accelerate_api::handle_storage(rhs_handle) == GpuTensorStorage::Real
        {
            if integer && target.physical().element_type() == rhs_handle.physical().element_type() {
                if let Ok(rhs_indices) = integer_gpu_rhs_indices_for_plan(rhs_handle.shape(), plan)
                {
                    let identity = !rhs_handle.shares_allocation_with(target)
                        && rhs_indices
                            .iter()
                            .enumerate()
                            .all(|(index, &actual)| actual as usize == index);
                    let values = if identity {
                        Some(rhs_handle.clone())
                    } else {
                        match optional(
                            provider.gather_linear(rhs_handle, &rhs_indices, &plan.output_shape),
                            "integer RHS expansion",
                        )? {
                            Some(output) => {
                                if output.shares_allocation_with(target) {
                                    return Err(gpu_helpers::reject_gpu_output(
                                        output,
                                        mex(
                                            "AccelerationOperationFailed",
                                            "RHS expansion aliased the assignment target",
                                        ),
                                    ));
                                }
                                let Value::GpuTensor(output) =
                                    ownership::admit_read(output, rhs_handle, &plan.output_shape)?
                                else {
                                    unreachable!("resident admission");
                                };
                                Some(output)
                            }
                            None => None,
                        }
                    };
                    if let Some(values) = values {
                        if optional(
                            provider.scatter_linear(target, &plan.indices, &values),
                            "integer slice assignment",
                        )?
                        .is_some()
                        {
                            return Ok(Value::GpuTensor(target.clone()));
                        }
                    }
                }
            } else if !integer {
                let rows = plan.base_shape.first().copied().unwrap_or(1);
                let cols = plan.base_shape.get(1).copied().unwrap_or(1);
                if let Some(col) = plan.properties.full_column {
                    if col < cols
                        && matches!(rhs_handle.shape(), [length] | [length, 1] if *length == rows)
                    {
                        if let Some(output) = optional(
                            provider.scatter_column(target, col, rhs_handle),
                            "column assignment",
                        )? {
                            return functional_result(output, target, rhs_handle);
                        }
                    }
                }
                if let Some(row) = plan.properties.full_row {
                    if row < rows
                        && matches!(rhs_handle.shape(), [length] | [1, length] if *length == cols)
                    {
                        if let Some(output) = optional(
                            provider.scatter_row(target, row, rhs_handle),
                            "row assignment",
                        )? {
                            return functional_result(output, target, rhs_handle);
                        }
                    }
                }
            }
        }
    }
    let rhs_complex = matches!(rhs, Value::Complex(..) | Value::ComplexTensor(_))
        || matches!(rhs, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved);
    if real && !integer && !rhs_complex {
        let values = materialize_rhs_real_for_plan(rhs, plan).await?;
        let shape = vec![values.len(), 1];
        let tensor =
            if runmat_accelerate_api::handle_precision(target) == Some(ProviderPrecision::F32) {
                Tensor::from_f32(
                    values.into_iter().map(|value| value as f32).collect(),
                    shape,
                )
            } else {
                Tensor::new(values, shape)
            }
            .map_err(|cause| map_slice_shape_error("assignment RHS", cause))?;
        let prepared = match ownership::restore(target, Value::Tensor(tensor), &protected) {
            Ok(Value::GpuTensor(handle)) => Some(handle),
            Ok(_) => unreachable!("resident restore"),
            Err(cause) => {
                let cause = ProviderExecutionError::from_source(cause);
                if cause.is_unsupported() {
                    None
                } else {
                    return Err(ownership::failure("assignment RHS upload", cause));
                }
            }
        };
        if let Some(values) = prepared {
            if optional(
                provider.scatter_linear(target, &plan.indices, &values),
                "slice assignment",
            )?
            .is_some()
            {
                return Ok(Value::GpuTensor(target.clone()));
            }
        }
    }
    let host = ownership::gather_numeric(target).await?;
    let result = match host {
        Value::Tensor(tensor) => assign_tensor_with_plan(tensor, plan, rhs).await?,
        Value::ComplexTensor(tensor) => assign_complex_with_plan(tensor, plan, rhs).await?,
        _ => {
            return Err(mex(
                "AccelerationOperationFailed",
                "Invalid resident slice assignment carrier",
            ))
        }
    };
    ownership::restore(target, result, &protected)
}

pub async fn delete_gpu_slice_with_plan(
    target: &GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        return Ok(Value::GpuTensor(target.clone()));
    }
    let host = ownership::gather_numeric(target).await?;
    let result = match host {
        Value::Tensor(tensor) => delete_tensor_with_plan(tensor, plan, rhs)?,
        Value::ComplexTensor(tensor) => delete_complex_with_plan(tensor, plan, rhs)?,
        _ => {
            return Err(mex(
                "AccelerationOperationFailed",
                "Invalid resident slice deletion carrier",
            ))
        }
    };
    ownership::restore(target, result, &[])
}
