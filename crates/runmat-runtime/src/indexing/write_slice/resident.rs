//! Device slice mutation with request-local owning intermediates.
use super::*;
use crate::builtins::common::gpu_helpers;
use crate::indexing::resident as ownership;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ProviderExecutionError, ProviderPrecision,
    ProviderScatterValueMapping,
};

#[cfg(test)]
#[path = "resident/functional_scatter_tests.rs"]
mod functional_scatter_tests;
#[cfg(test)]
mod tests;

fn optional_execution<T>(
    result: runmat_accelerate_api::ProviderExecutionResult<T>,
    context: &str,
) -> Result<Option<T>, RuntimeError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(cause) if cause.is_unsupported() => Ok(None),
        Err(cause) => Err(ownership::failure(context, cause)),
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
    protected_rhs: &[&GpuTensorHandle],
) -> Result<Value, RuntimeError> {
    if protected_rhs
        .iter()
        .any(|rhs| output.shares_allocation_with(rhs))
    {
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
    let context = crate::context::legacy::active();
    assign_gpu_slice_with_plan_in_context(context.as_ref(), target, plan, rhs).await
}

pub async fn assign_gpu_slice_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    target: &GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        super::empty_rhs::validate_resident_rhs(target, rhs, plan)?;
        return Ok(Value::GpuTensor(target.clone()));
    }
    // Only the resident branch owns this admitted continuation. Host-only
    // indexing keeps its allocation-free unique mutation path and does not
    // embed the provider's entire fallback future in every caller's frame.
    let result = crate::execution::rooted_continuation::admit_rooted_future_with(
        context.cloned(),
        "resident slice assign",
        crate::builtin::builtin_control_error,
        move |context| async move {
            let value = assign_resident(context.as_ref(), target, plan, rhs).await?;
            crate::execution::RootedValue::in_optional_context(value, context.as_ref())
        },
    )
    .await?;
    Ok(result.with_owned(|value| value))
}

async fn assign_resident(
    context: Option<&crate::context::RuntimeContext>,
    target: &GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let managed = crate::context::host_allocation::HostAllocationReceipt::policy(context).map_err(
        |source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                context,
                "RunMat:Indexing:Ownership",
                "resident indexing policy",
                source,
            )
        },
    )? != crate::context::host_allocation::HostAllocationPolicy::Unbounded;
    let mut indices =
        crate::indexing::plan::admission::Scratch::<u32>::new(context, plan.indices.len())?;
    indices.as_mut_slice().copy_from_slice(&plan.indices);
    owner.check()?;
    // The existing sort is an indivisible CPU operation; check both sides.
    indices.as_mut_slice().sort_unstable();
    owner.check()?;
    if indices.as_slice().windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(mex(
            "RepeatedGpuAssignmentIndex",
            "gpuArray indexed assignment does not support repeated target subscripts",
        ));
    }
    let growing = plan.base_shape != target.shape();
    if growing {
        super::empty_rhs::validate_resident_rhs(target, rhs, plan)?;
    }
    let provider = ownership::owner(target)?;
    let protected = match rhs {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    };
    let protected = protected.as_slice();
    let _target_residency = gpu_helpers::ResidencyRestoreGuard::new(target);
    let _rhs_residency = protected
        .first()
        .map(|input| gpu_helpers::ResidencyRestoreGuard::new(input));
    if growing {
        // Same-shape scatter hooks cannot publish the new shape. Preserve the
        // input leases, gather once, and restore the class-checked grown host
        // result to the original provider with both operands protected.
        let host = owner
            .await_stage(ownership::gather_in_context(context, target))
            .await??;
        let result = assign_host(context, host, plan, rhs).await?;
        return ownership::restore_in_context(context, target, result, protected);
    }
    let integer = runmat_accelerate_api::handle_integer_type(target).is_some();
    let real = runmat_accelerate_api::handle_storage(target) == GpuTensorStorage::Real;
    if let Value::GpuTensor(rhs_handle) = rhs {
        if same_owner(target, rhs_handle)
            && real
            && runmat_accelerate_api::handle_storage(rhs_handle) == GpuTensorStorage::Real
        {
            if target.physical().element_type() == rhs_handle.physical().element_type() {
                let rhs_len = super::checked_assignment_element_count(rhs_handle.shape())?;
                let layout = if plan.dims == 1 {
                    super::rhs_shape::AssignmentRhsLayout::linear(rhs_len, plan.indices.len())
                } else {
                    super::rhs_shape::AssignmentRhsLayout::multidimensional(
                        rhs_handle.shape(),
                        rhs_len,
                        &plan.selection_lengths,
                    )
                };
                // Non-scalar native operands already have canonical linear
                // order. Functional scatter never mutates either operand,
                // including when RHS aliases target. Scalar replication uses
                // the borrowed host mapping below, not a full index mirror.
                if layout.is_ok_and(|layout| layout.count() == plan.indices.len())
                    && (rhs_len == plan.indices.len() || rhs_len == 1)
                {
                    if rhs_len == plan.indices.len() || managed {
                        let mapping = if rhs_len == 1 && plan.indices.len() != 1 {
                            ProviderScatterValueMapping::Scalar
                        } else {
                            ProviderScatterValueMapping::Linear
                        };
                        if let Some(output) = optional_execution(
                            ownership::functional_scatter_in_context(
                                context,
                                target,
                                &plan.indices,
                                rhs_handle,
                                mapping,
                            )
                            .await,
                            "integer slice assignment",
                        )? {
                            return functional_result(output, target, &[rhs_handle]);
                        }
                    }
                    if rhs_len == 1 && plan.indices.len() != 1 {
                        let expansion = if managed {
                            ownership::expand_scalar_in_context(context, rhs_handle, plan)
                        } else {
                            // Ordinary compatibility hook has no scalar mapping
                            // port: keep its original required gather selector.
                            let rhs_indices =
                                integer_gpu_rhs_indices_for_plan(rhs_handle.shape(), plan)?;
                            provider
                                .gather_linear(rhs_handle, &rhs_indices, &plan.output_shape)
                                .map_err(|cause| {
                                    ownership::failure("scalar RHS expansion", cause.into())
                                })
                                .and_then(|output| {
                                    ownership::admit_read(output, rhs_handle, &plan.output_shape)
                                })
                                .map(|value| {
                                    let Value::GpuTensor(output) = value else {
                                        unreachable!("resident scalar expansion")
                                    };
                                    output
                                })
                        };
                        let expanded = match expansion {
                            Ok(values) => Some(values),
                            Err(cause) => {
                                let cause = ProviderExecutionError::from_source(cause);
                                if cause.is_unsupported() {
                                    None
                                } else {
                                    return Err(ownership::failure("scalar RHS expansion", cause));
                                }
                            }
                        };
                        if let Some(values) = expanded {
                            if let Some(output) = optional_execution(
                                ownership::functional_scatter_in_context(
                                    context,
                                    target,
                                    &plan.indices,
                                    &values,
                                    ProviderScatterValueMapping::Linear,
                                )
                                .await,
                                "expanded slice assignment",
                            )? {
                                return functional_result(output, target, &[&values, rhs_handle]);
                            }
                        }
                    }
                }
            } else if !integer && !managed {
                let rows = plan.base_shape.first().copied().unwrap_or(1);
                let cols = plan.base_shape.get(1).copied().unwrap_or(1);
                if let Some(col) = plan.properties.full_column {
                    if col < cols
                        && matches!(rhs_handle.shape(), [length] | [length, 1] if *length == rows)
                    {
                        if let Some(output) = optional_execution(
                            provider.scatter_column(target, col, rhs_handle).await,
                            "column assignment",
                        )? {
                            return functional_result(output, target, &[rhs_handle]);
                        }
                    }
                }
                if let Some(row) = plan.properties.full_row {
                    if row < rows
                        && matches!(rhs_handle.shape(), [length] | [1, length] if *length == cols)
                    {
                        if let Some(output) = optional_execution(
                            provider.scatter_row(target, row, rhs_handle).await,
                            "row assignment",
                        )? {
                            return functional_result(output, target, &[rhs_handle]);
                        }
                    }
                }
            }
        }
    }
    let rhs_complex = matches!(rhs, Value::Complex(..) | Value::ComplexTensor(_))
        || matches!(rhs, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved);
    let downloaded_rhs = match rhs {
        Value::GpuTensor(handle) => Some(
            owner
                .await_stage(ownership::gather_in_context(context, handle))
                .await??,
        ),
        _ => None,
    };
    let host_rhs = downloaded_rhs.as_ref().unwrap_or(rhs);
    let scalar_rhs = match host_rhs {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(value) => value.len() == 1,
        Value::LogicalArray(value) => value.data.len() == 1,
        _ => false,
    };
    if real && !integer && !rhs_complex {
        let view = super::host_rhs::HostRhs::for_plan(host_rhs, plan)?;
        if view.len() != plan.indices.len() {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
        for mapping in [
            ProviderScatterValueMapping::Scalar,
            ProviderScatterValueMapping::Linear,
        ] {
            if mapping == ProviderScatterValueMapping::Scalar
                && !(managed && scalar_rhs && plan.indices.len() != 1)
            {
                continue;
            }
            let upload_count = if mapping == ProviderScatterValueMapping::Scalar {
                1
            } else {
                view.len()
            };
            let shape = crate::indexing::plan::admission::shape(context, 2, |axis| {
                Ok(if axis == 0 { upload_count } else { 1 })
            })?;
            let dtype = if runmat_accelerate_api::handle_precision(target)
                == Some(ProviderPrecision::F32)
            {
                NumericDType::F32
            } else {
                NumericDType::F64
            };
            let tensor = crate::indexing::read_slice::payload::output::real(
                context,
                dtype,
                shape,
                |output| {
                    for ordinal in 0..upload_count {
                        if ordinal % 4096 == 0 {
                            owner.check()?;
                        }
                        output.set(ordinal, view.numeric_at(ordinal))?;
                    }
                    Ok(())
                },
            )?;
            let prepared = match ownership::restore_in_context(
                context,
                target,
                Value::Tensor(tensor),
                protected,
            ) {
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
                if let Some(output) = optional_execution(
                    ownership::functional_scatter_in_context(
                        context,
                        target,
                        &plan.indices,
                        &values,
                        mapping,
                    )
                    .await,
                    "slice assignment",
                )? {
                    return match protected.first() {
                        Some(rhs) => functional_result(output, target, &[&values, rhs]),
                        None => functional_result(output, target, &[&values]),
                    };
                }
            }
        }
    }
    let host = owner
        .await_stage(ownership::gather_in_context(context, target))
        .await??;
    let result = assign_host(context, host, plan, host_rhs).await?;
    ownership::restore_in_context(context, target, result, protected)
}

async fn assign_host(
    context: Option<&crate::context::RuntimeContext>,
    host: Value,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    match host {
        Value::Tensor(tensor) => {
            assign_tensor_with_plan_in_context(context, tensor, plan, rhs).await
        }
        Value::ComplexTensor(tensor) => {
            assign_complex_with_plan_in_context(context, tensor, plan, rhs).await
        }
        Value::LogicalArray(array) => {
            super::assign_logical_with_plan_in_context(context, array, plan, rhs).await
        }
        _ => Err(mex(
            "AccelerationOperationFailed",
            "Invalid resident slice assignment carrier",
        )),
    }
}

pub async fn delete_gpu_slice_with_plan(
    target: &GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    delete_gpu_slice_with_plan_in_context(context.as_ref(), target, plan, rhs).await
}

pub async fn delete_gpu_slice_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
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
    // Only the resident branch owns this admitted continuation. Host-only
    // indexing keeps its allocation-free unique mutation path and does not
    // embed the provider's entire fallback future in every caller's frame.
    let result = crate::execution::rooted_continuation::admit_rooted_future_with(
        context.cloned(),
        "resident slice delete",
        crate::builtin::builtin_control_error,
        move |context| async move {
            let value = delete_resident(context.as_ref(), target, plan, rhs).await?;
            crate::execution::RootedValue::in_optional_context(value, context.as_ref())
        },
    )
    .await?;
    Ok(result.with_owned(|value| value))
}

async fn delete_resident(
    context: Option<&crate::context::RuntimeContext>,
    target: &GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let host = owner
        .await_stage(ownership::gather_in_context(context, target))
        .await??;
    let result = match host {
        Value::Tensor(tensor) => delete_tensor_with_plan_in_context(context, tensor, plan, rhs)?,
        Value::ComplexTensor(tensor) => {
            delete_complex_with_plan_in_context(context, tensor, plan, rhs)?
        }
        Value::LogicalArray(array) => {
            super::delete_logical_with_plan(context, array, plan, rhs).await?
        }
        _ => {
            return Err(mex(
                "AccelerationOperationFailed",
                "Invalid resident slice deletion carrier",
            ))
        }
    };
    ownership::restore_in_context(context, target, result, &[])
}
