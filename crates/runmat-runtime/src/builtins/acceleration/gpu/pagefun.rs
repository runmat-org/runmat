use runmat_builtins::*;
// MATLAB-compatible `pagefun` builtin.
//
// The `pagefun` builtin applies a MATLAB operator to every 2-D page across the
// trailing dimensions of the supplied inputs. This mirrors MathWorks MATLAB
// semantics for GPU arrays while retaining host fallbacks when GPU providers
// do not expose specialised kernels.

use crate::builtins::common::{gpu_helpers, tensor as tensor_utils};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, PagefunOp, PagefunRequest};
#[cfg(test)]
use runmat_accelerate_api::{HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView};
use runmat_macros::runtime_builtin;
use runmat_value::NumericStorage;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

type ComplexMatrixData = (Vec<(f64, f64)>, usize, usize, NumericDType);
const BUILTIN_NAME: &str = "pagefun";
mod shape;
use shape::*;

fn pagefun_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pagefun_error_with_message(error.message, error)
}

fn pagefun_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn pagefun_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    pagefun_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn pagefun_internal_error(message: impl Into<String>) -> RuntimeError {
    pagefun_error_with_detail(&PAGEFUN_ERROR_INTERNAL, message.into())
}

/// Operational failures remain outside the shared product's domain-error
/// channel. Add this public identity without erasing typed retry or causes.
fn pagefun_operational_error(error: RuntimeError) -> RuntimeError {
    if error.failure_origin()
        == crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation
    {
        return error;
    }
    let mut builder = build_runtime_error(error.message())
        .with_builtin(BUILTIN_NAME)
        .with_failure_origin(error.failure_origin())
        .with_gpu_gather_retry(error.gpu_gather_retry);
    if let Some(identifier) = error.identifier().or(PAGEFUN_ERROR_INTERNAL.identifier) {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(span) = error.span {
        builder = builder.with_span(span);
    }
    if let Some(task) = error.context.task_id.clone() {
        builder = builder.with_task_id(task);
    }
    if let Some(phase) = error.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    if !error.context.call_stack.is_empty() {
        builder = builder.with_call_stack(error.context.call_stack.clone());
    }
    if !error.context.call_frames.is_empty() {
        builder = builder.with_call_frames(error.context.call_frames.clone());
    }
    builder = builder.with_call_frames_elided(error.context.call_frames_elided);
    builder.with_source(error).build()
}

#[runtime_builtin(
    name = "pagefun",
    binding_variant = "default",
    builtin_path = "crate::builtins::acceleration::gpu::pagefun"
)]
async fn pagefun_builtin(
    func: Value,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &PAGEFUN_DESCRIPTOR,
        &PAGEFUN_ERROR_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if matches!(
        &func,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PAGEFUN_TEXT_CALLABLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !matches!(&first, Value::GpuTensor(_))
        && !rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PAGEFUN_HOST_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let result = pagefun_impl(func, first, rest).await?;
    if crate::output_count::current_output_count() == Some(0) {
        // Discard our owning result, not the allocation behind a borrowed
        // handle. Any independently published aliases keep their own leases.
        drop(result);
        Ok(Value::output_list(Vec::new()))
    } else {
        Ok(result)
    }
}

async fn pagefun_impl(
    func: Value,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let operation = PageOperation::from_callable(func)?;
    let mut operands = Vec::with_capacity(rest.len() + 1);
    operands.push(first);
    operands.extend(rest);
    if operands.is_empty() {
        return Err(pagefun_error(&PAGEFUN_ERROR_INVALID_ARITY));
    }

    operation.validate_arity(operands.len())?;
    validate_operand_metadata(&operands)?;

    let resident_source = gpu_helpers::select_resident_output_source(
        operands.iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        BUILTIN_NAME,
    )?;

    if let Some(value) = try_pagefun_gpu(&operation, &operands)? {
        return Ok(value);
    }

    let protected: Vec<_> = operands
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect();
    let mut host_values = Vec::with_capacity(operands.len());
    for value in &operands {
        host_values.push(gather_if_needed_async(value).await?);
    }

    let mut page_inputs = Vec::with_capacity(host_values.len());
    for value in host_values {
        page_inputs.push(PageInput::from_value(value)?);
    }

    let rank = page_inputs
        .iter()
        .map(|input| input.page_dims.len())
        .max()
        .unwrap_or(0);

    let result_page_dims =
        broadcast_page_dims(&page_inputs[0].page_dims, &page_inputs[1].page_dims)?;

    let page_volume = checked_volume(&result_page_dims)?;

    let mut prepared_inputs = Vec::with_capacity(page_inputs.len());
    for input in page_inputs {
        prepared_inputs.push(PreparedInput::new(input, rank));
    }

    operation.validate_shapes(&prepared_inputs)?;
    let output_kind = operation.output_kind(&prepared_inputs);
    let (mut result_rows, mut result_cols) =
        operation.output_matrix_shape(&prepared_inputs, output_kind)?;

    if page_volume == 0 {
        return finalise_empty_output(
            &operation,
            &prepared_inputs,
            &result_page_dims,
            output_kind,
            resident_source.as_ref(),
            &protected,
        );
    }

    let mut real_data: Option<NumericStorage> = None;
    let mut complex_data: Option<Vec<(f64, f64)>> = None;
    let mut complex_dtype = None;
    let mut multi_index = vec![0usize; rank];

    let mut page_counter = 0usize;
    loop {
        let mut page_args = Vec::with_capacity(prepared_inputs.len());
        for input in &prepared_inputs {
            page_args.push(input.page_value(&multi_index)?);
        }

        let mut evaluated = operation.evaluate(&page_args).await?;
        evaluated = gather_if_needed_async(&evaluated).await?;
        match output_kind {
            OutputKind::Real => {
                let (data, rows, cols) = tensor_matrix_storage(evaluated)?;
                if real_data.is_none() {
                    result_rows = rows;
                    result_cols = cols;
                    real_data = Some(empty_storage_like(
                        &data,
                        checked_volume(&[rows, cols, page_volume])?,
                    ));
                } else if rows != result_rows || cols != result_cols {
                    return Err(pagefun_error_with_detail(
                        &PAGEFUN_ERROR_RESULT_SHAPE,
                        "result matrices must be the same size",
                    ));
                }
                if let Some(storage) = real_data.as_mut() {
                    extend_same_class(storage, data)?;
                }
            }
            OutputKind::Complex => {
                let (data, rows, cols, dtype) = complex_matrix_data(evaluated)?;
                if complex_dtype.is_some_and(|prior| prior != dtype) {
                    return Err(pagefun_error(&PAGEFUN_ERROR_RESULT_TYPE));
                }
                complex_dtype = Some(dtype);
                if complex_data.is_none() {
                    result_rows = rows;
                    result_cols = cols;
                    complex_data = Some(Vec::with_capacity(checked_volume(&[
                        rows,
                        cols,
                        page_volume,
                    ])?));
                } else if rows != result_rows || cols != result_cols {
                    return Err(pagefun_error_with_detail(
                        &PAGEFUN_ERROR_RESULT_SHAPE,
                        "result matrices must be the same size",
                    ));
                }
                if let Some(vec) = complex_data.as_mut() {
                    vec.extend(data);
                }
            }
        }

        page_counter += 1;
        if page_counter == page_volume {
            break;
        }
        increment_multi_index(&mut multi_index, &result_page_dims)?;
    }

    let final_shape = assemble_shape(result_rows, result_cols, &result_page_dims);
    let output = match output_kind {
        OutputKind::Real => {
            let data = real_data.unwrap_or_else(|| NumericStorage::F64(Vec::new()));
            let tensor = Tensor::from_numeric_storage(data, final_shape).map_err(|e| {
                pagefun_error_with_detail(
                    &PAGEFUN_ERROR_INTERNAL,
                    format!("failed to construct result tensor ({e})"),
                )
            })?;
            FinalOutput::Real(tensor)
        }
        OutputKind::Complex => {
            let data = complex_data.unwrap_or_default();
            let tensor = ComplexTensor::from_f64_values_with_dtype(
                data,
                final_shape,
                complex_dtype.unwrap_or(NumericDType::F64),
            )
            .map_err(|e| {
                pagefun_error_with_detail(
                    &PAGEFUN_ERROR_INTERNAL,
                    format!("failed to construct complex result tensor ({e})"),
                )
            })?;
            FinalOutput::Complex(tensor)
        }
    };

    output.into_value(resident_source.as_ref(), &protected)
}

mod provider;
mod output;
mod input;
mod storage;
mod operation;
mod type_names;
use input::*;
use operation::*;
use output::*;
use provider::*;
use storage::*;
use type_names::TypeName;
#[cfg(test)]
pub(crate) mod tests;
