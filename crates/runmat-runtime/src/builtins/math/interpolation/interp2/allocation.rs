//! Spline admission before any resident argument or numeric option download.
//! The method projection uses the same typed parser and last-wins order as
//! ParsedInterp2; numeric extrapolation is deliberately not materialized here.

use super::*;
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};

pub(super) fn admit_before_gather(
    args: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&RuntimeContext>,
) -> crate::BuiltinResult<Option<HostAllocationReceipt>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(parallel) = runtime.service_ports().parallel() else {
        return Ok(None);
    };
    if parallel
        .placement_resources()
        .memory_available_bytes
        .is_none()
    {
        return Ok(None);
    }
    if args.len() < 3 {
        return Ok(None);
    }
    let explicit = args.len() >= 5 && !is_option_arg(&args[3])?;
    let option_start = if explicit { 5 } else { 3 };
    let mut method = InterpMethod::Linear;
    for option in args.iter().skip(option_start) {
        if crate::builtins::common::random_args::keyword_of(option)?.as_deref() == Some("extrap") {
            continue;
        }
        if let Some(parsed) = parse_method(option, NAME)? {
            method = parsed;
        }
    }
    if method != InterpMethod::Spline {
        return Ok(None);
    }
    let z_index = if explicit { 2 } else { 0 };
    let (rows, columns) = raw_grid_dimensions(&args[z_index]);
    let (dtype, lanes) = raw_grid_class(&args[z_index])?;
    let xq_index = if explicit { 3 } else { 1 };
    let yq_index = xq_index + 1;
    let x_count = raw_count(&args[xq_index])
        .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?;
    let y_count = raw_count(&args[yq_index])
        .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?;
    let x_shape = raw_shape(&args[xq_index]);
    let y_shape = raw_shape(&args[yq_index]);
    let output_count = if x_count == 1 {
        y_count
    } else if y_count == 1 {
        x_count
    } else if x_shape.len() == 2 && y_shape.len() == 2 && x_shape[0] == 1 && y_shape[1] == 1 {
        x_count
            .checked_mul(y_count)
            .ok_or(HostAllocationError::Overflow)
            .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?
    } else {
        x_count.max(y_count)
    };
    let output_rank = x_shape.len().max(y_shape.len()).max(2);
    let scratch = argument_envelope(args, rows, columns, output_count, explicit)
        .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?;
    let plan = super::super::pp::tensor_spline_allocation_plan(
        rows,
        columns,
        lanes,
        output_count,
        output_rank,
        dtype,
        scratch,
    )
    .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?;
    receipt.map(Some).ok_or_else(|| {
        super::super::pp::host_allocation_error(NAME, HostAllocationError::StaleAuthority)
    })
}

fn raw_grid_dimensions(value: &Value) -> (usize, usize) {
    match raw_shape(value) {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

fn raw_shape(value: &Value) -> &[usize] {
    match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::ComplexTensor(tensor) => tensor.shape(),
        Value::GpuTensor(handle) => handle.shape(),
        _ => &[],
    }
}

fn raw_count(value: &Value) -> Result<usize, HostAllocationError> {
    raw_shape(value)
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or(HostAllocationError::Overflow)
}

fn raw_grid_class(value: &Value) -> crate::BuiltinResult<(NumericDType, usize)> {
    match value {
        Value::ComplexTensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) =>
        {
            Ok((tensor.numeric_dtype(), 2))
        }
        Value::Tensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) =>
        {
            Ok((tensor.numeric_dtype(), 1))
        }
        Value::ComplexTensor(_) | Value::Tensor(_) => Err(interp2_invalid_input(
            "spline sample values must be single or double",
        )),
        Value::GpuTensor(handle) => {
            let dtype = match handle.physical().element_type() {
                NumericElementType::F32 => NumericDType::F32,
                NumericElementType::F64 => NumericDType::F64,
                _ => {
                    return Err(interp2_invalid_input(
                        "spline sample values must be single or double",
                    ))
                }
            };
            let lanes = if handle.physical().storage() == GpuTensorStorage::ComplexInterleaved {
                2
            } else {
                1
            };
            Ok((dtype, lanes))
        }
        _ => Ok((NumericDType::F64, 1)),
    }
}

fn argument_envelope(
    args: crate::call::FunctionArgumentView<'_>,
    rows: usize,
    columns: usize,
    output_count: usize,
    explicit: bool,
) -> Result<u64, HostAllocationError> {
    let mut bytes = 0u64;
    for value in args {
        if let Value::GpuTensor(handle) = value {
            // A download produces one host buffer from this immutable view.
            // Provider staging remains the provider's own resource domain.
            let count = raw_count(value)? as u64;
            let lanes = if handle.physical().storage() == GpuTensorStorage::ComplexInterleaved {
                2
            } else {
                1
            };
            let download = count
                .checked_mul(handle.physical().element_type().element_size() as u64)
                .and_then(|n| n.checked_mul(lanes))
                .ok_or(HostAllocationError::Overflow)?;
            bytes = bytes
                .checked_add(download)
                .ok_or(HostAllocationError::Overflow)?;
        }
    }
    // X/Y extracted vectors, both query numeric projections, and the two
    // aligned query vectors coexist with the downloaded Z through evaluation.
    let (xq, yq) = if explicit { (3, 4) } else { (1, 2) };
    let axis = rows
        .checked_add(columns)
        .ok_or(HostAllocationError::Overflow)?;
    let query = raw_count(&args[xq])?
        .checked_add(raw_count(&args[yq])?)
        .and_then(|n| n.checked_add(output_count.checked_mul(2)?))
        .ok_or(HostAllocationError::Overflow)?;
    let projected = axis
        .checked_add(query)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .ok_or(HostAllocationError::Overflow)?;
    bytes
        .checked_add(u64::try_from(projected).map_err(|_| HostAllocationError::Overflow)?)
        .ok_or(HostAllocationError::Overflow)
}
