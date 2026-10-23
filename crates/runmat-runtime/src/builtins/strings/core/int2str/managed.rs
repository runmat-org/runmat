//! Admitted two-pass integer formatting with borrowed native-class input.

use super::{
    int2str_allocation_error, int2str_error, matlab_round, INT2STR_ERROR_INTERNAL,
    INT2STR_ERROR_INVALID_INPUT,
};
use crate::builtins::strings::core::numeric_format_work::{FormatBuffer, FormatCapacityExceeded};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, GpuGatherRetry};
use runmat_value::{
    CharArray, ExactArrayStorage, NumericScalar, SharedShape, Tensor, Utf16CodeUnit, Value,
};

pub(super) const CELL_TEXT_CAPACITY: usize = 512;

enum BorrowedNumeric<'a> {
    Scalar(NumericScalar),
    Tensor(&'a Tensor),
    Logical(&'a runmat_value::LogicalArray),
}

impl BorrowedNumeric<'_> {
    fn get(&self, index: usize) -> Option<NumericScalar> {
        match self {
            Self::Scalar(value) => (index == 0).then_some(*value),
            Self::Tensor(tensor) => tensor.host_buffer().value_at(index),
            Self::Logical(values) => values
                .data
                .as_slice()
                .get(index)
                .map(|&value| NumericScalar::F64(if value != 0 { 1.0 } else { 0.0 })),
        }
    }
}

fn borrowed_numeric(value: &Value) -> BuiltinResult<(BorrowedNumeric<'_>, usize, usize)> {
    match value {
        Value::Num(value) => Ok((BorrowedNumeric::Scalar(NumericScalar::F64(*value)), 1, 1)),
        Value::Int(value) => Ok((BorrowedNumeric::Scalar(value.clone().into()), 1, 1)),
        Value::Bool(value) => Ok((
            BorrowedNumeric::Scalar(NumericScalar::F64(if *value { 1.0 } else { 0.0 })),
            1,
            1,
        )),
        Value::Tensor(tensor) if tensor.shape().len() <= 2 => Ok((
            BorrowedNumeric::Tensor(tensor),
            tensor.rows(),
            tensor.cols(),
        )),
        Value::LogicalArray(logical) if logical.shape.len() <= 2 => {
            let rows = logical.shape.first().copied().unwrap_or(1);
            let cols = logical.shape.get(1).copied().unwrap_or(1);
            Ok((BorrowedNumeric::Logical(logical), rows, cols))
        }
        _ => Err(int2str_error(&INT2STR_ERROR_INVALID_INPUT)),
    }
}

pub(super) fn write_managed_cell(
    value: NumericScalar,
    text: &mut FormatBuffer<'_>,
) -> Result<(), FormatCapacityExceeded> {
    use std::fmt::Write as _;
    text.clear();
    match value {
        NumericScalar::F64(value) => write_managed_real(value, text),
        NumericScalar::F32(value) => write_managed_real(f64::from(value), text),
        NumericScalar::I8(value) => write!(text, "{value}").map_err(|_| FormatCapacityExceeded),
        NumericScalar::I16(value) => write!(text, "{value}").map_err(|_| FormatCapacityExceeded),
        NumericScalar::I32(value) => write!(text, "{value}").map_err(|_| FormatCapacityExceeded),
        NumericScalar::I64(value) => write!(text, "{value}").map_err(|_| FormatCapacityExceeded),
        NumericScalar::U8(value) => write!(text, "{value}").map_err(|_| FormatCapacityExceeded),
        NumericScalar::U16(value) => write!(text, "{value}").map_err(|_| FormatCapacityExceeded),
        NumericScalar::U32(value) => write!(text, "{value}").map_err(|_| FormatCapacityExceeded),
        NumericScalar::U64(value) => write!(text, "{value}").map_err(|_| FormatCapacityExceeded),
    }
}

fn write_managed_real(
    value: f64,
    text: &mut FormatBuffer<'_>,
) -> Result<(), FormatCapacityExceeded> {
    use std::fmt::Write as _;
    let value = matlab_round(value);
    if value.is_nan() {
        return text.append("NaN");
    }
    if value.is_infinite() {
        return text.append(if value.is_sign_negative() {
            "-Inf"
        } else {
            "Inf"
        });
    }
    if value == 0.0 {
        return text.append("0");
    }
    if value >= i64::MIN as f64 && value <= i64::MAX as f64 {
        write!(text, "{}", value as i64).map_err(|_| FormatCapacityExceeded)
    } else {
        write!(text, "{value:.0}").map_err(|_| FormatCapacityExceeded)
    }
}

pub(super) async fn format_managed(
    value: Value,
    runtime: &RuntimeContext,
) -> BuiltinResult<CharArray> {
    let (source, rows, cols) = borrowed_numeric(&value)?;
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| int2str_error(&INT2STR_ERROR_INTERNAL))?;
    if count == 0 {
        return crate::builtins::strings::core::numeric_format_work::admitted_empty(runtime, 0)
            .map_err(|cause| int2str_allocation_error(runtime, cause));
    }
    let width_bytes = HostTypedWorkspace::<usize>::quoted_bytes(cols)
        .map_err(|cause| int2str_allocation_error(runtime, cause))?;
    let text_bytes = HostTypedWorkspace::<u8>::quoted_bytes(CELL_TEXT_CAPACITY)
        .map_err(|cause| int2str_allocation_error(runtime, cause))?;
    let scratch = width_bytes
        .checked_add(text_bytes)
        .ok_or_else(|| int2str_allocation_error(runtime, HostAllocationError::Overflow))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(scratch, 0)
            .map_err(|cause| int2str_allocation_error(runtime, cause))?,
    )
    .map_err(|cause| int2str_allocation_error(runtime, cause))?
    .ok_or_else(|| int2str_allocation_error(runtime, HostAllocationError::MissingManagedOwner))?;
    let mut widths = receipt
        .temporary_typed::<usize>(runtime, cols)
        .map_err(|cause| int2str_allocation_error(runtime, cause))?;
    let mut text_bytes = receipt
        .temporary_typed::<u8>(runtime, CELL_TEXT_CAPACITY)
        .map_err(|cause| int2str_allocation_error(runtime, cause))?;
    let mut budget = runtime.cpu_cooperation();
    for col in 0..cols {
        let mut width = 0usize;
        for row in 0..rows {
            let index = row + col * rows;
            let value = source
                .get(index)
                .ok_or_else(|| int2str_error(&INT2STR_ERROR_INTERNAL))?;
            let mut text = FormatBuffer::new(text_bytes.as_mut_slice());
            write_managed_cell(value, &mut text)
                .map_err(|_| int2str_error(&INT2STR_ERROR_INTERNAL))?;
            width = width.max(text.as_str().encode_utf16().count());
            budget.checkpoint(1).await.map_err(|source| {
                let mut error = source.clone().into_runtime_error();
                error.source = Some(Box::new(source));
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            })?;
            receipt
                .checkpoint(runtime)
                .map_err(|cause| int2str_allocation_error(runtime, cause))?;
        }
        widths.as_mut_slice()[col] = width;
    }
    let columns = widths.as_slice();
    let row_width = columns
        .iter()
        .enumerate()
        .try_fold(0usize, |width, (col, &cell)| {
            width.checked_add(cell)?.checked_add(usize::from(col > 0))
        })
        .ok_or_else(|| int2str_error(&INT2STR_ERROR_INTERNAL))?;
    let output_len = rows
        .checked_mul(row_width)
        .ok_or_else(|| int2str_error(&INT2STR_ERROR_INTERNAL))?;
    let payload = ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(output_len)
        .map_err(|cause| int2str_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    let shape = SharedShape::requested_bytes_for_rank(2)
        .map_err(|cause| int2str_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    let physical = payload
        .checked_add(shape)
        .ok_or_else(|| int2str_allocation_error(runtime, HostAllocationError::Overflow))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical)
            .map_err(|_| int2str_allocation_error(runtime, HostAllocationError::Overflow))?,
        usize::from(payload != 0) + usize::from(shape != 0),
    )
    .map_err(|cause| int2str_allocation_error(runtime, cause))?;
    let output_receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained)
            .map_err(|cause| int2str_allocation_error(runtime, cause))?,
    )
    .map_err(|cause| int2str_allocation_error(runtime, cause))?
    .ok_or_else(|| int2str_allocation_error(runtime, HostAllocationError::MissingManagedOwner))?;
    let permit = output_receipt
        .output_permit(runtime)
        .map_err(|cause| int2str_allocation_error(runtime, cause))?;
    let shape = SharedShape::new_admitted_from_slice(&[rows, row_width], &permit)
        .map_err(|cause| int2str_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    let mut builder = ExactArrayStorage::<Utf16CodeUnit>::try_begin(output_len, &permit)
        .map_err(|cause| int2str_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    for row in 0..rows {
        for (col, &cell_width) in columns.iter().enumerate() {
            if col > 0 {
                builder
                    .try_push(Utf16CodeUnit::from_raw(u16::from(b' ')))
                    .map_err(|_| int2str_error(&INT2STR_ERROR_INTERNAL))?;
            }
            let value = source
                .get(row + col * rows)
                .ok_or_else(|| int2str_error(&INT2STR_ERROR_INTERNAL))?;
            let mut text = FormatBuffer::new(text_bytes.as_mut_slice());
            write_managed_cell(value, &mut text)
                .map_err(|_| int2str_error(&INT2STR_ERROR_INTERNAL))?;
            let width = text.as_str().encode_utf16().count();
            for _ in 0..cell_width.saturating_sub(width) {
                builder
                    .try_push(Utf16CodeUnit::from_raw(u16::from(b' ')))
                    .map_err(|_| int2str_error(&INT2STR_ERROR_INTERNAL))?;
            }
            for unit in text.as_str().encode_utf16() {
                builder
                    .try_push(Utf16CodeUnit::from_raw(unit))
                    .map_err(|_| int2str_error(&INT2STR_ERROR_INTERNAL))?;
            }
            budget.checkpoint(1).await.map_err(|source| {
                let mut error = source.clone().into_runtime_error();
                error.source = Some(Box::new(source));
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            })?;
            receipt
                .checkpoint(runtime)
                .map_err(|cause| int2str_allocation_error(runtime, cause))?;
            output_receipt
                .checkpoint(runtime)
                .map_err(|cause| int2str_allocation_error(runtime, cause))?;
        }
    }
    let data = builder
        .try_finish()
        .map_err(|_| int2str_error(&INT2STR_ERROR_INTERNAL))?;
    let output = CharArray::from_admitted(data, shape)
        .map_err(|_| int2str_error(&INT2STR_ERROR_INTERNAL))?;
    output_receipt
        .checkpoint(runtime)
        .map_err(|cause| int2str_allocation_error(runtime, cause))?;
    Ok(output)
}
