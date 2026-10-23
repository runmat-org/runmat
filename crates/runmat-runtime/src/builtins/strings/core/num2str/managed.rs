//! Borrowed numeric cells and admitted incremental UTF-16 output.

use super::managed_format;
use super::{
    num2str_allocation_error, num2str_error, FormatOptions, NUM2STR_ERROR_INTERNAL,
    NUM2STR_ERROR_INVALID_INPUT,
};
use crate::builtins::strings::core::numeric_format_work::{FormatBuffer, FormatCapacityExceeded};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_value::{
    CharArray, ComplexTensor, ExactArrayStorage, NumericScalar, SharedShape, Tensor, Utf16CodeUnit,
    Value,
};

// Each component is bounded by the parsed field width (or, without padding,
// by f64's <= 309 integer digits plus <= MAX_PRECISION fractional digits).
// A complex cell has at most two fields, a three-byte separator, `i`, and a
// possible imaginary sign when the real component is zero.
const CELL_TEXT_CAPACITY: usize = 2 * super::MAX_FORMAT_FIELD + 8;

enum BorrowedNumeric<'a> {
    Scalar(NumericScalar),
    ComplexScalar(NumericScalar, NumericScalar),
    Tensor(&'a Tensor),
    ComplexTensor(&'a ComplexTensor),
    Logical(&'a runmat_value::LogicalArray),
}

impl BorrowedNumeric<'_> {
    fn get(&self, index: usize) -> Option<(NumericScalar, Option<NumericScalar>)> {
        match self {
            Self::Scalar(value) => (index == 0).then_some((*value, None)),
            Self::ComplexScalar(real, imag) => (index == 0).then_some((*real, Some(*imag))),
            Self::Tensor(tensor) => tensor
                .host_buffer()
                .value_at(index)
                .map(|value| (value, None)),
            Self::ComplexTensor(tensor) => tensor
                .numeric_value_at(index)
                .map(|(real, imag)| (real, Some(imag))),
            Self::Logical(values) => values
                .data
                .as_slice()
                .get(index)
                .map(|&value| (NumericScalar::F64(if value != 0 { 1.0 } else { 0.0 }), None)),
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
        Value::Complex(real, imag) => Ok((
            BorrowedNumeric::ComplexScalar(NumericScalar::F64(*real), NumericScalar::F64(*imag)),
            1,
            1,
        )),
        Value::Tensor(tensor) if tensor.shape().len() <= 2 => Ok((
            BorrowedNumeric::Tensor(tensor),
            tensor.rows(),
            tensor.cols(),
        )),
        Value::ComplexTensor(tensor) if tensor.shape().len() <= 2 => Ok((
            BorrowedNumeric::ComplexTensor(tensor),
            tensor.rows(),
            tensor.cols(),
        )),
        Value::LogicalArray(logical) if logical.shape.len() <= 2 => {
            let rows = logical.shape.first().copied().unwrap_or(1);
            let cols = logical.shape.get(1).copied().unwrap_or(1);
            Ok((BorrowedNumeric::Logical(logical), rows, cols))
        }
        _ => Err(num2str_error(&NUM2STR_ERROR_INVALID_INPUT)),
    }
}

fn scalar_as_real(value: NumericScalar) -> Option<f64> {
    match value {
        NumericScalar::F64(value) => Some(value),
        NumericScalar::F32(value) => Some(f64::from(value)),
        _ => None,
    }
}

fn write_integer_raw(
    value: NumericScalar,
    text: &mut FormatBuffer<'_>,
) -> Result<(), FormatCapacityExceeded> {
    use std::fmt::Write as _;
    text.clear();
    match value {
        NumericScalar::I8(value) => write!(text, "{value}"),
        NumericScalar::I16(value) => write!(text, "{value}"),
        NumericScalar::I32(value) => write!(text, "{value}"),
        NumericScalar::I64(value) => write!(text, "{value}"),
        NumericScalar::U8(value) => write!(text, "{value}"),
        NumericScalar::U16(value) => write!(text, "{value}"),
        NumericScalar::U32(value) => write!(text, "{value}"),
        NumericScalar::U64(value) => write!(text, "{value}"),
        NumericScalar::F32(_) | NumericScalar::F64(_) => return Err(FormatCapacityExceeded),
    }
    .map_err(|_| FormatCapacityExceeded)
}

fn write_managed_cell(
    real: NumericScalar,
    imag: Option<NumericScalar>,
    options: &FormatOptions,
    output: &mut FormatBuffer<'_>,
    scratch: &mut FormatBuffer<'_>,
) -> Result<(), FormatCapacityExceeded> {
    if let Some(imag) = imag {
        if let (Some(real), Some(imag)) = (scalar_as_real(real), scalar_as_real(imag)) {
            return managed_format::complex(real, imag, options, output, scratch);
        }
        let mut raw_real_bytes = [0u8; 32];
        let mut raw_real = FormatBuffer::new(&mut raw_real_bytes);
        write_integer_raw(real, &mut raw_real)?;
        let real_is_zero = raw_real.as_str() == "0";
        managed_format::integer(raw_real.as_str(), options, output)?;
        let mut raw_imag_bytes = [0u8; 32];
        let mut raw_imag = FormatBuffer::new(&mut raw_imag_bytes);
        write_integer_raw(imag, &mut raw_imag)?;
        if raw_imag.as_str() == "0" {
            return Ok(());
        }
        let negative = raw_imag.as_str().starts_with('-');
        let magnitude = raw_imag
            .as_str()
            .strip_prefix('-')
            .unwrap_or(raw_imag.as_str());
        managed_format::integer(magnitude, options, scratch)?;
        if real_is_zero {
            output.clear();
            if negative {
                output.ascii(b'-')?;
            }
        } else {
            output.append(if negative { " - " } else { " + " })?;
        }
        output.append(scratch.as_str())?;
        return output.ascii(b'i');
    }
    if let Some(value) = scalar_as_real(real) {
        return managed_format::real(value, options, output);
    }
    write_integer_raw(real, scratch)?;
    managed_format::integer(scratch.as_str(), options, output)
}

fn formatted_width(text: &str, decimal: char) -> usize {
    text.encode_utf16().count()
        + if decimal.len_utf16() == 2 {
            text.bytes().filter(|&byte| byte == b'.').count()
        } else {
            0
        }
}

fn formatted_units(text: &str, decimal: char, mut unit: impl FnMut(u16)) {
    for character in text.chars() {
        let character = if character == '.' { decimal } else { character };
        let mut units = [0u16; 2];
        for encoded in character.encode_utf16(&mut units).iter().copied() {
            unit(encoded);
        }
    }
}

fn cpu_error(source: crate::execution::ExecutionServiceError) -> RuntimeError {
    let mut error = source.clone().into_runtime_error();
    error.source = Some(Box::new(source));
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

pub(super) async fn format_managed(
    value: Value,
    options: &FormatOptions,
    runtime: &RuntimeContext,
) -> BuiltinResult<CharArray> {
    let (source, rows, cols) = borrowed_numeric(&value)?;
    if rows == 0 {
        return crate::builtins::strings::core::numeric_format_work::admitted_empty(runtime, 0)
            .map_err(|cause| num2str_allocation_error(runtime, cause));
    }
    if cols == 0 {
        return crate::builtins::strings::core::numeric_format_work::admitted_empty(runtime, rows)
            .map_err(|cause| num2str_allocation_error(runtime, cause));
    }
    let width_bytes = HostTypedWorkspace::<usize>::quoted_bytes(cols)
        .map_err(|cause| num2str_allocation_error(runtime, cause))?;
    let text_bytes = HostTypedWorkspace::<u8>::quoted_bytes(CELL_TEXT_CAPACITY * 2)
        .map_err(|cause| num2str_allocation_error(runtime, cause))?;
    let scratch = width_bytes
        .checked_add(text_bytes)
        .ok_or_else(|| num2str_allocation_error(runtime, HostAllocationError::Overflow))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(scratch, 0)
            .map_err(|cause| num2str_allocation_error(runtime, cause))?,
    )
    .map_err(|cause| num2str_allocation_error(runtime, cause))?
    .ok_or_else(|| num2str_allocation_error(runtime, HostAllocationError::MissingManagedOwner))?;
    let mut widths = receipt
        .temporary_typed::<usize>(runtime, cols)
        .map_err(|cause| num2str_allocation_error(runtime, cause))?;
    let mut text_bytes = receipt
        .temporary_typed::<u8>(runtime, CELL_TEXT_CAPACITY * 2)
        .map_err(|cause| num2str_allocation_error(runtime, cause))?;
    let mut budget = runtime.cpu_cooperation();
    for col in 0..cols {
        let mut width = 0usize;
        for row in 0..rows {
            let (real, imag) = source
                .get(row + col * rows)
                .ok_or_else(|| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
            let (first, second) = text_bytes.as_mut_slice().split_at_mut(CELL_TEXT_CAPACITY);
            let mut output = FormatBuffer::new(first);
            let mut secondary = FormatBuffer::new(second);
            write_managed_cell(real, imag, options, &mut output, &mut secondary)
                .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
            width = width.max(formatted_width(output.as_str(), options.decimal));
            budget.checkpoint(1).await.map_err(cpu_error)?;
            receipt
                .checkpoint(runtime)
                .map_err(|cause| num2str_allocation_error(runtime, cause))?;
        }
        widths.as_mut_slice()[col] = width;
    }
    let row_width = widths
        .as_slice()
        .iter()
        .enumerate()
        .try_fold(0usize, |width, (col, &cell)| {
            width.checked_add(cell)?.checked_add(usize::from(col > 0))
        })
        .ok_or_else(|| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
    let output_len = rows
        .checked_mul(row_width)
        .ok_or_else(|| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
    let payload = ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(output_len)
        .map_err(|cause| num2str_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    let shape = SharedShape::requested_bytes_for_rank(2)
        .map_err(|cause| num2str_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    let physical = payload
        .checked_add(shape)
        .ok_or_else(|| num2str_allocation_error(runtime, HostAllocationError::Overflow))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical)
            .map_err(|_| num2str_allocation_error(runtime, HostAllocationError::Overflow))?,
        usize::from(payload != 0) + usize::from(shape != 0),
    )
    .map_err(|cause| num2str_allocation_error(runtime, cause))?;
    let output_receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained)
            .map_err(|cause| num2str_allocation_error(runtime, cause))?,
    )
    .map_err(|cause| num2str_allocation_error(runtime, cause))?
    .ok_or_else(|| num2str_allocation_error(runtime, HostAllocationError::MissingManagedOwner))?;
    let permit = output_receipt
        .output_permit(runtime)
        .map_err(|cause| num2str_allocation_error(runtime, cause))?;
    let shape = SharedShape::new_admitted_from_slice(&[rows, row_width], &permit)
        .map_err(|cause| num2str_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    let mut builder = ExactArrayStorage::<Utf16CodeUnit>::try_begin(output_len, &permit)
        .map_err(|cause| num2str_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    for row in 0..rows {
        for (col, &cell_width) in widths.as_slice().iter().enumerate() {
            if col > 0 {
                builder
                    .try_push(Utf16CodeUnit::from_raw(u16::from(b' ')))
                    .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
            }
            let (real, imag) = source
                .get(row + col * rows)
                .ok_or_else(|| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
            let (first, second) = text_bytes.as_mut_slice().split_at_mut(CELL_TEXT_CAPACITY);
            let mut output = FormatBuffer::new(first);
            let mut secondary = FormatBuffer::new(second);
            write_managed_cell(real, imag, options, &mut output, &mut secondary)
                .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
            let text = output.as_str();
            let width = formatted_width(text, options.decimal);
            for _ in 0..cell_width.saturating_sub(width) {
                builder
                    .try_push(Utf16CodeUnit::from_raw(u16::from(b' ')))
                    .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
            }
            let mut overflow = false;
            formatted_units(text, options.decimal, |unit| {
                overflow |= builder.try_push(Utf16CodeUnit::from_raw(unit)).is_err();
            });
            if overflow {
                return Err(num2str_error(&NUM2STR_ERROR_INTERNAL));
            }
            budget.checkpoint(1).await.map_err(cpu_error)?;
            receipt
                .checkpoint(runtime)
                .map_err(|cause| num2str_allocation_error(runtime, cause))?;
            output_receipt
                .checkpoint(runtime)
                .map_err(|cause| num2str_allocation_error(runtime, cause))?;
        }
    }
    let data = builder
        .try_finish()
        .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
    let output = CharArray::from_admitted(data, shape)
        .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
    output_receipt
        .checkpoint(runtime)
        .map_err(|cause| num2str_allocation_error(runtime, cause))?;
    Ok(output)
}
