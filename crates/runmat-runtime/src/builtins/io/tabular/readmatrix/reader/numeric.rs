use super::super::output::integer_element_dtype;
use super::*;
use std::borrow::Cow;

mod diagnostic;
use diagnostic::NumericTokenDiagnostic;

pub(in crate::builtins::io::tabular::readmatrix::reader) fn maximum_numeric_diagnostic_bytes(
    opened_extent: usize,
) -> Option<usize> {
    NumericTokenDiagnostic::maximum_total_bytes(opened_extent)
}

fn invalid_numeric_token(
    token: &str,
    line: usize,
    column: usize,
    reservation: Option<&dyn DiagnosticReservation>,
) -> RuntimeError {
    let diagnostic = NumericTokenDiagnostic::new(token, line, column);
    if let Some(reservation) = reservation {
        return diagnostic
            .try_render_charged(reservation)
            .unwrap_or_else(|source| admission::allocation_error(BUILTIN_NAME, source));
    }
    match diagnostic.try_render() {
        Ok(message) => readmatrix_error_with(&READMATRIX_ERROR_NON_NUMERIC_TOKEN, message),
        Err(source) => readmatrix_error_with_source(
            &READMATRIX_ERROR_TENSOR_BUILD,
            "readmatrix: unable to allocate numeric token diagnostic",
            source,
        ),
    }
}

pub(super) fn parse_numeric_token(
    token: &str,
    options: &ReadMatrixOptions,
    line_number: usize,
    column: MatrixFieldColumn,
) -> BuiltinResult<f64> {
    parse_numeric_token_with_reservation(token, options, line_number, column, None)
}

pub(in crate::builtins::io::tabular::readmatrix::reader) fn parse_numeric_token_with_reservation(
    token: &str,
    options: &ReadMatrixOptions,
    line_number: usize,
    column: MatrixFieldColumn,
    reservation: Option<&dyn DiagnosticReservation>,
) -> BuiltinResult<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Ok(options.empty_value_for(column)?.materialize_f64());
    }
    let unquoted = unquote(trimmed);
    let inner = unquoted.trim();
    if inner.is_empty() {
        return Ok(options.empty_value_for(column)?.materialize_f64());
    }
    if options.is_missing_token_for(inner, column) {
        return Ok(options.missing_value_for(column)?.materialize_f64());
    }
    let normalized = normalize_numeric_token(inner, options, column);
    if normalized.is_empty() {
        return Ok(options.empty_value_for(column)?.materialize_f64());
    }
    if normalized.eq_ignore_ascii_case("nan") {
        return Ok(f64::NAN);
    }
    if ["inf", "+inf", "infinity", "+infinity"]
        .iter()
        .any(|value| normalized.eq_ignore_ascii_case(value))
    {
        return Ok(f64::INFINITY);
    }
    if ["-inf", "-infinity"]
        .iter()
        .any(|value| normalized.eq_ignore_ascii_case(value))
    {
        return Ok(f64::NEG_INFINITY);
    }
    normalized.parse::<f64>().map_err(|_| {
        invalid_numeric_token(inner, line_number, column.diagnostic_number(), reservation)
    })
}

pub(super) fn validate_numeric_token(
    token: &str,
    options: &ReadMatrixOptions,
    line_number: usize,
    column: MatrixFieldColumn,
) -> BuiltinResult<()> {
    parse_numeric_token(token, options, line_number, column).map(|_| ())
}

pub(in crate::builtins::io::tabular::readmatrix::reader) fn validate_numeric_token_with_reservation(
    token: &str,
    options: &ReadMatrixOptions,
    line_number: usize,
    column: MatrixFieldColumn,
    reservation: Option<&dyn DiagnosticReservation>,
) -> BuiltinResult<()> {
    parse_numeric_token_with_reservation(token, options, line_number, column, reservation)
        .map(|_| ())
}

pub(super) fn integer_output_dtype(template: &OutputTemplate) -> Option<NumericDType> {
    let dtype = match template {
        OutputTemplate::Numeric(dtype) => *dtype,
        OutputTemplate::Like(Value::Tensor(tensor)) => tensor.numeric_dtype(),
        OutputTemplate::Like(Value::Int(value)) => {
            IntegerStorage::from_scalar(value.clone()).numeric_dtype()
        }
        OutputTemplate::Like(Value::GpuTensor(handle)) => {
            integer_element_dtype(runmat_accelerate_api::handle_integer_type(handle)?)
        }
        OutputTemplate::Logical | OutputTemplate::Like(_) => return None,
    };
    matches!(
        dtype,
        NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64
    )
    .then_some(dtype)
}

pub(in crate::builtins::io::tabular::readmatrix) fn parse_integer_numeric_token(
    token: &str,
    options: &ReadMatrixOptions,
    dtype: NumericDType,
    line_number: usize,
    column: MatrixFieldColumn,
) -> BuiltinResult<NumericScalar> {
    parse_integer_numeric_token_with_reservation(token, options, dtype, line_number, column, None)
}

pub(in crate::builtins::io::tabular::readmatrix::reader) fn parse_integer_numeric_token_with_reservation(
    token: &str,
    options: &ReadMatrixOptions,
    dtype: NumericDType,
    line_number: usize,
    column: MatrixFieldColumn,
    reservation: Option<&dyn DiagnosticReservation>,
) -> BuiltinResult<NumericScalar> {
    let trimmed = token.trim();
    let unquoted = unquote(trimmed);
    let inner = unquoted.trim();
    if inner.is_empty() {
        return Ok(integer_scalar_from_numeric(
            dtype,
            options.empty_value_for(column)?,
        ));
    }
    if options.is_missing_token_for(inner, column) {
        return Ok(integer_scalar_from_numeric(
            dtype,
            options.missing_value_for(column)?,
        ));
    }
    let normalized = normalize_numeric_token(inner, options, column);
    if normalized.is_empty() {
        return Ok(integer_scalar_from_numeric(
            dtype,
            options.empty_value_for(column)?,
        ));
    }
    if normalized.eq_ignore_ascii_case("nan") {
        return Ok(integer_scalar_from_f64(dtype, f64::NAN));
    }
    if ["inf", "+inf", "infinity", "+infinity"]
        .iter()
        .any(|value| normalized.eq_ignore_ascii_case(value))
    {
        return Ok(integer_scalar_from_f64(dtype, f64::INFINITY));
    }
    if ["-inf", "-infinity"]
        .iter()
        .any(|value| normalized.eq_ignore_ascii_case(value))
    {
        return Ok(integer_scalar_from_f64(dtype, f64::NEG_INFINITY));
    }
    let invalid =
        || invalid_numeric_token(inner, line_number, column.diagnostic_number(), reservation);
    if normalized.chars().any(|ch| matches!(ch, '.' | 'e' | 'E')) {
        return normalized
            .parse::<f64>()
            .map(|value| integer_scalar_from_f64(dtype, value))
            .map_err(|_| invalid());
    }
    parse_decimal_integer_saturating(&normalized, dtype).ok_or_else(invalid)
}

fn parse_decimal_integer_saturating(text: &str, dtype: NumericDType) -> Option<NumericScalar> {
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        Some(_) => (false, text),
        None => return None,
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let magnitude = digits.parse::<u128>().ok();
    Some(if negative {
        match magnitude {
            Some(value) if value <= i128::MAX as u128 => {
                integer_scalar_from_signed(dtype, -(value as i128))
            }
            _ => integer_scalar_from_signed(dtype, i128::MIN),
        }
    } else {
        integer_scalar_from_unsigned(dtype, magnitude.unwrap_or(u128::MAX))
    })
}

fn integer_scalar_from_numeric(dtype: NumericDType, value: NumericScalar) -> NumericScalar {
    match value {
        NumericScalar::F64(value) => integer_scalar_from_f64(dtype, value),
        NumericScalar::F32(value) => integer_scalar_from_f64(dtype, f64::from(value)),
        NumericScalar::I8(value) => integer_scalar_from_signed(dtype, i128::from(value)),
        NumericScalar::I16(value) => integer_scalar_from_signed(dtype, i128::from(value)),
        NumericScalar::I32(value) => integer_scalar_from_signed(dtype, i128::from(value)),
        NumericScalar::I64(value) => integer_scalar_from_signed(dtype, i128::from(value)),
        NumericScalar::U8(value) => integer_scalar_from_unsigned(dtype, u128::from(value)),
        NumericScalar::U16(value) => integer_scalar_from_unsigned(dtype, u128::from(value)),
        NumericScalar::U32(value) => integer_scalar_from_unsigned(dtype, u128::from(value)),
        NumericScalar::U64(value) => integer_scalar_from_unsigned(dtype, u128::from(value)),
    }
}

fn integer_scalar_from_signed(dtype: NumericDType, value: i128) -> NumericScalar {
    match dtype {
        NumericDType::I8 | NumericDType::I16 | NumericDType::I32 | NumericDType::I64 => {
            integer_scalar_from_i128(dtype, value)
        }
        NumericDType::U8 | NumericDType::U16 | NumericDType::U32 | NumericDType::U64 => {
            integer_scalar_from_u128(dtype, value.max(0) as u128)
        }
        NumericDType::F64 | NumericDType::F32 => {
            unreachable!("integer conversion requested floating dtype")
        }
    }
}

fn integer_scalar_from_unsigned(dtype: NumericDType, value: u128) -> NumericScalar {
    match dtype {
        NumericDType::I8 | NumericDType::I16 | NumericDType::I32 | NumericDType::I64 => {
            integer_scalar_from_i128(dtype, value.min(i128::MAX as u128) as i128)
        }
        NumericDType::U8 | NumericDType::U16 | NumericDType::U32 | NumericDType::U64 => {
            integer_scalar_from_u128(dtype, value)
        }
        NumericDType::F64 | NumericDType::F32 => {
            unreachable!("integer conversion requested floating dtype")
        }
    }
}

fn integer_scalar_from_i128(dtype: NumericDType, value: i128) -> NumericScalar {
    match dtype {
        NumericDType::I8 => NumericScalar::I8(value.clamp(i8::MIN as i128, i8::MAX as i128) as i8),
        NumericDType::I16 => {
            NumericScalar::I16(value.clamp(i16::MIN as i128, i16::MAX as i128) as i16)
        }
        NumericDType::I32 => {
            NumericScalar::I32(value.clamp(i32::MIN as i128, i32::MAX as i128) as i32)
        }
        NumericDType::I64 => {
            NumericScalar::I64(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
        }
        _ => unreachable!("signed conversion requested non-signed dtype"),
    }
}

fn integer_scalar_from_u128(dtype: NumericDType, value: u128) -> NumericScalar {
    match dtype {
        NumericDType::U8 => NumericScalar::U8(value.min(u8::MAX as u128) as u8),
        NumericDType::U16 => NumericScalar::U16(value.min(u16::MAX as u128) as u16),
        NumericDType::U32 => NumericScalar::U32(value.min(u32::MAX as u128) as u32),
        NumericDType::U64 => NumericScalar::U64(value.min(u64::MAX as u128) as u64),
        _ => unreachable!("unsigned conversion requested non-unsigned dtype"),
    }
}

fn integer_scalar_from_f64(dtype: NumericDType, value: f64) -> NumericScalar {
    if value.is_nan() {
        return match dtype {
            NumericDType::I8 => NumericScalar::I8(0),
            NumericDType::I16 => NumericScalar::I16(0),
            NumericDType::I32 => NumericScalar::I32(0),
            NumericDType::I64 => NumericScalar::I64(0),
            NumericDType::U8 => NumericScalar::U8(0),
            NumericDType::U16 => NumericScalar::U16(0),
            NumericDType::U32 => NumericScalar::U32(0),
            NumericDType::U64 => NumericScalar::U64(0),
            _ => unreachable!("integer conversion requested floating dtype"),
        };
    }
    let rounded = value.round();
    match dtype {
        NumericDType::I8 => NumericScalar::I8(rounded.clamp(i8::MIN as f64, i8::MAX as f64) as i8),
        NumericDType::I16 => {
            NumericScalar::I16(rounded.clamp(i16::MIN as f64, i16::MAX as f64) as i16)
        }
        NumericDType::I32 => {
            NumericScalar::I32(rounded.clamp(i32::MIN as f64, i32::MAX as f64) as i32)
        }
        NumericDType::I64 => {
            NumericScalar::I64(rounded.clamp(i64::MIN as f64, i64::MAX as f64) as i64)
        }
        NumericDType::U8 => NumericScalar::U8(rounded.clamp(0.0, u8::MAX as f64) as u8),
        NumericDType::U16 => NumericScalar::U16(rounded.clamp(0.0, u16::MAX as f64) as u16),
        NumericDType::U32 => NumericScalar::U32(rounded.clamp(0.0, u32::MAX as f64) as u32),
        NumericDType::U64 => NumericScalar::U64(rounded.clamp(0.0, u64::MAX as f64) as u64),
        NumericDType::F64 | NumericDType::F32 => {
            unreachable!("integer conversion requested floating dtype")
        }
    }
}

fn normalize_numeric_token<'a>(
    token: &'a str,
    options: &ReadMatrixOptions,
    column: MatrixFieldColumn,
) -> Cow<'a, str> {
    let (decimal_separator, thousands_separator) = options
        .variable(column)
        .map(|variable| {
            (
                variable
                    .decimal_separator
                    .unwrap_or(options.decimal_separator),
                variable
                    .thousands_separator
                    .unwrap_or(options.thousands_separator),
            )
        })
        .unwrap_or((options.decimal_separator, options.thousands_separator));
    super::super::super::import_options::normalize_numeric_token(
        token,
        decimal_separator,
        thousands_separator,
    )
}

fn unquote(token: &str) -> &str {
    if token.len() >= 2 {
        let bytes = token.as_bytes();
        if (bytes[0] == b'"' && bytes[token.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[token.len() - 1] == b'\'')
        {
            return &token[1..token.len() - 1];
        }
    }
    token
}
