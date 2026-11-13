//! Primitive string-conversion spelling over a caller-owned UTF-8 sink.
use std::fmt::{self, Write};

use runmat_value::NumericScalar;

pub(crate) fn numeric_is_missing(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_nan(),
        NumericScalar::F32(value) => value.is_nan(),
        NumericScalar::I8(_)
        | NumericScalar::I16(_)
        | NumericScalar::I32(_)
        | NumericScalar::I64(_)
        | NumericScalar::U8(_)
        | NumericScalar::U16(_)
        | NumericScalar::U32(_)
        | NumericScalar::U64(_) => false,
    }
}

pub(crate) fn write_number_text(output: &mut impl Write, value: f64) -> fmt::Result {
    if value.is_nan() {
        output.write_str("NaN")
    } else if value.is_infinite() {
        output.write_str(if value.is_sign_negative() {
            "-Inf"
        } else {
            "Inf"
        })
    } else if value == 0.0 {
        output.write_str("0")
    } else {
        write!(output, "{value}")
    }
}

pub(crate) fn write_complex_text(
    output: &mut impl Write,
    real: f64,
    imaginary: f64,
) -> fmt::Result {
    if imaginary == 0.0 {
        return write_number_text(output, real);
    }
    if real != 0.0 {
        write_number_text(output, real)?;
        output.write_char(if imaginary < 0.0 { '-' } else { '+' })?;
        write_number_text(output, imaginary.abs())?;
    } else {
        write_number_text(output, imaginary)?;
    }
    output.write_char('i')
}

/// Integer payloads never cross a floating-point presentation boundary.
pub(crate) fn write_numeric_text(output: &mut impl Write, value: NumericScalar) -> fmt::Result {
    match value {
        NumericScalar::F64(value) => write_number_text(output, value),
        // Preserve the existing string conversion's promoted-single spelling.
        NumericScalar::F32(value) => write_number_text(output, f64::from(value)),
        NumericScalar::I8(value) => write!(output, "{value}"),
        NumericScalar::I16(value) => write!(output, "{value}"),
        NumericScalar::I32(value) => write!(output, "{value}"),
        NumericScalar::I64(value) => write!(output, "{value}"),
        NumericScalar::U8(value) => write!(output, "{value}"),
        NumericScalar::U16(value) => write!(output, "{value}"),
        NumericScalar::U32(value) => write!(output, "{value}"),
        NumericScalar::U64(value) => write!(output, "{value}"),
    }
}
