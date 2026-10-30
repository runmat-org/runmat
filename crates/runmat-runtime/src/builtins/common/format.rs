//! Shared formatting helpers for string-producing builtins.

use std::char;
use std::iter::Peekable;
use std::str::Chars;

use runmat_value::{decode_utf16_code_units, IntValue, IntegerStorage, LogicalArray, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod conversion;
mod floating;
pub(crate) mod scalar_text;
use conversion::{FloatStyle, FormatConversion};
pub(crate) use floating::format_delimited_numeric;
use floating::format_float;

const FORMAT_UNSUPPORTED_SPECIFIER_IDENTIFIER: &str = "RunMat:format:UnsupportedSpecifier";

/// Stateful cursor over formatting arguments.
#[derive(Debug)]
pub struct ArgCursor<'a> {
    args: crate::call::FunctionArgumentView<'a>,
    index: usize,
}

impl<'a> ArgCursor<'a> {
    pub fn new(args: &'a [Value]) -> Self {
        Self::from_arguments(crate::call::FunctionArgumentView::from_slice(args))
    }

    pub fn from_arguments(args: crate::call::FunctionArgumentView<'a>) -> Self {
        Self { args, index: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.args.len().saturating_sub(self.index)
    }

    pub fn index(&self) -> usize {
        self.index
    }

    fn next(&mut self) -> BuiltinResult<Value> {
        if self.index >= self.args.len() {
            return Err(format_error(
                "sprintf: not enough input arguments for format specifier",
            ));
        }
        let value = self.args[self.index].clone();
        self.index += 1;
        Ok(value)
    }
}

fn format_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).build()
}

fn format_utf16_error(context: &str, error: runmat_value::Utf16DecodeError) -> RuntimeError {
    build_runtime_error(format!("{context}: {error}"))
        .with_source(error)
        .build()
}

fn format_error_with_identifier(
    message: impl Into<String>,
    identifier: &'static str,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(identifier)
        .build()
}

fn map_control_flow_with_context(err: RuntimeError, context: &str) -> RuntimeError {
    crate::builtins::common::map_control_flow_with_builtin(err, context)
}

/// Result of formatting a string with the current cursor state.
#[derive(Debug, Default, Clone)]
pub struct FormatStepResult {
    pub output: String,
    pub consumed: usize,
}

#[derive(Clone, Copy, Default)]
struct FormatFlags {
    alternate: bool,
    zero_pad: bool,
    left_align: bool,
    sign_plus: bool,
    sign_space: bool,
    grouping: bool,
}

#[derive(Clone, Copy)]
enum Count {
    Value(isize),
    FromArgument,
}

#[derive(Clone, Copy)]
struct FormatSpec {
    flags: FormatFlags,
    width: Option<Count>,
    precision: Option<Count>,
    conversion: FormatConversion,
}

/// Parsed grammar facts reusable by a sink before it materializes its values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FormatSummary {
    conversion: ConversionCount,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum ConversionCount {
    #[default]
    Zero,
    SoleStringWithoutStar,
    OtherOrMultiple,
}

impl FormatSummary {
    pub fn sole_string_conversion_without_star(self) -> bool {
        self.conversion == ConversionCount::SoleStringWithoutStar
    }

    fn record(&mut self, spec: FormatSpec) {
        self.conversion = match (self.conversion, spec.conversion) {
            (ConversionCount::Zero, FormatConversion::String)
                if !matches!(spec.width, Some(Count::FromArgument))
                    && !matches!(spec.precision, Some(Count::FromArgument)) =>
            {
                ConversionCount::SoleStringWithoutStar
            }
            _ => ConversionCount::OtherOrMultiple,
        };
    }
}

/// Format a MATLAB-style format string with the provided arguments.
///
/// This function consumes arguments in column-major order, supports field widths,
/// precision (including the `*` form), and honours the usual printf flags for the
/// subset required by MATLAB builtins. It is intentionally strict: errors are
/// reported when format specifiers cannot be satisfied by the provided arguments.
pub fn format_variadic(fmt: &str, args: &[Value]) -> BuiltinResult<String> {
    format_variadic_arguments(fmt, crate::call::FunctionArgumentView::from_slice(args))
}

pub fn format_variadic_arguments(
    fmt: &str,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<String> {
    let mut cursor = ArgCursor::from_arguments(args);
    let step = format_variadic_with_cursor(fmt, &mut cursor)?;
    Ok(step.output)
}

/// Format a string using the supplied cursor, returning the formatted text along
/// with the number of arguments consumed during this pass.
pub fn format_variadic_with_cursor(
    fmt: &str,
    cursor: &mut ArgCursor<'_>,
) -> BuiltinResult<FormatStepResult> {
    format_once(fmt, cursor)
}

/// Validate only the format grammar, without consuming or materializing values.
/// A sink can call this before reading a resident formatting argument.
pub fn validate_format_syntax(fmt: &str) -> BuiltinResult<FormatSummary> {
    let mut chars = fmt.chars().peekable();
    let mut summary = FormatSummary::default();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
        } else {
            summary.record(parse_format_spec(&mut chars)?);
        }
    }
    Ok(summary)
}

fn format_once(fmt: &str, cursor: &mut ArgCursor<'_>) -> BuiltinResult<FormatStepResult> {
    let mut chars = fmt.chars().peekable();
    let mut out = String::with_capacity(fmt.len());
    let mut consumed = 0usize;

    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }

        if let Some('%') = chars.peek() {
            chars.next();
            out.push('%');
            continue;
        }

        let spec = parse_format_spec(&mut chars)?;
        let (formatted, used) = apply_format_spec(spec, cursor)?;
        consumed += used;
        out.push_str(&formatted);
    }

    Ok(FormatStepResult {
        output: out,
        consumed,
    })
}

fn parse_format_spec(chars: &mut Peekable<Chars<'_>>) -> BuiltinResult<FormatSpec> {
    let mut flags = FormatFlags::default();
    loop {
        match chars.peek().copied() {
            Some('#') => {
                flags.alternate = true;
                chars.next();
            }
            Some('0') => {
                flags.zero_pad = true;
                chars.next();
            }
            Some('-') => {
                flags.left_align = true;
                chars.next();
            }
            Some(' ') => {
                flags.sign_space = true;
                chars.next();
            }
            Some('+') => {
                flags.sign_plus = true;
                chars.next();
            }
            Some('\'') => {
                flags.grouping = true;
                chars.next();
            }
            Some('I') => {
                // Locale-specific alternative digits are not implemented yet.
                // Consume the flag to keep format parsing compatible.
                chars.next();
            }
            _ => break,
        }
    }

    let width = if let Some('*') = chars.peek() {
        chars.next();
        Some(Count::FromArgument)
    } else {
        parse_number(chars).map(Count::Value)
    };

    let precision = if let Some('.') = chars.peek() {
        chars.next();
        if let Some('*') = chars.peek() {
            chars.next();
            Some(Count::FromArgument)
        } else {
            Some(Count::Value(parse_number(chars).unwrap_or(0)))
        }
    } else {
        None
    };

    // Length modifiers are ignored, but we must consume them to remain compatible.
    if let Some(&('h' | 'l' | 'L' | 'z' | 'j' | 't')) = chars.peek() {
        let current = chars.next().unwrap();
        if matches!(current, 'h' | 'l') && chars.peek() == Some(&current) {
            chars.next();
        }
    }

    let symbol = chars
        .next()
        .ok_or_else(|| format_error("sprintf: incomplete format specifier"))?;
    let conversion = FormatConversion::from_symbol(symbol).ok_or_else(|| {
        format_error_with_identifier(
            format!("sprintf: unsupported format %{symbol}"),
            FORMAT_UNSUPPORTED_SPECIFIER_IDENTIFIER,
        )
    })?;
    Ok(FormatSpec {
        flags,
        width,
        precision,
        conversion,
    })
}

fn parse_number(chars: &mut Peekable<Chars<'_>>) -> Option<isize> {
    let mut value: i128 = 0;
    let mut seen = false;
    while let Some(&ch) = chars.peek() {
        if !ch.is_ascii_digit() {
            break;
        }
        seen = true;
        value = value * 10 + i128::from((ch as u8 - b'0') as i16);
        chars.next();
    }
    if seen {
        let capped = value
            .clamp(isize::MIN as i128, isize::MAX as i128)
            .try_into()
            .unwrap_or(isize::MAX);
        Some(capped)
    } else {
        None
    }
}

fn apply_format_spec(
    spec: FormatSpec,
    cursor: &mut ArgCursor<'_>,
) -> BuiltinResult<(String, usize)> {
    let mut consumed = 0usize;
    let mut flags = spec.flags;

    let mut width = match spec.width {
        Some(Count::Value(w)) => Some(w),
        Some(Count::FromArgument) => {
            let value = cursor.next()?;
            consumed += 1;
            let w = value_to_isize(&value)?;
            Some(w)
        }
        None => None,
    };

    let precision = match spec.precision {
        Some(Count::Value(p)) => Some(p),
        Some(Count::FromArgument) => {
            let value = cursor.next()?;
            consumed += 1;
            let p = value_to_isize(&value)?;
            if p < 0 {
                None
            } else {
                Some(p)
            }
        }
        None => None,
    };

    if let Some(w) = width {
        if w < 0 {
            flags.left_align = true;
            width = Some(-w);
        }
    }

    let conversion = spec.conversion;
    let formatted = match conversion {
        FormatConversion::SignedDecimal => {
            let value = cursor.next()?;
            consumed += 1;
            match value {
                // An incompatible finite floating value cannot use an integer
                // conversion without losing information; preserve formatting
                // flags and use the documented conversion-override path.
                Value::Num(n) if n.is_finite() && !signed_decimal_float_fits_i128(n) => {
                    format_float(
                        n,
                        FloatStyle::ExponentLower,
                        flags,
                        width,
                        precision,
                        flags.alternate,
                    )
                }
                value => {
                    let int_value = value_to_i128(&value)?;
                    format_integer(
                        int_value,
                        int_value.is_negative(),
                        10,
                        flags,
                        width,
                        precision,
                        false,
                        false,
                    )
                }
            }
        }
        FormatConversion::UnsignedDecimal => {
            let value = cursor.next()?;
            consumed += 1;
            let uint_value = value_to_u128(&value)?;
            format_unsigned(uint_value, 10, flags, width, precision, false, false)
        }
        FormatConversion::Octal => {
            let value = cursor.next()?;
            consumed += 1;
            let uint_value = value_to_u128(&value)?;
            format_unsigned(
                uint_value,
                8,
                flags,
                width,
                precision,
                spec.flags.alternate,
                false,
            )
        }
        FormatConversion::HexLower => {
            let value = cursor.next()?;
            consumed += 1;
            let uint_value = value_to_u128(&value)?;
            format_unsigned(
                uint_value,
                16,
                flags,
                width,
                precision,
                spec.flags.alternate,
                false,
            )
        }
        FormatConversion::HexUpper => {
            let value = cursor.next()?;
            consumed += 1;
            let uint_value = value_to_u128(&value)?;
            format_unsigned(
                uint_value,
                16,
                flags,
                width,
                precision,
                spec.flags.alternate,
                true,
            )
        }
        FormatConversion::Binary => {
            let value = cursor.next()?;
            consumed += 1;
            let uint_value = value_to_u128(&value)?;
            format_unsigned(
                uint_value,
                2,
                flags,
                width,
                precision,
                spec.flags.alternate,
                false,
            )
        }
        FormatConversion::Float(style) => {
            let value = cursor.next()?;
            consumed += 1;
            let float_value = value_to_f64(&value)?;
            format_float(
                float_value,
                style,
                flags,
                width,
                precision,
                spec.flags.alternate,
            )
        }
        FormatConversion::String => {
            let value = cursor.next()?;
            consumed += 1;
            format_string(value, flags, width, precision)
        }
        FormatConversion::Character => {
            let value = cursor.next()?;
            consumed += 1;
            format_char(value, flags, width)
        }
    }?;

    Ok((formatted, consumed))
}

#[allow(clippy::too_many_arguments)]
fn format_integer(
    value: i128,
    is_negative: bool,
    base: u32,
    mut flags: FormatFlags,
    width: Option<isize>,
    precision: Option<isize>,
    alternate: bool,
    uppercase: bool,
) -> BuiltinResult<String> {
    let mut sign = String::new();
    let abs_val = value.unsigned_abs();

    if is_negative {
        sign.push('-');
    } else if flags.sign_plus {
        sign.push('+');
    } else if flags.sign_space {
        sign.push(' ');
    }

    if precision.is_some() {
        flags.zero_pad = false;
    }

    let mut digits = to_base_string(abs_val, base, uppercase);
    let precision_value = precision.unwrap_or(-1);
    if precision_value == 0 && abs_val == 0 {
        digits.clear();
    }
    if precision_value > 0 {
        let required = precision_value as usize;
        if digits.len() < required {
            let mut buf = String::with_capacity(required);
            for _ in 0..(required - digits.len()) {
                buf.push('0');
            }
            buf.push_str(&digits);
            digits = buf;
        }
    }

    let mut prefix = String::new();
    if alternate && abs_val != 0 {
        match base {
            8 => prefix.push('0'),
            16 => {
                prefix.push('0');
                prefix.push(if uppercase { 'X' } else { 'x' });
            }
            2 => {
                prefix.push('0');
                prefix.push('b');
            }
            _ => {}
        }
    }

    if flags.grouping && base == 10 {
        digits = group_decimal_digits(&digits);
    }

    apply_width(sign, prefix, digits, flags, width, flags.zero_pad)
}

fn format_unsigned(
    value: u128,
    base: u32,
    mut flags: FormatFlags,
    width: Option<isize>,
    precision: Option<isize>,
    alternate: bool,
    uppercase: bool,
) -> BuiltinResult<String> {
    if precision.is_some() {
        flags.zero_pad = false;
    }

    let mut digits = to_base_string(value, base, uppercase);
    let precision_value = precision.unwrap_or(-1);
    if precision_value == 0 && value == 0 {
        digits.clear();
    }
    if precision_value > 0 {
        let required = precision_value as usize;
        if digits.len() < required {
            let mut buf = String::with_capacity(required);
            for _ in 0..(required - digits.len()) {
                buf.push('0');
            }
            buf.push_str(&digits);
            digits = buf;
        }
    }

    let mut prefix = String::new();
    if alternate && value != 0 {
        match base {
            8 => prefix.push('0'),
            16 => {
                prefix.push_str(if uppercase { "0X" } else { "0x" });
            }
            2 => prefix.push_str("0b"),
            _ => {}
        }
    }

    if flags.grouping && base == 10 {
        digits = group_decimal_digits(&digits);
    }

    apply_width(String::new(), prefix, digits, flags, width, flags.zero_pad)
}

fn format_string(
    value: Value,
    flags: FormatFlags,
    width: Option<isize>,
    precision: Option<isize>,
) -> BuiltinResult<String> {
    let mut text = value_to_string(&value)?;
    if let Some(p) = precision {
        if p >= 0 {
            let mut chars = text.chars();
            let mut truncated = String::with_capacity(text.len());
            for _ in 0..(p as usize) {
                if let Some(ch) = chars.next() {
                    truncated.push(ch);
                } else {
                    break;
                }
            }
            text = truncated;
        }
    }

    apply_width(String::new(), String::new(), text, flags, width, false)
}

fn format_char(value: Value, flags: FormatFlags, width: Option<isize>) -> BuiltinResult<String> {
    let ch = value_to_char(&value)?;
    let text = ch.to_string();
    apply_width(String::new(), String::new(), text, flags, width, false)
}

fn apply_width(
    sign: String,
    prefix: String,
    digits: String,
    flags: FormatFlags,
    width: Option<isize>,
    zero_pad: bool,
) -> BuiltinResult<String> {
    let mut result = String::new();
    let sign_prefix_len = sign.len() + prefix.len();
    let total_len = sign_prefix_len + digits.len();
    let target_width = width.unwrap_or(0).max(0) as usize;

    if target_width <= total_len {
        result.push_str(&sign);
        result.push_str(&prefix);
        result.push_str(&digits);
        return Ok(result);
    }

    let pad_len = target_width - total_len;
    if flags.left_align {
        result.push_str(&sign);
        result.push_str(&prefix);
        result.push_str(&digits);
        for _ in 0..pad_len {
            result.push(' ');
        }
        return Ok(result);
    }

    if zero_pad {
        result.push_str(&sign);
        result.push_str(&prefix);
        for _ in 0..pad_len {
            result.push('0');
        }
        result.push_str(&digits);
    } else {
        for _ in 0..pad_len {
            result.push(' ');
        }
        result.push_str(&sign);
        result.push_str(&prefix);
        result.push_str(&digits);
    }
    Ok(result)
}

fn value_to_isize(value: &Value) -> BuiltinResult<isize> {
    match value {
        Value::Int(i) => Ok(i.to_i64().clamp(isize::MIN as i64, isize::MAX as i64) as isize),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format_error(
                    "sprintf: width/precision specifier must be finite",
                ));
            }
            Ok(n.trunc().clamp(isize::MIN as f64, isize::MAX as f64) as isize)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        other => Err(format_error(format!(
            "sprintf: width/precision specifier expects numeric value, got {other:?}"
        ))),
    }
}

fn value_to_i128(value: &Value) -> BuiltinResult<i128> {
    match value {
        Value::Int(i) => Ok(match i {
            IntValue::I8(v) => i128::from(*v),
            IntValue::I16(v) => i128::from(*v),
            IntValue::I32(v) => i128::from(*v),
            IntValue::I64(v) => i128::from(*v),
            IntValue::U8(v) => i128::from(*v),
            IntValue::U16(v) => i128::from(*v),
            IntValue::U32(v) => i128::from(*v),
            IntValue::U64(v) => i128::from(*v),
        }),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format_error(
                    "sprintf: numeric conversion requires finite input",
                ));
            }
            if !signed_decimal_float_fits_i128(*n) {
                return Err(format_error(
                    "sprintf: integer conversion requires a value in signed range",
                ));
            }
            Ok(*n as i128)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        other => Err(format_error(format!(
            "sprintf: expected numeric argument, got {other:?}"
        ))),
    }
}

fn signed_decimal_float_fits_i128(value: f64) -> bool {
    // i128::MAX rounds UP to 2^127 as f64, which is outside the signed range.
    // The lower endpoint (-2^127) is exactly representable and inclusive.
    value.is_finite()
        && value.fract() == 0.0
        && value >= i128::MIN as f64
        && value < i128::MAX as f64
}

fn value_to_u128(value: &Value) -> BuiltinResult<u128> {
    match value {
        Value::Int(i) => match i {
            IntValue::I8(v) if *v < 0 => Err(format_error("sprintf: expected non-negative value")),
            IntValue::I16(v) if *v < 0 => Err(format_error("sprintf: expected non-negative value")),
            IntValue::I32(v) if *v < 0 => Err(format_error("sprintf: expected non-negative value")),
            IntValue::I64(v) if *v < 0 => Err(format_error("sprintf: expected non-negative value")),
            IntValue::I8(v) => Ok((*v) as u128),
            IntValue::I16(v) => Ok((*v) as u128),
            IntValue::I32(v) => Ok((*v) as u128),
            IntValue::I64(v) => Ok((*v) as u128),
            IntValue::U8(v) => Ok((*v) as u128),
            IntValue::U16(v) => Ok((*v) as u128),
            IntValue::U32(v) => Ok((*v) as u128),
            IntValue::U64(v) => Ok((*v) as u128),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format_error(
                    "sprintf: numeric conversion requires finite input",
                ));
            }
            if *n < 0.0 {
                return Err(format_error("sprintf: expected non-negative value"));
            }
            Ok(n.trunc().clamp(0.0, u128::MAX as f64) as u128)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        other => Err(format_error(format!(
            "sprintf: expected non-negative numeric value, got {other:?}"
        ))),
    }
}

fn value_to_f64(value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(format_error(format!(
            "sprintf: expected numeric value, got {other:?}"
        ))),
    }
}

pub(crate) fn number_to_string(value: f64) -> String {
    let mut output = String::new();
    scalar_text::write_number_text(&mut output, value).expect("String formatting is infallible");
    output
}

pub(crate) fn complex_to_string(re: f64, im: f64) -> String {
    let mut output = String::new();
    scalar_text::write_complex_text(&mut output, re, im).expect("String formatting is infallible");
    output
}

fn value_to_string(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) => decode_utf16_code_units(ca.data()).map_err(|error| {
            format_utf16_error("sprintf: character value is not valid UTF-16 text", error)
        }),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::Num(n) => Ok(number_to_string(*n)),
        Value::Int(i) => Ok(int_value_to_decimal(i)),
        Value::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
        Value::Complex(re, im) => Ok(complex_to_string(*re, *im)),
        other => Err(format_error(format!(
            "sprintf: expected text or scalar value for %s conversion, got {other:?}"
        ))),
    }
}

fn value_to_char(value: &Value) -> BuiltinResult<char> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => s.chars().next().ok_or_else(|| {
            format_error("sprintf: %c conversion requires non-empty character input")
        }),
        Value::CharArray(ca) => {
            let mut decoded = char::decode_utf16(ca.data().iter().map(|unit| unit.raw()));
            match decoded.next() {
                Some(Ok(character)) => Ok(character),
                Some(Err(source)) => Err(build_runtime_error(format!(
                    "sprintf: %c character is not valid UTF-16 text: {source}"
                ))
                .with_source(source)
                .build()),
                None => Err(format_error(
                    "sprintf: %c conversion requires non-empty char input",
                )),
            }
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format_error(
                    "sprintf: %c conversion needs finite numeric value",
                ));
            }
            let code = n.trunc() as u32;
            std::char::from_u32(code)
                .ok_or_else(|| format_error("sprintf: numeric value outside valid character range"))
        }
        Value::Int(i) => {
            let code = i.to_i64();
            if code < 0 {
                return Err(format_error("sprintf: negative value for %c conversion"));
            }
            std::char::from_u32(code as u32)
                .ok_or_else(|| format_error("sprintf: numeric value outside valid character range"))
        }
        other => Err(format_error(format!(
            "sprintf: %c conversion expects character data, got {other:?}"
        ))),
    }
}

fn to_base_string(mut value: u128, base: u32, uppercase: bool) -> String {
    if value == 0 {
        return "0".to_string();
    }
    let mut buf = Vec::new();
    while value > 0 {
        let digit = (value % base as u128) as u8;
        let ch = match digit {
            0..=9 => b'0' + digit,
            _ => {
                if uppercase {
                    b'A' + (digit - 10)
                } else {
                    b'a' + (digit - 10)
                }
            }
        };
        buf.push(ch as char);
        value /= base as u128;
    }
    buf.iter().rev().collect()
}

fn group_decimal_digits(digits: &str) -> String {
    if digits.len() <= 3 {
        return digits.to_string();
    }
    let chars: Vec<char> = digits.chars().collect();
    let mut out = String::with_capacity(digits.len() + (digits.len() - 1) / 3);
    for (idx, ch) in chars.iter().enumerate() {
        if idx > 0 && (chars.len() - idx).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*ch);
    }
    out
}

fn group_float_mantissa(text: &str) -> String {
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(idx) => (&text[..idx], &text[idx..]),
        None => (text, ""),
    };

    let mut parts = mantissa.splitn(2, '.');
    let int_part = parts.next().unwrap_or_default();
    let frac_part = parts.next();
    let grouped_int = group_decimal_digits(int_part);

    let mut out = grouped_int;
    if let Some(frac) = frac_part {
        out.push('.');
        out.push_str(frac);
    }
    out.push_str(exponent);
    out
}

/// Extract a printf-style format string from a MATLAB value, validating that it
/// is a character row vector or string scalar.
pub fn extract_format_string(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) => {
            if ca.rows != 1 {
                return Err(format_error(format!(
                    "{context}: formatSpec must be a character row vector or string scalar"
                )));
            }
            ca.try_row_string()
                .map_err(|error| format_utf16_error(context, error))?
                .ok_or_else(|| {
                    format_error(format!(
                        "{context}: formatSpec must be a character row vector or string scalar"
                    ))
                })
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        _ => Err(format_error(format!(
            "{context}: formatSpec must be a character row vector or string scalar"
        ))),
    }
}

/// Decode MATLAB-compatible escape sequences within a format specification.
pub fn decode_escape_sequences(context: &str, input: &str) -> BuiltinResult<String> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        let Some(next) = chars.next() else {
            result.push('\\');
            break;
        };
        match next {
            '\\' => result.push('\\'),
            'a' => result.push('\u{0007}'),
            'b' => result.push('\u{0008}'),
            'f' => result.push('\u{000C}'),
            'n' => result.push('\n'),
            'r' => result.push('\r'),
            't' => result.push('\t'),
            'v' => result.push('\u{000B}'),
            'x' => {
                let mut hex = String::new();
                for _ in 0..2 {
                    match chars.peek().copied() {
                        Some(c) if c.is_ascii_hexdigit() => {
                            hex.push(chars.next().unwrap());
                        }
                        _ => break,
                    }
                }
                if hex.is_empty() {
                    result.push('\\');
                    result.push('x');
                } else {
                    let value = u32::from_str_radix(&hex, 16).map_err(|_| {
                        format_error(format!("{context}: invalid hexadecimal escape \\x{hex}"))
                    })?;
                    if let Some(chr) = char::from_u32(value) {
                        result.push(chr);
                    } else {
                        return Err(format_error(format!(
                            "{context}: \\x{hex} escape outside valid Unicode range"
                        )));
                    }
                }
            }
            '0'..='7' => {
                let mut oct = String::new();
                oct.push(next);
                for _ in 0..2 {
                    match chars.peek().copied() {
                        Some(c) if ('0'..='7').contains(&c) => {
                            oct.push(chars.next().unwrap());
                        }
                        _ => break,
                    }
                }
                let value = u32::from_str_radix(&oct, 8).map_err(|_| {
                    format_error(format!("{context}: invalid octal escape \\{oct}"))
                })?;
                if let Some(chr) = char::from_u32(value) {
                    result.push(chr);
                } else {
                    return Err(format_error(format!(
                        "{context}: \\{oct} escape outside valid Unicode range"
                    )));
                }
            }
            other => {
                result.push('\\');
                result.push(other);
            }
        }
    }
    Ok(result)
}

/// Flatten MATLAB argument values into a linear vector suitable for repeated
/// printf-style formatting. Arrays are traversed in column-major order and GPU
/// tensors are gathered back to the host.
pub async fn flatten_arguments(args: &[Value], context: &str) -> BuiltinResult<Vec<Value>> {
    flatten_argument_elements(crate::call::FunctionArgumentView::from_slice(args), context).await
}

/// Expand numerical/text elements for formatting, borrowing the original
/// logical argument segments. This is formatting scratch, not a new call ABI.
pub async fn flatten_argument_elements(
    args: crate::call::FunctionArgumentView<'_>,
    context: &str,
) -> BuiltinResult<Vec<Value>> {
    let mut flattened = Vec::new();
    for value in args {
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(|flow| map_control_flow_with_context(flow, context))?;
        flatten_value(gathered, &mut flattened, context).await?;
    }
    Ok(flattened)
}

#[async_recursion::async_recursion(?Send)]
async fn flatten_value(value: Value, output: &mut Vec<Value>, context: &str) -> BuiltinResult<()> {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::String(_)
        | Value::Complex(_, _)
        | Value::Symbolic(_) => {
            output.push(value);
        }
        Value::SymbolicArray(array) => {
            for expr in array.data() {
                output.push(Value::Symbolic(expr.clone()));
            }
        }
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                for index in 0..storage.len() {
                    output.push(Value::Int(integer_storage_value(storage, index)));
                }
            } else {
                let values = tensor::tensor_values_f64_cow(&tensor);
                for &elem in values.as_ref() {
                    output.push(Value::Num(elem));
                }
            }
        }
        Value::ComplexTensor(tensor) => {
            if tensor.integer_storage().is_some() {
                for index in 0..tensor::complex_tensor_element_len(&tensor) {
                    output.push(Value::string(tensor.format_element(index)));
                }
            } else {
                for &(re, im) in &tensor.materialize_f64() {
                    output.push(Value::Complex(re, im));
                }
            }
        }
        Value::LogicalArray(LogicalArray { data, .. }) => {
            for byte in data {
                output.push(Value::Bool(byte != 0));
            }
        }
        Value::StringArray(array) => match array.into_parts().0.try_into_owned() {
            Ok(data) => output.extend(data.into_iter().map(|text| Value::String(text.into()))),
            Err(data) => output.extend(data.iter().cloned().map(|text| Value::String(text.into()))),
        },
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                let text = decode_utf16_code_units(ca.data()).map_err(|error| {
                    format_utf16_error("character value is not valid UTF-16 text", error)
                })?;
                output.push(Value::string(text));
            } else {
                for row in 0..ca.rows {
                    let start = row * ca.cols;
                    let line = decode_utf16_code_units(&ca.data()[start..start + ca.cols])
                        .map_err(|error| {
                            format_utf16_error("character row is not valid UTF-16 text", error)
                        })?;
                    output.push(Value::string(line));
                }
            }
        }
        Value::Cell(cell) => {
            for col in 0..cell.cols {
                for row in 0..cell.rows {
                    let idx = row * cell.cols + col;
                    let inner = cell.data()[idx].clone();
                    let gathered = gather_if_needed_async(&inner)
                        .await
                        .map_err(|flow| map_control_flow_with_context(flow, context))?;
                    flatten_value(gathered, output, context).await?;
                }
            }
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_if_needed_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_context(flow, context))?;
            flatten_value(gathered, output, context).await?;
        }
        Value::OutputList(values) => {
            let values = values.try_into_owned().map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
            for value in values {
                flatten_value(value, output, context).await?;
            }
        }
        Value::MException(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::SparseTensor(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_) => {
            return Err(format_error(format!(
                "{context}: unsupported argument type"
            )));
        }
    }
    Ok(())
}

fn integer_storage_value(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

fn int_value_to_decimal(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.to_string(),
        IntValue::I16(value) => value.to_string(),
        IntValue::I32(value) => value.to_string(),
        IntValue::I64(value) => value.to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{
        get_display_format, set_display_format, FormatMode, IntegerComplexStorage, NumericStorage,
        Tensor,
    };

    #[test]
    fn syntax_admission_and_typed_execution_share_conversion_grammar() {
        for (format, value) in [
            ("%d", Value::Num(12.0)),
            ("%i", Value::Num(12.0)),
            ("%u", Value::Num(12.0)),
            ("%o", Value::Num(12.0)),
            ("%x", Value::Num(12.0)),
            ("%X", Value::Num(12.0)),
            ("%b", Value::Num(12.0)),
            ("%f", Value::Num(12.0)),
            ("%F", Value::Num(12.0)),
            ("%e", Value::Num(12.0)),
            ("%E", Value::Num(12.0)),
            ("%g", Value::Num(12.0)),
            ("%G", Value::Num(12.0)),
            ("%s", Value::from("value")),
            ("%c", Value::Num(65.0)),
        ] {
            validate_format_syntax(format).expect("admitted conversion");
            format_variadic(format, &[value]).expect("same conversion executes");
        }
        for invalid in ["%q", "%", "%10.2q"] {
            assert!(validate_format_syntax(invalid).is_err());
        }
        assert_eq!(
            format_variadic("%#x", &[Value::Num(255.0)]).unwrap(),
            "0xff"
        );
        assert_eq!(
            format_variadic("%#X", &[Value::Num(255.0)]).unwrap(),
            "0XFF"
        );
        assert_eq!(
            format_variadic("%.1e", &[Value::Num(12.0)]).unwrap(),
            "1.2e+01"
        );
        assert_eq!(
            format_variadic("%.1E", &[Value::Num(12.0)]).unwrap(),
            "1.2E+01"
        );
    }

    #[test]
    fn floating_conversions_use_scientific_exponents_and_significant_digits() {
        for (spec, value, expected) in [
            ("%.1e", 8.8e8, "8.8e+08"),
            ("%e", 0.5, "5.000000e-01"),
            ("%E", 1234.5, "1.234500E+03"),
            ("%g", 20.0, "20"),
            ("%g", 1.0, "1"),
            ("%g", 0.0, "0"),
            ("%g", 1.0e6, "1e+06"),
            ("%.3g", 0.000123456, "0.000123"),
            ("%G", 1.0e-10, "1E-10"),
            ("%.6g", 999999.5, "1e+06"),
            ("%#g", 20.0, "20.0000"),
            ("%#.0e", 1.0, "1.e+00"),
            ("%#.0f", 1.0, "1."),
        ] {
            assert_eq!(
                format_variadic(spec, &[Value::Num(value)]).unwrap(),
                expected,
                "{spec}"
            );
        }
        assert_eq!(format_variadic("%g", &[Value::Num(-0.0)]).unwrap(), "-0");
    }

    #[test]
    fn signed_decimal_does_not_truncate_or_clamp_incompatible_floating_values() {
        assert_eq!(
            format_variadic("%d", &[Value::Num(2.5)]).unwrap(),
            "2.500000e+00"
        );
        assert_eq!(format_variadic("%d", &[Value::Num(2.0)]).unwrap(), "2");
        assert_eq!(
            format_variadic("%d", &[Value::Num(1.0e300)]).unwrap(),
            "1.000000e+300"
        );
        assert_eq!(
            format_variadic("%d", &[Value::Num(i128::MIN as f64)]).unwrap(),
            i128::MIN.to_string()
        );
        assert!(format_variadic("%d", &[Value::Num(i128::MAX as f64)])
            .unwrap()
            .ends_with("e+38"));
        assert!(format_variadic("%d", &[Value::Num(f64::INFINITY)]).is_err());
        assert_eq!(
            format_variadic("%d", &[Value::Int(IntValue::I64(i64::MAX))]).unwrap(),
            i64::MAX.to_string()
        );
        assert_eq!(
            format_variadic("%+012.1d", &[Value::Num(2.5)]).unwrap(),
            "+00002.5e+00"
        );
    }

    #[test]
    fn sole_string_query_uses_parsed_specifiers_not_text_scanning() {
        for format in ["%s", "[%10.2s]", "%% %hs"] {
            assert!(
                validate_format_syntax(format)
                    .unwrap()
                    .sole_string_conversion_without_star(),
                "{format}"
            );
        }
        for format in ["plain text", "%%", "%s %s", "%d%s", "%*s", "%.*s"] {
            assert!(
                !validate_format_syntax(format)
                    .unwrap()
                    .sole_string_conversion_without_star(),
                "{format}"
            );
        }
    }

    #[test]
    fn format_variadic_supports_thousands_grouping_flag() {
        let out = format_variadic("%'d %'.2f", &[Value::Num(1234567.0), Value::Num(12345.5)])
            .expect("grouped formatting should succeed");
        assert_eq!(out, "1,234,567 12,345.50");
    }

    #[test]
    fn format_variadic_consumes_i_flag_without_error() {
        let out = format_variadic("%Id", &[Value::Int(IntValue::I32(42))])
            .expect("I flag should be accepted as compatibility no-op");
        assert_eq!(out, "42");
    }

    #[test]
    fn percent_s_numeric_and_complex_ignore_display_format() {
        let previous = get_display_format();
        set_display_format(FormatMode::Hex);
        let result = format_variadic(
            "%s %s",
            &[Value::Num(std::f64::consts::PI), Value::Complex(1.5, -2.0)],
        );
        set_display_format(previous);

        let out = result.expect("%s formatting should succeed");
        assert_eq!(out, "3.141592653589793 1.5-2i");
    }

    #[test]
    fn typed_integer_tensors_keep_exact_values_through_formatting() {
        let tensor = runmat_value::Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
            vec![1, 2],
        )
        .expect("integer tensor");
        let flattened =
            futures::executor::block_on(flatten_arguments(&[Value::Tensor(tensor)], "sprintf"))
                .expect("flattened arguments");

        assert_eq!(
            format_variadic(
                "%u %x %s",
                &[
                    flattened[0].clone(),
                    flattened[0].clone(),
                    flattened[1].clone()
                ]
            )
            .expect("formatted integer values"),
            format!("{} ffffffffffffffff {}", u64::MAX, 1_u64 << 63)
        );
    }

    #[test]
    fn native_single_tensors_flatten_from_authoritative_storage() {
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, -2.5]), vec![1, 2])
                .expect("single tensor");
        let flattened =
            futures::executor::block_on(flatten_arguments(&[Value::Tensor(tensor)], "sprintf"))
                .expect("flattened arguments");
        assert_eq!(flattened, vec![Value::Num(1.25), Value::Num(-2.5)]);
    }

    #[test]
    fn typed_complex_integer_tensors_keep_exact_values_through_string_formatting() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
            IntegerStorage::U64(vec![7, 0]),
        )
        .expect("matching complex integer storage");
        let tensor = runmat_value::ComplexTensor::new_integer(storage, vec![1, 2])
            .expect("complex integer tensor");
        let flattened = futures::executor::block_on(flatten_arguments(
            &[Value::ComplexTensor(tensor)],
            "sprintf",
        ))
        .expect("flattened arguments");

        assert_eq!(
            format_variadic("%s %s", &flattened).expect("formatted complex integer values"),
            format!("{}+7i {}", u64::MAX, 1_u64 << 63)
        );
    }
}
