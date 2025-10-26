//! Shared formatting helpers for string-producing builtins.

use std::char;
use std::iter::Peekable;
use std::str::Chars;

use runmat_builtins::{IntValue, LogicalArray, StringArray, Value};

use crate::gather_if_needed;

/// Stateful cursor over formatting arguments.
#[derive(Debug)]
pub struct ArgCursor<'a> {
    args: &'a [Value],
    index: usize,
}

impl<'a> ArgCursor<'a> {
    pub fn new(args: &'a [Value]) -> Self {
        Self { args, index: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.args.len().saturating_sub(self.index)
    }

    pub fn index(&self) -> usize {
        self.index
    }

    fn next(&mut self) -> Result<Value, String> {
        if self.index >= self.args.len() {
            return Err("sprintf: not enough input arguments for format specifier".to_string());
        }
        let value = self.args[self.index].clone();
        self.index += 1;
        Ok(value)
    }
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
    conversion: char,
}

/// Format a MATLAB-style format string with the provided arguments.
///
/// This function consumes arguments in column-major order, supports field widths,
/// precision (including the `*` form), and honours the usual printf flags for the
/// subset required by MATLAB builtins. It is intentionally strict: errors are
/// reported when format specifiers cannot be satisfied by the provided arguments.
pub fn format_variadic(fmt: &str, args: &[Value]) -> Result<String, String> {
    let mut cursor = ArgCursor::new(args);
    let step = format_variadic_with_cursor(fmt, &mut cursor)?;
    Ok(step.output)
}

/// Format a string using the supplied cursor, returning the formatted text along
/// with the number of arguments consumed during this pass.
pub fn format_variadic_with_cursor(
    fmt: &str,
    cursor: &mut ArgCursor<'_>,
) -> Result<FormatStepResult, String> {
    format_once(fmt, cursor)
}

fn format_once(fmt: &str, cursor: &mut ArgCursor<'_>) -> Result<FormatStepResult, String> {
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

fn parse_format_spec(chars: &mut Peekable<Chars<'_>>) -> Result<FormatSpec, String> {
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
    if let Some(&modch) = chars.peek() {
        match modch {
            'h' | 'l' | 'L' | 'z' | 'j' | 't' => {
                let c = chars.next().unwrap();
                if (c == 'h' || c == 'l') && chars.peek() == Some(&c) {
                    chars.next();
                }
            }
            _ => {}
        }
    }

    let conversion = chars
        .next()
        .ok_or_else(|| "sprintf: incomplete format specifier".to_string())?;

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
) -> Result<(String, usize), String> {
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
        'd' | 'i' => {
            let value = cursor.next()?;
            consumed += 1;
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
        'u' => {
            let value = cursor.next()?;
            consumed += 1;
            let uint_value = value_to_u128(&value)?;
            format_unsigned(uint_value, 10, flags, width, precision, false, false)
        }
        'o' => {
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
        'x' => {
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
        'X' => {
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
        'b' => {
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
        'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
            let value = cursor.next()?;
            consumed += 1;
            let float_value = value_to_f64(&value)?;
            format_float(
                float_value,
                conversion,
                flags,
                width,
                precision,
                spec.flags.alternate,
            )
        }
        's' => {
            let value = cursor.next()?;
            consumed += 1;
            format_string(value, flags, width, precision)
        }
        'c' => {
            let value = cursor.next()?;
            consumed += 1;
            format_char(value, flags, width)
        }
        other => {
            return Err(format!("sprintf: unsupported format %{other}"));
        }
    }?;

    Ok((formatted, consumed))
}

fn format_integer(
    value: i128,
    is_negative: bool,
    base: u32,
    mut flags: FormatFlags,
    width: Option<isize>,
    precision: Option<isize>,
    alternate: bool,
    uppercase: bool,
) -> Result<String, String> {
    let mut sign = String::new();
    let abs_val = value.abs() as u128;

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
) -> Result<String, String> {
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

    apply_width(String::new(), prefix, digits, flags, width, flags.zero_pad)
}

fn format_float(
    value: f64,
    conversion: char,
    flags: FormatFlags,
    width: Option<isize>,
    precision: Option<isize>,
    alternate: bool,
) -> Result<String, String> {
    let mut sign = String::new();
    let mut magnitude = value;

    if value.is_nan() {
        return apply_width(
            String::new(),
            String::new(),
            "NaN".to_string(),
            flags,
            width,
            false,
        );
    }

    if value.is_infinite() {
        if value.is_sign_negative() {
            sign.push('-');
        } else if flags.sign_plus {
            sign.push('+');
        } else if flags.sign_space {
            sign.push(' ');
        }
        let text = "Inf".to_string();
        return apply_width(sign, String::new(), text, flags, width, false);
    }

    if value.is_sign_negative() || (value == 0.0 && (1.0 / value).is_sign_negative()) {
        sign.push('-');
        magnitude = -value;
    } else if flags.sign_plus {
        sign.push('+');
    } else if flags.sign_space {
        sign.push(' ');
    }

    let prec = precision.unwrap_or(6).max(0) as usize;
    let mut body = match conversion {
        'f' | 'F' => format!("{magnitude:.prec$}"),
        'e' => format!("{magnitude:.prec$e}"),
        'E' => format!("{magnitude:.prec$E}"),
        'g' | 'G' => format_float_general(magnitude, prec, conversion.is_uppercase()),
        _ => {
            return Err(format!(
                "sprintf: unsupported float conversion %{}",
                conversion
            ))
        }
    };

    if alternate && !body.contains('.') && matches!(conversion, 'f' | 'F' | 'g' | 'G') {
        body.push('.');
    }

    let zero_pad_allowed = flags.zero_pad && !flags.left_align;
    apply_width(sign, String::new(), body, flags, width, zero_pad_allowed)
}

fn format_float_general(value: f64, precision: usize, uppercase: bool) -> String {
    if value == 0.0 {
        if precision == 0 {
            return "0".to_string();
        }
        let mut zero = String::from("0");
        if precision > 0 {
            zero.push('.');
            zero.push_str(&"0".repeat(precision.saturating_sub(1)));
        }
        return zero;
    }

    let mut prec = precision;
    if prec == 0 {
        prec = 1;
    }

    let abs_val = value.abs();
    let exp = abs_val.log10().floor() as i32;
    let use_exp = exp < -4 || exp >= prec as i32;

    if use_exp {
        let mut s = format!("{:.*e}", prec - 1, value);
        if uppercase {
            s = s.to_uppercase();
        }
        trim_trailing_zeros(&mut s, true);
        s
    } else {
        let mut s = format!("{:.*}", prec.max(1) - 1, value);
        trim_trailing_zeros(&mut s, false);
        s
    }
}

fn trim_trailing_zeros(text: &mut String, keep_exponent: bool) {
    if let Some(dot_idx) = text.find('.') {
        let mut end = text.len();
        while end > dot_idx + 1 {
            let byte = text.as_bytes()[end - 1];
            if byte == b'0' {
                end -= 1;
            } else {
                break;
            }
        }
        if end > dot_idx + 1 && text.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        if keep_exponent {
            if let Some(exp_idx) = text.find(|c: char| c == 'e' || c == 'E') {
                let exponent = text[exp_idx..].to_string();
                text.truncate(end.min(exp_idx));
                text.push_str(&exponent);
                return;
            }
        }
        text.truncate(end);
    }
}

fn format_string(
    value: Value,
    flags: FormatFlags,
    width: Option<isize>,
    precision: Option<isize>,
) -> Result<String, String> {
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

fn format_char(value: Value, flags: FormatFlags, width: Option<isize>) -> Result<String, String> {
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
) -> Result<String, String> {
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

fn value_to_isize(value: &Value) -> Result<isize, String> {
    match value {
        Value::Int(i) => Ok(i.to_i64().clamp(isize::MIN as i64, isize::MAX as i64) as isize),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("sprintf: width/precision specifier must be finite".to_string());
            }
            Ok(n.trunc().clamp(isize::MIN as f64, isize::MAX as f64) as isize)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        other => Err(format!(
            "sprintf: width/precision specifier expects numeric value, got {other:?}"
        )),
    }
}

fn value_to_i128(value: &Value) -> Result<i128, String> {
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
                return Err("sprintf: numeric conversion requires finite input".to_string());
            }
            Ok(n.trunc().clamp(i128::MIN as f64, i128::MAX as f64) as i128)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        other => Err(format!("sprintf: expected numeric argument, got {other:?}")),
    }
}

fn value_to_u128(value: &Value) -> Result<u128, String> {
    match value {
        Value::Int(i) => match i {
            IntValue::I8(v) if *v < 0 => Err("sprintf: expected non-negative value".to_string()),
            IntValue::I16(v) if *v < 0 => Err("sprintf: expected non-negative value".to_string()),
            IntValue::I32(v) if *v < 0 => Err("sprintf: expected non-negative value".to_string()),
            IntValue::I64(v) if *v < 0 => Err("sprintf: expected non-negative value".to_string()),
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
                return Err("sprintf: numeric conversion requires finite input".to_string());
            }
            if *n < 0.0 {
                return Err("sprintf: expected non-negative value".to_string());
            }
            Ok(n.trunc().clamp(0.0, u128::MAX as f64) as u128)
        }
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        other => Err(format!(
            "sprintf: expected non-negative numeric value, got {other:?}"
        )),
    }
}

fn value_to_f64(value: &Value) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(format!("sprintf: expected numeric value, got {other:?}")),
    }
}

fn value_to_string(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => {
            let mut s = String::with_capacity(ca.data.len());
            for ch in &ca.data {
                s.push(*ch);
            }
            Ok(s)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::Num(n) => Ok(Value::Num(*n).to_string()),
        Value::Int(i) => Ok(i.to_i64().to_string()),
        Value::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
        Value::Complex(re, im) => Ok(Value::Complex(*re, *im).to_string()),
        other => Err(format!(
            "sprintf: expected text or scalar value for %s conversion, got {other:?}"
        )),
    }
}

fn value_to_char(value: &Value) -> Result<char, String> {
    match value {
        Value::String(s) => s
            .chars()
            .next()
            .ok_or_else(|| "sprintf: %c conversion requires non-empty character input".to_string()),
        Value::CharArray(ca) => ca
            .data
            .get(0)
            .copied()
            .ok_or_else(|| "sprintf: %c conversion requires non-empty char input".to_string()),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("sprintf: %c conversion needs finite numeric value".to_string());
            }
            let code = n.trunc() as u32;
            std::char::from_u32(code)
                .ok_or_else(|| "sprintf: numeric value outside valid character range".to_string())
        }
        Value::Int(i) => {
            let code = i.to_i64();
            if code < 0 {
                return Err("sprintf: negative value for %c conversion".to_string());
            }
            std::char::from_u32(code as u32)
                .ok_or_else(|| "sprintf: numeric value outside valid character range".to_string())
        }
        other => Err(format!(
            "sprintf: %c conversion expects character data, got {other:?}"
        )),
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

/// Extract a printf-style format string from a MATLAB value, validating that it
/// is a character row vector or string scalar.
pub fn extract_format_string(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => {
            if ca.rows != 1 {
                return Err(format!(
                    "{context}: formatSpec must be a character row vector or string scalar"
                ));
            }
            Ok(ca.data.iter().collect())
        }
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(format!(
            "{context}: formatSpec must be a character row vector or string scalar"
        )),
    }
}

/// Decode MATLAB-compatible escape sequences within a format specification.
pub fn decode_escape_sequences(context: &str, input: &str) -> Result<String, String> {
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
                    let value = u32::from_str_radix(&hex, 16)
                        .map_err(|_| format!("{context}: invalid hexadecimal escape \\x{hex}"))?;
                    if let Some(chr) = char::from_u32(value) {
                        result.push(chr);
                    } else {
                        return Err(format!(
                            "{context}: \\x{hex} escape outside valid Unicode range"
                        ));
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
                let value = u32::from_str_radix(&oct, 8)
                    .map_err(|_| format!("{context}: invalid octal escape \\{oct}"))?;
                if let Some(chr) = char::from_u32(value) {
                    result.push(chr);
                } else {
                    return Err(format!(
                        "{context}: \\{oct} escape outside valid Unicode range"
                    ));
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
pub fn flatten_arguments(args: &[Value], context: &str) -> Result<Vec<Value>, String> {
    let mut flattened = Vec::new();
    for value in args {
        let gathered = gather_if_needed(value).map_err(|e| format!("{context}: {e}"))?;
        flatten_value(gathered, &mut flattened, context)?;
    }
    Ok(flattened)
}

fn flatten_value(value: Value, output: &mut Vec<Value>, context: &str) -> Result<(), String> {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::String(_)
        | Value::Complex(_, _) => {
            output.push(value);
        }
        Value::Tensor(tensor) => {
            for &elem in &tensor.data {
                output.push(Value::Num(elem));
            }
        }
        Value::ComplexTensor(tensor) => {
            for &(re, im) in &tensor.data {
                output.push(Value::Complex(re, im));
            }
        }
        Value::LogicalArray(LogicalArray { data, .. }) => {
            for byte in data {
                output.push(Value::Bool(byte != 0));
            }
        }
        Value::StringArray(StringArray { data, .. }) => {
            for s in data {
                output.push(Value::String(s));
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                output.push(Value::String(ca.data.iter().collect()));
            } else {
                for row in 0..ca.rows {
                    let mut line = String::with_capacity(ca.cols);
                    for col in 0..ca.cols {
                        line.push(ca.data[row * ca.cols + col]);
                    }
                    output.push(Value::String(line));
                }
            }
        }
        Value::Cell(cell) => {
            for col in 0..cell.cols {
                for row in 0..cell.rows {
                    let idx = row * cell.cols + col;
                    let inner = (*cell.data[idx]).clone();
                    let gathered =
                        gather_if_needed(&inner).map_err(|e| format!("{context}: {e}"))?;
                    flatten_value(gathered, output, context)?;
                }
            }
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_if_needed(&Value::GpuTensor(handle))
                .map_err(|e| format!("{context}: {e}"))?;
            flatten_value(gathered, output, context)?;
        }
        Value::MException(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_) => {
            return Err(format!("{context}: unsupported argument type"));
        }
    }
    Ok(())
}
