use super::*;
use crate::builtins::strings::core::numeric_length::nonnegative_platform_usize;
use runmat_value::{NumericStorage, Utf16CodeUnit};

pub(super) struct SscanfResult {
    pub(super) value: Value,
    pub(super) count: usize,
    pub(super) errmsg: String,
    pub(super) next_index: usize,
}

/// A borrowed UTF-8 input position measured in MATLAB UTF-16 code units.
/// `trailing` means the high surrogate of the scalar at `byte` was consumed.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ScanCursor {
    byte: usize,
    units: usize,
    trailing: bool,
}

impl ScanCursor {
    const START: Self = Self {
        byte: 0,
        units: 0,
        trailing: false,
    };

    fn next_scalar(self, text: &str) -> Option<(char, Self)> {
        if self.trailing {
            return None;
        }
        let character = text[self.byte..].chars().next()?;
        Some((
            character,
            Self {
                byte: self.byte + character.len_utf8(),
                units: self.units + character.len_utf16(),
                trailing: false,
            },
        ))
    }

    fn next_unit(self, text: &str) -> Option<(Utf16CodeUnit, Self)> {
        let character = text[self.byte..].chars().next()?;
        let mut encoded = [0u16; 2];
        let units = character.encode_utf16(&mut encoded);
        let (unit, byte, trailing) = if self.trailing {
            if units.len() != 2 {
                return None;
            }
            (units[1], self.byte + character.len_utf8(), false)
        } else if units.len() == 2 {
            (units[0], self.byte, true)
        } else {
            (units[0], self.byte + character.len_utf8(), false)
        };
        Some((
            Utf16CodeUnit::from_raw(unit),
            Self {
                byte,
                units: self.units + 1,
                trailing,
            },
        ))
    }
}

#[derive(Clone, Copy)]
enum ScanKind {
    Float,
    SignedInteger {
        radix: IntegerRadix,
        exact_i64: bool,
    },
    UnsignedInteger {
        radix: IntegerRadix,
        exact_u64: bool,
    },
    String,
    Char,
}

#[derive(Clone, Copy)]
enum IntegerRadix {
    Decimal,
    Auto,
    Octal,
    Hex,
}

#[derive(Clone, Copy)]
enum ScanValue {
    F64(f64),
    I64(i64),
    U64(u64),
    Char(Utf16CodeUnit),
}

#[derive(Clone, Copy)]
enum ScanOutputClass {
    F64,
    I64,
    U64,
    Char,
}

#[derive(Clone)]
enum ScanToken {
    Whitespace,
    Literal(char),
    Spec {
        kind: ScanKind,
        width: Option<usize>,
        suppress: bool,
    },
}

pub(super) fn sscanf_scan(
    text: &str,
    format: &str,
    size: Option<Vec<usize>>,
) -> BuiltinResult<SscanfResult> {
    let tokens = parse_scan_format(format)?;
    if tokens.is_empty() {
        return Err(compat_error("sscanf", "sscanf: format must not be empty"));
    }

    // A size is a limit on conversions, not a resize of an already completed
    // scan. In particular, neither diagnostics nor nextindex may observe the
    // portion of the input after the requested output has been filled.
    let output_class = scan_output_class(&tokens);
    let limit = size.as_ref().map(|dims| {
        dims.iter()
            .try_fold(1usize, |total, dim| total.checked_mul(*dim))
            .unwrap_or(usize::MAX)
    });
    let mut values = Vec::new();
    let mut fields = 0usize;
    let mut pos = ScanCursor::START;
    let mut unmatched_input = false;
    loop {
        if scan_limit_reached(limit, size.as_deref(), output_class, fields, values.len()) {
            break;
        }
        let start_pos = pos;
        let start_len = values.len();
        let mut matched_all = true;
        for token in &tokens {
            if scan_limit_reached(limit, size.as_deref(), output_class, fields, values.len()) {
                break;
            }
            match token {
                ScanToken::Whitespace => {
                    pos = skip_whitespace(text, pos);
                }
                ScanToken::Literal(ch) => {
                    let Some((next, next_pos)) = pos.next_scalar(text) else {
                        matched_all = false;
                        break;
                    };
                    if next != *ch {
                        matched_all = false;
                        break;
                    }
                    pos = next_pos;
                }
                ScanToken::Spec {
                    kind,
                    width,
                    suppress,
                } => {
                    if !matches!(kind, ScanKind::Char) {
                        pos = skip_whitespace(text, pos);
                    }
                    // Suppressed conversions consume their entire field/width
                    // without occupying output. For character matrices, the
                    // remaining output capacity bounds the scan itself, so
                    // nextindex never skips an unreturned field suffix.
                    let matrix_character_capacity = if matches!(output_class, ScanOutputClass::Char)
                        && size
                            .as_ref()
                            .and_then(|dims| dims.get(1))
                            .is_some_and(|columns| *columns != 1)
                    {
                        limit.map(|capacity| capacity.saturating_sub(values.len()))
                    } else {
                        None
                    };
                    let remaining = if *suppress {
                        None
                    } else if matches!(output_class, ScanOutputClass::Char) {
                        matrix_character_capacity
                    } else {
                        limit.map(|limit| limit.saturating_sub(values.len()))
                    };
                    let capture_limit = (*suppress).then_some(0);
                    let Some((parsed, next_pos)) =
                        scan_one(text, pos, *kind, *width, remaining, capture_limit)
                    else {
                        matched_all = false;
                        break;
                    };
                    pos = next_pos;
                    if !*suppress {
                        fields += 1;
                        values.extend(parsed);
                    }
                }
            }
        }
        if !matched_all {
            unmatched_input = pos.byte < text.len();
            break;
        }
        if pos == start_pos || values.len() == start_len && pos.byte >= text.len() {
            break;
        }
        if pos.byte >= text.len() {
            break;
        }
    }

    let count = values.len();
    let shape = scan_output_shape(size.as_deref(), count, output_class)?;
    let required = shape.iter().product::<usize>();
    values.resize(required, scan_zero(output_class));
    let value = if matches!(output_class, ScanOutputClass::Char) {
        let chars = values
            .into_iter()
            .map(|value| match value {
                ScanValue::Char(ch) => Ok(ch),
                _ => Err(compat_error(
                    "sscanf",
                    "sscanf: incompatible character conversion",
                )),
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        CharArray::from_column_major(chars, shape)
            .map(Value::CharArray)
            .map_err(|e| compat_error("sscanf", e))?
    } else {
        let storage = scan_values_into_storage(values, output_class)?;
        Tensor::from_numeric_storage(storage, shape)
            .map(Value::Tensor)
            .map_err(|e| compat_error("sscanf", e))?
    };
    Ok(SscanfResult {
        value,
        count,
        errmsg: if unmatched_input {
            "Input file or string does not contain specified format.".into()
        } else {
            String::new()
        },
        // A width-limited character field may stop between surrogate units.
        next_index: pos.units.saturating_add(1),
    })
}

fn scan_limit_reached(
    limit: Option<usize>,
    size: Option<&[usize]>,
    class: ScanOutputClass,
    fields: usize,
    elements: usize,
) -> bool {
    let Some(limit) = limit else { return false };
    if matches!(class, ScanOutputClass::Char) {
        fields >= limit
            || size
                .and_then(|dims| dims.get(1))
                .is_some_and(|columns| *columns != 1 && elements >= limit)
    } else {
        elements >= limit
    }
}

fn scan_output_shape(
    size: Option<&[usize]>,
    count: usize,
    class: ScanOutputClass,
) -> BuiltinResult<Vec<usize>> {
    let Some(&[rows, columns]) = size else {
        return Ok(if matches!(class, ScanOutputClass::Char) {
            vec![1, count]
        } else {
            vec![count, 1]
        });
    };
    if columns == 1 || rows == 0 || columns == 0 || count < rows {
        return Ok(if matches!(class, ScanOutputClass::Char) {
            vec![1, count]
        } else {
            vec![count, 1]
        });
    }
    // An incomplete last column is padded, but absent columns are not.
    let used_columns = count.div_ceil(rows);
    let padded = rows.checked_mul(used_columns).ok_or_else(scan_size_error)?;
    if padded < count {
        return Err(scan_size_error());
    }
    Ok(vec![rows, used_columns])
}

fn scan_zero(class: ScanOutputClass) -> ScanValue {
    match class {
        ScanOutputClass::F64 => ScanValue::F64(0.0),
        ScanOutputClass::I64 => ScanValue::I64(0),
        ScanOutputClass::U64 => ScanValue::U64(0),
        ScanOutputClass::Char => ScanValue::Char(Utf16CodeUnit::from_raw(0)),
    }
}

fn parse_scan_format(format: &str) -> BuiltinResult<Vec<ScanToken>> {
    let mut chars = format.chars().peekable();
    let mut tokens = Vec::new();
    while let Some(ch) = chars.next() {
        if ch.is_whitespace() {
            while chars.peek().is_some_and(|next| next.is_whitespace()) {
                chars.next();
            }
            tokens.push(ScanToken::Whitespace);
            continue;
        }
        if ch != '%' {
            tokens.push(ScanToken::Literal(ch));
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            tokens.push(ScanToken::Literal('%'));
            continue;
        }
        let suppress = if chars.peek() == Some(&'*') {
            chars.next();
            true
        } else {
            false
        };
        let mut width = String::new();
        while chars.peek().is_some_and(|next| next.is_ascii_digit()) {
            width.push(chars.next().unwrap());
        }
        let width = if width.is_empty() {
            None
        } else {
            Some(
                width
                    .parse::<usize>()
                    .map_err(|_| compat_error("sscanf", "sscanf: invalid field width"))?,
            )
        };
        let long = if chars.peek() == Some(&'l') {
            chars.next();
            true
        } else {
            false
        };
        let Some(specifier) = chars.next() else {
            return Err(compat_error(
                "sscanf",
                "sscanf: incomplete format specifier",
            ));
        };
        let kind = match specifier {
            'f' | 'e' | 'E' | 'g' | 'G' => ScanKind::Float,
            'd' => ScanKind::SignedInteger {
                radix: IntegerRadix::Decimal,
                exact_i64: long,
            },
            'i' => ScanKind::SignedInteger {
                radix: IntegerRadix::Auto,
                exact_i64: long,
            },
            'u' => ScanKind::UnsignedInteger {
                radix: IntegerRadix::Decimal,
                exact_u64: long,
            },
            'o' => ScanKind::UnsignedInteger {
                radix: IntegerRadix::Octal,
                exact_u64: long,
            },
            'x' | 'X' => ScanKind::UnsignedInteger {
                radix: IntegerRadix::Hex,
                exact_u64: long,
            },
            's' => ScanKind::String,
            'c' => ScanKind::Char,
            other => {
                return Err(compat_error(
                    "sscanf",
                    format!("sscanf: unsupported format specifier %{other}"),
                ))
            }
        };
        tokens.push(ScanToken::Spec {
            kind,
            width,
            suppress,
        });
    }
    Ok(tokens)
}

fn scan_one(
    text: &str,
    pos: ScanCursor,
    kind: ScanKind,
    width: Option<usize>,
    remaining: Option<usize>,
    capture_limit: Option<usize>,
) -> Option<(Vec<ScanValue>, ScanCursor)> {
    if pos.byte > text.len() {
        return None;
    }
    let width = if matches!(kind, ScanKind::Char) {
        Some(width.unwrap_or(1).min(remaining.unwrap_or(usize::MAX)))
    } else if matches!(kind, ScanKind::String) {
        match (width, remaining) {
            (Some(width), Some(remaining)) => Some(width.min(remaining)),
            (None, Some(remaining)) => Some(remaining),
            (width, None) => width,
        }
    } else {
        width
    };
    match kind {
        ScanKind::Float => {
            let fragment = numeric_fragment(text, pos, width)?;
            let len = numeric_prefix_len(fragment, false)?;
            let token = &fragment[..len];
            Some((
                vec![ScanValue::F64(token.parse::<f64>().ok()?)],
                numeric_advance(pos, len),
            ))
        }
        ScanKind::SignedInteger { radix, exact_i64 } => {
            let fragment = numeric_fragment(text, pos, width)?;
            let (value, len) = scan_signed_integer(fragment, radix)?;
            let value = if exact_i64 {
                ScanValue::I64(value)
            } else {
                ScanValue::F64(value as f64)
            };
            Some((vec![value], numeric_advance(pos, len)))
        }
        ScanKind::UnsignedInteger { radix, exact_u64 } => {
            let fragment = numeric_fragment(text, pos, width)?;
            let (value, len) = scan_unsigned_integer(fragment, radix)?;
            let value = if exact_u64 {
                ScanValue::U64(value)
            } else {
                ScanValue::F64(value as f64)
            };
            Some((vec![value], numeric_advance(pos, len)))
        }
        ScanKind::String => {
            let mut next = pos;
            let mut parsed = Vec::new();
            while width.is_none_or(|limit| next.units - pos.units < limit) {
                if next
                    .next_scalar(text)
                    .is_some_and(|(character, _)| character.is_whitespace())
                {
                    break;
                }
                let Some((unit, advanced)) = next.next_unit(text) else {
                    break;
                };
                if parsed.len() < capture_limit.unwrap_or(usize::MAX) {
                    parsed.push(ScanValue::Char(unit));
                }
                next = advanced;
            }
            if next == pos {
                None
            } else {
                Some((parsed, next))
            }
        }
        ScanKind::Char => {
            let mut next = pos;
            let mut parsed = Vec::new();
            let limit = width.unwrap_or(1);
            while next.units - pos.units < limit {
                let Some((unit, advanced)) = next.next_unit(text) else {
                    break;
                };
                if parsed.len() < capture_limit.unwrap_or(usize::MAX) {
                    parsed.push(ScanValue::Char(unit));
                }
                next = advanced;
            }
            (next != pos).then_some((parsed, next))
        }
    }
}

fn numeric_fragment(text: &str, pos: ScanCursor, width: Option<usize>) -> Option<&str> {
    if pos.trailing {
        return None;
    }
    let end = if let Some(width) = width {
        let mut next = pos;
        while next.units - pos.units < width {
            let Some((character, advanced)) = next.next_scalar(text) else {
                break;
            };
            if advanced.units - pos.units > width {
                break;
            }
            next = advanced;
            if !character.is_ascii() {
                break;
            }
        }
        next.byte
    } else {
        text.len()
    };
    Some(&text[pos.byte..end])
}

fn numeric_advance(pos: ScanCursor, ascii_bytes: usize) -> ScanCursor {
    ScanCursor {
        byte: pos.byte + ascii_bytes,
        units: pos.units + ascii_bytes,
        trailing: false,
    }
}

fn scan_output_class(tokens: &[ScanToken]) -> ScanOutputClass {
    let mut character_conversion = false;
    let mut numeric_conversion = false;
    for token in tokens {
        if let ScanToken::Spec { kind, .. } = token {
            match kind {
                ScanKind::String | ScanKind::Char => character_conversion = true,
                _ => numeric_conversion = true,
            }
        }
    }
    if character_conversion {
        // A numeric conversion determines double output even when suppressed.
        return if numeric_conversion {
            ScanOutputClass::F64
        } else {
            ScanOutputClass::Char
        };
    }
    let mut class = None;
    for token in tokens {
        let ScanToken::Spec {
            kind,
            suppress: false,
            ..
        } = token
        else {
            continue;
        };
        let next = match kind {
            ScanKind::SignedInteger {
                exact_i64: true, ..
            } => ScanOutputClass::I64,
            ScanKind::UnsignedInteger {
                exact_u64: true, ..
            } => ScanOutputClass::U64,
            _ => return ScanOutputClass::F64,
        };
        match (class, next) {
            (None, next) => class = Some(next),
            (Some(ScanOutputClass::I64), ScanOutputClass::I64)
            | (Some(ScanOutputClass::U64), ScanOutputClass::U64) => {}
            _ => return ScanOutputClass::F64,
        }
    }
    class.unwrap_or(ScanOutputClass::F64)
}

fn scan_values_into_storage(
    values: Vec<ScanValue>,
    class: ScanOutputClass,
) -> BuiltinResult<NumericStorage> {
    match class {
        ScanOutputClass::F64 => Ok(NumericStorage::F64(
            values
                .into_iter()
                .map(|value| match value {
                    ScanValue::F64(value) => value,
                    ScanValue::I64(value) => value as f64,
                    ScanValue::U64(value) => value as f64,
                    ScanValue::Char(value) => f64::from(value.raw()),
                })
                .collect(),
        )),
        ScanOutputClass::Char => Err(compat_error(
            "sscanf",
            "sscanf: character scan requires character output",
        )),
        ScanOutputClass::I64 => values
            .into_iter()
            .map(|value| match value {
                ScanValue::I64(value) => Ok(value),
                _ => Err(compat_error(
                    "sscanf",
                    "sscanf: incompatible conversions in int64 scan format",
                )),
            })
            .collect::<BuiltinResult<Vec<_>>>()
            .map(NumericStorage::I64),
        ScanOutputClass::U64 => values
            .into_iter()
            .map(|value| match value {
                ScanValue::U64(value) => Ok(value),
                _ => Err(compat_error(
                    "sscanf",
                    "sscanf: incompatible conversions in uint64 scan format",
                )),
            })
            .collect::<BuiltinResult<Vec<_>>>()
            .map(NumericStorage::U64),
    }
}

fn scan_signed_integer(text: &str, radix: IntegerRadix) -> Option<(i64, usize)> {
    let (negative, digits, base, len) = integer_token_parts(text, radix, true)?;
    let magnitude = u64::from_str_radix(digits, base).ok()?;
    let value = if negative {
        if magnitude == (1_u64 << 63) {
            i64::MIN
        } else {
            -i64::try_from(magnitude).ok()?
        }
    } else {
        i64::try_from(magnitude).ok()?
    };
    Some((value, len))
}

fn scan_unsigned_integer(text: &str, radix: IntegerRadix) -> Option<(u64, usize)> {
    let (negative, digits, base, len) = integer_token_parts(text, radix, false)?;
    if negative {
        return None;
    }
    Some((u64::from_str_radix(digits, base).ok()?, len))
}

fn integer_token_parts(
    text: &str,
    radix: IntegerRadix,
    allow_negative: bool,
) -> Option<(bool, &str, u32, usize)> {
    let bytes = text.as_bytes();
    let mut start = 0usize;
    let mut negative = false;
    if let Some(sign) = bytes.first() {
        if *sign == b'+' {
            start = 1;
        } else if *sign == b'-' && allow_negative {
            start = 1;
            negative = true;
        }
    }
    let remaining = &text[start..];
    let (base, prefix_len) = match radix {
        IntegerRadix::Decimal => (10, 0),
        IntegerRadix::Octal => (8, 0),
        IntegerRadix::Hex => (
            16,
            usize::from(remaining.starts_with("0x") || remaining.starts_with("0X")) * 2,
        ),
        IntegerRadix::Auto if remaining.starts_with("0x") || remaining.starts_with("0X") => (16, 2),
        IntegerRadix::Auto if remaining.starts_with('0') => (8, 0),
        IntegerRadix::Auto => (10, 0),
    };
    let digit_start = start + prefix_len;
    let digit_len = text[digit_start..]
        .char_indices()
        .take_while(|(_, ch)| ch.is_digit(base))
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()?;
    let end = digit_start + digit_len;
    Some((negative, &text[digit_start..end], base, end))
}

fn numeric_prefix_len(text: &str, integer: bool) -> Option<usize> {
    let mut end = 0usize;
    for (idx, ch) in text.char_indices() {
        let allowed = if integer {
            ch.is_ascii_digit() || ((ch == '+' || ch == '-') && idx == 0)
        } else {
            ch.is_ascii_digit() || matches!(ch, '+' | '-' | '.' | 'e' | 'E')
        };
        if !allowed {
            break;
        }
        end = idx + ch.len_utf8();
    }
    (end > 0).then_some(end)
}

fn skip_whitespace(text: &str, mut pos: ScanCursor) -> ScanCursor {
    while let Some((character, next)) = pos.next_scalar(text) {
        if !character.is_whitespace() {
            break;
        }
        pos = next;
    }
    pos
}

pub(super) fn scan_size_from_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(n) => Ok(vec![scan_size_dim(*n, true)?, 1]),
        Value::Int(value) => Ok(vec![
            positive_scan_size(value.try_to_usize().ok_or_else(scan_size_error)?)?,
            1,
        ]),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(vec![scan_tensor_size_dim(tensor, 0, true)?, 1])
        }
        Value::Tensor(tensor) if tensor_element_len(tensor) == 2 => Ok(vec![
            scan_tensor_size_dim(tensor, 0, false)?,
            scan_tensor_size_dim(tensor, 1, true)?,
        ]),
        other => Err(compat_error(
            "sscanf",
            format!("sscanf: invalid size argument {other:?}"),
        )),
    }
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn scan_tensor_size_dim(tensor: &Tensor, index: usize, allow_inf: bool) -> BuiltinResult<usize> {
    match tensor.numeric_value_at(index).ok_or_else(scan_size_error)? {
        NumericScalar::F64(value) => scan_size_dim(value, allow_inf),
        NumericScalar::F32(value) => scan_size_dim(f64::from(value), allow_inf),
        integer => positive_scan_size(
            integer
                .into_int_value()
                .expect("non-floating numeric scalar is integer")
                .try_to_usize()
                .ok_or_else(scan_size_error)?,
        ),
    }
}

fn scan_size_error() -> crate::RuntimeError {
    compat_error(
        "sscanf",
        "sscanf: size dimensions must be positive integers",
    )
}

fn scan_size_dim(value: f64, allow_inf: bool) -> BuiltinResult<usize> {
    if value.is_infinite() && value.is_sign_positive() {
        return allow_inf
            .then_some(usize::MAX / 2)
            .ok_or_else(scan_size_error);
    }
    positive_scan_size(nonnegative_platform_usize(value).ok_or_else(scan_size_error)?)
}

fn positive_scan_size(value: usize) -> BuiltinResult<usize> {
    (value > 0).then_some(value).ok_or_else(scan_size_error)
}
