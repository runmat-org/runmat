//! MATLAB-compatible binary and hexadecimal conversion helpers.

use runmat_builtins::{
    CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, ResolveContext, Tensor, Type,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{gpu_helpers, tensor as tensor_utils};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BI2DE_NAME: &str = "bi2de";
const DE2BI_NAME: &str = "de2bi";
const DEC2BIN_NAME: &str = "dec2bin";
const DEC2HEX_NAME: &str = "dec2hex";
const BIN2DEC_NAME: &str = "bin2dec";
const HEX2DEC_NAME: &str = "hex2dec";
const INTEGER_TOL: f64 = 1e-9;
const MAX_EXACT_INTEGER: u128 = 1u128 << 52;
const MAX_DE2BI_OUTPUT_ELEMENTS: usize = 10_000_000;
const MAX_DECIMAL_TEXT_OUTPUT_CHARS: usize = 10_000_000;
const MAX_TEXT_CONVERSION_DIGITS: usize = 64;
const MAX_FLOAT_DECIMAL_INPUT: f64 = 18_446_744_073_709_551_616.0; // 2^64

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DigitOrder {
    RightMsb,
    LeftMsb,
}

#[derive(Debug)]
struct DigitMatrix {
    data: Vec<usize>,
    rows: usize,
    cols: usize,
}

fn bi2de_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BI2DE_NAME)
        .build()
}

fn de2bi_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(DE2BI_NAME)
        .build()
}

fn conversion_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
}

fn vector_or_scalar_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::tensor()
}

fn matrix_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::tensor()
}

fn char_matrix_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::String
}

#[runtime_builtin(
    name = "bi2de",
    category = "comms/conversion",
    summary = "Convert rows of base-p digits to decimal integers.",
    keywords = "bi2de,binary,decimal,base,communications,right-msb,left-msb",
    type_resolver(vector_or_scalar_type),
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn bi2de_builtin(b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let opts = Bi2deOptions::parse(&rest)?;
    let digits = digit_matrix_from_value(b, BI2DE_NAME).await?;
    validate_digit_matrix(&digits, opts.base, BI2DE_NAME)?;

    let mut out = Vec::with_capacity(digits.rows);
    for row in 0..digits.rows {
        let value = row_digits_to_decimal(&digits, row, opts.base, opts.order)?;
        out.push(value as f64);
    }

    if out.len() == 1 {
        Ok(Value::Num(out[0]))
    } else {
        let rows = out.len();
        Ok(Value::Tensor(
            Tensor::new(out, vec![rows, 1]).map_err(|err| bi2de_error(format!("bi2de: {err}")))?,
        ))
    }
}

#[runtime_builtin(
    name = "de2bi",
    category = "comms/conversion",
    summary = "Convert nonnegative decimal integers to rows of base-p digits.",
    keywords = "de2bi,decimal,binary,base,communications,right-msb,left-msb",
    type_resolver(matrix_type),
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn de2bi_builtin(d: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let opts = De2biOptions::parse(&rest)?;
    let decimals = decimal_vector_from_value(d).await?;
    let needed_width = decimals
        .iter()
        .copied()
        .map(|value| digits_required(value, opts.base))
        .max()
        .unwrap_or(0);
    let width = opts.width.unwrap_or(needed_width);
    if opts.width.is_some() && width < needed_width {
        return Err(de2bi_error(format!(
            "de2bi: N is too small to represent an input value in base {}",
            opts.base
        )));
    }

    let rows = decimals.len();
    let output_len = checked_de2bi_output_len(rows, width)?;
    let mut out = vec![0.0; output_len];
    for (row, value) in decimals.into_iter().enumerate() {
        write_digits(value, opts.base, width, opts.order, &mut out, row, rows);
    }

    Ok(Value::Tensor(
        Tensor::new(out, vec![rows, width]).map_err(|err| de2bi_error(format!("de2bi: {err}")))?,
    ))
}

#[runtime_builtin(
    name = "dec2bin",
    category = "comms/conversion",
    summary = "Convert decimal integers to binary character rows.",
    keywords = "dec2bin,decimal,binary,base conversion,twos complement",
    type_resolver(char_matrix_type),
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn dec2bin_builtin(d: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    decimal_to_text_builtin(d, rest, DEC2BIN_NAME, 2, true).await
}

#[runtime_builtin(
    name = "dec2hex",
    category = "comms/conversion",
    summary = "Convert decimal integers to hexadecimal character rows.",
    keywords = "dec2hex,decimal,hexadecimal,base conversion,twos complement",
    type_resolver(char_matrix_type),
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn dec2hex_builtin(d: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    decimal_to_text_builtin(d, rest, DEC2HEX_NAME, 16, false).await
}

#[runtime_builtin(
    name = "bin2dec",
    category = "comms/conversion",
    summary = "Convert binary text to decimal numbers.",
    keywords = "bin2dec,binary,decimal,base conversion,twos complement",
    type_resolver(vector_or_scalar_type),
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn bin2dec_builtin(bin_str: Value) -> BuiltinResult<Value> {
    text_to_decimal_builtin(bin_str, BIN2DEC_NAME, 2).await
}

#[runtime_builtin(
    name = "hex2dec",
    category = "comms/conversion",
    summary = "Convert hexadecimal text to decimal numbers.",
    keywords = "hex2dec,hexadecimal,decimal,base conversion,twos complement",
    type_resolver(vector_or_scalar_type),
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn hex2dec_builtin(hex_str: Value) -> BuiltinResult<Value> {
    text_to_decimal_builtin(hex_str, HEX2DEC_NAME, 16).await
}

#[derive(Debug)]
struct Bi2deOptions {
    base: usize,
    order: DigitOrder,
}

impl Bi2deOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut base = 2usize;
        let mut order = DigitOrder::RightMsb;
        match args {
            [] => {}
            [one] => {
                if let Some(parsed) = parse_order(one, BI2DE_NAME)? {
                    order = parsed;
                } else {
                    base = parse_base(one, BI2DE_NAME)?;
                }
            }
            [base_value, flag] => {
                base = parse_base(base_value, BI2DE_NAME)?;
                order = parse_required_order(flag, BI2DE_NAME)?;
            }
            _ => return Err(bi2de_error("bi2de: expected at most three inputs")),
        }
        Ok(Self { base, order })
    }
}

#[derive(Debug)]
struct De2biOptions {
    width: Option<usize>,
    base: usize,
    order: DigitOrder,
}

impl De2biOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut width = None;
        let mut base = 2usize;
        let mut order = DigitOrder::RightMsb;
        let mut idx = 0usize;

        if let Some(first) = args.get(idx) {
            if let Some(parsed) = parse_order(first, DE2BI_NAME)? {
                order = parsed;
                idx += 1;
                if idx != args.len() {
                    return Err(de2bi_error(
                        "de2bi: digit order flag must follow width and base arguments",
                    ));
                }
            } else {
                width = parse_width(first)?;
                idx += 1;
            }
        }

        if let Some(second) = args.get(idx) {
            if let Some(parsed) = parse_order(second, DE2BI_NAME)? {
                order = parsed;
                idx += 1;
            } else {
                base = parse_base(second, DE2BI_NAME)?;
                idx += 1;
            }
        }

        if let Some(third) = args.get(idx) {
            order = parse_required_order(third, DE2BI_NAME)?;
            idx += 1;
        }

        if idx != args.len() {
            return Err(de2bi_error("de2bi: expected at most four inputs"));
        }
        Ok(Self { width, base, order })
    }
}

async fn decimal_to_text_builtin(
    value: Value,
    rest: Vec<Value>,
    builtin: &'static str,
    radix: u32,
    floor_fractional: bool,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: expected D or D,minDigits"),
        ));
    }
    let min_digits = rest
        .first()
        .map(|value| parse_min_digits(value, builtin))
        .transpose()?
        .unwrap_or(0);
    let decimals = decimal_items_from_value(value, builtin, floor_fractional).await?;
    checked_decimal_text_output_len(decimals.len(), 1usize.max(min_digits), builtin)?;
    let mut rows = Vec::with_capacity(decimals.len());
    let mut width = 1usize.max(min_digits);
    for decimal in decimals {
        let row = format_decimal_item(decimal, radix, min_digits, builtin)?;
        width = width.max(row.text.chars().count());
        rows.push(row);
    }
    let output_len = checked_decimal_text_output_len(rows.len(), width, builtin)?;
    let mut chars = Vec::with_capacity(output_len);
    for row in &rows {
        let pad = width.saturating_sub(row.text.chars().count());
        chars.extend(std::iter::repeat_n(row.pad, pad));
        chars.extend(row.text.chars());
    }
    CharArray::new(chars, rows.len(), width)
        .map(Value::CharArray)
        .map_err(|err| conversion_error(builtin, format!("{builtin}: {err}")))
}

fn checked_decimal_text_output_len(
    rows: usize,
    width: usize,
    builtin: &'static str,
) -> BuiltinResult<usize> {
    let len = rows.checked_mul(width).ok_or_else(|| {
        conversion_error(builtin, format!("{builtin}: requested output is too large"))
    })?;
    if len > MAX_DECIMAL_TEXT_OUTPUT_CHARS {
        return Err(conversion_error(
            builtin,
            format!(
                "{builtin}: requested output has {len} characters; limit is {MAX_DECIMAL_TEXT_OUTPUT_CHARS}"
            ),
        ));
    }
    Ok(len)
}

async fn text_to_decimal_builtin(
    value: Value,
    builtin: &'static str,
    radix: u32,
) -> BuiltinResult<Value> {
    let parsed = text_items_from_value(value, builtin)?;
    let values = parsed
        .items
        .iter()
        .map(|text| parse_base_text(text, radix, builtin))
        .collect::<BuiltinResult<Vec<_>>>()?;
    value_from_decimal_text_values(values, parsed.shape, parsed.preserve_shape, parsed.column)
        .map_err(|err| conversion_error(builtin, format!("{builtin}: {err}")))
}

#[derive(Clone, Copy, Debug)]
enum DecimalItem {
    Signed {
        value: i128,
        storage_bits: Option<usize>,
    },
    Unsigned(u128),
}

#[derive(Debug)]
struct FormattedDecimal {
    text: String,
    pad: char,
}

async fn decimal_items_from_value(
    value: Value,
    builtin: &'static str,
    floor_fractional: bool,
) -> BuiltinResult<Vec<DecimalItem>> {
    match value {
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                Ok(decimal_items_from_integer_storage(storage))
            } else {
                tensor
                    .data
                    .into_iter()
                    .map(|value| decimal_item_from_f64(value, builtin, floor_fractional))
                    .collect()
            }
        }
        Value::LogicalArray(logical) => Ok(logical
            .data
            .into_iter()
            .map(|value| DecimalItem::Unsigned(u128::from(value != 0)))
            .collect()),
        Value::CharArray(chars) => Ok(chars
            .data
            .into_iter()
            .map(|ch| DecimalItem::Unsigned(u128::from(ch as u32)))
            .collect()),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            tensor
                .data
                .into_iter()
                .map(|value| decimal_item_from_f64(value, builtin, floor_fractional))
                .collect()
        }
        Value::Num(value) => Ok(vec![decimal_item_from_f64(
            value,
            builtin,
            floor_fractional,
        )?]),
        Value::Int(value) => Ok(vec![decimal_item_from_int(value)]),
        Value::Bool(value) => Ok(vec![DecimalItem::Unsigned(u128::from(value))]),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(conversion_error(
            builtin,
            format!("{builtin}: D must contain real integer values"),
        )),
        other => Err(conversion_error(
            builtin,
            format!("{builtin}: D must be numeric, logical, or char, got {other:?}"),
        )),
    }
}

fn decimal_item_from_int(value: IntValue) -> DecimalItem {
    match value {
        IntValue::I8(value) => DecimalItem::Signed {
            value: i128::from(value),
            storage_bits: Some(8),
        },
        IntValue::I16(value) => DecimalItem::Signed {
            value: i128::from(value),
            storage_bits: Some(16),
        },
        IntValue::I32(value) => DecimalItem::Signed {
            value: i128::from(value),
            storage_bits: Some(32),
        },
        IntValue::I64(value) => DecimalItem::Signed {
            value: i128::from(value),
            storage_bits: Some(64),
        },
        IntValue::U8(value) => DecimalItem::Unsigned(u128::from(value)),
        IntValue::U16(value) => DecimalItem::Unsigned(u128::from(value)),
        IntValue::U32(value) => DecimalItem::Unsigned(u128::from(value)),
        IntValue::U64(value) => DecimalItem::Unsigned(u128::from(value)),
    }
}

fn decimal_items_from_integer_storage(storage: &IntegerStorage) -> Vec<DecimalItem> {
    storage
        .exact_values()
        .into_iter()
        .map(decimal_item_from_int)
        .collect()
}

fn decimal_item_from_f64(
    value: f64,
    builtin: &'static str,
    floor_fractional: bool,
) -> BuiltinResult<DecimalItem> {
    if !value.is_finite() {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: D must contain finite values"),
        ));
    }
    let converted = if floor_fractional {
        value.floor()
    } else {
        let rounded = value.round();
        if (value - rounded).abs() > INTEGER_TOL {
            return Err(conversion_error(
                builtin,
                format!("{builtin}: D must contain integer values"),
            ));
        }
        rounded
    };
    if converted < i64::MIN as f64 || converted >= MAX_FLOAT_DECIMAL_INPUT {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: D is outside the supported integer range"),
        ));
    }
    if converted < 0.0 {
        Ok(DecimalItem::Signed {
            value: converted as i128,
            storage_bits: None,
        })
    } else {
        Ok(DecimalItem::Unsigned(converted as u128))
    }
}

fn format_decimal_item(
    item: DecimalItem,
    radix: u32,
    min_digits: usize,
    builtin: &'static str,
) -> BuiltinResult<FormattedDecimal> {
    match item {
        DecimalItem::Unsigned(value) => Ok(FormattedDecimal {
            text: format_unsigned_digits(value, radix, min_digits, '0'),
            pad: '0',
        }),
        DecimalItem::Signed { value, .. } if value >= 0 => Ok(FormattedDecimal {
            text: format_unsigned_digits(value as u128, radix, min_digits, '0'),
            pad: '0',
        }),
        DecimalItem::Signed {
            value,
            storage_bits,
        } => {
            let bits = storage_bits
                .or_else(|| signed_storage_bits(value))
                .ok_or_else(|| {
                    conversion_error(
                        builtin,
                        format!("{builtin}: D is outside the supported signed integer range"),
                    )
                })?;
            let unsigned = (1u128 << bits) - value.unsigned_abs();
            let min_width = match radix {
                2 => bits,
                16 => bits / 4,
                _ => unreachable!("unsupported radix"),
            };
            let pad = match radix {
                2 => '1',
                16 => 'F',
                _ => unreachable!("unsupported radix"),
            };
            Ok(FormattedDecimal {
                text: format_unsigned_digits(unsigned, radix, min_digits.max(min_width), pad),
                pad,
            })
        }
    }
}

fn signed_storage_bits(value: i128) -> Option<usize> {
    [8usize, 16, 32, 64]
        .into_iter()
        .find(|&bits| value >= -(1i128 << (bits - 1)) && value < (1i128 << (bits - 1)))
}

fn format_unsigned_digits(mut value: u128, radix: u32, min_digits: usize, pad: char) -> String {
    let mut chars = Vec::new();
    if value == 0 {
        chars.push('0');
    } else {
        let radix_u = u128::from(radix);
        while value > 0 {
            let digit = (value % radix_u) as u8;
            chars.push(match digit {
                0..=9 => char::from(b'0' + digit),
                10..=15 => char::from(b'A' + digit - 10),
                _ => unreachable!("unsupported digit"),
            });
            value /= radix_u;
        }
        chars.reverse();
    }
    let pad_count = min_digits.saturating_sub(chars.len());
    let mut out = String::with_capacity(chars.len() + pad_count);
    out.extend(std::iter::repeat_n(pad, pad_count));
    out.extend(chars);
    out
}

fn parse_min_digits(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    let parsed = parse_nonnegative_integer(value, builtin, "minDigits")?;
    if parsed > usize::MAX as u128 {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: minDigits is too large for this platform"),
        ));
    }
    Ok(parsed as usize)
}

#[derive(Debug)]
struct ParsedTextItems {
    items: Vec<String>,
    shape: Vec<usize>,
    preserve_shape: bool,
    column: bool,
}

fn text_items_from_value(value: Value, builtin: &'static str) -> BuiltinResult<ParsedTextItems> {
    match value {
        Value::CharArray(chars) => {
            let mut items = Vec::with_capacity(chars.rows);
            for row in 0..chars.rows {
                let start = row * chars.cols;
                let end = start + chars.cols;
                items.push(chars.data[start..end].iter().collect::<String>());
            }
            Ok(ParsedTextItems {
                items,
                shape: vec![chars.rows, 1],
                preserve_shape: false,
                column: true,
            })
        }
        Value::String(text) => Ok(ParsedTextItems {
            items: vec![text],
            shape: vec![1, 1],
            preserve_shape: true,
            column: false,
        }),
        Value::StringArray(array) => {
            let shape = array.shape.clone();
            Ok(ParsedTextItems {
                items: array.data,
                shape,
                preserve_shape: true,
                column: false,
            })
        }
        Value::Cell(cell) => text_items_from_cell(cell, builtin),
        other => Err(conversion_error(
            builtin,
            format!("{builtin}: input must be text, got {other:?}"),
        )),
    }
}

fn text_items_from_cell(cell: CellArray, builtin: &'static str) -> BuiltinResult<ParsedTextItems> {
    let mut items = Vec::with_capacity(cell.data.len());
    for value in cell.data {
        let Some(text) = value_as_string(&value) else {
            return Err(conversion_error(
                builtin,
                format!("{builtin}: cell array elements must be text"),
            ));
        };
        items.push(text);
    }
    Ok(ParsedTextItems {
        shape: vec![items.len(), 1],
        items,
        preserve_shape: false,
        column: true,
    })
}

fn parse_base_text(text: &str, radix: u32, builtin: &'static str) -> BuiltinResult<f64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: input text must not be empty"),
        ));
    }
    let (digits, suffix) = strip_base_prefix_and_suffix(trimmed, radix);
    if digits.is_empty() {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: input text must contain digits"),
        ));
    }
    if digits.len() > MAX_TEXT_CONVERSION_DIGITS {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: input text has more than {MAX_TEXT_CONVERSION_DIGITS} digits"),
        ));
    }

    let mut value = 0u128;
    for ch in digits.chars() {
        let Some(digit) = ch.to_digit(radix) else {
            return Err(conversion_error(
                builtin,
                format!("{builtin}: input text contains invalid digits"),
            ));
        };
        value = value
            .checked_mul(u128::from(radix))
            .and_then(|acc| acc.checked_add(u128::from(digit)))
            .ok_or_else(|| conversion_error(builtin, format!("{builtin}: decimal overflow")))?;
    }

    if let Some(suffix) = suffix {
        let (bits, signed) = match suffix {
            TextSuffix::Signed(bits) => (bits, true),
            TextSuffix::Unsigned(bits) => (bits, false),
        };
        if bits > 64 {
            return Err(conversion_error(
                builtin,
                format!("{builtin}: suffix width is too large"),
            ));
        }
        let sign_bit = 1u128 << (bits - 1);
        let modulus = 1u128 << bits;
        if value >= modulus {
            return Err(conversion_error(
                builtin,
                format!("{builtin}: value does not fit in suffix width"),
            ));
        }
        if signed && value & sign_bit != 0 {
            let signed = value as i128 - modulus as i128;
            if signed.unsigned_abs() > MAX_EXACT_INTEGER {
                return Err(conversion_error(
                    builtin,
                    format!(
                        "{builtin}: result exceeds the maximum exact integer supported by RunMat"
                    ),
                ));
            }
            return Ok(signed as f64);
        }
        if value > MAX_EXACT_INTEGER {
            return Err(conversion_error(
                builtin,
                format!("{builtin}: result exceeds the maximum exact integer supported by RunMat"),
            ));
        }
        return Ok(value as f64);
    }

    if value > MAX_EXACT_INTEGER {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: result exceeds the maximum exact integer supported by RunMat"),
        ));
    }

    Ok(value as f64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextSuffix {
    Signed(usize),
    Unsigned(usize),
}

fn strip_base_prefix_and_suffix(text: &str, radix: u32) -> (String, Option<TextSuffix>) {
    let mut body = if radix == 2 {
        text.strip_prefix("0b")
            .or_else(|| text.strip_prefix("0B"))
            .unwrap_or(text)
    } else {
        text.strip_prefix("0x")
            .or_else(|| text.strip_prefix("0X"))
            .unwrap_or(text)
    };
    let mut suffix_bits = None;
    for suffix in ["s8", "s16", "s32", "s64"] {
        if let Some(stripped) = body
            .strip_suffix(suffix)
            .or_else(|| body.strip_suffix(&suffix.to_uppercase()))
        {
            suffix_bits = suffix[1..].parse::<usize>().ok().map(TextSuffix::Signed);
            body = stripped;
            break;
        }
    }
    if suffix_bits.is_none() {
        for suffix in ["u8", "u16", "u32", "u64"] {
            if let Some(stripped) = body
                .strip_suffix(suffix)
                .or_else(|| body.strip_suffix(&suffix.to_uppercase()))
            {
                suffix_bits = suffix[1..].parse::<usize>().ok().map(TextSuffix::Unsigned);
                body = stripped;
                break;
            }
        }
    }
    (body.to_string(), suffix_bits)
}

fn value_from_decimal_text_values(
    values: Vec<f64>,
    shape: Vec<usize>,
    preserve_shape: bool,
    column: bool,
) -> Result<Value, String> {
    if values.len() == 1 {
        return Ok(Value::Num(values[0]));
    }
    if preserve_shape {
        return Tensor::new(values, shape).map(Value::Tensor);
    }
    if values.len() == 1 && !column {
        Ok(Value::Num(values[0]))
    } else {
        let rows = values.len();
        Tensor::new(values, vec![rows, 1]).map(Value::Tensor)
    }
}

async fn digit_matrix_from_value(
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<DigitMatrix> {
    match value {
        Value::Tensor(tensor) => digit_matrix_from_tensor(tensor, builtin),
        Value::LogicalArray(logical) => digit_matrix_from_logical(logical, builtin),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            digit_matrix_from_tensor(tensor, builtin)
        }
        Value::Num(n) => Ok(DigitMatrix {
            data: vec![number_to_digit(n, builtin, "B")?],
            rows: 1,
            cols: 1,
        }),
        Value::Int(i) => Ok(DigitMatrix {
            data: vec![integer_to_digit(&i, builtin, "B")?],
            rows: 1,
            cols: 1,
        }),
        Value::Bool(b) => Ok(DigitMatrix {
            data: vec![usize::from(b)],
            rows: 1,
            cols: 1,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(bi2de_error("bi2de: B must contain real digits"))
        }
        other => Err(build_runtime_error(format!(
            "{builtin}: B must be a numeric or logical matrix, got {other:?}"
        ))
        .with_builtin(builtin)
        .build()),
    }
}

fn digit_matrix_from_tensor(tensor: Tensor, builtin: &'static str) -> BuiltinResult<DigitMatrix> {
    ensure_matrix_shape(&tensor.shape, builtin, "B")?;
    let data = if let Some(storage) = tensor.integer_storage() {
        integer_storage_to_digits(storage, builtin, "B")?
    } else {
        tensor
            .data
            .into_iter()
            .map(|value| number_to_digit(value, builtin, "B"))
            .collect::<BuiltinResult<Vec<_>>>()?
    };
    Ok(DigitMatrix {
        data,
        rows: tensor.rows,
        cols: tensor.cols,
    })
}

fn digit_matrix_from_logical(
    logical: LogicalArray,
    builtin: &'static str,
) -> BuiltinResult<DigitMatrix> {
    ensure_matrix_shape(&logical.shape, builtin, "B")?;
    let rows = logical.shape.first().copied().unwrap_or(1);
    let cols = logical.shape.get(1).copied().unwrap_or(logical.data.len());
    Ok(DigitMatrix {
        data: logical.data.into_iter().map(usize::from).collect(),
        rows,
        cols,
    })
}

async fn decimal_vector_from_value(value: Value) -> BuiltinResult<Vec<u128>> {
    match value {
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                integer_storage_to_nonnegative(storage, DE2BI_NAME, "D")
            } else {
                tensor
                    .data
                    .into_iter()
                    .map(|value| number_to_decimal(value, "D"))
                    .collect()
            }
        }
        Value::LogicalArray(logical) => Ok(logical.data.into_iter().map(u128::from).collect()),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            tensor
                .data
                .into_iter()
                .map(|value| number_to_decimal(value, "D"))
                .collect()
        }
        Value::Num(n) => Ok(vec![number_to_decimal(n, "D")?]),
        Value::Int(i) => Ok(vec![integer_to_nonnegative(&i, DE2BI_NAME, "D")?]),
        Value::Bool(b) => Ok(vec![if b { 1 } else { 0 }]),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(de2bi_error(
            "de2bi: D must contain real nonnegative integers",
        )),
        other => Err(de2bi_error(format!(
            "de2bi: D must be numeric or logical, got {other:?}"
        ))),
    }
}

fn ensure_matrix_shape(shape: &[usize], builtin: &'static str, name: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(
            build_runtime_error(format!("{builtin}: {name} must be a vector or 2-D matrix"))
                .with_builtin(builtin)
                .build(),
        )
    }
}

fn validate_digit_matrix(
    digits: &DigitMatrix,
    base: usize,
    builtin: &'static str,
) -> BuiltinResult<()> {
    for &digit in &digits.data {
        if digit >= base {
            return Err(build_runtime_error(format!(
                "{builtin}: digit values must be integers in the range [0, {}]",
                base - 1
            ))
            .with_builtin(builtin)
            .build());
        }
    }
    Ok(())
}

fn row_digits_to_decimal(
    digits: &DigitMatrix,
    row: usize,
    base: usize,
    order: DigitOrder,
) -> BuiltinResult<u128> {
    let base_u = base as u128;
    let mut acc = 0u128;
    match order {
        DigitOrder::LeftMsb => {
            for col in 0..digits.cols {
                let digit = digits.data[row + col * digits.rows] as u128;
                acc = acc
                    .checked_mul(base_u)
                    .and_then(|value| value.checked_add(digit))
                    .ok_or_else(|| bi2de_error("bi2de: decimal value overflow"))?;
                ensure_exact_range(acc, BI2DE_NAME)?;
            }
        }
        DigitOrder::RightMsb => {
            let mut place = 1u128;
            for col in 0..digits.cols {
                let digit = digits.data[row + col * digits.rows] as u128;
                let term = digit
                    .checked_mul(place)
                    .ok_or_else(|| bi2de_error("bi2de: decimal value overflow"))?;
                acc = acc
                    .checked_add(term)
                    .ok_or_else(|| bi2de_error("bi2de: decimal value overflow"))?;
                ensure_exact_range(acc, BI2DE_NAME)?;
                if col + 1 < digits.cols {
                    place = place
                        .checked_mul(base_u)
                        .ok_or_else(|| bi2de_error("bi2de: decimal value overflow"))?;
                    ensure_exact_range(place, BI2DE_NAME)?;
                }
            }
        }
    }
    Ok(acc)
}

fn write_digits(
    mut value: u128,
    base: usize,
    width: usize,
    order: DigitOrder,
    out: &mut [f64],
    row: usize,
    rows: usize,
) {
    let base_u = base as u128;
    for offset in 0..width {
        let digit = (value % base_u) as f64;
        value /= base_u;
        let col = match order {
            DigitOrder::RightMsb => offset,
            DigitOrder::LeftMsb => width - 1 - offset,
        };
        out[row + col * rows] = digit;
    }
}

fn digits_required(mut value: u128, base: usize) -> usize {
    if value == 0 {
        return 1;
    }
    let base_u = base as u128;
    let mut count = 0usize;
    while value > 0 {
        value /= base_u;
        count += 1;
    }
    count
}

fn checked_de2bi_output_len(rows: usize, width: usize) -> BuiltinResult<usize> {
    let len = rows
        .checked_mul(width)
        .ok_or_else(|| de2bi_error("de2bi: requested output exceeds the maximum supported size"))?;
    if len > MAX_DE2BI_OUTPUT_ELEMENTS {
        return Err(de2bi_error(format!(
            "de2bi: requested output has {len} elements; limit is {MAX_DE2BI_OUTPUT_ELEMENTS}"
        )));
    }
    Ok(len)
}

fn parse_width(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_numeric(value) {
        return Ok(None);
    }
    let width = parse_nonnegative_integer(value, DE2BI_NAME, "N")?;
    if width > usize::MAX as u128 {
        return Err(de2bi_error("de2bi: N is too large for this platform"));
    }
    Ok(Some(width as usize))
}

fn parse_base(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    let base = parse_nonnegative_integer(value, builtin, "P")?;
    if base < 2 {
        return Err(
            build_runtime_error(format!("{builtin}: P must be an integer greater than 1"))
                .with_builtin(builtin)
                .build(),
        );
    }
    if base > MAX_EXACT_INTEGER || base > usize::MAX as u128 {
        return Err(
            build_runtime_error(format!("{builtin}: P is too large for this platform"))
                .with_builtin(builtin)
                .build(),
        );
    }
    Ok(base as usize)
}

fn parse_nonnegative_integer(
    value: &Value,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<u128> {
    if let Value::Int(integer) = value {
        return integer_to_nonnegative(integer, builtin, name);
    }
    if let Value::Tensor(tensor) = value {
        if tensor_utils::is_scalar_tensor(tensor) {
            if let Some(storage) = tensor.integer_storage() {
                let integer = storage.value_at(0).ok_or_else(|| {
                    build_runtime_error(format!("{builtin}: {name} must be a scalar"))
                        .with_builtin(builtin)
                        .build()
                })?;
                return integer_to_nonnegative(&integer, builtin, name);
            }
        }
    }
    let n = scalar_number(value, builtin, name)?;
    if !n.is_finite() {
        return Err(
            build_runtime_error(format!("{builtin}: {name} must be finite"))
                .with_builtin(builtin)
                .build(),
        );
    }
    let rounded = n.round();
    if (n - rounded).abs() > INTEGER_TOL || rounded < 0.0 {
        return Err(build_runtime_error(format!(
            "{builtin}: {name} must be a nonnegative integer"
        ))
        .with_builtin(builtin)
        .build());
    }
    if rounded > MAX_EXACT_INTEGER as f64 {
        return Err(build_runtime_error(format!(
            "{builtin}: {name} exceeds the maximum exact integer supported by RunMat"
        ))
        .with_builtin(builtin)
        .build());
    }
    Ok(rounded as u128)
}

fn scalar_number(value: &Value, builtin: &'static str, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(
            build_runtime_error(format!("{builtin}: {name} must be a scalar"))
                .with_builtin(builtin)
                .build(),
        ),
    }
}

fn number_to_digit(value: f64, builtin: &'static str, name: &str) -> BuiltinResult<usize> {
    let integer = parse_nonnegative_integer(&Value::Num(value), builtin, name)?;
    if integer > usize::MAX as u128 {
        return Err(build_runtime_error(format!(
            "{builtin}: {name} value is too large for this platform"
        ))
        .with_builtin(builtin)
        .build());
    }
    Ok(integer as usize)
}

fn integer_to_digit(value: &IntValue, builtin: &'static str, name: &str) -> BuiltinResult<usize> {
    let integer = integer_to_nonnegative(value, builtin, name)?;
    if integer > usize::MAX as u128 {
        return Err(build_runtime_error(format!(
            "{builtin}: {name} value is too large for this platform"
        ))
        .with_builtin(builtin)
        .build());
    }
    Ok(integer as usize)
}

fn integer_storage_to_digits(
    storage: &IntegerStorage,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<Vec<usize>> {
    storage
        .exact_values()
        .iter()
        .map(|value| integer_to_digit(value, builtin, name))
        .collect()
}

fn integer_storage_to_nonnegative(
    storage: &IntegerStorage,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<Vec<u128>> {
    storage
        .exact_values()
        .iter()
        .map(|value| integer_to_nonnegative(value, builtin, name))
        .collect()
}

fn integer_to_nonnegative(
    value: &IntValue,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<u128> {
    value.try_to_u64().map(u128::from).ok_or_else(|| {
        build_runtime_error(format!("{builtin}: {name} must be a nonnegative integer"))
            .with_builtin(builtin)
            .build()
    })
}

fn number_to_decimal(value: f64, name: &str) -> BuiltinResult<u128> {
    parse_nonnegative_integer(&Value::Num(value), DE2BI_NAME, name)
}

fn ensure_exact_range(value: u128, builtin: &'static str) -> BuiltinResult<()> {
    if value > MAX_EXACT_INTEGER {
        Err(build_runtime_error(format!(
            "{builtin}: result exceeds the maximum exact integer supported by RunMat"
        ))
        .with_builtin(builtin)
        .build())
    } else {
        Ok(())
    }
}

fn parse_order(value: &Value, builtin: &'static str) -> BuiltinResult<Option<DigitOrder>> {
    let Some(text) = value_as_string(value) else {
        return Ok(None);
    };
    match normalize_flag(&text).as_str() {
        "rightmsb" => Ok(Some(DigitOrder::RightMsb)),
        "leftmsb" => Ok(Some(DigitOrder::LeftMsb)),
        other => Err(
            build_runtime_error(format!("{builtin}: unsupported digit order '{other}'"))
                .with_builtin(builtin)
                .build(),
        ),
    }
}

fn parse_required_order(value: &Value, builtin: &'static str) -> BuiltinResult<DigitOrder> {
    parse_order(value, builtin)?.ok_or_else(|| {
        build_runtime_error(format!("{builtin}: digit order flag must be a string"))
            .with_builtin(builtin)
            .build()
    })
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn normalize_flag(flag: &str) -> String {
    flag.chars()
        .filter(|ch| *ch != '-' && *ch != '_' && !ch.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor_utils::tensor_element_len(tensor) == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::StringArray;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).expect("tensor"))
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    fn bi2de(value: Value, rest: Vec<Value>) -> Value {
        block_on(super::bi2de_builtin(value, rest)).expect("bi2de")
    }

    fn de2bi(value: Value, rest: Vec<Value>) -> Tensor {
        match block_on(super::de2bi_builtin(value, rest)).expect("de2bi") {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    fn dec2bin(value: Value, rest: Vec<Value>) -> CharArray {
        match block_on(super::dec2bin_builtin(value, rest)).expect("dec2bin") {
            Value::CharArray(chars) => chars,
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    fn dec2hex(value: Value, rest: Vec<Value>) -> CharArray {
        match block_on(super::dec2hex_builtin(value, rest)).expect("dec2hex") {
            Value::CharArray(chars) => chars,
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    fn bin2dec(value: Value) -> Value {
        block_on(super::bin2dec_builtin(value)).expect("bin2dec")
    }

    fn hex2dec(value: Value) -> Value {
        block_on(super::hex2dec_builtin(value)).expect("hex2dec")
    }

    fn value_data(value: Value) -> Vec<f64> {
        match value {
            Value::Num(n) => vec![n],
            Value::Tensor(tensor) => tensor.data,
            other => panic!("expected numeric output, got {other:?}"),
        }
    }

    fn char_rows(chars: &CharArray) -> Vec<String> {
        (0..chars.rows)
            .map(|row| {
                let start = row * chars.cols;
                let end = start + chars.cols;
                chars.data[start..end].iter().collect()
            })
            .collect()
    }

    #[test]
    fn bi2de_row_vector_defaults_to_right_msb() {
        assert_eq!(
            value_data(bi2de(tensor(vec![1.0, 0.0, 1.0, 0.0], vec![1, 4]), vec![])),
            vec![5.0]
        );
    }

    #[test]
    fn bi2de_left_msb_flag() {
        assert_eq!(
            value_data(bi2de(
                tensor(vec![1.0, 0.0, 1.0, 0.0], vec![1, 4]),
                vec![Value::from("left-msb")]
            )),
            vec![10.0]
        );
    }

    #[test]
    fn bi2de_converts_each_matrix_row() {
        let out = bi2de(
            tensor(
                vec![
                    1.0, 0.0, 1.0, // first column
                    0.0, 1.0, 1.0, // second column
                    1.0, 1.0, 0.0, // third column
                ],
                vec![3, 3],
            ),
            vec![],
        );
        let Value::Tensor(tensor) = out else {
            panic!("expected column vector")
        };
        assert_eq!(tensor.shape, vec![3, 1]);
        assert_eq!(tensor.data, vec![5.0, 6.0, 3.0]);
    }

    #[test]
    fn bi2de_accepts_arbitrary_base() {
        assert_eq!(
            value_data(bi2de(
                tensor(vec![2.0, 1.0, 0.0], vec![1, 3]),
                vec![Value::Num(3.0)]
            )),
            vec![5.0]
        );
        assert_eq!(
            value_data(bi2de(
                tensor(vec![2.0, 1.0, 0.0], vec![1, 3]),
                vec![Value::Num(3.0), Value::from("left-msb")]
            )),
            vec![21.0]
        );
    }

    #[test]
    fn bi2de_rejects_digits_outside_base() {
        let err = block_on(super::bi2de_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            vec![],
        ))
        .expect_err("invalid binary digit should fail");
        assert!(err.to_string().contains("digit values must be integers"));
    }

    #[test]
    fn de2bi_uses_minimum_width_right_msb() {
        let out = de2bi(tensor(vec![0.0, 1.0, 2.0, 5.0], vec![1, 4]), vec![]);
        assert_eq!(out.shape, vec![4, 3]);
        assert_eq!(
            out.data,
            vec![
                0.0, 1.0, 0.0, 1.0, // 2^0 column
                0.0, 0.0, 1.0, 0.0, // 2^1 column
                0.0, 0.0, 0.0, 1.0, // 2^2 column
            ]
        );
    }

    #[test]
    fn de2bi_respects_width_base_and_left_msb() {
        let out = de2bi(
            tensor(vec![5.0, 21.0], vec![2, 1]),
            vec![Value::Num(4.0), Value::Num(3.0), Value::from("left-msb")],
        );
        assert_eq!(out.shape, vec![2, 4]);
        assert_eq!(
            out.data,
            vec![
                0.0, 0.0, // 3^3 column
                0.0, 2.0, // 3^2 column
                1.0, 1.0, // 3^1 column
                2.0, 0.0, // 3^0 column
            ]
        );
    }

    #[test]
    fn de2bi_round_trips_with_bi2de() {
        let decimals = tensor(vec![0.0, 1.0, 2.0, 7.0, 15.0], vec![5, 1]);
        let bits = de2bi(
            decimals,
            vec![Value::Num(4.0), Value::Num(2.0), Value::from("right-msb")],
        );
        let back = bi2de(Value::Tensor(bits), vec![Value::Num(2.0)]);
        let Value::Tensor(back) = back else {
            panic!("expected tensor")
        };
        assert_eq!(back.data, vec![0.0, 1.0, 2.0, 7.0, 15.0]);
    }

    #[test]
    fn de2bi_accepts_empty_width_placeholder_with_base_and_order() {
        let out = de2bi(
            tensor(vec![5.0, 21.0], vec![2, 1]),
            vec![
                tensor(vec![], vec![0, 0]),
                Value::Num(3.0),
                Value::from("left-msb"),
            ],
        );
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(
            out.data,
            vec![
                0.0, 2.0, // 3^2 column
                1.0, 1.0, // 3^1 column
                2.0, 0.0, // 3^0 column
            ]
        );
    }

    #[test]
    fn de2bi_rejects_too_small_width() {
        let err = block_on(super::de2bi_builtin(Value::Num(8.0), vec![Value::Num(3.0)]))
            .expect_err("width too small should fail");
        assert!(err.to_string().contains("N is too small"));
    }

    #[test]
    fn de2bi_rejects_excessive_output_before_allocation() {
        let err = block_on(super::de2bi_builtin(
            Value::Num(0.0),
            vec![Value::Num((MAX_DE2BI_OUTPUT_ELEMENTS + 1) as f64)],
        ))
        .expect_err("oversized output should fail");
        assert!(err.to_string().contains("requested output"));
    }

    #[test]
    fn dec2bin_converts_scalars_vectors_and_min_digits() {
        assert_eq!(char_rows(&dec2bin(Value::Num(23.0), vec![])), vec!["10111"]);
        assert_eq!(
            char_rows(&dec2bin(Value::Num(23.0), vec![Value::Num(8.0)])),
            vec!["00010111"]
        );
        assert_eq!(
            char_rows(&dec2bin(Value::Num(0.0), vec![Value::Num(0.0)])),
            vec!["0"]
        );
        assert_eq!(
            char_rows(&dec2bin(
                tensor(vec![1023.0, 122.0, 14.0], vec![1, 3]),
                vec![]
            )),
            vec!["1111111111", "0001111010", "0000001110"]
        );
    }

    #[test]
    fn dec2bin_floors_fractional_numeric_input() {
        assert_eq!(char_rows(&dec2bin(Value::Num(12.5), vec![])), vec!["1100"]);
        assert_eq!(
            char_rows(&dec2bin(Value::Num(-12.5), vec![])),
            vec!["11110011"]
        );
    }

    #[test]
    fn dec2bin_uses_twos_complement_for_negative_values() {
        assert_eq!(
            char_rows(&dec2bin(Value::Num(-1.0), vec![])),
            vec!["11111111"]
        );
        assert_eq!(
            char_rows(&dec2bin(Value::Num(-16.0), vec![])),
            vec!["11110000"]
        );
        assert_eq!(
            char_rows(&dec2bin(Value::Num(-16.0), vec![Value::Num(12.0)])),
            vec!["111111110000"]
        );
        assert_eq!(
            char_rows(&dec2bin(
                tensor(vec![-1.0, 1.0], vec![1, 2]),
                vec![Value::Num(4.0)]
            )),
            vec!["11111111", "00000001"]
        );
    }

    #[test]
    fn dec2bin_preserves_signed_integer_storage_width() {
        assert_eq!(
            char_rows(&dec2bin(Value::Int(IntValue::I16(-1)), vec![])),
            vec!["1111111111111111"]
        );
        assert_eq!(
            char_rows(&dec2hex(Value::Int(IntValue::I32(-1)), vec![])),
            vec!["FFFFFFFF"]
        );
    }

    #[test]
    fn dec2bin_rejects_floating_two_to_sixty_four() {
        let err = block_on(super::dec2bin_builtin(
            Value::Num(MAX_FLOAT_DECIMAL_INPUT),
            vec![],
        ))
        .expect_err("2^64 should fail");
        assert!(err
            .to_string()
            .contains("outside the supported integer range"));
    }

    #[test]
    fn dec2bin_rejects_excessive_output_before_allocation() {
        let err = block_on(super::dec2bin_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            vec![Value::Num((MAX_DECIMAL_TEXT_OUTPUT_CHARS + 1) as f64)],
        ))
        .expect_err("oversized output should fail");
        assert!(err.to_string().contains("requested output"));
    }

    #[test]
    fn dec2hex_converts_scalars_vectors_and_min_digits() {
        assert_eq!(char_rows(&dec2hex(Value::Num(1023.0), vec![])), vec!["3FF"]);
        assert_eq!(
            char_rows(&dec2hex(Value::Num(1023.0), vec![Value::Num(6.0)])),
            vec!["0003FF"]
        );
        assert_eq!(
            char_rows(&dec2hex(
                tensor(vec![1023.0, 122.0, 14.0], vec![1, 3]),
                vec![]
            )),
            vec!["3FF", "07A", "00E"]
        );
    }

    #[test]
    fn dec2hex_uses_twos_complement_and_requires_integers() {
        assert_eq!(char_rows(&dec2hex(Value::Num(-1.0), vec![])), vec!["FF"]);
        assert_eq!(char_rows(&dec2hex(Value::Num(-16.0), vec![])), vec!["F0"]);
        assert_eq!(
            char_rows(&dec2hex(Value::Num(-16.0), vec![Value::Num(4.0)])),
            vec!["FFF0"]
        );

        let err = block_on(super::dec2hex_builtin(Value::Num(12.5), vec![]))
            .expect_err("fractional dec2hex input should fail");
        assert!(err.to_string().contains("integer values"));
    }

    #[test]
    fn bin2dec_accepts_scalar_text_char_matrices_and_string_arrays() {
        assert_eq!(
            value_data(bin2dec(Value::CharArray(CharArray::new_row("10111")))),
            vec![23.0]
        );

        let chars = CharArray::new("111111111100011110100000001110".chars().collect(), 3, 10)
            .expect("char matrix");
        let Value::Tensor(out) = bin2dec(Value::CharArray(chars)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![3, 1]);
        assert_eq!(out.data, vec![1023.0, 122.0, 14.0]);

        let strings = StringArray::new(
            vec!["1111111111".into(), "1111010".into(), "1110".into()],
            vec![1, 3],
        )
        .expect("string array");
        let Value::Tensor(out) = bin2dec(Value::StringArray(strings)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.data, vec![1023.0, 122.0, 14.0]);
    }

    #[test]
    fn bin2dec_accepts_prefixes_suffixes_and_cell_text() {
        assert_eq!(value_data(bin2dec(Value::from("0b111"))), vec![7.0]);
        assert_eq!(value_data(bin2dec(Value::from("0b111s32"))), vec![7.0]);
        assert_eq!(value_data(bin2dec(Value::from("0b11111111s8"))), vec![-1.0]);
        assert_eq!(
            value_data(bin2dec(Value::from(
                "0b1111111111111111111111111111111111111111111111111111111111111111s64"
            ))),
            vec![-1.0]
        );

        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("101")),
                Value::from("0b1000"),
            ],
            1,
            2,
        )
        .expect("cell");
        let Value::Tensor(out) = bin2dec(Value::Cell(cell)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.data, vec![5.0, 8.0]);
    }

    #[test]
    fn bin2dec_rejects_invalid_digits_and_inexact_results() {
        let err = block_on(super::bin2dec_builtin(Value::from("102")))
            .expect_err("invalid digit should fail");
        assert!(err.to_string().contains("invalid digits"));

        let err = block_on(super::bin2dec_builtin(Value::from("10_10")))
            .expect_err("underscore separators should fail");
        assert!(err.to_string().contains("invalid digits"));

        let err = block_on(super::bin2dec_builtin(Value::from("0b100000000u8")))
            .expect_err("unsigned suffix overflow should fail");
        assert!(err.to_string().contains("does not fit in suffix width"));

        let err = block_on(super::bin2dec_builtin(Value::from(
            "100000000000000000000000000000000000000000000000000000",
        )))
        .expect_err("inexact binary value should fail");
        assert!(err
            .to_string()
            .contains("maximum exact integer supported by RunMat"));
    }

    #[test]
    fn hex2dec_accepts_text_shapes_prefixes_and_suffixes() {
        assert_eq!(
            value_data(hex2dec(Value::CharArray(CharArray::new_row("3FF")))),
            vec![1023.0]
        );

        let strings = StringArray::new(vec!["3FF".into(), "7A".into(), "E".into()], vec![1, 3])
            .expect("string array");
        let Value::Tensor(out) = hex2dec(Value::StringArray(strings)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.data, vec![1023.0, 122.0, 14.0]);

        assert_eq!(value_data(hex2dec(Value::from("0xFF"))), vec![255.0]);
        assert_eq!(value_data(hex2dec(Value::from("0xFFs32"))), vec![255.0]);
        assert_eq!(value_data(hex2dec(Value::from("0xFFs8"))), vec![-1.0]);
    }

    #[test]
    fn hex2dec_rejects_invalid_digits() {
        let err = block_on(super::hex2dec_builtin(Value::from("0x1G")))
            .expect_err("invalid digit should fail");
        assert!(err.to_string().contains("invalid digits"));

        let err = block_on(super::hex2dec_builtin(Value::from("0x100u8")))
            .expect_err("unsigned suffix overflow should fail");
        assert!(err.to_string().contains("does not fit in suffix width"));

        let err = block_on(super::hex2dec_builtin(Value::from("0x20000000000000")))
            .expect_err("inexact hex value should fail");
        assert!(err
            .to_string()
            .contains("maximum exact integer supported by RunMat"));
    }

    #[test]
    fn bi2de_and_de2bi_scalar_integers_remain_exact() {
        assert_eq!(
            integer_to_nonnegative(&IntValue::U64(u64::MAX), DE2BI_NAME, "D").unwrap(),
            u128::from(u64::MAX)
        );
        assert_eq!(
            block_on(decimal_vector_from_value(Value::Int(IntValue::U64(
                u64::MAX
            ))))
            .unwrap(),
            vec![u128::from(u64::MAX)]
        );
        let digits = block_on(digit_matrix_from_value(
            Value::Int(IntValue::U8(1)),
            BI2DE_NAME,
        ))
        .unwrap();
        assert_eq!(digits.data, vec![1]);
        assert!(integer_to_nonnegative(&IntValue::I8(-1), DE2BI_NAME, "D").is_err());
        assert!(block_on(decimal_vector_from_value(Value::Int(IntValue::I8(-1)))).is_err());
    }

    #[test]
    fn bi2de_and_de2bi_typed_integer_tensors_remain_exact() {
        let digits = integer_tensor(IntegerStorage::U64(vec![1, 0, 1]), vec![1, 3]);
        assert_eq!(value_data(bi2de(digits, vec![])), vec![5.0]);

        let mut width = Tensor::new_integer(IntegerStorage::U16(vec![12]), vec![1, 1]).unwrap();
        width.data.clear();
        assert_eq!(
            scalar_number(&Value::Tensor(width), DE2BI_NAME, "N").unwrap(),
            12.0
        );

        let mut empty = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();
        empty.data = vec![1.0];
        assert!(is_empty_numeric(&Value::Tensor(empty)));

        assert_eq!(
            block_on(decimal_vector_from_value(integer_tensor(
                IntegerStorage::U64(vec![u64::MAX]),
                vec![1, 1],
            )))
            .unwrap(),
            vec![u128::from(u64::MAX)]
        );

        let err = block_on(super::de2bi_builtin(
            integer_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
            vec![],
        ))
        .expect_err("negative signed integer tensor should fail");
        assert!(err.to_string().contains("nonnegative integer"));
    }

    #[test]
    fn dec2bin_and_dec2hex_typed_integer_tensors_preserve_storage_width() {
        assert_eq!(
            char_rows(&dec2bin(
                integer_tensor(IntegerStorage::I16(vec![-1, 1]), vec![1, 2]),
                vec![],
            )),
            vec!["1111111111111111", "0000000000000001"]
        );
        assert_eq!(
            char_rows(&dec2hex(
                integer_tensor(IntegerStorage::I32(vec![-1, 255]), vec![1, 2]),
                vec![],
            )),
            vec!["FFFFFFFF", "000000FF"]
        );
        assert_eq!(
            char_rows(&dec2hex(
                integer_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]),
                vec![],
            )),
            vec!["FFFFFFFFFFFFFFFF"]
        );
    }
}
