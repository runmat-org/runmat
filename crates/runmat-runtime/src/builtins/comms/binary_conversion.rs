//! MATLAB-compatible binary and hexadecimal conversion helpers.

use runmat_builtins::{
    BuiltinErrorDescriptor, BI2DE_ERROR_BASE, BI2DE_ERROR_INPUT, BI2DE_ERROR_ORDER,
    BI2DE_ERROR_PROVIDER, BI2DE_ERROR_RANGE, BI2DE_GPU_INPUT_EXTENSION,
    BI2DE_INTEGER_BASE_EXTENSION, DE2BI_ERROR_BASE, DE2BI_ERROR_INPUT, DE2BI_ERROR_ORDER,
    DE2BI_ERROR_PROVIDER, DE2BI_ERROR_RESOURCE, DE2BI_ERROR_WIDTH, DE2BI_GPU_INPUT_EXTENSION,
    DE2BI_INTEGER_CONTROL_EXTENSION, DE2BI_LOGICAL_INPUT_EXTENSION, DE2BI_ND_INPUT_EXTENSION,
    DE2BI_WIDE_INTEGER_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericStorage,
    Tensor, Value,
};

use crate::builtins::common::{gpu_helpers, tensor as tensor_utils};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod typed_output;
use typed_output::{digit_decimal_result, digit_output_storage};

const BI2DE_NAME: &str = "bi2de";
const DE2BI_NAME: &str = "de2bi";
const DEC2BIN_NAME: &str = "dec2bin";
const DEC2HEX_NAME: &str = "dec2hex";
const BIN2DEC_NAME: &str = "bin2dec";
const HEX2DEC_NAME: &str = "hex2dec";
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DigitConversion {
    Bi2de,
    De2bi,
}

#[derive(Clone, Copy)]
enum DigitError {
    Bi2deInput,
    Bi2deBase,
    Bi2deOrder,
    Bi2deRange,
    De2biInput,
    De2biBase,
    De2biOrder,
    De2biWidth,
    De2biResource,
}

impl DigitConversion {
    fn name(self) -> &'static str {
        match self {
            Self::Bi2de => BI2DE_NAME,
            Self::De2bi => DE2BI_NAME,
        }
    }

    fn base_error(self) -> DigitError {
        match self {
            Self::Bi2de => DigitError::Bi2deBase,
            Self::De2bi => DigitError::De2biBase,
        }
    }

    fn order_error(self) -> DigitError {
        match self {
            Self::Bi2de => DigitError::Bi2deOrder,
            Self::De2bi => DigitError::De2biOrder,
        }
    }
}

impl DigitError {
    fn operation(self) -> DigitConversion {
        match self {
            Self::Bi2deInput | Self::Bi2deBase | Self::Bi2deOrder | Self::Bi2deRange => {
                DigitConversion::Bi2de
            }
            Self::De2biInput
            | Self::De2biBase
            | Self::De2biOrder
            | Self::De2biWidth
            | Self::De2biResource => DigitConversion::De2bi,
        }
    }

    fn descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Bi2deInput => &BI2DE_ERROR_INPUT,
            Self::Bi2deBase => &BI2DE_ERROR_BASE,
            Self::Bi2deOrder => &BI2DE_ERROR_ORDER,
            Self::Bi2deRange => &BI2DE_ERROR_RANGE,
            Self::De2biInput => &DE2BI_ERROR_INPUT,
            Self::De2biBase => &DE2BI_ERROR_BASE,
            Self::De2biOrder => &DE2BI_ERROR_ORDER,
            Self::De2biWidth => &DE2BI_ERROR_WIDTH,
            Self::De2biResource => &DE2BI_ERROR_RESOURCE,
        }
    }
}

fn digit_error(error: DigitError, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(error.operation().name());
    if let Some(identifier) = error.descriptor().identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Copy)]
enum NumericParseOwner {
    Digit(DigitError),
    Text(&'static str),
}

const BI2DE_INPUT_OWNER: NumericParseOwner = NumericParseOwner::Digit(DigitError::Bi2deInput);
const DE2BI_INPUT_OWNER: NumericParseOwner = NumericParseOwner::Digit(DigitError::De2biInput);

impl NumericParseOwner {
    fn name(self) -> &'static str {
        match self {
            Self::Digit(error) => error.operation().name(),
            Self::Text(name) => name,
        }
    }

    fn error(self, message: impl Into<String>) -> RuntimeError {
        match self {
            Self::Digit(error) => digit_error(error, message),
            Self::Text(name) => conversion_error(name, message),
        }
    }
}

#[derive(Debug)]
struct DigitMatrix {
    data: Vec<usize>,
    rows: usize,
    cols: usize,
    dtype: Option<NumericDType>,
}

fn bi2de_error(message: impl Into<String>) -> RuntimeError {
    digit_error(DigitError::Bi2deInput, message)
}

fn de2bi_error(message: impl Into<String>) -> RuntimeError {
    digit_error(DigitError::De2biInput, message)
}

fn conversion_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextConversion {
    Dec2Bin,
    Dec2Hex,
    Bin2Dec,
    Hex2Dec,
}

impl TextConversion {
    const fn name(self) -> &'static str {
        match self {
            Self::Dec2Bin => DEC2BIN_NAME,
            Self::Dec2Hex => DEC2HEX_NAME,
            Self::Bin2Dec => BIN2DEC_NAME,
            Self::Hex2Dec => HEX2DEC_NAME,
        }
    }

    const fn radix(self) -> u32 {
        match self {
            Self::Dec2Bin | Self::Bin2Dec => 2,
            Self::Dec2Hex | Self::Hex2Dec => 16,
        }
    }
}

#[runtime_builtin(
    name = "bi2de",
    binding_variant = "default",
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn bi2de_builtin(b: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let opts = Bi2deOptions::parse(rest.view())?;
    if rest.first().is_some_and(is_fixed_width_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BI2DE_INTEGER_BASE_EXTENSION,
            BI2DE_NAME,
        )?;
    }
    if let Value::GpuTensor(handle) = &b {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BI2DE_GPU_INPUT_EXTENSION,
            BI2DE_NAME,
        )?;
        if runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
        {
            return Err(bi2de_error("bi2de: B must contain real digits"));
        }
    }
    let digits = digit_matrix_from_value(b, BI2DE_NAME).await?;
    validate_digit_matrix(&digits, opts.base, BI2DE_NAME)?;

    let mut out = Vec::with_capacity(digits.rows);
    for row in 0..digits.rows {
        let value = row_digits_to_decimal(&digits, row, opts.base, opts.order)?;
        out.push(value as f64);
    }

    digit_decimal_result(out, digits.dtype)
}

#[runtime_builtin(
    name = "de2bi",
    binding_variant = "default",
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn de2bi_builtin(d: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let opts = De2biOptions::parse(rest.view())?;
    if rest.iter().any(is_fixed_width_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DE2BI_INTEGER_CONTROL_EXTENSION,
            DE2BI_NAME,
        )?;
    }
    if matches!(&d, Value::Tensor(tensor) if tensor.shape().len() > 2)
        || matches!(&d, Value::GpuTensor(handle) if handle.shape().len() > 2)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DE2BI_ND_INPUT_EXTENSION,
            DE2BI_NAME,
        )?;
    }
    if matches!(d, Value::Bool(_) | Value::LogicalArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DE2BI_LOGICAL_INPUT_EXTENSION,
            DE2BI_NAME,
        )?;
    }
    if let Value::GpuTensor(handle) = &d {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DE2BI_GPU_INPUT_EXTENSION,
            DE2BI_NAME,
        )?;
        if runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
        {
            return Err(de2bi_error(
                "de2bi: D must contain real nonnegative integers",
            ));
        }
    }
    let (decimals, dtype) = decimal_vector_from_value(d).await?;
    if decimals.iter().any(|&value| value > MAX_EXACT_INTEGER) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DE2BI_WIDE_INTEGER_EXTENSION,
            DE2BI_NAME,
        )?;
    }
    let needed_width = decimals
        .iter()
        .copied()
        .map(|value| digits_required(value, opts.base))
        .max()
        .unwrap_or(0);
    let width = opts.width.unwrap_or(needed_width);
    if opts.width.is_some() && width < needed_width {
        return Err(digit_error(
            DigitError::De2biWidth,
            format!(
                "de2bi: N is too small to represent an input value in base {}",
                opts.base
            ),
        ));
    }

    let rows = decimals.len();
    let output_len = checked_de2bi_output_len(rows, width)?;
    let mut out = vec![0.0; output_len];
    for (row, value) in decimals.into_iter().enumerate() {
        write_digits(value, opts.base, width, opts.order, &mut out, row, rows);
    }

    let storage = digit_output_storage(dtype, out);
    Ok(Value::Tensor(
        Tensor::from_numeric_storage(storage, vec![rows, width])
            .map_err(|err| de2bi_error(format!("de2bi: {err}")))?,
    ))
}

fn is_fixed_width_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.numeric_dtype().integer_class().is_some(),
        _ => false,
    }
}

#[runtime_builtin(
    name = "dec2bin",
    binding_variant = "default",
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn dec2bin_builtin(d: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    decimal_to_text_builtin(d, rest, TextConversion::Dec2Bin).await
}

#[runtime_builtin(
    name = "dec2hex",
    binding_variant = "default",
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn dec2hex_builtin(d: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    decimal_to_text_builtin(d, rest, TextConversion::Dec2Hex).await
}

#[runtime_builtin(
    name = "bin2dec",
    binding_variant = "default",
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn bin2dec_builtin(bin_str: Value) -> BuiltinResult<Value> {
    text_to_decimal_builtin(bin_str, TextConversion::Bin2Dec).await
}

#[runtime_builtin(
    name = "hex2dec",
    binding_variant = "default",
    builtin_path = "crate::builtins::comms::binary_conversion"
)]
async fn hex2dec_builtin(hex_str: Value) -> BuiltinResult<Value> {
    text_to_decimal_builtin(hex_str, TextConversion::Hex2Dec).await
}

#[derive(Debug)]
struct Bi2deOptions {
    base: usize,
    order: DigitOrder,
}

impl Bi2deOptions {
    fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        let mut base = 2usize;
        let mut order = DigitOrder::RightMsb;
        match (args.len(), args.get(0), args.get(1)) {
            (0, _, _) => {}
            (1, Some(one), _) => {
                if let Some(parsed) = parse_order(one, DigitConversion::Bi2de)? {
                    order = parsed;
                } else {
                    base = parse_base(one, DigitConversion::Bi2de)?;
                }
            }
            (2, Some(base_value), Some(flag)) => {
                base = parse_base(base_value, DigitConversion::Bi2de)?;
                order = parse_required_order(flag, DigitConversion::Bi2de)?;
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
    fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        let mut width = None;
        let mut base = 2usize;
        let mut order = DigitOrder::RightMsb;
        let mut idx = 0usize;

        if let Some(first) = args.get(idx) {
            if let Some(parsed) = parse_order(first, DigitConversion::De2bi)? {
                order = parsed;
                idx += 1;
                if idx != args.len() {
                    return Err(digit_error(
                        DigitError::De2biOrder,
                        "de2bi: digit order flag must follow width and base arguments",
                    ));
                }
            } else {
                width = parse_width(first)?;
                idx += 1;
            }
        }

        if let Some(second) = args.get(idx) {
            if let Some(parsed) = parse_order(second, DigitConversion::De2bi)? {
                order = parsed;
                idx += 1;
            } else {
                base = parse_base(second, DigitConversion::De2bi)?;
                idx += 1;
            }
        }

        if let Some(third) = args.get(idx) {
            order = parse_required_order(third, DigitConversion::De2bi)?;
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
    rest: crate::call::BuiltinArguments,
    conversion: TextConversion,
) -> BuiltinResult<Value> {
    let builtin = conversion.name();
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
    let decimals = decimal_items_from_value(
        value,
        builtin,
        matches!(conversion, TextConversion::Dec2Bin),
    )
    .await?;
    checked_decimal_text_output_len(decimals.len(), 1usize.max(min_digits), builtin)?;
    let mut rows = Vec::with_capacity(decimals.len());
    let mut width = 1usize.max(min_digits);
    for decimal in decimals {
        let row = format_decimal_item(decimal, conversion.radix(), min_digits, builtin)?;
        width = width.max(row.text.encode_utf16().count());
        rows.push(row);
    }
    let output_len = checked_decimal_text_output_len(rows.len(), width, builtin)?;
    let mut chars = Vec::with_capacity(output_len);
    for row in &rows {
        let pad = width.saturating_sub(row.text.encode_utf16().count());
        chars.extend(std::iter::repeat_n(
            // Radix formatting admits only ASCII padding digits.
            runmat_value::Utf16CodeUnit::from_raw(row.pad as u16),
            pad,
        ));
        chars.extend(
            row.text
                .encode_utf16()
                .map(runmat_value::Utf16CodeUnit::from_raw),
        );
    }
    CharArray::from_code_units(chars, rows.len(), width)
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

async fn text_to_decimal_builtin(value: Value, conversion: TextConversion) -> BuiltinResult<Value> {
    let builtin = conversion.name();
    let parsed = text_items_from_value(value, builtin)?;
    let values = parsed
        .items
        .iter()
        .map(|text| {
            parse_base_text(
                text.as_str().ok_or_else(|| {
                    conversion_error(
                        builtin,
                        format!("{builtin}: input text must contain digits"),
                    )
                })?,
                conversion,
            )
        })
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
        Value::Tensor(tensor) => decimal_items_from_tensor(tensor, builtin, floor_fractional),
        Value::LogicalArray(logical) => Ok(logical
            .data
            .into_iter()
            .map(|value| DecimalItem::Unsigned(u128::from(value != 0)))
            .collect()),
        Value::CharArray(chars) => Ok(chars
            .iter_column_major()
            .map(|ch| DecimalItem::Unsigned(u128::from(ch.raw())))
            .collect()),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            decimal_items_from_tensor(tensor, builtin, floor_fractional)
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

fn decimal_items_from_tensor(
    tensor: Tensor,
    builtin: &'static str,
    floor_fractional: bool,
) -> BuiltinResult<Vec<DecimalItem>> {
    match tensor
        .into_numeric_storage()
        .map_err(|err| conversion_error(builtin, format!("{builtin}: {err}")))?
    {
        NumericStorage::F64(values) => values
            .into_iter()
            .map(|value| decimal_item_from_f64(value, builtin, floor_fractional))
            .collect(),
        NumericStorage::F32(values) => values
            .into_iter()
            .map(|value| decimal_item_from_f64(f64::from(value), builtin, floor_fractional))
            .collect(),
        storage => Ok(decimal_items_from_integer_storage(
            &storage
                .into_integer_storage()
                .expect("non-floating numeric storage is integer"),
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
        if value.fract() != 0.0 {
            return Err(conversion_error(
                builtin,
                format!("{builtin}: D must contain integer values"),
            ));
        }
        value
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
    let parsed = parse_nonnegative_integer(value, NumericParseOwner::Text(builtin), "minDigits")?;
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
    items: runmat_value::ExactArrayStorage<runmat_value::StringElement>,
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
                items.push(
                    runmat_value::decode_utf16_code_units(&chars.data()[start..end])
                        .map_err(|source| {
                            build_runtime_error(format!("{builtin}: invalid UTF-16 text: {source}"))
                                .with_builtin(builtin)
                                .with_source(source)
                                .build()
                        })?
                        .into(),
                );
            }
            Ok(ParsedTextItems {
                items: runmat_value::ExactArrayStorage::unbounded(items),
                shape: vec![chars.rows, 1],
                preserve_shape: false,
                column: true,
            })
        }
        Value::String(text) => Ok(ParsedTextItems {
            items: runmat_value::ExactArrayStorage::unbounded(vec![text]),
            shape: vec![1, 1],
            preserve_shape: true,
            column: false,
        }),
        Value::StringArray(array) => {
            let shape = array.shape.to_vec();
            Ok(ParsedTextItems {
                items: array.into_parts().0,
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
    let mut items = Vec::with_capacity(cell.data().len());
    for value in cell.data() {
        let Some(text) = value_as_string(value, builtin)? else {
            return Err(conversion_error(
                builtin,
                format!("{builtin}: cell array elements must be text"),
            ));
        };
        items.push(text.into());
    }
    Ok(ParsedTextItems {
        shape: vec![items.len(), 1],
        items: runmat_value::ExactArrayStorage::unbounded(items),
        preserve_shape: false,
        column: true,
    })
}

fn parse_base_text(text: &str, conversion: TextConversion) -> BuiltinResult<f64> {
    let builtin = conversion.name();
    let radix = conversion.radix();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(conversion_error(
            builtin,
            format!("{builtin}: input text must not be empty"),
        ));
    }
    let (digits, suffix) = strip_base_prefix_and_suffix(trimmed, conversion);
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

fn strip_base_prefix_and_suffix(
    text: &str,
    conversion: TextConversion,
) -> (String, Option<TextSuffix>) {
    let mut body = match conversion {
        TextConversion::Bin2Dec => text
            .strip_prefix("0b")
            .or_else(|| text.strip_prefix("0B"))
            .unwrap_or(text),
        TextConversion::Hex2Dec => text
            .strip_prefix("0x")
            .or_else(|| text.strip_prefix("0X"))
            .unwrap_or(text),
        TextConversion::Dec2Bin | TextConversion::Dec2Hex => text,
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
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|cause| {
                    build_runtime_error("bi2de: failed to gather resident digits")
                        .with_builtin(BI2DE_NAME)
                        .with_identifier(
                            BI2DE_ERROR_PROVIDER
                                .identifier
                                .expect("provider identifier"),
                        )
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .with_source(cause)
                        .build()
                })?;
            digit_matrix_from_tensor(tensor, builtin)
        }
        Value::Num(n) => Ok(DigitMatrix {
            data: vec![number_to_digit(n, "B")?],
            rows: 1,
            cols: 1,
            dtype: Some(NumericDType::F64),
        }),
        Value::Int(i) => {
            let dtype = i.numeric_dtype();
            Ok(DigitMatrix {
                data: vec![integer_to_digit(&i, "B")?],
                rows: 1,
                cols: 1,
                dtype: Some(dtype),
            })
        }
        Value::Bool(b) => Ok(DigitMatrix {
            data: vec![usize::from(b)],
            rows: 1,
            cols: 1,
            dtype: None,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(bi2de_error("bi2de: B must contain real digits"))
        }
        other => Err(bi2de_error(format!(
            "{builtin}: B must be a numeric or logical matrix, got {other:?}"
        ))),
    }
}

fn digit_matrix_from_tensor(tensor: Tensor, builtin: &'static str) -> BuiltinResult<DigitMatrix> {
    ensure_matrix_shape(tensor.shape(), builtin, "B")?;
    let (rows, cols) = matrix_extents(tensor.shape());
    let dtype = tensor.numeric_dtype();
    let data = match tensor
        .into_numeric_storage()
        .map_err(|err| bi2de_error(format!("{builtin}: {err}")))?
    {
        NumericStorage::F64(values) => values
            .into_iter()
            .map(|value| number_to_digit(value, "B"))
            .collect::<BuiltinResult<Vec<_>>>()?,
        NumericStorage::F32(values) => values
            .into_iter()
            .map(|value| number_to_digit(f64::from(value), "B"))
            .collect::<BuiltinResult<Vec<_>>>()?,
        storage => integer_storage_to_digits(
            &storage
                .into_integer_storage()
                .expect("non-floating numeric storage is integer"),
            "B",
        )?,
    };
    Ok(DigitMatrix {
        data,
        rows,
        cols,
        dtype: Some(dtype),
    })
}

fn matrix_extents(shape: &[usize]) -> (usize, usize) {
    match shape {
        [rows, cols, ..] => (*rows, *cols),
        [cols] => (1, *cols),
        [] => (0, 0),
    }
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
        dtype: None,
    })
}

async fn decimal_vector_from_value(value: Value) -> BuiltinResult<(Vec<u128>, NumericDType)> {
    match value {
        Value::Tensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            Ok((decimal_vector_from_tensor(tensor)?, dtype))
        }
        Value::LogicalArray(logical) => Ok((
            logical.data.into_iter().map(u128::from).collect(),
            NumericDType::F64,
        )),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|cause| {
                    build_runtime_error("de2bi: failed to gather resident decimals")
                        .with_builtin(DE2BI_NAME)
                        .with_identifier(
                            DE2BI_ERROR_PROVIDER
                                .identifier
                                .expect("provider identifier"),
                        )
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .with_source(cause)
                        .build()
                })?;
            let dtype = tensor.numeric_dtype();
            Ok((decimal_vector_from_tensor(tensor)?, dtype))
        }
        Value::Num(n) => Ok((vec![number_to_decimal(n, "D")?], NumericDType::F64)),
        Value::Int(i) => Ok((
            vec![integer_to_nonnegative(&i, DE2BI_INPUT_OWNER, "D")?],
            i.numeric_dtype(),
        )),
        Value::Bool(b) => Ok((vec![u128::from(b)], NumericDType::F64)),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(de2bi_error(
            "de2bi: D must contain real nonnegative integers",
        )),
        other => Err(de2bi_error(format!(
            "de2bi: D must be numeric or logical, got {other:?}"
        ))),
    }
}

fn decimal_vector_from_tensor(tensor: Tensor) -> BuiltinResult<Vec<u128>> {
    match tensor
        .into_numeric_storage()
        .map_err(|err| de2bi_error(format!("de2bi: {err}")))?
    {
        NumericStorage::F64(values) => values
            .into_iter()
            .map(|value| number_to_decimal(value, "D"))
            .collect(),
        NumericStorage::F32(values) => values
            .into_iter()
            .map(|value| number_to_decimal(f64::from(value), "D"))
            .collect(),
        storage => integer_storage_to_nonnegative(
            &storage
                .into_integer_storage()
                .expect("non-floating numeric storage is integer"),
            "D",
        ),
    }
}

fn ensure_matrix_shape(shape: &[usize], builtin: &'static str, name: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(bi2de_error(format!(
            "{builtin}: {name} must be a vector or 2-D matrix"
        )))
    }
}

fn validate_digit_matrix(
    digits: &DigitMatrix,
    base: usize,
    builtin: &'static str,
) -> BuiltinResult<()> {
    for &digit in &digits.data {
        if digit >= base {
            return Err(bi2de_error(format!(
                "{builtin}: digit values must be integers in the range [0, {}]",
                base - 1
            )));
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
                    .ok_or_else(|| {
                        digit_error(DigitError::Bi2deRange, "bi2de: decimal value overflow")
                    })?;
                ensure_exact_range(acc, BI2DE_NAME)?;
            }
        }
        DigitOrder::RightMsb => {
            // Read most-significant columns first. Horner evaluation ignores
            // leading zero columns without overflowing a positional multiplier.
            for col in (0..digits.cols).rev() {
                let digit = digits.data[row + col * digits.rows] as u128;
                acc = acc
                    .checked_mul(base_u)
                    .and_then(|value| value.checked_add(digit))
                    .ok_or_else(|| {
                        digit_error(DigitError::Bi2deRange, "bi2de: decimal value overflow")
                    })?;
                ensure_exact_range(acc, BI2DE_NAME)?;
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
    let len = rows.checked_mul(width).ok_or_else(|| {
        digit_error(
            DigitError::De2biResource,
            "de2bi: requested output exceeds the maximum supported size",
        )
    })?;
    if len > MAX_DE2BI_OUTPUT_ELEMENTS {
        return Err(digit_error(
            DigitError::De2biResource,
            format!(
                "de2bi: requested output has {len} elements; limit is {MAX_DE2BI_OUTPUT_ELEMENTS}"
            ),
        ));
    }
    Ok(len)
}

fn parse_width(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_numeric(value) {
        return Ok(None);
    }
    let width =
        parse_nonnegative_integer(value, NumericParseOwner::Digit(DigitError::De2biWidth), "N")?;
    if width == 0 {
        return Err(digit_error(
            DigitError::De2biWidth,
            "de2bi: N must be a positive integer",
        ));
    }
    if width > usize::MAX as u128 {
        return Err(digit_error(
            DigitError::De2biWidth,
            "de2bi: N is too large for this platform",
        ));
    }
    Ok(Some(width as usize))
}

fn parse_base(value: &Value, operation: DigitConversion) -> BuiltinResult<usize> {
    let builtin = operation.name();
    let base =
        parse_nonnegative_integer(value, NumericParseOwner::Digit(operation.base_error()), "P")?;
    if base < 2 {
        return Err(digit_error(
            operation.base_error(),
            format!("{builtin}: P must be an integer greater than 1"),
        ));
    }
    if base > MAX_EXACT_INTEGER || base > usize::MAX as u128 {
        return Err(digit_error(
            operation.base_error(),
            format!("{builtin}: P is too large for this platform"),
        ));
    }
    Ok(base as usize)
}

fn parse_nonnegative_integer(
    value: &Value,
    owner: NumericParseOwner,
    name: &str,
) -> BuiltinResult<u128> {
    let builtin = owner.name();
    if let Value::Int(integer) = value {
        return integer_to_nonnegative(integer, owner, name);
    }
    if let Value::Tensor(tensor) = value {
        if tensor_utils::is_scalar_tensor(tensor) {
            if let Some(storage) = tensor.integer_storage() {
                let integer = storage
                    .value_at(0)
                    .ok_or_else(|| owner.error(format!("{builtin}: {name} must be a scalar")))?;
                return integer_to_nonnegative(&integer, owner, name);
            }
        }
    }
    let n = scalar_number(value, owner, name)?;
    if !n.is_finite() {
        return Err(owner.error(format!("{builtin}: {name} must be finite")));
    }
    if n.fract() != 0.0 || n < 0.0 {
        return Err(owner.error(format!("{builtin}: {name} must be a nonnegative integer")));
    }
    if n > MAX_EXACT_INTEGER as f64 {
        return Err(owner.error(format!(
            "{builtin}: {name} exceeds the maximum exact integer supported by RunMat"
        )));
    }
    Ok(n as u128)
}

fn scalar_number(value: &Value, owner: NumericParseOwner, name: &str) -> BuiltinResult<f64> {
    let builtin = owner.name();
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
        _ => Err(owner.error(format!("{builtin}: {name} must be a scalar"))),
    }
}

fn number_to_digit(value: f64, name: &str) -> BuiltinResult<usize> {
    let integer = parse_nonnegative_integer(&Value::Num(value), BI2DE_INPUT_OWNER, name)?;
    if integer > usize::MAX as u128 {
        return Err(BI2DE_INPUT_OWNER.error(format!(
            "bi2de: {name} value is too large for this platform"
        )));
    }
    Ok(integer as usize)
}

fn integer_to_digit(value: &IntValue, name: &str) -> BuiltinResult<usize> {
    let integer = integer_to_nonnegative(value, BI2DE_INPUT_OWNER, name)?;
    if integer > usize::MAX as u128 {
        return Err(BI2DE_INPUT_OWNER.error(format!(
            "bi2de: {name} value is too large for this platform"
        )));
    }
    Ok(integer as usize)
}

fn integer_storage_to_digits(storage: &IntegerStorage, name: &str) -> BuiltinResult<Vec<usize>> {
    storage
        .exact_values()
        .iter()
        .map(|value| integer_to_digit(value, name))
        .collect()
}

fn integer_storage_to_nonnegative(
    storage: &IntegerStorage,
    name: &str,
) -> BuiltinResult<Vec<u128>> {
    storage
        .exact_values()
        .iter()
        .map(|value| integer_to_nonnegative(value, DE2BI_INPUT_OWNER, name))
        .collect()
}

fn integer_to_nonnegative(
    value: &IntValue,
    owner: NumericParseOwner,
    name: &str,
) -> BuiltinResult<u128> {
    let builtin = owner.name();
    value
        .try_to_u64()
        .map(u128::from)
        .ok_or_else(|| owner.error(format!("{builtin}: {name} must be a nonnegative integer")))
}

fn number_to_decimal(value: f64, name: &str) -> BuiltinResult<u128> {
    parse_nonnegative_integer(&Value::Num(value), DE2BI_INPUT_OWNER, name)
}

fn ensure_exact_range(value: u128, builtin: &'static str) -> BuiltinResult<()> {
    if value > MAX_EXACT_INTEGER {
        Err(digit_error(
            DigitError::Bi2deRange,
            format!("{builtin}: result exceeds the maximum exact integer supported by RunMat"),
        ))
    } else {
        Ok(())
    }
}

fn parse_order(value: &Value, operation: DigitConversion) -> BuiltinResult<Option<DigitOrder>> {
    let builtin = operation.name();
    let Some(text) = value_as_string(value, builtin)? else {
        return Ok(None);
    };
    match normalize_flag(&text).as_str() {
        "rightmsb" => Ok(Some(DigitOrder::RightMsb)),
        "leftmsb" => Ok(Some(DigitOrder::LeftMsb)),
        other => Err(digit_error(
            operation.order_error(),
            format!("{builtin}: unsupported digit order '{other}'"),
        )),
    }
}

fn parse_required_order(value: &Value, operation: DigitConversion) -> BuiltinResult<DigitOrder> {
    parse_order(value, operation)?.ok_or_else(|| {
        digit_error(
            operation.order_error(),
            format!("{}: digit order flag must be a string", operation.name()),
        )
    })
}

fn value_as_string(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Option<runmat_value::Utf8Text>> {
    Ok(match value {
        Value::String(s) => s.text_ref().cloned(),
        Value::StringArray(sa) if sa.data().len() == 1 => sa.data()[0].text_ref().cloned(),
        Value::CharArray(ca) if ca.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(ca.data())
                .map_err(|source| {
                    build_runtime_error(format!("{builtin}: invalid UTF-16 text: {source}"))
                        .with_builtin(builtin)
                        .with_source(source)
                        .build()
                })?
                .into(),
        ),
        _ => None,
    })
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
    use runmat_value::StringArray;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).expect("tensor"))
    }

    fn single_tensor(data: Vec<f32>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::from_f32(data, shape).expect("single tensor"))
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    fn bi2de(value: Value, rest: Vec<Value>) -> Value {
        block_on(super::bi2de_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect("bi2de")
    }

    fn de2bi(value: Value, rest: Vec<Value>) -> Tensor {
        match block_on(super::de2bi_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect("de2bi")
        {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    fn dec2bin(value: Value, rest: Vec<Value>) -> CharArray {
        match block_on(super::dec2bin_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect("dec2bin")
        {
            Value::CharArray(chars) => chars,
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    fn dec2hex(value: Value, rest: Vec<Value>) -> CharArray {
        match block_on(super::dec2hex_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect("dec2hex")
        {
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
            Value::Int(integer) => vec![integer.to_f64()],
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected numeric output, got {other:?}"),
        }
    }

    #[test]
    fn paired_digit_conversions_preserve_numeric_input_classes() {
        let single = single_tensor(vec![1.0, 0.0, 1.0], vec![1, 3]);
        let Value::Tensor(single_result) = bi2de(single, vec![]) else {
            panic!("single result must retain single storage");
        };
        assert_eq!(single_result.numeric_dtype(), NumericDType::F32);
        assert_eq!(single_result.materialize_f64(), vec![5.0]);

        assert_eq!(
            bi2de(
                integer_tensor(IntegerStorage::U8(vec![1, 0, 1]), vec![1, 3]),
                vec![]
            ),
            Value::Int(IntValue::U8(5))
        );
        assert_eq!(
            bi2de(
                integer_tensor(IntegerStorage::U8(vec![1; 9]), vec![1, 9]),
                vec![]
            ),
            Value::Int(IntValue::U16(511))
        );
        let digits = de2bi(
            integer_tensor(IntegerStorage::I16(vec![2, 3]), vec![2, 1]),
            vec![Value::Num(3.0)],
        );
        assert_eq!(digits.numeric_dtype(), NumericDType::I16);
        assert_eq!(digits.materialize_f64(), vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let unsigned = de2bi(Value::Int(IntValue::U64(u64::MAX)), vec![]);
        assert_eq!(unsigned.numeric_dtype(), NumericDType::U64);
        assert_eq!(unsigned.shape(), &[1, 64]);
        assert_eq!(unsigned.materialize_f64().iter().sum::<f64>(), 64.0);
        let nd = de2bi(tensor(vec![1.0, 2.0], vec![1, 1, 2]), vec![]);
        assert_eq!(nd.shape(), &[2, 2]);
        assert_eq!(nd.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn low_rank_digit_inputs_follow_conversion_matrix_geometry() {
        let Value::Tensor(empty) = bi2de(tensor(vec![1.0], vec![]), vec![]) else {
            panic!("rank-zero digit geometry has no matrix rows")
        };
        assert_eq!(empty.shape(), &[0, 1]);
        assert!(empty.materialize_f64().is_empty());
        assert_eq!(
            bi2de(tensor(vec![1.0, 0.0, 1.0], vec![3]), vec![]),
            Value::Num(5.0)
        );
        for shape in [vec![], vec![1]] {
            let digits = de2bi(tensor(vec![5.0], shape), vec![]);
            assert_eq!(digits.shape(), &[1, 3]);
            assert_eq!(digits.materialize_f64(), vec![1.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn empty_and_nd_digit_shapes_are_explicit() {
        let empty_digits = bi2de(tensor(vec![], vec![0, 3]), vec![]);
        let Value::Tensor(empty_decimals) = empty_digits else {
            panic!("empty digit matrix should return an empty column");
        };
        assert_eq!(empty_decimals.shape(), &[0, 1]);
        let empty_expansion = de2bi(tensor(vec![], vec![0, 3]), vec![]);
        assert_eq!(empty_expansion.shape(), &[0, 0]);
        let err = block_on(super::bi2de_builtin(
            tensor(vec![1.0, 0.0], vec![1, 1, 2]),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("bi2de rejects N-D digit arrays");
        assert!(err.to_string().contains("vector or 2-D matrix"));
    }

    #[test]
    fn extension_forms_reject_in_matlab_mode_before_conversion() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(super::bi2de_builtin(
            tensor(vec![1.0, 0.0, 1.0], vec![1, 3]),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(2),
            )]),
        ))
        .expect_err("typed base should be mode gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:Bi2deIntegerBaseExtension")
        );

        let err = block_on(super::de2bi_builtin(
            Value::Int(IntValue::U64(u64::MAX)),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("wide integer should be mode gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:De2biWideIntegerExtension")
        );

        let err = block_on(super::de2bi_builtin(
            Value::Bool(true),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("logical data should be mode gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:De2biLogicalInputExtension")
        );

        let err = block_on(super::de2bi_builtin(
            tensor(vec![1.0, 2.0], vec![1, 1, 2]),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("N-D input should be mode gated");
        assert_eq!(err.identifier(), DE2BI_ND_INPUT_EXTENSION.error_identifier);
    }

    #[test]
    fn resident_conversion_extension_rejects_before_provider_download() {
        use crate::builtins::common::{gpu_helpers, test_support};

        test_support::with_test_provider(|provider| {
            let digits = Tensor::new(vec![1.0, 0.0, 1.0], vec![1, 3]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &digits).expect("upload");
            provider.reset_telemetry();
            let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
            let bi_error = block_on(super::bi2de_builtin(
                Value::GpuTensor(handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect_err("resident input must reject");
            assert_eq!(
                bi_error.identifier(),
                BI2DE_GPU_INPUT_EXTENSION.error_identifier
            );
            let de_error = block_on(super::de2bi_builtin(
                Value::GpuTensor(handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect_err("resident input must reject");
            assert_eq!(
                de_error.identifier(),
                DE2BI_GPU_INPUT_EXTENSION.error_identifier
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            drop(_matlab);
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert_eq!(
                value_data(
                    block_on(super::bi2de_builtin(
                        Value::GpuTensor(handle.clone()),
                        crate::builtins::common::test_support::builtin_arguments(vec![]),
                    ))
                    .expect("RunMat resident digit conversion")
                ),
                vec![5.0]
            );
            let expanded = block_on(super::de2bi_builtin(
                Value::GpuTensor(handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("RunMat resident decimal conversion");
            assert_eq!(value_data(expanded), vec![1.0, 0.0, 1.0]);
            assert!(provider.telemetry_snapshot().download_bytes > 0);
            handle.release().expect("release fixture handle");
        });
    }

    fn char_rows(chars: &CharArray) -> Vec<String> {
        (0..chars.rows)
            .map(|row| {
                let start = row * chars.cols;
                let end = start + chars.cols;
                runmat_value::decode_utf16_code_units(&chars.data()[start..end])
                    .expect("valid text fixture")
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
    fn bi2de_long_zero_padding_does_not_overflow() {
        let mut right_msb = vec![0.0; 140];
        right_msb[0] = 1.0;
        assert_eq!(
            value_data(bi2de(tensor(right_msb, vec![1, 140]), vec![])),
            vec![1.0]
        );

        let mut left_msb = vec![0.0; 140];
        left_msb[139] = 1.0;
        assert_eq!(
            value_data(bi2de(
                tensor(left_msb, vec![1, 140]),
                vec![Value::String("left-msb".to_string().into())],
            )),
            vec![1.0]
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
        assert_eq!(tensor.shape(), &[3, 1]);
        assert_eq!(tensor.materialize_f64(), vec![5.0, 6.0, 3.0]);
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
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("invalid binary digit should fail");
        assert!(err.to_string().contains("digit values must be integers"));
        assert_eq!(err.identifier(), BI2DE_ERROR_INPUT.identifier);
    }

    #[test]
    fn digit_conversion_errors_have_typed_operation_and_role_owners() {
        let digits = || tensor(vec![1.0], vec![1, 1]);
        for (input, controls, descriptor) in [
            (digits(), vec![Value::Num(1.0)], &BI2DE_ERROR_BASE),
            (digits(), vec![Value::from("sideways")], &BI2DE_ERROR_ORDER),
            (
                tensor(vec![1.0; 54], vec![1, 54]),
                vec![],
                &BI2DE_ERROR_RANGE,
            ),
        ] {
            let error = block_on(super::bi2de_builtin(
                input,
                crate::builtins::common::test_support::builtin_arguments(controls),
            ))
            .expect_err("bi2de domain error");
            assert_eq!(error.identifier(), descriptor.identifier);
        }
        for (input, controls, descriptor) in [
            (Value::Num(-1.0), vec![], &DE2BI_ERROR_INPUT),
            (Value::Num(0.0), vec![Value::Num(0.0)], &DE2BI_ERROR_WIDTH),
            (
                Value::Num(1.0),
                vec![tensor(vec![], vec![0, 0]), Value::Num(1.0)],
                &DE2BI_ERROR_BASE,
            ),
            (
                Value::Num(1.0),
                vec![Value::from("sideways")],
                &DE2BI_ERROR_ORDER,
            ),
            (
                Value::Num(0.0),
                vec![Value::Num((MAX_DE2BI_OUTPUT_ELEMENTS + 1) as f64)],
                &DE2BI_ERROR_RESOURCE,
            ),
        ] {
            let error = block_on(super::de2bi_builtin(
                input,
                crate::builtins::common::test_support::builtin_arguments(controls),
            ))
            .expect_err("de2bi domain error");
            assert_eq!(error.identifier(), descriptor.identifier);
        }
    }

    #[test]
    fn de2bi_uses_minimum_width_right_msb() {
        let out = de2bi(tensor(vec![0.0, 1.0, 2.0, 5.0], vec![1, 4]), vec![]);
        assert_eq!(out.shape(), &[4, 3]);
        assert_eq!(
            out.materialize_f64(),
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
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(
            out.materialize_f64(),
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
        assert_eq!(back.materialize_f64(), vec![0.0, 1.0, 2.0, 7.0, 15.0]);
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
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            out.materialize_f64(),
            vec![
                0.0, 2.0, // 3^2 column
                1.0, 1.0, // 3^1 column
                2.0, 0.0, // 3^0 column
            ]
        );
    }

    #[test]
    fn de2bi_rejects_too_small_width() {
        let err = block_on(super::de2bi_builtin(
            Value::Num(8.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
        ))
        .expect_err("width too small should fail");
        assert!(err.to_string().contains("N is too small"));
    }

    #[test]
    fn digit_conversions_reject_zero_width_and_fractional_values_exactly() {
        let width = block_on(super::de2bi_builtin(
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
        ))
        .expect_err("explicit zero width is invalid even for zero input");
        assert!(width.to_string().contains("N must be a positive integer"));

        let near_integer = 2.0 + 5e-10;
        let base = block_on(super::bi2de_builtin(
            tensor(vec![1.0], vec![1, 1]),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                near_integer,
            )]),
        ))
        .expect_err("fractional base is invalid");
        assert!(base.to_string().contains("nonnegative integer"));

        let width = block_on(super::de2bi_builtin(
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0 + 5e-10)]),
        ))
        .expect_err("fractional width is invalid");
        assert!(width.to_string().contains("nonnegative integer"));

        let data = block_on(super::de2bi_builtin(
            Value::Num(1.0 + 5e-10),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("fractional data is invalid");
        assert!(data.to_string().contains("nonnegative integer"));
    }

    #[test]
    fn de2bi_rejects_excessive_output_before_allocation() {
        let err = block_on(super::de2bi_builtin(
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                (MAX_DE2BI_OUTPUT_ELEMENTS + 1) as f64,
            )]),
        ))
        .expect_err("oversized output should fail");
        assert!(err.to_string().contains("requested output"));
    }

    #[test]
    fn dec2bin_converts_scalars_vectors_and_min_digits() {
        let raw = CharArray::from_code_units(
            [1, 2, 0xD800, 0xFFFF]
                .into_iter()
                .map(runmat_value::Utf16CodeUnit::from_raw)
                .collect(),
            2,
            2,
        )
        .unwrap();
        assert_eq!(
            char_rows(&dec2hex(Value::CharArray(raw), vec![])),
            ["0001", "D800", "0002", "FFFF"]
        );
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
    fn logical_min_digits_controls_are_admitted_for_both_radices() {
        for width in [
            Value::Bool(false),
            Value::Bool(true),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).expect("logical scalar")),
        ] {
            assert_eq!(
                char_rows(&dec2bin(Value::Num(23.0), vec![width.clone()])),
                vec!["10111"]
            );
            assert_eq!(
                char_rows(&dec2hex(Value::Num(23.0), vec![width])),
                vec!["17"]
            );
        }
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
            crate::builtins::common::test_support::builtin_arguments(vec![]),
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
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                (MAX_DECIMAL_TEXT_OUTPUT_CHARS + 1) as f64,
            )]),
        ))
        .expect_err("oversized output should fail");
        assert!(err.to_string().contains("requested output"));
    }

    #[test]
    fn decimal_text_conversion_flattens_nd_and_preserves_empty_rows() {
        let nd = dec2bin(tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]), vec![]);
        assert_eq!(char_rows(&nd), vec!["001", "010", "011", "100"]);
        let empty = dec2hex(tensor(vec![], vec![0, 3]), vec![]);
        assert_eq!((empty.rows, empty.cols), (0, 1));
        assert!(empty.data().is_empty());
    }

    #[test]
    fn invalid_decimal_text_control_rejects_before_resident_download() {
        use crate::builtins::common::{gpu_helpers, test_support};

        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![23.0], vec![1, 1]).expect("fixture tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            provider.reset_telemetry();
            for conversion in [TextConversion::Dec2Bin, TextConversion::Dec2Hex] {
                for invalid_width in [
                    Value::Num(-1.0),
                    Value::LogicalArray(
                        LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical vector"),
                    ),
                ] {
                    let error = block_on(decimal_to_text_builtin(
                        Value::GpuTensor(handle.clone()),
                        crate::builtins::common::test_support::builtin_arguments(vec![
                            invalid_width,
                        ]),
                        conversion,
                    ))
                    .expect_err("invalid width must reject");
                    assert!(error.to_string().contains("minDigits"));
                    assert_eq!(error.context.builtin.as_deref(), Some(conversion.name()));
                }
            }
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert!(block_on(decimal_to_text_builtin(
                Value::GpuTensor(handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)]),
                TextConversion::Dec2Bin,
            ))
            .is_ok());
            assert!(provider.telemetry_snapshot().download_bytes > 0);
            handle.release().expect("release fixture handle");
        });
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

        let err = block_on(super::dec2hex_builtin(
            Value::Num(12.5),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("fractional dec2hex input should fail");
        assert!(err.to_string().contains("integer values"));
    }

    #[test]
    fn bin2dec_accepts_scalar_text_char_matrices_and_string_arrays() {
        assert_eq!(
            value_data(bin2dec(Value::CharArray(CharArray::new_row("10111")))),
            vec![23.0]
        );

        let chars = CharArray::from_unicode_scalars(
            "111111111100011110100000001110".chars().collect(),
            3,
            10,
        )
        .expect("char matrix");
        let Value::Tensor(out) = bin2dec(Value::CharArray(chars)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape(), &[3, 1]);
        assert_eq!(out.materialize_f64(), vec![1023.0, 122.0, 14.0]);

        let strings = StringArray::new(
            vec!["1111111111".into(), "1111010".into(), "1110".into()],
            vec![1, 3],
        )
        .expect("string array");
        let Value::Tensor(out) = bin2dec(Value::StringArray(strings)) else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape(), &[1, 3]);
        assert_eq!(out.materialize_f64(), vec![1023.0, 122.0, 14.0]);
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
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(out.materialize_f64(), vec![5.0, 8.0]);
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
        assert_eq!(out.shape(), &[1, 3]);
        assert_eq!(out.materialize_f64(), vec![1023.0, 122.0, 14.0]);

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
    fn singleton_string_arrays_scalarize_and_empty_arrays_keep_their_shape() {
        for (text, radix) in [
            ("101", TextConversion::Bin2Dec),
            ("A", TextConversion::Hex2Dec),
        ] {
            for shape in [vec![1], vec![1, 1, 1]] {
                let input = StringArray::new(vec![text.into()], shape).expect("singleton string");
                let output = block_on(text_to_decimal_builtin(Value::StringArray(input), radix))
                    .expect("singleton conversion");
                assert!(matches!(output, Value::Num(_)));
            }
            let input = StringArray::new(vec![], vec![0, 2]).expect("empty string array");
            let Value::Tensor(output) =
                block_on(text_to_decimal_builtin(Value::StringArray(input), radix))
                    .expect("empty conversion")
            else {
                panic!("empty string array must produce an empty tensor");
            };
            assert_eq!(output.shape(), &[0, 2]);
        }
    }

    #[test]
    fn nested_text_cells_reject_multi_item_string_and_multirow_character_values() {
        let bad_string = Value::StringArray(
            StringArray::new(vec!["1".into(), "0".into()], vec![1, 2])
                .expect("two-item string array"),
        );
        let bad_char = Value::CharArray(
            CharArray::from_unicode_scalars("1010".chars().collect(), 2, 2)
                .expect("two-row char array"),
        );
        for nested in [bad_string, bad_char] {
            let cell = CellArray::new(vec![nested], 1, 1).expect("cell");
            for radix in [TextConversion::Bin2Dec, TextConversion::Hex2Dec] {
                let error = block_on(text_to_decimal_builtin(Value::Cell(cell.clone()), radix))
                    .expect_err("nested non-scalar text must reject");
                assert!(error
                    .to_string()
                    .contains("cell array elements must be text"));
            }
        }
    }

    #[test]
    fn bi2de_and_de2bi_scalar_integers_remain_exact() {
        assert_eq!(
            integer_to_nonnegative(&IntValue::U64(u64::MAX), DE2BI_INPUT_OWNER, "D").unwrap(),
            u128::from(u64::MAX)
        );
        assert_eq!(
            block_on(decimal_vector_from_value(Value::Int(IntValue::U64(
                u64::MAX
            ))))
            .unwrap(),
            (vec![u128::from(u64::MAX)], NumericDType::U64)
        );
        let digits = block_on(digit_matrix_from_value(
            Value::Int(IntValue::U8(1)),
            BI2DE_NAME,
        ))
        .unwrap();
        assert_eq!(digits.data, vec![1]);
        assert!(integer_to_nonnegative(&IntValue::I8(-1), DE2BI_INPUT_OWNER, "D").is_err());
        assert!(block_on(decimal_vector_from_value(Value::Int(IntValue::I8(-1)))).is_err());
    }

    #[test]
    fn dec_text_min_digits_rejects_exact_integer_resource_overflow() {
        let too_wide = IntValue::U64(MAX_DECIMAL_TEXT_OUTPUT_CHARS as u64 + 1);
        let err = block_on(super::dec2bin_builtin(
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(too_wide)]),
        ))
        .expect_err("wide typed minDigits should reject before allocation");
        assert!(err.to_string().contains("requested output"));
        assert_eq!(err.identifier(), None);

        let width = Tensor::new_integer(
            IntegerStorage::U64(vec![MAX_DECIMAL_TEXT_OUTPUT_CHARS as u64 + 1]),
            vec![1, 1],
        )
        .unwrap();
        let err = block_on(super::dec2hex_builtin(
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(width)]),
        ))
        .expect_err("wide typed tensor minDigits should reject before allocation");
        assert!(err.to_string().contains("requested output"));
        assert_eq!(err.identifier(), None);
    }

    #[test]
    fn bi2de_and_de2bi_typed_integer_tensors_remain_exact() {
        let digits = integer_tensor(IntegerStorage::U64(vec![1, 0, 1]), vec![1, 3]);
        assert_eq!(value_data(bi2de(digits, vec![])), vec![5.0]);

        let width = Tensor::new_integer(IntegerStorage::U16(vec![12]), vec![1, 1]).unwrap();
        assert_eq!(
            scalar_number(
                &Value::Tensor(width),
                NumericParseOwner::Digit(DigitError::De2biWidth),
                "N"
            )
            .unwrap(),
            12.0
        );

        for storage in [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ] {
            let width = Tensor::new_integer(storage, vec![1, 1]).expect("typed width");
            assert_eq!(parse_width(&Value::Tensor(width)).unwrap(), Some(2));
        }

        let empty = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();
        assert!(is_empty_numeric(&Value::Tensor(empty)));

        assert_eq!(
            block_on(decimal_vector_from_value(integer_tensor(
                IntegerStorage::U64(vec![u64::MAX]),
                vec![1, 1],
            )))
            .unwrap(),
            (vec![u128::from(u64::MAX)], NumericDType::U64)
        );

        let err = block_on(super::de2bi_builtin(
            integer_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
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

    #[test]
    fn binary_conversions_accept_native_single_storage() {
        assert_eq!(
            value_data(bi2de(
                single_tensor(vec![1.0, 0.0, 1.0], vec![1, 3]),
                vec![],
            )),
            vec![5.0]
        );
        assert_eq!(
            de2bi(single_tensor(vec![5.0], vec![1, 1]), vec![]).materialize_f64(),
            vec![1.0, 0.0, 1.0]
        );
        assert_eq!(
            char_rows(&dec2hex(single_tensor(vec![15.0], vec![1, 1]), vec![])),
            vec!["F"]
        );
    }
}
