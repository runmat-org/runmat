//! MATLAB-compatible `num2str` builtin with GPU-aware semantics for RunMat.

use regex::Regex;
use runmat_builtins::{
    BuiltinErrorDescriptor, NUM2STR_DESCRIPTOR, NUM2STR_ERROR_INTERNAL,
    NUM2STR_ERROR_INVALID_FORMAT, NUM2STR_ERROR_INVALID_INPUT, NUM2STR_ERROR_INVALID_OPTION,
    NUM2STR_ERROR_INVALID_PRECISION, NUM2STR_ERROR_TOO_MANY_OUTPUTS,
    NUM2STR_EXPLICIT_GPU_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, Tensor, Value,
};

use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::GpuGatherRetry;

#[path = "num2str/managed.rs"]
mod managed;
#[path = "num2str/managed_format.rs"]
mod managed_format;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_PRECISION: usize = 15;
const MAX_PRECISION: usize = 52;
// Match the existing bounded numeric-format field convention in dlmwrite.
const MAX_FORMAT_FIELD: usize = 4096;

const BUILTIN_NAME: &str = "num2str";

fn num2str_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    num2str_error_with_message(error.message, error)
}

fn num2str_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_num2str_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::num2str")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "num2str",
    op_kind: GpuOpKind::Custom("conversion"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Always gathers GPU data to host memory before formatting numeric text.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::num2str")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "num2str",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Conversion builtin; not eligible for fusion and always materialises host character arrays.",
};

#[runtime_builtin(
    name = "num2str",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::num2str"
)]
async fn num2str_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &NUM2STR_DESCRIPTOR,
        &NUM2STR_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    let resident_control = rest
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NUM2STR_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !matches!(
        &value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    ) {
        return Err(num2str_error(&NUM2STR_ERROR_INVALID_INPUT));
    }
    validate_option_structure(&rest)?;
    let capture = || {
        HostAllocationReceipt::capture_owner(runtime.as_ref()).map_err(|cause| {
            let runtime = runtime.as_ref().expect("managed capture has runtime");
            num2str_allocation_error(runtime, cause)
        })
    };
    let early_owner = if resident_control { capture()? } else { None };
    // Parse and validate controls before downloading a resident data operand.
    let options = parse_options_validated(rest).await.map_err(|mut error| {
        if resident_control {
            error.gpu_gather_retry = GpuGatherRetry::Never;
        }
        error
    })?;
    let original_owner = if resident_control {
        early_owner
    } else {
        capture()?
    };
    if let (Some(owner), Some(runtime)) = (original_owner.as_ref(), runtime.as_ref()) {
        owner.check(runtime).map_err(|cause| {
            let mut error = num2str_allocation_error(runtime, cause);
            if resident_control {
                error.gpu_gather_retry = GpuGatherRetry::Never;
            }
            error
        })?;
    }
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(remap_num2str_flow)?;
    let gathered = if original_owner.is_some() {
        if let Value::GpuTensor(handle) = gathered {
            Value::Tensor(
                gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(remap_num2str_flow)
                    .map_err(|mut error| {
                        error.gpu_gather_retry = GpuGatherRetry::Never;
                        error
                    })?,
            )
        } else {
            gathered
        }
    } else {
        gathered
    };
    if let Some(owner) = original_owner {
        let runtime = runtime.as_ref().expect("managed owner has runtime");
        owner
            .check(runtime)
            .map_err(|cause| num2str_allocation_error(runtime, cause))
            .map_err(|mut error| {
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            })?;
        return managed::format_managed(gathered, &options, runtime)
            .await
            .map(Value::CharArray)
            .map_err(|mut error| {
                error.gpu_gather_retry = GpuGatherRetry::Never;
                error
            });
    }
    let data = extract_numeric_data(gathered).await?;

    let char_array = format_numeric_data(data, &options)?;
    Ok(Value::CharArray(char_array))
}

fn num2str_allocation_error(runtime: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:num2str:HostFormatting",
        "num2str: host formatting allocation failed",
        cause,
    )
}

struct FormatOptions {
    spec: FormatSpec,
    decimal: char,
}

#[derive(Clone)]
enum FormatSpec {
    General { digits: usize },
    Custom(CustomFormat),
}

#[derive(Clone)]
struct CustomFormat {
    kind: CustomKind,
    width: Option<usize>,
    precision: Option<usize>,
    sign_always: bool,
    left_align: bool,
    zero_pad: bool,
    uppercase: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CustomKind {
    Fixed,
    Exponent,
    General,
}

enum NumericData {
    Real {
        data: Vec<f64>,
        rows: usize,
        cols: usize,
    },
    Complex {
        data: Vec<(f64, f64)>,
        rows: usize,
        cols: usize,
    },
    IntegerComplex {
        storage: IntegerComplexStorage,
        rows: usize,
        cols: usize,
    },
    Integer {
        storage: IntegerStorage,
        rows: usize,
        cols: usize,
    },
}

fn validate_option_structure(args: &crate::call::BuiltinArguments) -> BuiltinResult<()> {
    if args.len() > 2 {
        return Err(num2str_error_with_message(
            "num2str: too many input arguments",
            &NUM2STR_ERROR_INVALID_OPTION,
        ));
    }
    // An invalid final host option is knowable without gathering an earlier
    // resident precision argument. Resident controls are resolved below.
    if let Some(second) = args.get(1) {
        if !matches!(second, Value::GpuTensor(_)) && !is_local_token(second)? {
            return Err(num2str_error_with_message(
                "num2str: expected 'local' as the third argument",
                &NUM2STR_ERROR_INVALID_OPTION,
            ));
        }
    }
    if let Some(first) = args.first() {
        if !matches!(
            first,
            Value::Num(_)
                | Value::Int(_)
                | Value::Bool(_)
                | Value::Tensor(_)
                | Value::LogicalArray(_)
                | Value::String(_)
                | Value::StringArray(_)
                | Value::CharArray(_)
                | Value::GpuTensor(_)
        ) {
            return Err(num2str_error_with_message(
                "num2str: second argument must be a precision or format string",
                &NUM2STR_ERROR_INVALID_OPTION,
            ));
        }
    }
    Ok(())
}

async fn parse_options_validated(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<FormatOptions> {
    if args.is_empty() {
        return Ok(FormatOptions {
            spec: FormatSpec::General {
                digits: DEFAULT_PRECISION,
            },
            decimal: '.',
        });
    }

    let mut iter = args.into_iter();
    let mut spec = FormatSpec::General {
        digits: DEFAULT_PRECISION,
    };
    let mut decimal = '.';

    if let Some(first) = iter.next() {
        let first = gather_if_needed_async(&first)
            .await
            .map_err(remap_num2str_flow)?;
        if is_local_token(&first)? {
            decimal = detect_decimal_separator(true);
            if iter.next().is_some() {
                return Err(num2str_error_with_message(
                    "num2str: too many input arguments",
                    &NUM2STR_ERROR_INVALID_OPTION,
                ));
            }
            return Ok(FormatOptions { spec, decimal });
        }

        spec = if let Some(digits) = try_extract_precision(&first)? {
            FormatSpec::General { digits }
        } else if let Some(text) = value_to_text(&first)? {
            FormatSpec::Custom(parse_custom_format(&text)?)
        } else {
            return Err(num2str_error_with_message(
                "num2str: second argument must be a precision or format string",
                &NUM2STR_ERROR_INVALID_OPTION,
            ));
        };
    }

    if let Some(second) = iter.next() {
        let second = gather_if_needed_async(&second)
            .await
            .map_err(remap_num2str_flow)?;
        if !is_local_token(&second)? {
            return Err(num2str_error_with_message(
                "num2str: expected 'local' as the third argument",
                &NUM2STR_ERROR_INVALID_OPTION,
            ));
        }
        decimal = detect_decimal_separator(true);
    }

    Ok(FormatOptions { spec, decimal })
}

fn is_local_token(value: &Value) -> BuiltinResult<bool> {
    let Some(text) = value_to_text(value)? else {
        return Ok(false);
    };
    Ok(text.trim().eq_ignore_ascii_case("local"))
}

fn try_extract_precision(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Int(i) => {
            let digits = i.try_to_i64().ok_or_else(|| {
                num2str_error_with_message(
                    format!("num2str: precision must satisfy 0 <= p <= {MAX_PRECISION}"),
                    &NUM2STR_ERROR_INVALID_PRECISION,
                )
            })?;
            validate_precision(digits)?;
            Ok(Some(digits as usize))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(num2str_error_with_message(
                    "num2str: precision must be finite",
                    &NUM2STR_ERROR_INVALID_PRECISION,
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(num2str_error_with_message(
                    "num2str: precision must be an integer",
                    &NUM2STR_ERROR_INVALID_PRECISION,
                ));
            }
            validate_precision(rounded as i64)?;
            Ok(Some(rounded as usize))
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                let digits = int.try_to_i64().ok_or_else(|| {
                    num2str_error_with_message(
                        format!("num2str: precision must satisfy 0 <= p <= {MAX_PRECISION}"),
                        &NUM2STR_ERROR_INVALID_PRECISION,
                    )
                })?;
                validate_precision(digits)?;
                return Ok(Some(digits as usize));
            }
            let value = tensor::tensor_value_f64(t, 0);
            if !value.is_finite() {
                return Err(num2str_error_with_message(
                    "num2str: precision must be finite",
                    &NUM2STR_ERROR_INVALID_PRECISION,
                ));
            }
            let rounded = value.round();
            if (rounded - value).abs() > f64::EPSILON {
                return Err(num2str_error_with_message(
                    "num2str: precision must be an integer",
                    &NUM2STR_ERROR_INVALID_PRECISION,
                ));
            }
            validate_precision(rounded as i64)?;
            Ok(Some(rounded as usize))
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            let digits = if la.data[0] != 0 { 1 } else { 0 };
            validate_precision(digits)?;
            Ok(Some(digits as usize))
        }
        Value::Bool(b) => {
            let digits = if *b { 1 } else { 0 };
            Ok(Some(digits))
        }
        _ => Ok(None),
    }
}

fn validate_precision(value: i64) -> BuiltinResult<()> {
    if value < 0 || value > MAX_PRECISION as i64 {
        return Err(num2str_error_with_message(
            format!("num2str: precision must satisfy 0 <= p <= {MAX_PRECISION}"),
            &NUM2STR_ERROR_INVALID_PRECISION,
        ));
    }
    Ok(())
}

fn value_to_text(value: &Value) -> BuiltinResult<Option<std::borrow::Cow<'_, str>>> {
    Ok(match value {
        Value::String(s) => s.as_str().map(std::borrow::Cow::Borrowed),
        Value::StringArray(sa) if sa.data().len() == 1 => {
            sa.data()[0].as_str().map(std::borrow::Cow::Borrowed)
        }
        Value::CharArray(ca) if ca.rows == 1 => Some(std::borrow::Cow::Owned(
            runmat_value::decode_utf16_code_units(ca.data()).map_err(|source| {
                let mut builder =
                    build_runtime_error(format!("num2str: invalid UTF-16 format text: {source}"))
                        .with_builtin(BUILTIN_NAME)
                        .with_source(source);
                if let Some(identifier) = NUM2STR_ERROR_INVALID_OPTION.identifier {
                    builder = builder.with_identifier(identifier);
                }
                builder.build()
            })?,
        )),
        _ => None,
    })
}

fn detect_decimal_separator(local: bool) -> char {
    if !local {
        return '.';
    }

    if let Ok(custom) = crate::builtins::common::env::var("RUNMAT_DECIMAL_SEPARATOR") {
        let trimmed = custom.trim();
        if let Some(ch) = trimmed.chars().next() {
            return ch;
        }
    }

    let locale = crate::builtins::common::env::var("LC_NUMERIC")
        .or_else(|_| crate::builtins::common::env::var("RUNMAT_LOCALE"))
        .or_else(|_| crate::builtins::common::env::var("LANG"))
        .unwrap_or_default()
        .to_lowercase();

    if locale.is_empty() {
        return '.';
    }

    let comma_locales = [
        "af", "bs", "ca", "cs", "da", "de", "el", "es", "eu", "fi", "fr", "gl", "hr", "hu", "id",
        "is", "it", "lt", "lv", "nb", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sr", "sv", "tr",
        "uk", "vi",
    ];
    let locale_prefix = locale.split(['.', '_', '@']).next().unwrap_or(&locale);
    for prefix in &comma_locales {
        if locale_prefix.starts_with(prefix) {
            return ',';
        }
    }
    '.'
}

fn parse_custom_format(text: &str) -> BuiltinResult<CustomFormat> {
    if !text.starts_with('%') {
        return Err(num2str_error_with_message(
            "num2str: format must start with '%'",
            &NUM2STR_ERROR_INVALID_FORMAT,
        ));
    }
    if text == "%%" {
        return Err(num2str_error_with_message(
            "num2str: '%' escape is not supported for numeric conversion",
            &NUM2STR_ERROR_INVALID_FORMAT,
        ));
    }

    static FORMAT_RE: once_cell::sync::Lazy<Regex> = once_cell::sync::Lazy::new(|| {
        Regex::new(r"^%([+\-0]*)(\d+)?(?:\.(\d*))?([fFeEgG])$").expect("format regex")
    });

    let captures = FORMAT_RE.captures(text).ok_or_else(|| {
        num2str_error_with_message(
            format!(
                "{}; expected variants like '%0.3f' or '%.5g'",
                NUM2STR_ERROR_INVALID_FORMAT.message
            ),
            &NUM2STR_ERROR_INVALID_FORMAT,
        )
    })?;

    let flags = captures.get(1).map(|m| m.as_str()).unwrap_or("");
    let width = captures
        .get(2)
        .map(|m| {
            m.as_str().parse::<usize>().map_err(|_| {
                num2str_error_with_message(
                    "num2str: format width is too large",
                    &NUM2STR_ERROR_INVALID_FORMAT,
                )
            })
        })
        .transpose()?;
    if width.is_some_and(|width| width > MAX_FORMAT_FIELD) {
        return Err(num2str_error_with_message(
            format!("num2str: format width exceeds {MAX_FORMAT_FIELD}"),
            &NUM2STR_ERROR_INVALID_FORMAT,
        ));
    }
    let precision = captures
        .get(3)
        .map(|m| {
            if m.as_str().is_empty() {
                Ok(0usize)
            } else {
                m.as_str().parse::<usize>().map_err(|_| {
                    num2str_error_with_message(
                        "num2str: format precision is too large",
                        &NUM2STR_ERROR_INVALID_PRECISION,
                    )
                })
            }
        })
        .transpose()?;
    let conversion = captures
        .get(4)
        .map(|m| m.as_str().chars().next().unwrap())
        .unwrap();

    let mut sign_always = false;
    let mut left_align = false;
    let mut zero_pad = false;

    for ch in flags.chars() {
        match ch {
            '+' => sign_always = true,
            '-' => left_align = true,
            '0' => zero_pad = true,
            _ => {
                return Err(num2str_error_with_message(
                    format!(
                    "num2str: unsupported format flag '{}'; only '+', '-', and '0' are supported",
                    ch
                ),
                    &NUM2STR_ERROR_INVALID_FORMAT,
                ))
            }
        }
    }

    if let Some(p) = precision {
        if p > MAX_PRECISION {
            return Err(num2str_error_with_message(
                format!("num2str: precision must satisfy 0 <= p <= {MAX_PRECISION}"),
                &NUM2STR_ERROR_INVALID_PRECISION,
            ));
        }
    }

    let (kind, uppercase) = match conversion {
        'f' => (CustomKind::Fixed, false),
        'F' => (CustomKind::Fixed, true),
        'e' => (CustomKind::Exponent, false),
        'E' => (CustomKind::Exponent, true),
        'g' => (CustomKind::General, false),
        'G' => (CustomKind::General, true),
        _ => unreachable!(),
    };

    Ok(CustomFormat {
        kind,
        width,
        precision,
        sign_always,
        left_align,
        zero_pad,
        uppercase,
    })
}

async fn extract_numeric_data(value: Value) -> BuiltinResult<NumericData> {
    match value {
        Value::Num(n) => Ok(NumericData::Real {
            data: vec![n],
            rows: 1,
            cols: 1,
        }),
        Value::Int(i) => Ok(NumericData::Integer {
            storage: integer_storage_from_scalar(i),
            rows: 1,
            cols: 1,
        }),
        Value::Bool(b) => Ok(NumericData::Real {
            data: vec![if b { 1.0 } else { 0.0 }],
            rows: 1,
            cols: 1,
        }),
        Value::Tensor(t) => tensor_to_numeric_data(t),
        Value::LogicalArray(la) => {
            let tensor = tensor::logical_to_tensor(&la)
                .map_err(|_| num2str_error(&NUM2STR_ERROR_INVALID_INPUT))?;
            tensor_to_numeric_data(tensor)
        }
        Value::Complex(re, im) => Ok(NumericData::Complex {
            data: vec![(re, im)],
            rows: 1,
            cols: 1,
        }),
        Value::ComplexTensor(t) => complex_tensor_to_data(t),
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(remap_num2str_flow)?;
            tensor_to_numeric_data(gathered)
        }
        other => Err(num2str_error_with_message(
            format!(
                "{} {:?}; expected numeric or logical values",
                NUM2STR_ERROR_INVALID_INPUT.message, other
            ),
            &NUM2STR_ERROR_INVALID_INPUT,
        )),
    }
}

fn tensor_to_numeric_data(tensor: Tensor) -> BuiltinResult<NumericData> {
    if tensor.shape().len() > 2 {
        return Err(num2str_error_with_message(
            "num2str: input must be scalar, vector, or 2-D matrix",
            &NUM2STR_ERROR_INVALID_INPUT,
        ));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|_| num2str_error(&NUM2STR_ERROR_INVALID_INPUT))?;
    match storage.into_integer_storage() {
        Ok(storage) => Ok(NumericData::Integer {
            storage: storage.clone(),
            rows,
            cols,
        }),
        Err(storage) => Ok(NumericData::Real {
            data: storage.materialize_f64(),
            rows,
            cols,
        }),
    }
}

fn complex_tensor_to_data(tensor: ComplexTensor) -> BuiltinResult<NumericData> {
    if tensor.shape().len() > 2 {
        return Err(num2str_error_with_message(
            "num2str: complex input must be scalar, vector, or 2-D matrix",
            &NUM2STR_ERROR_INVALID_INPUT,
        ));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    if let Some(storage) = tensor.integer_storage() {
        return Ok(NumericData::IntegerComplex {
            storage: storage.clone(),
            rows,
            cols,
        });
    }
    Ok(NumericData::Complex {
        data: tensor.materialize_f64(),
        rows,
        cols,
    })
}

#[derive(Clone)]
struct CellEntry {
    text: String,
    width: usize,
}

fn format_numeric_data(data: NumericData, options: &FormatOptions) -> BuiltinResult<CharArray> {
    match data {
        NumericData::Real { data, rows, cols } => format_real_matrix(&data, rows, cols, options),
        NumericData::Complex { data, rows, cols } => {
            format_complex_matrix(&data, rows, cols, options)
        }
        NumericData::IntegerComplex {
            storage,
            rows,
            cols,
        } => format_integer_complex_matrix(&storage, rows, cols, options),
        NumericData::Integer {
            storage,
            rows,
            cols,
        } => format_integer_matrix(&storage, rows, cols, options),
    }
}

fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

fn format_real_matrix(
    data: &[f64],
    rows: usize,
    cols: usize,
    options: &FormatOptions,
) -> BuiltinResult<CharArray> {
    if rows == 0 {
        return CharArray::from_code_units(Vec::new(), 0, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }
    if cols == 0 {
        return CharArray::from_code_units(Vec::new(), rows, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }

    let mut entries = vec![
        vec![
            CellEntry {
                text: String::new(),
                width: 0
            };
            cols
        ];
        rows
    ];
    let mut col_widths = vec![0usize; cols];

    for (col, width) in col_widths.iter_mut().enumerate() {
        for (row, row_entries) in entries.iter_mut().enumerate() {
            let idx = row + col * rows;
            let value = data.get(idx).copied().unwrap_or(0.0);
            let text = format_real(value, &options.spec, options.decimal);
            let entry_width = text.encode_utf16().count();
            row_entries[col] = CellEntry {
                text,
                width: entry_width,
            };
            if entry_width > *width {
                *width = entry_width;
            }
        }
    }

    if cols > 1 {
        for (idx, width) in col_widths.iter_mut().enumerate() {
            if idx > 0 {
                *width += 1;
            }
        }
    }

    let rows_str = assemble_rows(entries, col_widths);
    rows_to_char_array(rows_str)
}

fn format_integer_matrix(
    storage: &IntegerStorage,
    rows: usize,
    cols: usize,
    options: &FormatOptions,
) -> BuiltinResult<CharArray> {
    if rows == 0 {
        return CharArray::from_code_units(Vec::new(), 0, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }
    if cols == 0 {
        return CharArray::from_code_units(Vec::new(), rows, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }

    let mut entries = vec![
        vec![
            CellEntry {
                text: String::new(),
                width: 0,
            };
            cols
        ];
        rows
    ];
    let mut col_widths = vec![0usize; cols];

    for (col, width) in col_widths.iter_mut().enumerate() {
        for (row, row_entries) in entries.iter_mut().enumerate() {
            let text = format_integer(
                integer_storage_decimal(storage, row + col * rows),
                &options.spec,
                options.decimal,
            );
            let entry_width = text.encode_utf16().count();
            row_entries[col] = CellEntry {
                text,
                width: entry_width,
            };
            *width = (*width).max(entry_width);
        }
    }

    if cols > 1 {
        for (idx, width) in col_widths.iter_mut().enumerate() {
            if idx > 0 {
                *width += 1;
            }
        }
    }

    rows_to_char_array(assemble_rows(entries, col_widths))
}

fn integer_storage_decimal(storage: &IntegerStorage, index: usize) -> String {
    match storage {
        IntegerStorage::I8(values) => values[index].to_string(),
        IntegerStorage::I16(values) => values[index].to_string(),
        IntegerStorage::I32(values) => values[index].to_string(),
        IntegerStorage::I64(values) => values[index].to_string(),
        IntegerStorage::U8(values) => values[index].to_string(),
        IntegerStorage::U16(values) => values[index].to_string(),
        IntegerStorage::U32(values) => values[index].to_string(),
        IntegerStorage::U64(values) => values[index].to_string(),
    }
}

fn format_integer_complex_matrix(
    storage: &IntegerComplexStorage,
    rows: usize,
    cols: usize,
    options: &FormatOptions,
) -> BuiltinResult<CharArray> {
    if rows == 0 {
        return CharArray::from_code_units(Vec::new(), 0, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }
    if cols == 0 {
        return CharArray::from_code_units(Vec::new(), rows, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }

    let mut entries = vec![
        vec![
            CellEntry {
                text: String::new(),
                width: 0
            };
            cols
        ];
        rows
    ];
    let mut col_widths = vec![0usize; cols];

    for (col, width) in col_widths.iter_mut().enumerate() {
        for (row, row_entries) in entries.iter_mut().enumerate() {
            let text = format_integer_complex(storage, row + col * rows, options);
            let entry_width = text.encode_utf16().count();
            row_entries[col] = CellEntry {
                text,
                width: entry_width,
            };
            *width = (*width).max(entry_width);
        }
    }

    if cols > 1 {
        for (idx, width) in col_widths.iter_mut().enumerate() {
            if idx > 0 {
                *width += 1;
            }
        }
    }

    rows_to_char_array(assemble_rows(entries, col_widths))
}

fn format_integer_complex(
    storage: &IntegerComplexStorage,
    index: usize,
    options: &FormatOptions,
) -> String {
    let real_raw = integer_storage_decimal(storage.real(), index);
    let imag_raw = integer_storage_decimal(storage.imag(), index);
    let real_is_zero = real_raw == "0";
    let real_str = format_integer(real_raw, &options.spec, options.decimal);
    let Some(imag_abs_raw) = integer_abs_decimal(&imag_raw) else {
        return real_str;
    };
    let imag_str = format_integer(imag_abs_raw, &options.spec, options.decimal);
    let imag_sign = if imag_raw.starts_with('-') { '-' } else { '+' };

    if real_is_zero {
        if imag_sign == '-' {
            return format!("-{imag_str}i");
        }
        return format!("{imag_str}i");
    }

    format!("{real_str} {imag_sign} {imag_str}i")
}

fn integer_abs_decimal(value: &str) -> Option<String> {
    if value == "0" {
        None
    } else {
        Some(value.strip_prefix('-').unwrap_or(value).to_string())
    }
}

fn format_complex_matrix(
    data: &[(f64, f64)],
    rows: usize,
    cols: usize,
    options: &FormatOptions,
) -> BuiltinResult<CharArray> {
    if rows == 0 {
        return CharArray::from_code_units(Vec::new(), 0, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }
    if cols == 0 {
        return CharArray::from_code_units(Vec::new(), rows, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }

    let mut entries = vec![
        vec![
            CellEntry {
                text: String::new(),
                width: 0
            };
            cols
        ];
        rows
    ];
    let mut col_widths = vec![0usize; cols];

    for (col, width) in col_widths.iter_mut().enumerate() {
        for (row, row_entries) in entries.iter_mut().enumerate() {
            let idx = row + col * rows;
            let (re, im) = data.get(idx).copied().unwrap_or((0.0, 0.0));
            let text = format_complex(re, im, &options.spec, options.decimal);
            let entry_width = text.encode_utf16().count();
            row_entries[col] = CellEntry {
                text,
                width: entry_width,
            };
            if entry_width > *width {
                *width = entry_width;
            }
        }
    }

    if cols > 1 {
        for (idx, width) in col_widths.iter_mut().enumerate() {
            if idx > 0 {
                *width += 1;
            }
        }
    }

    let rows_str = assemble_rows(entries, col_widths);
    rows_to_char_array(rows_str)
}

fn assemble_rows(entries: Vec<Vec<CellEntry>>, col_widths: Vec<usize>) -> Vec<String> {
    entries
        .into_iter()
        .map(|row_entries| {
            row_entries
                .into_iter()
                .enumerate()
                .fold(String::new(), |mut acc, (col, entry)| {
                    if col > 0 {
                        acc.push(' ');
                    }
                    let target = col_widths[col];
                    let pad = target.saturating_sub(entry.width);
                    acc.extend(std::iter::repeat_n(' ', pad));
                    acc.push_str(&entry.text);
                    acc
                })
        })
        .collect()
}

fn rows_to_char_array(rows: Vec<String>) -> BuiltinResult<CharArray> {
    if rows.is_empty() {
        return CharArray::from_code_units(Vec::new(), 0, 0)
            .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL));
    }
    let row_count = rows.len();
    let col_count = rows
        .iter()
        .map(|row| row.encode_utf16().count())
        .max()
        .unwrap_or(0);

    let capacity = row_count
        .checked_mul(col_count)
        .ok_or_else(|| num2str_error(&NUM2STR_ERROR_INTERNAL))?;
    let mut data = Vec::with_capacity(capacity);
    for row in rows {
        let width = row.encode_utf16().count();
        data.extend(
            row.encode_utf16()
                .map(runmat_value::Utf16CodeUnit::from_raw),
        );
        data.extend(std::iter::repeat_n(
            runmat_value::Utf16CodeUnit::from_raw(u16::from(b' ')),
            col_count - width,
        ));
    }

    CharArray::from_code_units(data, row_count, col_count)
        .map_err(|_| num2str_error(&NUM2STR_ERROR_INTERNAL))
}

fn format_real(value: f64, spec: &FormatSpec, decimal: char) -> String {
    let text = match spec {
        FormatSpec::General { digits } => format_general(value, *digits, false),
        FormatSpec::Custom(custom) => format_custom(value, custom),
    };
    apply_decimal_locale(text, decimal)
}

fn format_integer(value: String, spec: &FormatSpec, decimal: char) -> String {
    let (negative, digits) = value
        .strip_prefix('-')
        .map_or((false, value.as_str()), |digits| (true, digits));
    let text = match spec {
        FormatSpec::General { digits: precision } => {
            format_integer_general(negative, digits, *precision, false)
        }
        FormatSpec::Custom(custom) => {
            let precision = custom.precision.unwrap_or(match custom.kind {
                CustomKind::Fixed | CustomKind::Exponent => 6,
                CustomKind::General => DEFAULT_PRECISION,
            });
            let text = match custom.kind {
                CustomKind::Fixed => format_integer_fixed(negative, digits, precision),
                CustomKind::Exponent => {
                    format_integer_exponent(negative, digits, precision + 1, custom.uppercase)
                }
                CustomKind::General => {
                    format_integer_general(negative, digits, precision.max(1), custom.uppercase)
                }
            };
            apply_format_flags(text, custom)
        }
    };
    apply_decimal_locale(text, decimal)
}

fn format_integer_fixed(negative: bool, digits: &str, precision: usize) -> String {
    let mut text = String::new();
    if negative {
        text.push('-');
    }
    text.push_str(digits);
    if precision > 0 {
        text.push('.');
        text.extend(std::iter::repeat_n('0', precision));
    }
    text
}

fn format_integer_general(
    negative: bool,
    digits: &str,
    precision: usize,
    uppercase: bool,
) -> String {
    let significant = precision.max(1);
    if digits.len() <= significant {
        let mut text = String::new();
        if negative {
            text.push('-');
        }
        text.push_str(digits);
        return text;
    }
    format_integer_exponent(negative, digits, significant, uppercase)
}

fn format_integer_exponent(
    negative: bool,
    digits: &str,
    significant: usize,
    uppercase: bool,
) -> String {
    let (rounded, exponent) = round_integer_significant(digits, significant.max(1));
    let mut text = String::new();
    if negative {
        text.push('-');
    }
    let first = rounded.as_bytes()[0] as char;
    text.push(first);
    let fraction = rounded.get(1..).unwrap_or("").trim_end_matches('0');
    if !fraction.is_empty() {
        text.push('.');
        text.push_str(fraction);
    }
    text.push(if uppercase { 'E' } else { 'e' });
    text.push('+');
    text.push_str(&exponent.to_string());
    text
}

fn round_integer_significant(digits: &str, significant: usize) -> (String, usize) {
    let exponent = digits.len() - 1;
    if digits.len() <= significant {
        return (digits.to_string(), exponent);
    }

    let mut rounded: Vec<u8> = digits.as_bytes()[..significant].to_vec();
    if digits.as_bytes()[significant] >= b'5' {
        for position in (0..rounded.len()).rev() {
            if rounded[position] < b'9' {
                rounded[position] += 1;
                break;
            }
            rounded[position] = b'0';
            if position == 0 {
                let mut carry = String::from("1");
                carry.extend(std::iter::repeat_n('0', significant.saturating_sub(1)));
                return (carry, exponent + 1);
            }
        }
    }
    (
        String::from_utf8(rounded).expect("integer digits are ascii"),
        exponent,
    )
}

fn format_complex(re: f64, im: f64, spec: &FormatSpec, decimal: char) -> String {
    let real_str = format_real(re, spec, decimal);
    let imag_sign = if im.is_sign_negative() { '-' } else { '+' };
    let abs_im = if im == 0.0 { 0.0 } else { im.abs() };
    let imag_str = format_real(abs_im, spec, decimal);

    if abs_im == 0.0 && !im.is_nan() {
        return real_str;
    }

    if re == 0.0 && !re.is_sign_negative() && !re.is_nan() {
        if im.is_sign_negative() && !im.is_nan() {
            return format!(
                "{}i",
                if imag_str.starts_with('-') {
                    imag_str.clone()
                } else {
                    format!("-{imag_str}")
                }
            );
        }
        return format!("{imag_str}i");
    }

    format!("{real_str} {imag_sign} {imag_str}i")
}

pub(crate) fn format_general(value: f64, digits: usize, uppercase: bool) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }
    if value == 0.0 {
        return "0".to_string();
    }

    let sig_digits = digits.max(1);
    let abs_val = value.abs();
    let exp10 = abs_val.log10().floor() as i32;
    let use_scientific = exp10 < -4 || exp10 >= sig_digits as i32;

    if use_scientific {
        let precision = sig_digits.saturating_sub(1);
        let s = if uppercase {
            format!("{:.*E}", precision, value)
        } else {
            format!("{:.*e}", precision, value)
        };
        let marker = if uppercase { 'E' } else { 'e' };
        if let Some(idx) = s.find(marker) {
            let (mantissa, exponent) = s.split_at(idx);
            let mut mant = mantissa.to_string();
            trim_trailing_zeros(&mut mant);
            normalize_negative_zero(&mut mant);
            let mut result = mant;
            result.push_str(exponent);
            return result;
        }
        s
    } else {
        let decimals = if sig_digits as i32 - 1 - exp10 < 0 {
            0
        } else {
            (sig_digits as i32 - 1 - exp10) as usize
        };
        let mut s = format!("{:.*}", decimals, value);
        trim_trailing_zeros(&mut s);
        normalize_negative_zero(&mut s);
        s
    }
}

fn trim_trailing_zeros(text: &mut String) {
    if let Some(dot_pos) = text.find('.') {
        let mut end = text.len();
        while end > dot_pos + 1 && text.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if end > dot_pos && text.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        text.truncate(end);
    }
}

fn normalize_negative_zero(text: &mut String) {
    if text.starts_with('-') && text.chars().skip(1).all(|ch| ch == '0') {
        *text = "0".to_string();
    }
}

fn format_custom(value: f64, fmt: &CustomFormat) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }

    let precision = fmt.precision.unwrap_or(match fmt.kind {
        CustomKind::Fixed | CustomKind::Exponent => 6,
        CustomKind::General => DEFAULT_PRECISION,
    });

    let mut text = match fmt.kind {
        CustomKind::Fixed => format!("{:.*}", precision, value),
        CustomKind::Exponent => {
            let mut s = format!("{:.*e}", precision, value);
            if fmt.uppercase {
                s = s.to_uppercase();
            }
            s
        }
        CustomKind::General => format_general(value, precision.max(1), fmt.uppercase),
    };

    if fmt.kind != CustomKind::Fixed {
        trim_trailing_zeros(&mut text);
        normalize_negative_zero(&mut text);
    }

    apply_format_flags(text, fmt)
}

fn apply_decimal_locale(text: String, decimal: char) -> String {
    if decimal == '.' {
        return text;
    }
    let mut replaced = false;
    text.chars()
        .map(|ch| {
            if ch == '.' && !replaced {
                replaced = true;
                decimal
            } else {
                ch
            }
        })
        .collect()
}

fn apply_format_flags(mut text: String, fmt: &CustomFormat) -> String {
    if fmt.sign_always && !text.starts_with('-') && !text.starts_with('+') && text != "NaN" {
        text.insert(0, '+');
    }

    let width = fmt.width.unwrap_or(0);
    if width == 0 {
        return text;
    }

    let len = text.encode_utf16().count();
    if len >= width {
        return text;
    }

    let pad_count = width - len;
    let pad_char = if fmt.zero_pad && !fmt.left_align {
        '0'
    } else {
        ' '
    };

    if fmt.left_align {
        let mut result = text.clone();
        result.extend(std::iter::repeat_n(' ', pad_count));
        return result;
    }

    if pad_char == '0' && (text.starts_with('+') || text.starts_with('-')) {
        let mut chars = text.chars();
        let sign = chars.next().unwrap();
        let remainder: String = chars.collect();
        let mut result = String::with_capacity(width);
        result.push(sign);
        result.extend(std::iter::repeat_n('0', pad_count));
        result.push_str(&remainder);
        return result;
    }

    let mut result = String::with_capacity(width);
    result.extend(std::iter::repeat_n(' ', pad_count));
    result.push_str(&text);
    result
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::strings::core::numeric_format_work::FormatBuffer;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }

        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    fn managed_runtime(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
            )
    }

    fn num2str_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::num2str_builtin(
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor};

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn admitted_cell_formatter_matches_original_spelling() {
        let options = [
            FormatOptions {
                spec: FormatSpec::General { digits: 15 },
                decimal: '.',
            },
            FormatOptions {
                spec: FormatSpec::General { digits: 3 },
                decimal: ',',
            },
            FormatOptions {
                spec: FormatSpec::General { digits: 4 },
                decimal: '🙂',
            },
            FormatOptions {
                spec: FormatSpec::Custom(CustomFormat {
                    kind: CustomKind::Fixed,
                    width: Some(32),
                    precision: Some(3),
                    sign_always: true,
                    left_align: false,
                    zero_pad: true,
                    uppercase: false,
                }),
                decimal: '.',
            },
            FormatOptions {
                spec: FormatSpec::Custom(CustomFormat {
                    kind: CustomKind::Exponent,
                    width: Some(20),
                    precision: Some(6),
                    sign_always: false,
                    left_align: true,
                    zero_pad: false,
                    uppercase: true,
                }),
                decimal: '.',
            },
            FormatOptions {
                spec: FormatSpec::Custom(CustomFormat {
                    kind: CustomKind::Fixed,
                    width: Some(20),
                    precision: Some(2),
                    sign_always: false,
                    left_align: false,
                    zero_pad: true,
                    uppercase: false,
                }),
                decimal: '.',
            },
        ];
        for options in &options {
            for &value in &[0.0, -0.0, 1e-7, -123.456, 1e100, f64::NAN, f64::INFINITY] {
                let mut backing = [0u8; 512];
                let mut text = FormatBuffer::new(&mut backing);
                managed_format::real(value, options, &mut text).unwrap();
                assert_eq!(
                    apply_decimal_locale(text.as_str().to_owned(), options.decimal),
                    format_real(value, &options.spec, options.decimal),
                    "value {value:?}"
                );
            }
            for raw in [
                i64::MIN.to_string(),
                u64::MAX.to_string(),
                "99999999999999999999".into(),
            ] {
                let mut backing = [0u8; 512];
                let mut text = FormatBuffer::new(&mut backing);
                managed_format::integer(&raw, options, &mut text).unwrap();
                assert_eq!(
                    apply_decimal_locale(text.as_str().to_owned(), options.decimal),
                    format_integer(raw.clone(), &options.spec, options.decimal),
                    "integer {raw}"
                );
            }
            for &(real, imag) in &[
                (0.0, -2.5),
                (-0.0, 2.5),
                (1.25, -0.0),
                (1.25, 2.5),
                (1e-7, f64::NAN),
            ] {
                let mut first = [0u8; 512];
                let mut second = [0u8; 512];
                let mut output = FormatBuffer::new(&mut first);
                let mut scratch = FormatBuffer::new(&mut second);
                managed_format::complex(real, imag, options, &mut output, &mut scratch).unwrap();
                assert_eq!(
                    output.as_str().replace('.', &options.decimal.to_string()),
                    format_complex(real, imag, &options.spec, options.decimal),
                    "complex ({real:?},{imag:?})"
                );
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn managed_num2str_borrows_wide_integer_and_retains_exact_output() {
        let denied = ManagedHostEffectLedger::new(1);
        let denied_options = FormatOptions {
            spec: FormatSpec::General { digits: 20 },
            decimal: '.',
        };
        let denied_result = futures::executor::block_on(managed::format_managed(
            Value::Int(IntValue::U64(u64::MAX)),
            &denied_options,
            &managed_runtime(&denied),
        ));
        assert!(denied_result.is_err());
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);

        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let runtime = managed_runtime(&ledger);
        let input = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
        );
        let options = FormatOptions {
            spec: FormatSpec::General { digits: 20 },
            decimal: '.',
        };
        let output =
            futures::executor::block_on(managed::format_managed(input, &options, &runtime))
                .expect("managed num2str");
        assert!(output.data_owner().is_exact());
        assert_eq!(
            output.try_row_string().unwrap().as_deref(),
            Some("18446744073709551615")
        );
        let retained = ledger.snapshot().live_reserved_bytes;
        assert!(retained > 0);
        let alias = output.clone();
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let local = FormatOptions {
            spec: FormatSpec::Custom(CustomFormat {
                kind: CustomKind::Fixed,
                width: None,
                precision: Some(2),
                sign_always: false,
                left_align: false,
                zero_pad: false,
                uppercase: false,
            }),
            decimal: '🙂',
        };
        let localized = futures::executor::block_on(managed::format_managed(
            Value::Num(1.5),
            &local,
            &managed_runtime(&ledger),
        ))
        .expect("supplementary decimal separator");
        assert_eq!(localized.cols, 5);
        assert_eq!(
            localized.try_row_string().unwrap().as_deref(),
            Some("1🙂50")
        );
        drop(localized);
        let complex = futures::executor::block_on(managed::format_managed(
            Value::Complex(1.5, 2.25),
            &local,
            &managed_runtime(&ledger),
        ))
        .expect("two supplementary decimal separators");
        let expected = "1🙂50 + 2🙂25i";
        assert_eq!(complex.cols, expected.encode_utf16().count());
        assert_eq!(complex.try_row_string().unwrap().as_deref(), Some(expected));
        drop(complex);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_scalar_default_precision() {
        let value = Value::Num(std::f64::consts::PI);
        let out = num2str_builtin(value, Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(ca.rows, 1);
                assert!(text.starts_with("3.1415926535897"));
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_precision_argument() {
        let value = Value::Num(std::f64::consts::PI);
        let out = num2str_builtin(value, vec![Value::Int(IntValue::I32(4))]).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text.trim(), "3.142");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn num2str_precision_parser_preserves_typed_integer_tensor_bounds() {
        let precision =
            Tensor::new_integer(IntegerStorage::U64(vec![52]), vec![1, 1]).expect("precision");
        assert_eq!(
            try_extract_precision(&Value::Tensor(precision)).unwrap(),
            Some(52)
        );

        let too_large =
            Tensor::new_integer(IntegerStorage::U64(vec![53]), vec![1, 1]).expect("precision");
        assert!(try_extract_precision(&Value::Tensor(too_large)).is_err());

        let negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("precision");
        assert!(try_extract_precision(&Value::Tensor(negative)).is_err());

        assert!(try_extract_precision(&Value::Int(IntValue::U64(u64::MAX))).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_matrix_alignment() {
        let tensor =
            Tensor::new(vec![1.0, 78.0, 23.0, 9.0, 456.0, 10.0], vec![2, 3]).expect("tensor");
        let out = num2str_builtin(Value::Tensor(tensor), Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 11);
                let rows: Vec<String> = ca
                    .data()
                    .chunks(ca.cols)
                    .map(|chunk| {
                        runmat_value::decode_utf16_code_units(chunk).expect("valid UTF-16 fixture")
                    })
                    .collect();
                assert_eq!(rows[0], " 1  23  456");
                assert_eq!(rows[1], "78   9   10");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_custom_format() {
        let tensor = Tensor::new(vec![1.234, 5.678], vec![1, 2]).expect("tensor");
        let fmt = Value::String("%.2f".to_string().into());
        let out = num2str_builtin(Value::Tensor(tensor), vec![fmt]).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, "1.23  5.68");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_preserves_exact_uint64_across_format_modes() {
        let scalar = num2str_builtin(Value::Int(IntValue::U64(u64::MAX)), Vec::new())
            .expect("default integer format");
        match scalar {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, "1.84467440737096e+19");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let fixed = num2str_builtin(
            Value::Int(IntValue::U64(u64::MAX)),
            vec![Value::String("%.2f".into())],
        )
        .expect("fixed integer format");
        match fixed {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, format!("{}.00", u64::MAX));
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let rounded = num2str_builtin(
            Value::Int(IntValue::U64(u64::MAX)),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("precision integer format");
        match rounded {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, "1.845e+19");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_complex_values() {
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (5.0, -6.0)], vec![1, 2]).expect("cplx");
        let out = num2str_builtin(Value::ComplexTensor(complex), Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, "3 + 4i  5 - 6i");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn num2str_preserves_real_and_complex_rank_projections() {
        let Value::CharArray(real_scalar) = num2str_builtin(
            Value::Tensor(Tensor::new(vec![7.0], Vec::new()).unwrap()),
            Vec::new(),
        )
        .expect("rank-zero real num2str") else {
            panic!("expected char array");
        };
        assert_eq!((real_scalar.rows, real_scalar.cols), (1, 1));
        assert_eq!(
            real_scalar.data(),
            runmat_value::encode_utf16_code_units("7")
        );

        let Value::CharArray(real_vector) = num2str_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
            Vec::new(),
        )
        .expect("rank-one real num2str") else {
            panic!("expected char array");
        };
        assert_eq!((real_vector.rows, real_vector.cols), (2, 1));
        assert_eq!(
            real_vector.data(),
            runmat_value::encode_utf16_code_units("12")
        );

        let Value::CharArray(complex_scalar) = num2str_builtin(
            Value::ComplexTensor(ComplexTensor::new(vec![(7.0, 1.0)], Vec::new()).unwrap()),
            Vec::new(),
        )
        .expect("rank-zero complex num2str") else {
            panic!("expected char array");
        };
        assert_eq!((complex_scalar.rows, complex_scalar.cols), (0, 0));
        assert!(complex_scalar.data().is_empty());

        let Value::CharArray(complex_vector) = num2str_builtin(
            Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2]).unwrap(),
            ),
            Vec::new(),
        )
        .expect("rank-one complex num2str") else {
            panic!("expected char array");
        };
        assert_eq!(complex_vector.rows, 1);
        assert_eq!(
            runmat_value::decode_utf16_code_units(complex_vector.data())
                .expect("valid text fixture"),
            "1 + 2i  3 + 4i"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_complex_integer_tensor_reads_exact_storage_without_mirror() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            IntegerStorage::U64(vec![7, 0]),
        )
        .expect("complex integer storage");
        let tensor =
            ComplexTensor::new_integer(storage, vec![1, 2]).expect("complex integer tensor");

        let out = num2str_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, "1.84467440737096e+19 + 7i  9.00719925474099e+15");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_complex_integer_tensor_fixed_format_keeps_exact_digits() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            IntegerStorage::U64(vec![7, 0]),
        )
        .expect("complex integer storage");
        let tensor =
            ComplexTensor::new_integer(storage, vec![1, 2]).expect("complex integer tensor");

        let out = num2str_builtin(
            Value::ComplexTensor(tensor),
            vec![Value::String("%.0f".to_string().into())],
        )
        .expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, format!("{} + 7i  9007199254740993", u64::MAX));
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_local_decimal() {
        let _guard = test_support::global_state_test_guard();
        let previous = crate::builtins::common::env::var("RUNMAT_DECIMAL_SEPARATOR").ok();
        crate::builtins::common::env::set_var("RUNMAT_DECIMAL_SEPARATOR", ",");
        let out =
            num2str_builtin(Value::Num(0.5), vec![Value::String("local".into())]).expect("num2str");
        crate::builtins::common::env::set_var("RUNMAT_DECIMAL_SEPARATOR", "\u{1F642}");
        let supplementary = num2str_builtin(Value::Num(0.5), vec![Value::String("local".into())])
            .expect("supplementary decimal separator");
        match previous {
            Some(value) => {
                crate::builtins::common::env::set_var("RUNMAT_DECIMAL_SEPARATOR", &value)
            }
            None => crate::builtins::common::env::remove_var("RUNMAT_DECIMAL_SEPARATOR"),
        }
        match out {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, "0,5");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        let Value::CharArray(supplementary) = supplementary else {
            panic!("expected character output")
        };
        assert_eq!((supplementary.rows, supplementary.cols), (1, 4));
        assert_eq!(
            supplementary
                .data()
                .iter()
                .map(|unit| unit.raw())
                .collect::<Vec<_>>(),
            [0x30, 0xD83D, 0xDE42, 0x35]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_logical_array() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).expect("logical");
        let out = num2str_builtin(Value::LogicalArray(logical), Vec::new()).expect("num2str");
        match out {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid text fixture");
                assert_eq!(text, "1  0  1");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_gpu_tensor_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![10.5, 20.5], vec![1, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let out = num2str_builtin(Value::GpuTensor(handle), vec![Value::String("%.1f".into())])
                .expect("num2str");
            match out {
                Value::CharArray(ca) => {
                    let text: String = runmat_value::decode_utf16_code_units(ca.data())
                        .expect("valid text fixture");
                    assert_eq!(text, "10.5  20.5");
                }
                other => panic!("expected char array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_invalid_input_type() {
        let err =
            error_message(num2str_builtin(Value::String("hello".into()), Vec::new()).unwrap_err());
        assert!(err.contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_invalid_format_string() {
        let err = error_message(
            num2str_builtin(Value::Num(1.0), vec![Value::String("%q".into())]).unwrap_err(),
        );
        assert!(err.contains("unsupported format string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn num2str_oversized_format_controls_return_errors() {
        let width = error_message(
            num2str_builtin(
                Value::Num(1.0),
                vec![Value::String("%9999999999999999999999999999999999f".into())],
            )
            .unwrap_err(),
        );
        assert!(width.contains("format width is too large"));

        let precision = error_message(
            num2str_builtin(
                Value::Num(1.0),
                vec![Value::String(
                    "%.9999999999999999999999999999999999f".into(),
                )],
            )
            .unwrap_err(),
        );
        assert!(precision.contains("format precision is too large"));

        let bounded_width = error_message(
            num2str_builtin(
                Value::Num(1.0),
                vec![Value::String(format!("%{}f", MAX_FORMAT_FIELD + 1).into())],
            )
            .unwrap_err(),
        );
        assert!(bounded_width.contains("format width exceeds"));
        let at_limit = num2str_builtin(
            Value::Num(1.0),
            vec![Value::String(format!("%{}f", MAX_FORMAT_FIELD).into())],
        )
        .expect("supported format width");
        let Value::CharArray(chars) = at_limit else {
            panic!("expected character array");
        };
        assert_eq!(chars.cols, MAX_FORMAT_FIELD);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_controls_are_rejected_before_resident_data_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
            let data = tensor.materialize_f64();
            let view = runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let error = error_message(
                num2str_builtin(Value::GpuTensor(handle), vec![Value::String("%q".into())])
                    .unwrap_err(),
            );
            assert!(error.contains("unsupported format string"));
        });
    }
}
