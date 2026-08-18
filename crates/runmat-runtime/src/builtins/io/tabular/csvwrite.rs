//! MATLAB-compatible `csvwrite` builtin for RunMat.
//!
//! `csvwrite` is an older convenience wrapper that persists numeric matrices to
//! comma-separated text files. Modern MATLAB code typically prefers
//! `writematrix`, but many legacy scripts still depend on `csvwrite`'s terse
//! API and zero-based offset arguments. This implementation mirrors those
//! semantics while integrating with RunMat's builtin framework.

use std::io::Write;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_filesystem::OpenOptions;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "csvwrite";

const CSVWRITE_BYTES_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "csvwrite-bytes-written-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "requesting a bytes-written output from csvwrite is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CsvwriteBytesOutputExtension"),
};
const CSVWRITE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "csvwrite-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "direct csvwrite of resident data or controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CsvwriteResidentInputExtension"),
};
pub const CSVWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CSVWRITE_BYTES_OUTPUT_EXTENSION,
    CSVWRITE_RESIDENT_INPUT_EXTENSION,
];
const CSVWRITE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "M", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight real integer classes are documented and serialize directly from authoritative storage." },
    BuiltinIntegerInputCapability { name: "row", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The zero-based row offset accepts all eight integer classes and is checked exactly." },
    BuiltinIntegerInputCapability { name: "col", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The zero-based column offset accepts all eight integer classes and is checked exactly." },
];
pub const CSVWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "csvwrite(filename, integer_M, integer_row?, integer_col?)", inputs: &CSVWRITE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer matrix elements serialize with their exact values. Resident input is independently gated and gathered through the handle owner." }];

const CSVWRITE_INPUTS_FILENAME_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CSV output path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix data to write.",
    },
];
const CSVWRITE_INPUTS_FILENAME_DATA_ROW_COL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CSV output path.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical matrix data to write.",
    },
    BuiltinParamDescriptor {
        name: "row",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based row offset before writing values.",
    },
    BuiltinParamDescriptor {
        name: "col",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based column offset before writing values.",
    },
];
const CSVWRITE_NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];
const CSVWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "csvwrite(filename, M)",
        inputs: &CSVWRITE_INPUTS_FILENAME_DATA,
        outputs: &CSVWRITE_NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "csvwrite(filename, M, row, col)",
        inputs: &CSVWRITE_INPUTS_FILENAME_DATA_ROW_COL,
        outputs: &CSVWRITE_NO_OUTPUT,
    },
];
const CSVWRITE_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.FILENAME",
    identifier: None,
    when: "Filename argument is not a scalar string/char vector.",
    message: "csvwrite: invalid filename input",
};
const CSVWRITE_ERROR_FILENAME_EMPTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.FILENAME_EMPTY",
    identifier: None,
    when: "Filename resolves to an empty string.",
    message: "csvwrite: filename must not be empty",
};
const CSVWRITE_ERROR_OFFSETS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.OFFSETS",
    identifier: None,
    when: "Offset arguments are missing, malformed, or out of bounds.",
    message: "csvwrite: invalid row/column offsets",
};
const CSVWRITE_ERROR_DATA_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.DATA_SHAPE",
    identifier: None,
    when: "Input data is not a 2-D matrix.",
    message: "csvwrite: input must be 2-D",
};
const CSVWRITE_ERROR_DATA_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.DATA_INPUT",
    identifier: None,
    when: "Input data cannot be converted to a numeric/logical tensor.",
    message: "csvwrite: input must be numeric or logical",
};
const CSVWRITE_ERROR_IO_OPEN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.IO_OPEN",
    identifier: None,
    when: "Output file cannot be opened.",
    message: "csvwrite: unable to open file for writing",
};
const CSVWRITE_ERROR_IO_WRITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CSVWRITE.IO_WRITE",
    identifier: None,
    when: "Output file write/flush fails.",
    message: "csvwrite: write failed",
};
const CSVWRITE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    CSVWRITE_ERROR_FILENAME,
    CSVWRITE_ERROR_FILENAME_EMPTY,
    CSVWRITE_ERROR_OFFSETS,
    CSVWRITE_ERROR_DATA_INPUT,
    CSVWRITE_ERROR_DATA_SHAPE,
    CSVWRITE_ERROR_IO_OPEN,
    CSVWRITE_ERROR_IO_WRITE,
];
pub const CSVWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CSVWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CSVWRITE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::csvwrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "csvwrite",
    op_kind: GpuOpKind::Custom("io-csvwrite"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host; gpuArray inputs are gathered before serialisation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::csvwrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "csvwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side file I/O.",
};

fn csvwrite_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn csvwrite_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "csvwrite",
    category = "io/tabular",
    summary = "Write numeric matrices to CSV files.",
    keywords = "csvwrite,csv,write,row offset,column offset",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::tabular::csvwrite::CSVWRITE_DESCRIPTOR),
    extensions(crate::builtins::io::tabular::csvwrite::CSVWRITE_EXTENSIONS),
    integer_capabilities(crate::builtins::io::tabular::csvwrite::CSVWRITE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::tabular::csvwrite"
)]
async fn csvwrite_builtin(
    filename: Value,
    data: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_DATA_INPUT,
            "csvwrite: too many output arguments",
        ));
    }
    if requested_outputs.is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CSVWRITE_BYTES_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(filename, Value::GpuTensor(_))
        || matches!(data, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CSVWRITE_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let filename_value = gather_if_needed_async(&filename)
        .await
        .map_err(map_control_flow)?;
    let path = resolve_path(&filename_value)?;

    let mut gathered_offsets = Vec::with_capacity(rest.len());
    for value in &rest {
        gathered_offsets.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    let (row_offset, col_offset) = parse_offsets(&gathered_offsets)?;

    let gathered_data = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let matrix = CsvMatrix::from_value(gathered_data)?;
    matrix.ensure_matrix_shape()?;

    let bytes = write_csv(&path, &matrix, row_offset, col_offset).await?;
    match requested_outputs {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![Value::Num(bytes as f64)])),
        Some(_) => Err(csvwrite_error_with(
            &CSVWRITE_ERROR_DATA_INPUT,
            "csvwrite: too many output arguments",
        )),
        None => Ok(Value::Num(bytes as f64)),
    }
}

enum CsvMatrix {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl CsvMatrix {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map(Self::Complex)
                .map_err(|e| {
                    csvwrite_error_with(&CSVWRITE_ERROR_DATA_INPUT, format!("csvwrite: {e}"))
                }),
            Value::ComplexTensor(tensor) if tensor.integer_storage().is_none() => {
                Ok(Self::Complex(tensor))
            }
            Value::ComplexTensor(_) => Err(csvwrite_error_with(
                &CSVWRITE_ERROR_DATA_INPUT,
                "csvwrite: complex integer input is not supported",
            )),
            value => tensor::value_into_tensor_for("csvwrite", value)
                .map(Self::Real)
                .map_err(|msg| {
                    csvwrite_error_with(&CSVWRITE_ERROR_DATA_INPUT, format!("csvwrite: {msg}"))
                }),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Real(t) => &t.shape,
            Self::Complex(t) => &t.shape,
        }
    }

    fn ensure_matrix_shape(&self) -> BuiltinResult<()> {
        ensure_matrix_dims(self.shape())
    }

    fn rows(&self) -> usize {
        self.shape().first().copied().unwrap_or(1)
    }
    fn cols(&self) -> usize {
        self.shape().get(1).copied().unwrap_or(1)
    }

    fn format_at(&self, idx: usize) -> String {
        match self {
            Self::Real(tensor) => format_tensor_value(tensor, idx),
            Self::Complex(tensor) => {
                let (re, im) = tensor
                    .numeric_value_at(idx)
                    .expect("index within authoritative complex storage");
                format_complex(re.materialize_f64(), im.materialize_f64())
            }
        }
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        _ => Err(csvwrite_error_with(
            &CSVWRITE_ERROR_FILENAME,
            "csvwrite: filename must be a string scalar or character vector",
        ))?,
    };

    if raw.trim().is_empty() {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_FILENAME_EMPTY,
            CSVWRITE_ERROR_FILENAME_EMPTY.message,
        ));
    }

    let expanded = expand_user_path(&raw, BUILTIN_NAME)
        .map_err(|msg| csvwrite_error_with(&CSVWRITE_ERROR_FILENAME, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn parse_offsets(args: &[Value]) -> BuiltinResult<(usize, usize)> {
    match args.len() {
        0 => Ok((0, 0)),
        2 => {
            let row = parse_offset(&args[0], "row offset")?;
            let col = parse_offset(&args[1], "column offset")?;
            Ok((row, col))
        }
        _ => Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            "csvwrite: offsets must be provided as two numeric arguments (row, column)",
        )),
    }
}

fn parse_offset(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(i) => i.try_to_usize().ok_or_else(|| {
            csvwrite_error_with(
                &CSVWRITE_ERROR_OFFSETS,
                format!("csvwrite: {context} must be >= 0"),
            )
        }),
        Value::Num(n) => coerce_offset_from_float(*n, context),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            let len = tensor::tensor_element_len(t);
            if len != 1 {
                return Err(csvwrite_error_with(
                    &CSVWRITE_ERROR_OFFSETS,
                    format!("csvwrite: {context} must be a scalar, got {} elements", len),
                ));
            }
            let value = t
                .numeric_value_at(0)
                .expect("one-element authoritative numeric storage");
            if let Some(value) = value.into_int_value() {
                return value.try_to_usize().ok_or_else(|| {
                    csvwrite_error_with(
                        &CSVWRITE_ERROR_OFFSETS,
                        format!("csvwrite: {context} must be >= 0"),
                    )
                });
            }
            coerce_offset_from_float(value.materialize_f64(), context)
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != 1 {
                return Err(csvwrite_error_with(
                    &CSVWRITE_ERROR_OFFSETS,
                    format!(
                        "csvwrite: {context} must be a scalar, got {} elements",
                        logical.data.len()
                    ),
                ));
            }
            Ok(if logical.data[0] != 0 { 1 } else { 0 })
        }
        other => Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} must be numeric, got {:?}", other),
        )),
    }
}

fn coerce_offset_from_float(value: f64, context: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} must be finite"),
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} must be an integer"),
        ));
    }
    if rounded < 0.0 {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} must be >= 0"),
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} is too large"),
        ));
    }
    Ok(rounded as usize)
}

fn ensure_matrix_dims(shape: &[usize]) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err(csvwrite_error_with(
        &CSVWRITE_ERROR_DATA_SHAPE,
        "csvwrite: input must be 2-D; reshape before writing",
    ))
}

async fn write_csv(
    path: &Path,
    matrix: &CsvMatrix,
    row_offset: usize,
    col_offset: usize,
) -> BuiltinResult<usize> {
    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    let mut file = options.open_async(path).await.map_err(|err| {
        csvwrite_error_with_source(
            &CSVWRITE_ERROR_IO_OPEN,
            format!(
                "csvwrite: unable to open \"{}\" for writing ({err})",
                path.display()
            ),
            err,
        )
    })?;

    let line_ending = "\n";
    let rows = matrix.rows();
    let cols = matrix.cols();

    let mut bytes_written = 0usize;

    for _ in 0..row_offset {
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                format!("csvwrite: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
    }

    if rows == 0 || cols == 0 {
        file.flush_async().await.map_err(|err| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                format!("csvwrite: failed to flush output ({err})"),
                err,
            )
        })?;
        return Ok(bytes_written);
    }

    for row in 0..rows {
        let mut fields = Vec::with_capacity(col_offset + cols);
        for _ in 0..col_offset {
            fields.push(String::new());
        }
        for col in 0..cols {
            let idx = row + col * rows;
            fields.push(matrix.format_at(idx));
        }
        let line = fields.join(",");
        if !line.is_empty() {
            file.write_all(line.as_bytes()).map_err(|err| {
                csvwrite_error_with_source(
                    &CSVWRITE_ERROR_IO_WRITE,
                    format!("csvwrite: failed to write value ({err})"),
                    err,
                )
            })?;
            bytes_written += line.len();
        }
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_WRITE,
                format!("csvwrite: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
    }

    file.flush_async().await.map_err(|err| {
        csvwrite_error_with_source(
            &CSVWRITE_ERROR_IO_WRITE,
            format!("csvwrite: failed to flush output ({err})"),
            err,
        )
    })?;

    Ok(bytes_written)
}

fn format_complex(re: f64, im: f64) -> String {
    let real = format_numeric(re);
    let imag = format_numeric(im.abs());
    if im.is_sign_negative() {
        format!("{real}-{imag}i")
    } else {
        format!("{real}+{imag}i")
    }
}

fn format_numeric(value: f64) -> String {
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

    let precision: i32 = 5;
    let abs = value.abs();
    let exp10 = abs.log10().floor() as i32;
    let use_scientific = exp10 < -4 || exp10 >= precision;

    let raw = if use_scientific {
        let digits_after = (precision - 1).max(0) as usize;
        format!("{:.*e}", digits_after, value)
    } else {
        let decimals = (precision - 1 - exp10).max(0) as usize;
        format!("{:.*}", decimals, value)
    };

    let mut trimmed = trim_trailing_zeros(raw);
    if trimmed == "-0" {
        trimmed = "0".to_string();
    }
    trimmed
}

fn format_tensor_value(tensor: &Tensor, idx: usize) -> String {
    let value = tensor
        .numeric_value_at(idx)
        .expect("index within authoritative numeric storage");
    if let Some(value) = value.into_int_value() {
        return value.decimal_string();
    }
    format_numeric(value.materialize_f64())
}

fn trim_trailing_zeros(mut value: String) -> String {
    if let Some(exp_pos) = value.find(['e', 'E']) {
        let exponent = value.split_off(exp_pos);
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
        value.push_str(&normalize_exponent(&exponent));
        value
    } else {
        if value.contains('.') {
            while value.ends_with('0') {
                value.pop();
            }
            if value.ends_with('.') {
                value.pop();
            }
        }
        if value.is_empty() {
            "0".to_string()
        } else {
            value
        }
    }
}

fn normalize_exponent(exponent: &str) -> String {
    if exponent.len() <= 1 {
        return exponent.to_string();
    }
    let mut chars = exponent.chars();
    let marker = chars.next().unwrap();
    let rest: String = chars.collect();
    match rest.parse::<i32>() {
        Ok(parsed) => format!("{}{:+03}", marker, parsed),
        Err(_) => exponent.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, IntegerStorage, LogicalArray};

    use crate::builtins::common::fs as fs_helpers;
    use crate::builtins::common::test_support;

    fn csvwrite_builtin(filename: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _provider_lock = runmat_filesystem::provider_override_lock();
        futures::executor::block_on(super::csvwrite_builtin(filename, data, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CSVWRITE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"csvwrite(filename, M)"));
        assert!(labels.contains(&"csvwrite(filename, M, row, col)"));
    }

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_csvwrite_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    fn line_ending() -> &'static str {
        "\n"
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_writes_basic_matrix() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, format!("1,2,3{le}4,5,6{le}", le = line_ending()));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_preserves_typed_integer_matrix_values_exactly() {
        let path = temp_path("csv");
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 17, (1_u64 << 53) + 1, 29]),
            vec![2, 2],
        )
        .expect("typed integer matrix");
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(
            contents,
            format!(
                "18446744073709551615,9007199254740993{le}17,29{le}",
                le = line_ending()
            )
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn csvwrite_serializes_all_eight_integer_classes() {
        let cases = [
            (IntegerStorage::I8(vec![-8]), "-8\n"),
            (IntegerStorage::I16(vec![-16]), "-16\n"),
            (IntegerStorage::I32(vec![-32]), "-32\n"),
            (
                IntegerStorage::I64(vec![i64::MIN]),
                "-9223372036854775808\n",
            ),
            (IntegerStorage::U8(vec![8]), "8\n"),
            (IntegerStorage::U16(vec![16]), "16\n"),
            (IntegerStorage::U32(vec![32]), "32\n"),
            (
                IntegerStorage::U64(vec![u64::MAX]),
                "18446744073709551615\n",
            ),
        ];
        for (storage, expected) in cases {
            let path = temp_path("csv");
            let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            csvwrite_builtin(
                Value::from(path.to_string_lossy().into_owned()),
                Value::Tensor(tensor),
                Vec::new(),
            )
            .unwrap();
            assert_eq!(fs::read_to_string(&path).unwrap(), expected);
            let _ = fs::remove_file(path);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_honours_offsets() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))],
        )
        .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(
            contents,
            format!("{le},,1,3{le},,2,4{le}", le = line_ending())
        );
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_handles_gpu_tensors() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let path = temp_path("csv");
            let tensor = Tensor::new(vec![0.5, 1.5], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let filename = path.to_string_lossy().into_owned();

            csvwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new())
                .expect("csvwrite");

            let contents = fs::read_to_string(&path).expect("read contents");
            assert_eq!(contents, format!("0.5,1.5{le}", le = line_ending()));
            let _ = fs::remove_file(path);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_formats_with_short_g_precision() {
        let path = temp_path("csv");
        let values =
            Tensor::new(vec![12.3456, 1_234_567.0, 0.000123456, -0.0], vec![1, 4]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(Value::from(filename), Value::Tensor(values), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(
            contents,
            format!("12.346,1.2346e+06,0.00012346,0{le}", le = line_ending())
        );
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_rejects_negative_offsets() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let err = csvwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::Num(-1.0), Value::Num(0.0)],
        )
        .expect_err("negative offsets should be rejected");
        let message = err.message().to_string();
        assert!(
            message.contains("row offset"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn csvwrite_offset_parser_preserves_typed_integer_tensor_bounds() {
        let offset =
            Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed offset");
        assert_eq!(
            parse_offset(&Value::Tensor(offset), "row offset").unwrap(),
            7
        );

        let negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("negative");
        assert!(parse_offset(&Value::Tensor(negative), "row offset").is_err());

        let too_large = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("too large");
        let parsed = parse_offset(&Value::Tensor(too_large), "row offset");
        if usize::try_from(u64::MAX).is_ok() {
            assert_eq!(parsed.unwrap(), usize::MAX);
        } else {
            assert!(parsed.is_err());
        }

        assert!(parse_offset(&Value::Num(usize::MAX as f64), "row offset").is_err());
        assert!(parse_offset(&Value::Num((usize::MAX as f64) + 1.0), "row offset").is_err());
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_handles_wgpu_provider_gather() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            panic!("wgpu provider not registered");
        };

        let path = temp_path("csv");
        let tensor = Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, format!("2,4{le}", le = line_ending()));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_expands_home_directory() {
        let Some(mut home) = fs_helpers::home_directory() else {
            // Skip when home directory cannot be determined.
            return;
        };
        let filename = format!(
            "runmat_csvwrite_home_{}_{}.csv",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        );
        home.push(&filename);

        let tilde_path = format!("~/{}", filename);
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).unwrap();

        if let Err(error) =
            csvwrite_builtin(Value::from(tilde_path), Value::Tensor(tensor), Vec::new())
        {
            if error.message().contains("Operation not permitted")
                || error.message().contains("Permission denied")
            {
                return;
            }
            panic!("csvwrite: {error}");
        }

        let contents = fs::read_to_string(&home).expect("read contents");
        assert_eq!(contents, format!("42{le}", le = line_ending()));
        let _ = fs::remove_file(home);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_rejects_non_numeric_inputs() {
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        let err = csvwrite_builtin(
            Value::from(filename),
            Value::String("abc".into()),
            Vec::new(),
        )
        .expect_err("csvwrite should fail");
        let message = err.message().to_string();
        assert!(
            message.contains("csvwrite"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvwrite_accepts_logical_arrays() {
        let path = temp_path("csv");
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(
            Value::from(filename),
            Value::LogicalArray(logical),
            Vec::new(),
        )
        .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, format!("1,1{le}0,0{le}", le = line_ending()));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn csvwrite_writes_complex_double_and_single_with_lf() {
        for complex in [
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap(),
            ComplexTensor::from_complex_storage(
                runmat_builtins::ComplexStorage::F32(vec![(1.0, 2.0), (3.0, -4.0)]),
                vec![1, 2],
            )
            .unwrap(),
        ] {
            let path = temp_path("csv");
            csvwrite_builtin(
                Value::from(path.to_string_lossy().into_owned()),
                Value::ComplexTensor(complex),
                Vec::new(),
            )
            .expect("complex csvwrite");
            assert_eq!(fs::read(&path).unwrap(), b"1+2i,3-4i\n");
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn csvwrite_declares_independent_output_and_resident_extensions() {
        assert_eq!(CSVWRITE_EXTENSIONS[0].id, "csvwrite-bytes-written-output");
        assert_eq!(CSVWRITE_EXTENSIONS[1].id, "csvwrite-resident-input");
        assert_eq!(CSVWRITE_INTEGER_CAPABILITIES[0].inputs.len(), 3);
        assert!(CSVWRITE_INTEGER_CAPABILITIES[0]
            .inputs
            .iter()
            .all(|input| input.classes.len() == 8));
    }

    #[test]
    fn csvwrite_output_and_resident_gates_run_before_side_effects_or_provider_access() {
        let path = temp_path("csv");
        let filename = Value::from(path.to_string_lossy().into_owned());
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let outputs = crate::output_count::push_output_count(Some(1));
        let error = csvwrite_builtin(
            filename.clone(),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            CSVWRITE_BYTES_OUTPUT_EXTENSION.error_identifier
        );
        assert!(!path.exists(), "output gate must precede file creation");
        drop(outputs);

        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 396,
            descriptor: Default::default(),
        });
        let error = csvwrite_builtin(filename.clone(), resident, Vec::new()).unwrap_err();
        assert_eq!(
            error.identifier(),
            CSVWRITE_RESIDENT_INPUT_EXTENSION.error_identifier
        );
        assert!(!path.exists(), "resident gate must precede file creation");
        drop(strict);

        let no_outputs = crate::output_count::push_output_count(Some(0));
        let value = csvwrite_builtin(
            filename,
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(value, Value::OutputList(Vec::new()));
        drop(no_outputs);
        let _ = fs::remove_file(path);

        let path = temp_path("csv");
        let filename = Value::from(path.to_string_lossy().into_owned());
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let too_many_outputs = crate::output_count::push_output_count(Some(2));
        let error = csvwrite_builtin(
            filename,
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.message().contains("too many output arguments"));
        assert!(!path.exists(), "output arity must precede file creation");
        drop(too_many_outputs);
    }
}
