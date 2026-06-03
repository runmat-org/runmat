//! MATLAB-compatible `fwrite` builtin for RunMat.
use std::io::{Seek, SeekFrom, Write};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem::File;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fwrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fwrite",
    op_kind: GpuOpKind::Custom("file-io-write"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only binary file I/O; GPU arguments are gathered to the CPU prior to writing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fwrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is never fused; metadata recorded for completeness.",
};

const BUILTIN_NAME: &str = "fwrite";

const FWRITE_OUTPUT_COUNT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "count",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of elements successfully written.",
}];
const FWRITE_INPUTS_FID_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
];
const FWRITE_INPUTS_FID_DATA_PRECISION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Write precision label (for example \"uint8\", \"double\").",
    },
];
const FWRITE_INPUTS_FID_DATA_PRECISION_SKIP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Write precision label (for example \"uint8\", \"double\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element written.",
    },
];
const FWRITE_INPUTS_FID_DATA_PRECISION_MACHINEFMT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Write precision label (for example \"uint8\", \"double\").",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FWRITE_INPUTS_FID_DATA_PRECISION_SKIP_MACHINEFMT: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Write precision label (for example \"uint8\", \"double\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element written.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data)",
        inputs: &FWRITE_INPUTS_FID_DATA,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data, precision)",
        inputs: &FWRITE_INPUTS_FID_DATA_PRECISION,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data, precision, skip)",
        inputs: &FWRITE_INPUTS_FID_DATA_PRECISION_SKIP,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data, precision, machinefmt)",
        inputs: &FWRITE_INPUTS_FID_DATA_PRECISION_MACHINEFMT,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data, precision, skip, machinefmt)",
        inputs: &FWRITE_INPUTS_FID_DATA_PRECISION_SKIP_MACHINEFMT,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
];

const FWRITE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.INVALID_INPUT",
    identifier: Some("RunMat:fwrite:InvalidInput"),
    when: "Identifier, payload, or argument cardinality/type constraints are violated.",
    message: "fwrite: invalid input arguments",
};
const FWRITE_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fwrite:InvalidIdentifier"),
    when: "Identifier does not refer to a writable open file.",
    message: "fwrite: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FWRITE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.INVALID_OPTION",
    identifier: Some("RunMat:fwrite:InvalidOption"),
    when: "Precision, skip, or machine format options are invalid.",
    message: "fwrite: invalid option configuration",
};
const FWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.IO",
    identifier: Some("RunMat:fwrite:IoFailure"),
    when: "Write/seek operation fails.",
    message: "fwrite: file write failed",
};
const FWRITE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow conversion fails.",
    message: "fwrite: internal error",
};
const FWRITE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FWRITE_ERROR_INVALID_INPUT,
    FWRITE_ERROR_INVALID_IDENTIFIER,
    FWRITE_ERROR_INVALID_OPTION,
    FWRITE_ERROR_IO,
    FWRITE_ERROR_INTERNAL,
];
pub const FWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FWRITE_ERRORS,
};

fn fwrite_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("fwrite: ").unwrap_or(detail);
    fwrite_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn fwrite_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FWRITE_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_string_result<T>(
    result: Result<T, String>,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    result.map_err(|detail| fwrite_error_with_detail(error, detail))
}

#[runtime_builtin(
    name = "fwrite",
    category = "io/filetext",
    summary = "Write binary data to file identifiers.",
    keywords = "fwrite,file,io,binary,precision",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fwrite_type),
    descriptor(crate::builtins::io::filetext::fwrite::FWRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::filetext::fwrite"
)]
async fn fwrite_builtin(fid: Value, data: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &data, &rest).await?;
    Ok(Value::Num(eval.count as f64))
}

/// Result of an `fwrite` evaluation.
#[derive(Debug, Clone)]
pub struct FwriteEval {
    count: usize,
}

impl FwriteEval {
    fn new(count: usize) -> Self {
        Self { count }
    }

    /// Number of elements successfully written.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Evaluate the `fwrite` builtin without invoking the runtime dispatcher.
pub async fn evaluate(
    fid_value: &Value,
    data_value: &Value,
    rest: &[Value],
) -> BuiltinResult<FwriteEval> {
    let fid_host = gather_value(fid_value).await?;
    let fid = map_string_result(parse_fid(&fid_host), &FWRITE_ERROR_INVALID_INPUT)?;
    if fid < 0 {
        return Err(fwrite_error_with_detail(
            &FWRITE_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fwrite_error_with_detail(
            &FWRITE_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let info = registry::info_for(fid).ok_or_else(|| {
        fwrite_error_with_message(
            FWRITE_ERROR_INVALID_IDENTIFIER.message,
            &FWRITE_ERROR_INVALID_IDENTIFIER,
        )
    })?;
    let handle = registry::take_handle(fid).ok_or_else(|| {
        fwrite_error_with_message(
            FWRITE_ERROR_INVALID_IDENTIFIER.message,
            &FWRITE_ERROR_INVALID_IDENTIFIER,
        )
    })?;

    let data_host = gather_value(data_value).await?;
    let rest_host = gather_args(rest).await?;
    let (precision_arg, skip_arg, machine_arg) =
        map_string_result(classify_arguments(&rest_host), &FWRITE_ERROR_INVALID_INPUT)?;

    let precision_spec =
        map_string_result(parse_precision(precision_arg), &FWRITE_ERROR_INVALID_OPTION)?;
    let skip_bytes = map_string_result(parse_skip(skip_arg), &FWRITE_ERROR_INVALID_OPTION)?;
    let machine_format = map_string_result(
        parse_machine_format(machine_arg, &info.machinefmt),
        &FWRITE_ERROR_INVALID_OPTION,
    )?;

    let mut guard = handle.lock().map_err(|_| {
        fwrite_error_with_detail(
            &FWRITE_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard.as_mut().ok_or_else(|| {
        fwrite_error_with_message(
            FWRITE_ERROR_INVALID_IDENTIFIER.message,
            &FWRITE_ERROR_INVALID_IDENTIFIER,
        )
    })?;

    let elements = map_string_result(flatten_elements(&data_host), &FWRITE_ERROR_INVALID_INPUT)?;
    let count = map_string_result(
        write_elements(file, &elements, precision_spec, skip_bytes, machine_format),
        &FWRITE_ERROR_IO,
    )?;
    Ok(FwriteEval::new(count))
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(gathered)
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    let scalar = match value {
        Value::Num(n) => *n,
        Value::Int(int) => int.to_f64(),
        _ => return Err("fwrite: file identifier must be numeric".to_string()),
    };
    if !scalar.is_finite() {
        return Err("fwrite: file identifier must be finite".to_string());
    }
    if scalar.fract().abs() > f64::EPSILON {
        return Err("fwrite: file identifier must be an integer".to_string());
    }
    Ok(scalar as i32)
}

type FwriteArgs<'a> = (Option<&'a Value>, Option<&'a Value>, Option<&'a Value>);

fn classify_arguments(args: &[Value]) -> Result<FwriteArgs<'_>, String> {
    match args.len() {
        0 => Ok((None, None, None)),
        1 => {
            if is_string_like(&args[0]) {
                Ok((Some(&args[0]), None, None))
            } else {
                Err(
                    "fwrite: precision argument must be a string scalar or character vector"
                        .to_string(),
                )
            }
        }
        2 => {
            if !is_string_like(&args[0]) {
                return Err(
                    "fwrite: precision argument must be a string scalar or character vector"
                        .to_string(),
                );
            }
            if is_numeric_like(&args[1]) {
                Ok((Some(&args[0]), Some(&args[1]), None))
            } else if is_string_like(&args[1]) {
                Ok((Some(&args[0]), None, Some(&args[1])))
            } else {
                Err("fwrite: invalid argument combination (expected numeric skip or machine format string)".to_string())
            }
        }
        3 => {
            if !is_string_like(&args[0]) || !is_numeric_like(&args[1]) || !is_string_like(&args[2])
            {
                return Err("fwrite: expected arguments (precision, skip, machinefmt)".to_string());
            }
            Ok((Some(&args[0]), Some(&args[1]), Some(&args[2])))
        }
        _ => Err("fwrite: too many input arguments".to_string()),
    }
}

fn is_string_like(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::CharArray(ca) => ca.rows == 1,
        Value::StringArray(sa) => sa.data.len() == 1,
        _ => false,
    }
}

fn is_numeric_like(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(t) => t.data.len() == 1,
        Value::LogicalArray(la) => la.data.len() == 1,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
struct WriteSpec {
    input: InputType,
}

impl WriteSpec {
    fn default() -> Self {
        Self {
            input: InputType::UInt8,
        }
    }
}

fn parse_precision(arg: Option<&Value>) -> Result<WriteSpec, String> {
    match arg {
        None => Ok(WriteSpec::default()),
        Some(value) => {
            let text = scalar_string(
                value,
                "fwrite: precision argument must be a string scalar or character vector",
            )?;
            parse_precision_string(&text)
        }
    }
}

fn parse_precision_string(raw: &str) -> Result<WriteSpec, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("fwrite: precision argument must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some((lhs, rhs)) = lower.split_once("=>") {
        let lhs = lhs.trim();
        let rhs = rhs.trim();
        let input = parse_input_label(lhs)?;
        let output = parse_input_label(rhs)?;
        if input != output {
            return Err(
                "fwrite: differing input/output precisions are not implemented yet".to_string(),
            );
        }
        Ok(WriteSpec { input })
    } else {
        parse_input_label(lower.trim()).map(|input| WriteSpec { input })
    }
}

fn parse_skip(arg: Option<&Value>) -> Result<usize, String> {
    match arg {
        None => Ok(0),
        Some(value) => {
            let scalar = numeric_scalar(value, "fwrite: skip must be numeric")?;
            if !scalar.is_finite() {
                return Err("fwrite: skip value must be finite".to_string());
            }
            if scalar < 0.0 {
                return Err("fwrite: skip value must be non-negative".to_string());
            }
            let rounded = scalar.round();
            if (rounded - scalar).abs() > f64::EPSILON {
                return Err("fwrite: skip value must be an integer".to_string());
            }
            if rounded > i64::MAX as f64 {
                return Err("fwrite: skip value is too large".to_string());
            }
            Ok(rounded as usize)
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum MachineFormat {
    Native,
    LittleEndian,
    BigEndian,
}

impl MachineFormat {
    fn to_endianness(self) -> Endianness {
        match self {
            MachineFormat::Native => {
                if cfg!(target_endian = "little") {
                    Endianness::Little
                } else {
                    Endianness::Big
                }
            }
            MachineFormat::LittleEndian => Endianness::Little,
            MachineFormat::BigEndian => Endianness::Big,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Endianness {
    Little,
    Big,
}

fn parse_machine_format(arg: Option<&Value>, default_label: &str) -> Result<MachineFormat, String> {
    match arg {
        Some(value) => {
            let text = scalar_string(
                value,
                "fwrite: machine format must be a string scalar or character vector",
            )?;
            machine_format_from_label(&text)
        }
        None => machine_format_from_label(default_label),
    }
}

fn machine_format_from_label(label: &str) -> Result<MachineFormat, String> {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return Err("fwrite: machine format must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    let collapsed: String = lower
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .collect();
    if matches!(collapsed.as_str(), "native" | "n" | "system" | "default") {
        return Ok(MachineFormat::Native);
    }
    if matches!(
        collapsed.as_str(),
        "l" | "le" | "littleendian" | "pc" | "intel"
    ) {
        return Ok(MachineFormat::LittleEndian);
    }
    if matches!(
        collapsed.as_str(),
        "b" | "be" | "bigendian" | "mac" | "motorola"
    ) {
        return Ok(MachineFormat::BigEndian);
    }
    if lower.starts_with("ieee-le") {
        return Ok(MachineFormat::LittleEndian);
    }
    if lower.starts_with("ieee-be") {
        return Ok(MachineFormat::BigEndian);
    }
    Err(format!("fwrite: unsupported machine format '{trimmed}'"))
}

fn scalar_string(value: &Value, err: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(err.to_string()),
    }
}

fn numeric_scalar(value: &Value, err: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(int) => Ok(int.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::LogicalArray(la) if la.data.len() == 1 => {
            Ok(if la.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(err.to_string()),
    }
}

fn flatten_elements(value: &Value) -> Result<Vec<f64>, String> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.data.clone()),
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(int) => Ok(vec![int.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|bit| if *bit != 0 { 1.0 } else { 0.0 })
            .collect()),
        Value::CharArray(ca) => Ok(flatten_char_array(ca)),
        Value::String(text) => Ok(text.chars().map(|ch| ch as u32 as f64).collect()),
        Value::StringArray(sa) => Ok(flatten_string_array(sa)),
        Value::GpuTensor(_) => Err("fwrite: expected host tensor data after gathering".to_string()),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("fwrite: complex values are not supported yet".to_string())
        }
        _ => Err(format!("fwrite: unsupported data type {:?}", value)),
    }
}

fn flatten_char_array(ca: &CharArray) -> Vec<f64> {
    let mut values = Vec::with_capacity(ca.rows.saturating_mul(ca.cols));
    for c in 0..ca.cols {
        for r in 0..ca.rows {
            let idx = r * ca.cols + c;
            values.push(ca.data[idx] as u32 as f64);
        }
    }
    values
}

fn flatten_string_array(sa: &runmat_builtins::StringArray) -> Vec<f64> {
    if sa.data.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::new();
    for (idx, text) in sa.data.iter().enumerate() {
        if idx > 0 {
            values.push('\n' as u32 as f64);
        }
        values.extend(text.chars().map(|ch| ch as u32 as f64));
    }
    values
}

fn write_elements(
    file: &mut File,
    values: &[f64],
    spec: WriteSpec,
    skip: usize,
    machine: MachineFormat,
) -> Result<usize, String> {
    let endianness = machine.to_endianness();
    let skip_offset = skip as i64;
    for &value in values {
        match spec.input {
            InputType::UInt8 => {
                let byte = to_u8(value);
                write_bytes(file, &[byte])?;
            }
            InputType::Int8 => {
                let byte = to_i8(value) as u8;
                write_bytes(file, &[byte])?;
            }
            InputType::UInt16 => {
                let bytes = encode_u16(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Int16 => {
                let bytes = encode_i16(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::UInt32 => {
                let bytes = encode_u32(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Int32 => {
                let bytes = encode_i32(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::UInt64 => {
                let bytes = encode_u64(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Int64 => {
                let bytes = encode_i64(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Float32 => {
                let bytes = encode_f32(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Float64 => {
                let bytes = encode_f64(value, endianness);
                write_bytes(file, &bytes)?;
            }
        }

        if skip > 0 {
            file.seek(SeekFrom::Current(skip_offset))
                .map_err(|err| format!("fwrite: failed to seek while applying skip ({err})"))?;
        }
    }
    Ok(values.len())
}

fn write_bytes(file: &mut File, bytes: &[u8]) -> Result<(), String> {
    file.write_all(bytes)
        .map_err(|err| format!("fwrite: failed to write to file ({err})"))
}

fn to_u8(value: f64) -> u8 {
    if !value.is_finite() {
        return if value.is_sign_negative() { 0 } else { u8::MAX };
    }
    let mut rounded = value.round();
    if rounded.is_nan() {
        return 0;
    }
    if rounded < 0.0 {
        rounded = 0.0;
    }
    if rounded > u8::MAX as f64 {
        rounded = u8::MAX as f64;
    }
    rounded as u8
}

fn to_i8(value: f64) -> i8 {
    saturating_round(value, i8::MIN as f64, i8::MAX as f64) as i8
}

fn encode_u16(value: f64, endianness: Endianness) -> [u8; 2] {
    let rounded = saturating_round(value, 0.0, u16::MAX as f64) as u16;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

fn encode_i16(value: f64, endianness: Endianness) -> [u8; 2] {
    let rounded = saturating_round(value, i16::MIN as f64, i16::MAX as f64) as i16;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

fn encode_u32(value: f64, endianness: Endianness) -> [u8; 4] {
    let rounded = saturating_round(value, 0.0, u32::MAX as f64) as u32;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

fn encode_i32(value: f64, endianness: Endianness) -> [u8; 4] {
    let rounded = saturating_round(value, i32::MIN as f64, i32::MAX as f64) as i32;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

fn encode_u64(value: f64, endianness: Endianness) -> [u8; 8] {
    let rounded = saturating_round(value, 0.0, u64::MAX as f64);
    let as_u64 = if rounded.is_finite() {
        rounded as u64
    } else if rounded.is_sign_negative() {
        0
    } else {
        u64::MAX
    };
    match endianness {
        Endianness::Little => as_u64.to_le_bytes(),
        Endianness::Big => as_u64.to_be_bytes(),
    }
}

fn encode_i64(value: f64, endianness: Endianness) -> [u8; 8] {
    let rounded = saturating_round(value, i64::MIN as f64, i64::MAX as f64);
    let as_i64 = if rounded.is_finite() {
        rounded as i64
    } else if rounded.is_sign_negative() {
        i64::MIN
    } else {
        i64::MAX
    };
    match endianness {
        Endianness::Little => as_i64.to_le_bytes(),
        Endianness::Big => as_i64.to_be_bytes(),
    }
}

fn encode_f32(value: f64, endianness: Endianness) -> [u8; 4] {
    let as_f32 = value as f32;
    let bits = as_f32.to_bits();
    match endianness {
        Endianness::Little => bits.to_le_bytes(),
        Endianness::Big => bits.to_be_bytes(),
    }
}

fn encode_f64(value: f64, endianness: Endianness) -> [u8; 8] {
    let bits = value.to_bits();
    match endianness {
        Endianness::Little => bits.to_le_bytes(),
        Endianness::Big => bits.to_be_bytes(),
    }
}

fn saturating_round(value: f64, min: f64, max: f64) -> f64 {
    if !value.is_finite() {
        return if value.is_sign_negative() { min } else { max };
    }
    let mut rounded = value.round();
    if rounded.is_nan() {
        return 0.0;
    }
    if rounded < min {
        rounded = min;
    }
    if rounded > max {
        rounded = max;
    }
    rounded
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
}

fn parse_input_label(label: &str) -> Result<InputType, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(InputType::Float64),
        "single" | "float32" | "real*4" => Ok(InputType::Float32),
        "int8" | "schar" | "integer*1" => Ok(InputType::Int8),
        "uint8" | "uchar" | "unsignedchar" | "char" | "byte" => Ok(InputType::UInt8),
        "int16" | "short" | "integer*2" => Ok(InputType::Int16),
        "uint16" | "ushort" | "unsignedshort" => Ok(InputType::UInt16),
        "int32" | "integer*4" | "long" => Ok(InputType::Int32),
        "uint32" | "unsignedint" | "unsignedlong" => Ok(InputType::UInt32),
        "int64" | "integer*8" | "longlong" => Ok(InputType::Int64),
        "uint64" | "unsignedlonglong" => Ok(InputType::UInt64),
        other => Err(format!("fwrite: unsupported precision '{other}'")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::registry;
    use crate::builtins::io::filetext::{fclose, fopen};
    use crate::RuntimeError;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::Tensor;
    use runmat_filesystem::File;
    use runmat_time::system_time_now;
    use std::io::Read;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(
        fid_value: &Value,
        data_value: &Value,
        rest: &[Value],
    ) -> BuiltinResult<FwriteEval> {
        futures::executor::block_on(evaluate(fid_value, data_value, rest))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn run_fclose(args: &[Value]) -> BuiltinResult<fclose::FcloseEval> {
        futures::executor::block_on(fclose::evaluate(args))
    }

    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        registry::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FWRITE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"count = fwrite(fid, data)"));
        assert!(labels.contains(&"count = fwrite(fid, data, precision, skip)"));
        assert!(labels.contains(&"count = fwrite(fid, data, precision, machinefmt)"));
        assert!(labels.contains(&"count = fwrite(fid, data, precision, skip, machinefmt)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_default_uint8_bytes() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fwrite_uint8");
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0, 2.0, 255.0], vec![3, 1]).unwrap();
        let eval = run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &Vec::new())
            .expect("fwrite");
        assert_eq!(eval.count(), 3);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();

        let bytes = test_support::fs::read(&path).expect("read");
        assert_eq!(bytes, vec![1u8, 2, 255]);
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_double_precision_writes_native_endian() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fwrite_double");
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.5, -2.25], vec![2, 1]).unwrap();
        let args = vec![Value::from("double")];
        let eval =
            run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 2);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();

        let bytes = test_support::fs::read(&path).expect("read");
        let expected: Vec<u8> = if cfg!(target_endian = "little") {
            [1.5f64.to_le_bytes(), (-2.25f64).to_le_bytes()].concat()
        } else {
            [1.5f64.to_be_bytes(), (-2.25f64).to_be_bytes()].concat()
        };
        assert_eq!(bytes, expected);
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_big_endian_uint16() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fwrite_be");
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
            Value::from("ieee-be"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![258.0, 772.0], vec![2, 1]).unwrap();
        let args = vec![Value::from("uint16")];
        let eval =
            run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 2);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();

        let bytes = test_support::fs::read(&path).expect("read");
        assert_eq!(bytes, vec![0x01, 0x02, 0x03, 0x04]);
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_skip_inserts_padding() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fwrite_skip");
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("uint8"), Value::Num(1.0)];
        let eval =
            run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 3);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();

        let bytes = test_support::fs::read(&path).expect("read");
        assert_eq!(bytes, vec![10u8, 0, 20, 0, 30]);
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_gpu_tensor_gathers_before_write() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fwrite_gpu");

        test_support::with_test_provider(|provider| {
            registry::reset_for_tests();
            let open = run_fopen(&[
                Value::from(path.to_string_lossy().to_string()),
                Value::from("w+b"),
            ])
            .expect("fopen");
            let fid = open.as_open().unwrap().fid as i32;

            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::from("uint16")];
            let eval = run_evaluate(&Value::Num(fid as f64), &Value::GpuTensor(handle), &args)
                .expect("fwrite");
            assert_eq!(eval.count(), 4);

            run_fclose(&[Value::Num(fid as f64)]).unwrap();
        });

        let mut file = File::open(&path).expect("open");
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).expect("read");
        assert_eq!(bytes.len(), 8);
        let mut decoded = Vec::new();
        for chunk in bytes.chunks_exact(2) {
            let value = if cfg!(target_endian = "little") {
                u16::from_le_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_be_bytes([chunk[0], chunk[1]])
            };
            decoded.push(value);
        }
        assert_eq!(decoded, vec![1u16, 2, 3, 4]);
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_invalid_precision_errors() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fwrite_invalid_precision");
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let args = vec![Value::from("bogus-class")];
        let err = unwrap_error_message(
            run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).unwrap_err(),
        );
        assert!(err.contains("unsupported precision"));
        let _ = run_fclose(&[Value::Num(fid as f64)]);
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_negative_skip_errors() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fwrite_negative_skip");
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![10.0], vec![1, 1]).unwrap();
        let args = vec![Value::from("uint8"), Value::Num(-1.0)];
        let err = unwrap_error_message(
            run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).unwrap_err(),
        );
        assert!(err.contains("skip value must be non-negative"));
        let _ = run_fclose(&[Value::Num(fid as f64)]);
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fwrite_wgpu_tensor_roundtrip() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fwrite_wgpu_roundtrip");
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let provider = provider::register_wgpu_provider(provider::WgpuProviderOptions::default())
            .expect("wgpu provider");

        let tensor = Tensor::new(vec![0.5, -1.25, 3.75], vec![3, 1]).unwrap();
        let expected = tensor.data.clone();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload to gpu");
        let args = vec![Value::from("double")];
        let eval = run_evaluate(&Value::Num(fid as f64), &Value::GpuTensor(handle), &args)
            .expect("fwrite");
        assert_eq!(eval.count(), 3);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();

        let mut file = File::open(&path).expect("open");
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).expect("read");
        assert_eq!(bytes.len(), 24);
        for (chunk, expected_value) in bytes.chunks_exact(8).zip(expected.iter()) {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            let value = if cfg!(target_endian = "little") {
                f64::from_le_bytes(buf)
            } else {
                f64::from_be_bytes(buf)
            };
            assert!(
                (value - expected_value).abs() < 1e-12,
                "mismatch: {} vs {}",
                value,
                expected_value
            );
        }
        test_support::fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_invalid_identifier_errors() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(
            run_evaluate(&Value::Num(-1.0), &Value::Num(1.0), &Vec::new()).unwrap_err(),
        );
        assert!(err.contains("file identifier must be non-negative"));
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!(
            "runmat_{prefix}_{}_{}.tmp",
            now.as_secs(),
            now.subsec_nanos()
        );
        std::env::temp_dir().join(filename)
    }
}
