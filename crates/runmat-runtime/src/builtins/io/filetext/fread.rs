//! MATLAB-compatible `fread` builtin for RunMat.

use std::io::{ErrorKind, Read, Seek, SeekFrom};

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{CharArray, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::{helpers::extract_scalar_string, registry};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem::File;

const INVALID_IDENTIFIER_MESSAGE: &str =
    "Invalid file identifier. Use fopen to generate a valid file ID.";
const BUILTIN_NAME: &str = "fread";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fread",
    op_kind: GpuOpKind::Custom("file-io-read"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Host-only operation that reads from the shared file registry; GPU arguments are gathered to the CPU before I/O.",
};

fn fread_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_string_result<T>(result: Result<T, String>) -> BuiltinResult<T> {
    result.map_err(fread_error)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O cannot participate in fusion; metadata is registered for completeness.",
};

#[runtime_builtin(
    name = "fread",
    category = "io/filetext",
    summary = "Read binary data from a file identifier.",
    keywords = "fread,file,io,binary,precision",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fread_type),
    builtin_path = "crate::builtins::io::filetext::fread"
)]
async fn fread_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    Ok(eval.first_output())
}

#[derive(Debug, Clone)]
pub struct FreadEval {
    data: Value,
    count: usize,
}

impl FreadEval {
    fn new(data: Value, count: usize) -> Self {
        Self { data, count }
    }

    pub fn first_output(&self) -> Value {
        self.data.clone()
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![self.data.clone(), Value::Num(self.count as f64)]
    }

    fn apply_like(
        &mut self,
        like_proto: Option<&Value>,
        precision: PrecisionSpec,
    ) -> Result<(), String> {
        if let Some(proto) = like_proto {
            let adjusted = adjust_output_for_like(self.data.clone(), proto, precision)?;
            self.data = adjusted;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn data(&self) -> &Value {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FreadEval> {
    let fid_host = gather_value(fid_value).await?;
    let fid = map_string_result(parse_fid(&fid_host))?;
    if fid < 0 {
        return Err(fread_error("fread: file identifier must be non-negative"));
    }
    if fid < 3 {
        return Err(fread_error(
            "fread: standard input/output identifiers are not supported yet",
        ));
    }

    let info = registry::info_for(fid)
        .ok_or_else(|| fread_error(format!("fread: {INVALID_IDENTIFIER_MESSAGE}")))?;
    let handle = registry::take_handle(fid)
        .ok_or_else(|| fread_error(format!("fread: {INVALID_IDENTIFIER_MESSAGE}")))?;
    let mut file = handle
        .lock()
        .map_err(|_| fread_error("fread: failed to lock file handle (poisoned mutex)"))?;

    let arg_refs: Vec<&Value> = rest.iter().collect();
    let (size_arg, precision_arg, skip_arg, machine_arg, like_arg) =
        classify_arguments(&arg_refs).map_err(|e| fread_error(format!("fread: {e}")))?;

    let size_host = match size_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let precision_host = match precision_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let skip_host = match skip_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let machine_host = match machine_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };

    let size_spec = map_string_result(parse_size(size_host.as_ref()))?;
    let precision = map_string_result(parse_precision(precision_host.as_ref()))?;
    let skip_bytes = map_string_result(parse_skip(skip_host.as_ref()))?;
    let machine_format = map_string_result(parse_machine_format(
        machine_host.as_ref(),
        &info.machinefmt,
    ))?;

    let mut eval = map_string_result(read_from_handle(
        &mut file,
        &size_spec,
        &precision,
        skip_bytes,
        machine_format,
    ))?;
    map_string_result(eval.apply_like(like_arg, precision))?;
    Ok(eval)
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    let number = match value {
        Value::Num(n) => *n,
        Value::Int(int) => int.to_f64(),
        _ => {
            return Err("file identifier must be numeric".to_string());
        }
    };
    if !number.is_finite() {
        return Err("file identifier must be finite".to_string());
    }
    let rounded = number.round();
    if (rounded - number).abs() > f64::EPSILON {
        return Err("file identifier must be an integer".to_string());
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err("file identifier is out of range".to_string());
    }
    Ok(rounded as i32)
}

type ClassifiedArgs<'a> = (
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
);

fn classify_arguments<'a>(args: &'a [&'a Value]) -> Result<ClassifiedArgs<'a>, String> {
    let mut filtered_indices: Vec<usize> = Vec::with_capacity(args.len());
    let mut like_proto: Option<&Value> = None;
    let mut i = 0usize;

    while i < args.len() {
        let value = args[i];
        if matches_keyword(value, "like") {
            if like_proto.is_some() {
                return Err("multiple 'like' prototypes are not supported".to_string());
            }
            i += 1;
            let Some(proto_value) = args.get(i) else {
                return Err("expected prototype after 'like'".to_string());
            };
            like_proto = Some(*proto_value);
            i += 1;
            continue;
        }
        filtered_indices.push(i);
        i += 1;
    }

    if filtered_indices.len() > 4 {
        return Err("too many input arguments".to_string());
    }

    let (size_idx, precision_idx, skip_idx, machine_idx) =
        classify_ordered_indices(args, &filtered_indices)?;

    let size = size_idx.map(|index| args[index]);
    let precision = precision_idx.map(|index| args[index]);
    let skip = skip_idx.map(|index| args[index]);
    let machine = machine_idx.map(|index| args[index]);

    Ok((size, precision, skip, machine, like_proto))
}

type ClassifiedIndices = (Option<usize>, Option<usize>, Option<usize>, Option<usize>);

fn classify_ordered_indices(
    args: &[&Value],
    indices: &[usize],
) -> Result<ClassifiedIndices, String> {
    let mut position = 0usize;
    let mut size_idx: Option<usize> = None;
    let mut precision_idx: Option<usize> = None;
    let mut skip_idx: Option<usize> = None;
    let mut machine_idx: Option<usize> = None;

    if let Some(&first_index) = indices.get(position) {
        let first = args[first_index];
        if is_string_like(first) {
            precision_idx = Some(first_index);
        } else {
            size_idx = Some(first_index);
        }
        position += 1;
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if precision_idx.is_none() && is_string_like(candidate) {
            precision_idx = Some(index);
            position += 1;
        } else if is_numeric_like(candidate) {
            skip_idx = Some(index);
            position += 1;
        } else if is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("invalid argument combination".to_string());
        }
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if skip_idx.is_none() && is_numeric_like(candidate) {
            skip_idx = Some(index);
            position += 1;
        } else if machine_idx.is_none() && is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("invalid argument combination".to_string());
        }
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if machine_idx.is_none() && is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("too many input arguments".to_string());
        }
    }

    if position < indices.len() {
        return Err("too many input arguments".to_string());
    }

    Ok((size_idx, precision_idx, skip_idx, machine_idx))
}

fn is_string_like(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::CharArray(ca) if ca.rows == 1 => true,
        Value::StringArray(sa) if sa.data.len() == 1 => true,
        _ => false,
    }
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

fn is_numeric_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::LogicalArray(_)
    )
}

#[derive(Clone, Debug)]
enum SizeSpec {
    All,
    Count(usize),
    Matrix { rows: usize, cols: Option<usize> },
}

impl SizeSpec {
    fn element_limit(&self) -> Option<usize> {
        match self {
            SizeSpec::All => None,
            SizeSpec::Count(n) => Some(*n),
            SizeSpec::Matrix { rows, cols } => {
                if *rows == 0 {
                    return Some(0);
                }
                cols.map(|c| rows.saturating_mul(c))
            }
        }
    }
}

fn parse_size(arg: Option<&Value>) -> Result<SizeSpec, String> {
    match arg {
        None => Ok(SizeSpec::All),
        Some(Value::String(s)) => parse_size_string(s),
        Some(Value::CharArray(ca)) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_size_string(&text)
        }
        Some(Value::StringArray(sa)) if sa.data.len() == 1 => parse_size_string(&sa.data[0]),
        Some(Value::Tensor(t)) => parse_size_tensor(t),
        Some(value) => {
            let scalar = value_to_scalar(value, "size argument must be numeric or a size vector")?;
            scalar_to_size(scalar)
        }
    }
}

fn parse_size_string(text: &str) -> Result<SizeSpec, String> {
    if text.trim().is_empty() {
        return Err("size argument must not be empty".to_string());
    }
    let lower = text.trim().to_ascii_lowercase();
    if lower == "inf" {
        Ok(SizeSpec::All)
    } else {
        let number = lower
            .parse::<f64>()
            .map_err(|_| "size argument must be numeric or 'inf'".to_string())?;
        scalar_to_size(number)
    }
}

fn parse_size_tensor(t: &Tensor) -> Result<SizeSpec, String> {
    match t.data.len() {
        0 => Ok(SizeSpec::Count(0)),
        1 => scalar_to_size(t.data[0]),
        2 => {
            let rows = scalar_to_size_component(
                t.data[0],
                "size vector components must be non-negative integers or Inf",
            )?;
            let cols_raw = t.data[1];
            if cols_raw.is_infinite() && cols_raw.is_sign_positive() {
                Ok(SizeSpec::Matrix { rows, cols: None })
            } else {
                let cols = scalar_to_size_component(
                    cols_raw,
                    "size vector components must be non-negative integers or Inf",
                )?;
                Ok(SizeSpec::Matrix {
                    rows,
                    cols: Some(cols),
                })
            }
        }
        _ => Err("size vector must contain at most two elements".to_string()),
    }
}

fn scalar_to_size(value: f64) -> Result<SizeSpec, String> {
    if value.is_infinite() && value.is_sign_positive() {
        return Ok(SizeSpec::All);
    }
    let count = scalar_to_size_component(value, "size argument must be a non-negative integer")?;
    Ok(SizeSpec::Count(count))
}

fn scalar_to_size_component(value: f64, err: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(err.to_string());
    }
    if value < 0.0 {
        return Err(err.to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(err.to_string());
    }
    if rounded > usize::MAX as f64 {
        return Err("size argument is too large".to_string());
    }
    Ok(rounded as usize)
}

#[derive(Clone, Copy, Debug)]
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

impl InputType {
    fn byte_len(&self) -> usize {
        match self {
            InputType::UInt8 | InputType::Int8 => 1,
            InputType::UInt16 | InputType::Int16 => 2,
            InputType::UInt32 | InputType::Int32 | InputType::Float32 => 4,
            InputType::UInt64 | InputType::Int64 | InputType::Float64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OutputKind {
    Double,
    Char,
}

#[derive(Clone, Copy, Debug)]
struct PrecisionSpec {
    input: InputType,
    output: OutputKind,
}

impl PrecisionSpec {
    fn default() -> Self {
        Self {
            input: InputType::Float64,
            output: OutputKind::Double,
        }
    }
}

fn parse_precision(arg: Option<&Value>) -> Result<PrecisionSpec, String> {
    match arg {
        None => Ok(PrecisionSpec::default()),
        Some(value) => {
            let text = scalar_string(
                value,
                "precision argument must be a string scalar or character vector",
            )?;
            parse_precision_string(&text)
        }
    }
}

fn parse_precision_string(raw: &str) -> Result<PrecisionSpec, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("precision argument must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix('*') {
        parse_star_precision(rest.trim())
    } else if let Some((lhs, rhs)) = lower.split_once("=>") {
        let input = parse_input_label(lhs.trim())?;
        let output = parse_output_label(rhs.trim())?;
        if matches!(output, OutputKind::Char)
            && !matches!(input, InputType::UInt8 | InputType::UInt16)
        {
            return Err(
                "char output requires an unsigned byte or unsigned 16-bit input precision"
                    .to_string(),
            );
        }
        Ok(PrecisionSpec { input, output })
    } else {
        let input = parse_input_label(lower.trim())?;
        let wants_char =
            lower == "char" || (matches!(input, InputType::UInt8) && lower.contains("char"));
        let output = if wants_char {
            OutputKind::Char
        } else {
            OutputKind::Double
        };
        if matches!(output, OutputKind::Char)
            && !matches!(input, InputType::UInt8 | InputType::UInt16)
        {
            return Err(
                "char precision requires unsigned byte or unsigned 16-bit input".to_string(),
            );
        }
        Ok(PrecisionSpec { input, output })
    }
}

fn parse_star_precision(label: &str) -> Result<PrecisionSpec, String> {
    let output = parse_output_label(label)?;
    match output {
        OutputKind::Char => Ok(PrecisionSpec {
            input: InputType::UInt8,
            output,
        }),
        OutputKind::Double => Ok(PrecisionSpec {
            input: InputType::Float64,
            output,
        }),
    }
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
        other => Err(format!("unsupported precision '{other}'")),
    }
}

fn parse_output_label(label: &str) -> Result<OutputKind, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(OutputKind::Double),
        "char" => Ok(OutputKind::Char),
        other => Err(format!("output class '{other}' is not implemented yet")),
    }
}

fn parse_skip(arg: Option<&Value>) -> Result<usize, String> {
    match arg {
        None => Ok(0),
        Some(value) => {
            let scalar = value_to_scalar(value, "skip value must be numeric")?;
            if !scalar.is_finite() {
                return Err("skip value must be finite".to_string());
            }
            if scalar < 0.0 {
                return Err("skip value must be non-negative".to_string());
            }
            let rounded = scalar.round();
            if (rounded - scalar).abs() > f64::EPSILON {
                return Err("skip value must be an integer".to_string());
            }
            if rounded > i64::MAX as f64 {
                return Err("skip value is too large".to_string());
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

#[derive(Clone, Copy, Debug)]
enum Endianness {
    Little,
    Big,
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

fn parse_machine_format(arg: Option<&Value>, default_label: &str) -> Result<MachineFormat, String> {
    match arg {
        Some(value) => {
            let text = scalar_string(
                value,
                "machine format must be a string scalar or character vector",
            )?;
            machine_format_from_label(&text)
        }
        None => machine_format_from_label(default_label),
    }
}

fn machine_format_from_label(label: &str) -> Result<MachineFormat, String> {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return Err("machine format must not be empty".to_string());
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
    Err(format!("unsupported machine format '{trimmed}'"))
}

fn scalar_string(value: &Value, err: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(err.to_string()),
    }
}

fn value_to_scalar(value: &Value, err: &str) -> Result<f64, String> {
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

fn read_from_handle(
    file: &mut File,
    size_spec: &SizeSpec,
    precision: &PrecisionSpec,
    skip: usize,
    machine: MachineFormat,
) -> Result<FreadEval, String> {
    let endianness = machine.to_endianness();
    match precision.output {
        OutputKind::Double => {
            let limit = size_spec.element_limit();
            let (values, count) =
                read_numeric_values(file, precision.input, limit, skip, endianness)?;
            let (data, rows, cols) = finalize_numeric(size_spec, count, values);
            let tensor = Tensor::new(data, vec![rows, cols]).map_err(|e| format!("fread: {e}"))?;
            Ok(FreadEval::new(Value::Tensor(tensor), count))
        }
        OutputKind::Char => {
            let limit = size_spec.element_limit();
            let (values, count) = read_char_values(file, precision.input, limit, skip, endianness)?;
            let (row_major, rows, cols) = finalize_char(size_spec, count, values);
            let char_array =
                CharArray::new(row_major, rows, cols).map_err(|e| format!("fread: {e}"))?;
            Ok(FreadEval::new(Value::CharArray(char_array), count))
        }
    }
}

fn adjust_output_for_like(
    data: Value,
    prototype: &Value,
    precision: PrecisionSpec,
) -> Result<Value, String> {
    if matches!(prototype, Value::GpuTensor(_)) {
        return match data {
            Value::Tensor(tensor) => tensor_to_gpu_value(tensor),
            Value::CharArray(_) => {
                Err("fread: character output cannot be returned on the GPU via 'like'".to_string())
            }
            other => Ok(other),
        };
    }

    match prototype {
        Value::LogicalArray(_) | Value::Bool(_) => convert_to_logical_value(data),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            if !matches!(precision.output, OutputKind::Char) {
                return Err(
                    "fread: character prototypes require a character precision such as '*char'"
                        .to_string(),
                );
            }
            ensure_char_result(data)
        }
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(data),
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            Err("fread: complex prototypes are not supported yet".to_string())
        }
        Value::Cell(_) => Err("fread: cell prototypes are not supported".to_string()),
        _ => Ok(data),
    }
}

fn ensure_char_result(data: Value) -> Result<Value, String> {
    match data {
        Value::CharArray(_) => Ok(data),
        _ => Err("fread: expected character output when using a character prototype".to_string()),
    }
}

fn tensor_to_gpu_value(tensor: Tensor) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(handle) = provider.upload(&view) {
            return Ok(Value::GpuTensor(handle));
        }
    }
    Ok(Value::Tensor(tensor))
}

fn convert_to_logical_value(data: Value) -> Result<Value, String> {
    match data {
        Value::LogicalArray(_) => Ok(data),
        Value::Tensor(tensor) => {
            let mut bits = Vec::with_capacity(tensor.data.len());
            for &value in &tensor.data {
                bits.push(if value != 0.0 { 1 } else { 0 });
            }
            LogicalArray::new(bits, tensor.shape.clone())
                .map(Value::LogicalArray)
                .map_err(|e| format!("fread: {e}"))
        }
        Value::CharArray(ca) => {
            let total = ca.rows.saturating_mul(ca.cols);
            let mut bits = Vec::with_capacity(total);
            for c in 0..ca.cols {
                for r in 0..ca.rows {
                    let idx = r * ca.cols + c;
                    let ch = ca.data[idx];
                    bits.push(if ch != '\0' { 1 } else { 0 });
                }
            }
            LogicalArray::new(bits, vec![ca.rows, ca.cols])
                .map(Value::LogicalArray)
                .map_err(|e| format!("fread: {e}"))
        }
        _ => Err(
            "fread: logical prototypes require numeric or character output from the read"
                .to_string(),
        ),
    }
}

fn read_numeric_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
) -> Result<(Vec<f64>, usize), String> {
    if let Some(0) = limit {
        return Ok((Vec::new(), 0));
    }
    let element_size = input.byte_len();
    let mut buffer = vec![0u8; element_size];
    let mut values = Vec::new();
    let mut count = 0usize;
    let target = limit.unwrap_or(usize::MAX);

    'outer: loop {
        if count >= target {
            break;
        }
        let mut remaining = element_size;
        while remaining > 0 {
            match reader.read(&mut buffer[element_size - remaining..element_size]) {
                Ok(0) => break 'outer,
                Ok(n) => remaining -= n,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => {
                    return Err(format!("fread: failed to read from file ({err})"));
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let value = decode_to_f64(&buffer, input, endianness)?;
        values.push(value);
        count += 1;
        if skip > 0 {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|err| format!("fread: failed to skip bytes ({err})"))?;
        }
    }
    Ok((values, count))
}

fn read_char_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
) -> Result<(Vec<char>, usize), String> {
    if !matches!(input, InputType::UInt8 | InputType::UInt16) {
        return Err(
            "char output requires an unsigned byte or unsigned 16-bit input precision".to_string(),
        );
    }
    if let Some(0) = limit {
        return Ok((Vec::new(), 0));
    }
    let element_size = input.byte_len();
    let mut buffer = vec![0u8; element_size];
    let mut values = Vec::new();
    let mut count = 0usize;
    let target = limit.unwrap_or(usize::MAX);

    'outer: loop {
        if count >= target {
            break;
        }
        let mut remaining = element_size;
        while remaining > 0 {
            match reader.read(&mut buffer[element_size - remaining..element_size]) {
                Ok(0) => break 'outer,
                Ok(n) => remaining -= n,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => {
                    return Err(format!("fread: failed to read from file ({err})"));
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let ch = decode_to_char(&buffer, input, endianness)?;
        values.push(ch);
        count += 1;
        if skip > 0 {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|err| format!("fread: failed to skip bytes ({err})"))?;
        }
    }

    Ok((values, count))
}

fn decode_to_f64(bytes: &[u8], input: InputType, endianness: Endianness) -> Result<f64, String> {
    Ok(match input {
        InputType::UInt8 => bytes[0] as f64,
        InputType::Int8 => (bytes[0] as i8) as f64,
        InputType::UInt16 => read_u16(bytes, endianness) as f64,
        InputType::Int16 => read_u16(bytes, endianness) as i16 as f64,
        InputType::UInt32 => read_u32(bytes, endianness) as f64,
        InputType::Int32 => read_u32(bytes, endianness) as i32 as f64,
        InputType::UInt64 => read_u64(bytes, endianness) as f64,
        InputType::Int64 => read_u64(bytes, endianness) as i64 as f64,
        InputType::Float32 => {
            let bits = read_u32(bytes, endianness);
            f32::from_bits(bits) as f64
        }
        InputType::Float64 => {
            let bits = read_u64(bytes, endianness);
            f64::from_bits(bits)
        }
    })
}

fn decode_to_char(bytes: &[u8], input: InputType, endianness: Endianness) -> Result<char, String> {
    let code = match input {
        InputType::UInt8 => bytes[0] as u32,
        InputType::UInt16 => read_u16(bytes, endianness) as u32,
        _ => {
            return Err(
                "char output requires an unsigned byte or unsigned 16-bit input precision"
                    .to_string(),
            );
        }
    };
    char::from_u32(code).ok_or_else(|| {
        format!("value 0x{code:X} cannot be represented as a Unicode scalar for char output")
    })
}

fn read_u16(bytes: &[u8], endianness: Endianness) -> u16 {
    match endianness {
        Endianness::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
        Endianness::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
    }
}

fn read_u32(bytes: &[u8], endianness: Endianness) -> u32 {
    match endianness {
        Endianness::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        Endianness::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

fn read_u64(bytes: &[u8], endianness: Endianness) -> u64 {
    match endianness {
        Endianness::Little => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        Endianness::Big => u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

fn finalize_numeric(
    size_spec: &SizeSpec,
    count_read: usize,
    mut values: Vec<f64>,
) -> (Vec<f64>, usize, usize) {
    match size_spec {
        SizeSpec::All | SizeSpec::Count(_) => {
            let rows = count_read;
            let cols = if count_read == 0 { 0 } else { 1 };
            (values, rows, cols)
        }
        SizeSpec::Matrix {
            rows,
            cols: Some(c),
        } => {
            let target = rows.saturating_mul(*c);
            if values.len() < target {
                values.resize(target, 0.0);
            } else if values.len() > target {
                values.truncate(target);
            }
            (values, *rows, *c)
        }
        SizeSpec::Matrix { rows, cols: None } => {
            if *rows == 0 {
                values.clear();
                (values, 0, 0)
            } else {
                let cols = if count_read == 0 {
                    0
                } else {
                    count_read.div_ceil(*rows)
                };
                let target = rows.saturating_mul(cols);
                if values.len() < target {
                    values.resize(target, 0.0);
                } else if values.len() > target {
                    values.truncate(target);
                }
                (values, *rows, cols)
            }
        }
    }
}

fn finalize_char(
    size_spec: &SizeSpec,
    count_read: usize,
    mut column_major: Vec<char>,
) -> (Vec<char>, usize, usize) {
    match size_spec {
        SizeSpec::All | SizeSpec::Count(_) => {
            let rows = count_read;
            let cols = if count_read == 0 { 0 } else { 1 };
            let row_major = column_to_row_major(&column_major, rows, cols);
            (row_major, rows, cols)
        }
        SizeSpec::Matrix {
            rows,
            cols: Some(c),
        } => {
            let target = rows.saturating_mul(*c);
            if column_major.len() < target {
                column_major.resize(target, '\0');
            } else if column_major.len() > target {
                column_major.truncate(target);
            }
            let row_major = column_to_row_major(&column_major, *rows, *c);
            (row_major, *rows, *c)
        }
        SizeSpec::Matrix { rows, cols: None } => {
            if *rows == 0 {
                column_major.clear();
                (Vec::new(), 0, 0)
            } else {
                let cols = if count_read == 0 {
                    0
                } else {
                    count_read.div_ceil(*rows)
                };
                let target = rows.saturating_mul(cols);
                if column_major.len() < target {
                    column_major.resize(target, '\0');
                } else if column_major.len() > target {
                    column_major.truncate(target);
                }
                let row_major = column_to_row_major(&column_major, *rows, cols);
                (row_major, *rows, cols)
            }
        }
    }
}

fn column_to_row_major(data: &[char], rows: usize, cols: usize) -> Vec<char> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let mut output = vec!['\0'; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            let src = c * rows + r;
            let dst = r * cols + c;
            if let Some(ch) = data.get(src) {
                output[dst] = *ch;
            }
        }
    }
    output
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::registry;
    use crate::builtins::io::filetext::{fclose, fopen};
    use crate::RuntimeError;
    use runmat_filesystem::{self as fs, File};
    use runmat_time::system_time_now;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FreadEval> {
        futures::executor::block_on(evaluate(fid_value, rest))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn run_fclose(args: &[Value]) -> BuiltinResult<fclose::FcloseEval> {
        futures::executor::block_on(fclose::evaluate(args))
    }

    #[cfg(feature = "wgpu")]
    fn run_call_builtin(name: &str, args: &[Value]) -> BuiltinResult<Value> {
        crate::call_builtin(name, args)
    }

    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        registry::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_reads_default_double() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_default_double");
        let mut file = File::create(&path).expect("create");
        file.write_all(&1.5f64.to_le_bytes()).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let eval = run_evaluate(&Value::Num(fid as f64), &Vec::new()).expect("fread");
        assert_eq!(eval.count(), 1);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 1.5).abs() < 1e-12);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_uint8_vector_with_count() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_uint8_vector");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(4.0), Value::from("uint8")];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 4);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
                assert_eq!(t.shape, vec![4, 1]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_uint8_matrix_with_padding() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_uint8_matrix");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let size_tensor = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(size_tensor), Value::from("uint8")];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 5);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_char_output() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_char_output");
        let mut file = File::create(&path).expect("create");
        file.write_all(b"abc").expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let eval = run_evaluate(&Value::Num(fid as f64), &[Value::from("*char")]).expect("fread");
        assert_eq!(eval.count(), 3);
        match eval.data() {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 3);
                assert_eq!(ca.cols, 1);
                let collected: String = ca.data.iter().collect();
                assert_eq!(collected, "abc");
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_logical_output() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_like_logical_output");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[0u8, 3, 0, 4]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let prototype = LogicalArray::zeros(vec![2, 1]);
        let args = vec![
            Value::Num(2.0),
            Value::from("uint8"),
            Value::from("like"),
            Value::LogicalArray(prototype),
        ];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 2);
        match eval.data() {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 1]);
                assert_eq!(array.data, vec![0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_requires_prototype() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_like_requires_prototype");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::from("like")];
        let err = unwrap_error_message(run_evaluate(&Value::Num(fid as f64), &args).unwrap_err());
        assert!(err.contains("expected prototype after 'like'"));

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_char_requires_precision() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_like_char_requires_precision");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[65u8]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![
            Value::Num(1.0),
            Value::from("uint8"),
            Value::from("like"),
            Value::CharArray(CharArray::new_row("A")),
        ];
        let err = unwrap_error_message(run_evaluate(&Value::Num(fid as f64), &args).unwrap_err());
        assert!(err.contains("character prototypes require"));

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_gpu_provider_roundtrip() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_like_gpu_provider_roundtrip");
        let mut file = File::create(&path).expect("create");
        file.write_all(&1.5f64.to_le_bytes()).expect("write");
        drop(file);

        test_support::with_test_provider(|provider| {
            let open = run_fopen(&[
                Value::from(path.to_string_lossy().to_string()),
                Value::from("rb"),
            ])
            .expect("fopen");
            let fid = open.as_open().unwrap().fid as i32;

            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &proto.data,
                shape: &proto.shape,
            };
            let handle = provider.upload(&view).expect("upload prototype");

            let args = vec![
                Value::Num(1.0),
                Value::from("double"),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
            match eval.data() {
                Value::GpuTensor(result) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(result.clone())).expect("gather");
                    assert_eq!(gathered.data, vec![1.5]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }

            run_fclose(&[Value::Num(fid as f64)]).unwrap();
        });

        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fread_wgpu_like_uploads_gpu() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_wgpu_like_uploads_gpu");
        let mut file = File::create(&path).expect("create");
        file.write_all(&2.25f64.to_le_bytes()).expect("write");
        drop(file);

        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let gpu_proto = run_call_builtin("gpuArray", &[Value::Tensor(proto)]).expect("gpuArray");

        let args = vec![
            Value::Num(1.0),
            Value::from("double"),
            Value::from("like"),
            gpu_proto,
        ];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        match eval.data() {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle.clone())).expect("gather");
                assert_eq!(gathered.data, vec![2.25]);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_skip_bytes() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_skip_bytes");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5, 6]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(3.0), Value::from("uint8"), Value::Num(1.0)];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 3);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 3.0, 5.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_big_endian_machine_format() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fread_big_endian");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[0x01, 0x02, 0x03, 0x04]).expect("write");
        drop(file);

        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
            Value::from("ieee-be"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(2.0), Value::from("uint16")];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 2);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![258.0, 772.0]);
                assert_eq!(t.shape, vec![2, 1]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_invalid_fid_errors() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&Value::Num(9999.0), &Vec::new()).unwrap_err());
        assert!(err.contains("Invalid file identifier"));
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
