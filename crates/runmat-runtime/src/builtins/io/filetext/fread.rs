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
use crate::{gather_if_needed, build_runtime_error, BuiltinResult, RuntimeError};
use runmat_filesystem::File;

const INVALID_IDENTIFIER_MESSAGE: &str =
    "Invalid file identifier. Use fopen to generate a valid file ID.";
const BUILTIN_NAME: &str = "fread";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fread",
        builtin_path = "crate::builtins::io::filetext::fread"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fread"
category: "io/filetext"
keywords: ["fread", "binary read", "io", "precision", "machine format", "skip"]
summary: "Read binary data from a file identifier with MATLAB-compatible size, precision, skip, and machine-format semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/fread.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. When arguments reside on the GPU, RunMat gathers them first; providers do not implement hooks for fread."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 4
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::filetext::fread::tests"
  integration:
    - "builtins::io::filetext::fread::tests::fread_big_endian_machine_format"
    - "builtins::io::filetext::fread::tests::fread_like_gpu_provider_roundtrip"
---

# What does the `fread` function do in MATLAB / RunMat?
`fread` reads binary data from a file identifier returned by `fopen`. It mirrors MATLAB's handling of
size, precision, skip, and machine-format arguments so existing MATLAB code can be ported without changes.
The builtin tracks the file position, honours the machine format recorded by `fopen`, and returns both the
data array and the number of elements successfully read.

## How does the `fread` function behave in MATLAB / RunMat?
- `A = fread(fid)` reads to the end of the file, returning a column vector of doubles.
- `A = fread(fid, sizeA)` reads at most `prod(sizeA)` elements, filling the result in column-major order.
  Use `[m n]` to request a matrix or `[m Inf]` to keep filling columns until EOF.
- `A = fread(fid, precision)` controls how many bytes each element consumes and how the result is typed.
  RunMat supports the common precisions from MATLAB: `double`, `single`, `uint8`, `int8`, `uint16`,
  `int16`, `uint32`, `int32`, `uint64`, `int64`, and `char`. When no output class is specified, data is
  converted to double; `char` (and `*char`) return a MATLAB-style character array.
- `A = fread(fid, sizeA, precision, skip, machinefmt)` honours optional byte skipping and machine-format
  overrides (`'native'`, `'ieee-le'`, `'ieee-be'`). The machine format defaults to the value recorded by
  `fopen`.
- `[A, count] = fread(...)` returns the number of elements successfully read before encountering EOF.
- If MATLAB requests more elements than available, `fread` pads matrix outputs with zeros (or `'\0'` for
  character data) to satisfy the requested dimensions; the `count` output always reflects the number of
  real elements read from the file.
- `A = fread(___, 'like', prototype)` matches the residency and logical or numeric flavour of `prototype`.
  GPU prototypes trigger an upload after reading so the result remains on the device, while logical prototypes
  convert the output using MATLAB's non-zero rule.
- RunMat executes the builtin entirely on the host CPU. Arguments that reside on a GPU are gathered
  automatically before any I/O occurs.

## `fread` Function GPU Execution Behaviour
`fread` is a host-only operation. When the file identifier or optional arguments (size vectors, precision
strings, skip counts) live on the GPU, RunMat gathers them to the host first. File handles are managed by
the shared registry established by `fopen`, and GPU providers never participate in file I/O. When a `'like'`
prototype is a GPU tensor, the builtin uploads the host result after the read so that the output mirrors the
prototype's residency. Providers that fail the upload silently fall back to returning the host tensor.

## Examples of using the `fread` function in MATLAB / RunMat

### Reading double-precision values from a binary file
```matlab
fid = fopen('numbers.bin', 'w+b');
fwrite(fid, [1 2 3], 'double');
frewind(fid);

[values, count] = fread(fid);      % defaults to double precision
% values is a 3x1 column vector [1; 2; 3], count == 3

fclose(fid);
delete('numbers.bin');
```

### Reading bytes with a specific element count
```matlab
fid = fopen('payload.bin', 'w+b');
fwrite(fid, uint8(1:6), 'uint8');
frewind(fid);

[bytes, count] = fread(fid, 4, 'uint8');
% bytes == [1; 2; 3; 4], count == 4

fclose(fid);
delete('payload.bin');
```

### Loading a two-dimensional block of `uint8` data
```matlab
fid = fopen('frame.bin', 'w+b');
payload = uint8(reshape(1:12, 3, 4));
fwrite(fid, payload, 'uint8');
frewind(fid);

[frame, count] = fread(fid, [3 4], 'uint8');
frame = uint8(frame);              % convert back to uint8 if needed

fclose(fid);
delete('frame.bin');
```

### Reading characters using the `*char` precision form
```matlab
fid = fopen('message.txt', 'w+b');
fwrite(fid, 'RunMat', 'char');
frewind(fid);

[text, count] = fread(fid, '*char');
text = text.';                      % row string 'RunMat'

fclose(fid);
delete('message.txt');
```

### Skipping bytes between samples
```matlab
fid = fopen('interleaved.bin', 'w+b');
fwrite(fid, uint8(1:12), 'uint8');
frewind(fid);

[every_other, count] = fread(fid, 5, 'uint8', 1);   % read one byte, skip one byte
% every_other == [1; 3; 5; 7; 9]

fclose(fid);
delete('interleaved.bin');
```

### Respecting big-endian machine formats
```matlab
fid = fopen('sensors.be', 'w+b', 'ieee-be');
fwrite(fid, uint16([258 772]), 'uint16');
frewind(fid);

[values, count] = fread(fid, [2 1], 'uint16');
% values == [258; 772], count == 2

fclose(fid);
delete('sensors.be');
```

### Matching GPU residency with `'like'`
```matlab
fid = fopen('samples.bin', 'w+b');
fwrite(fid, [2.5 4.5 6.5 8.5], 'double');
frewind(fid);

prototype = gpuArray.zeros(4, 1);
[values, count] = fread(fid, 4, 'double', 'like', prototype);
% When a GPU provider is active, values stays on the GPU and count == 4.

fclose(fid);
delete('samples.bin');
```
`values` stays on the GPU when an acceleration provider is active. Without a provider the function still
returns correct data on the host.

## FAQ

### What precision strings are supported?
RunMat implements the commonly used MATLAB precisions: `double`, `single`, `uint8`, `int8`, `uint16`,
`int16`, `uint32`, `int32`, `uint64`, `int64`, and `char`. The short form `*char` is also recognised.
When an output class is not specified explicitly (with `=>` or the `*class` syntax), the result is converted
to double precision.

### How are partial reads handled?
`fread` stops when it encounters EOF. Matrices requested with `[m n]` are padded with zeros (or `'\0'`)
when the file does not contain enough elements to fill every column. The `count` output records the number
of real elements read before padding.

### How do size arguments work?
Pass a scalar `N` to request a column vector with up to `N` elements, `[M N]` to request a matrix with `M`
rows and `N` columns, or `[M Inf]` to keep reading additional columns until EOF. Omitting the size argument
is equivalent to using `Inf` (read everything).

### How does the skip parameter behave?
`skip` specifies the number of bytes to skip after reading each element. It must be a non-negative integer.
The file position advances by the element size plus the skip value for every element that is successfully read.

### What does the `'like'` prototype control?
The `'like', prototype` pair matches the output residency and high-level type of `prototype`. Pass a GPU tensor
to receive a GPU tensor, use a logical array to obtain logical output (non-zero becomes `true`), or use a
character prototype together with a character precision to obtain a `CharArray`.

### Which machine formats are supported?
The builtin recognises `'native'`, `'ieee-le'`, and `'ieee-be'` (including their MATLAB aliases such as
`'little-endian'`, `'pc'`, `'big-endian'`, and `'mac'`). Unsupported formats (`'vaxd'`, `'cray'`, etc.) raise
descriptive errors.

### Can `fread` operate on standard input?
Standard input/output/error identifiers (0, 1, 2) are currently not supported by RunMat's `fread`. Open files
explicitly with `fopen` before calling `fread`.

## See Also
[fopen](./fopen), [fclose](./fclose), [fwrite](./fwrite), [fileread](./fileread), [filewrite](./filewrite)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/io/filetext/fread.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/fread.rs)
- Found a behavioural mismatch? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

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
    builtin_path = "crate::builtins::io::filetext::fread"
)]
fn fread_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest)?;
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

pub fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FreadEval> {
    let fid_host = gather_value(fid_value)?;
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
        classify_arguments(&arg_refs)
            .map_err(|e| fread_error(format!("fread: {e}")))?;

    let size_host = size_arg.map(gather_value).transpose()?;
    let precision_host = precision_arg.map(gather_value).transpose()?;
    let skip_host = skip_arg.map(gather_value).transpose()?;
    let machine_host = machine_arg.map(gather_value).transpose()?;

    let size_spec = map_string_result(parse_size(size_host.as_ref()))?;
    let precision = map_string_result(parse_precision(precision_host.as_ref()))?;
    let skip_bytes = map_string_result(parse_skip(skip_host.as_ref()))?;
    let machine_format =
        map_string_result(parse_machine_format(machine_host.as_ref(), &info.machinefmt))?;

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

fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed(value).map_err(map_control_flow)
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_reads_default_double() {
        registry::reset_for_tests();
        let path = unique_path("fread_default_double");
        let mut file = File::create(&path).expect("create");
        file.write_all(&1.5f64.to_le_bytes()).expect("write");
        drop(file);

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let eval = evaluate(&Value::Num(fid as f64), &Vec::new()).expect("fread");
        assert_eq!(eval.count(), 1);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 1.5).abs() < 1e-12);
            }
            other => panic!("unexpected result {other:?}"),
        }

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_uint8_vector_with_count() {
        registry::reset_for_tests();
        let path = unique_path("fread_uint8_vector");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5]).expect("write");
        drop(file);

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(4.0), Value::from("uint8")];
        let eval = evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 4);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
                assert_eq!(t.shape, vec![4, 1]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_uint8_matrix_with_padding() {
        registry::reset_for_tests();
        let path = unique_path("fread_uint8_matrix");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5]).expect("write");
        drop(file);

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let size_tensor = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(size_tensor), Value::from("uint8")];
        let eval = evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 5);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_char_output() {
        registry::reset_for_tests();
        let path = unique_path("fread_char_output");
        let mut file = File::create(&path).expect("create");
        file.write_all(b"abc").expect("write");
        drop(file);

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let eval = evaluate(&Value::Num(fid as f64), &[Value::from("*char")]).expect("fread");
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

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_logical_output() {
        registry::reset_for_tests();
        let path = unique_path("fread_like_logical_output");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[0u8, 3, 0, 4]).expect("write");
        drop(file);

        let open = fopen::evaluate(&[
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
        let eval = evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 2);
        match eval.data() {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 1]);
                assert_eq!(array.data, vec![0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_requires_prototype() {
        registry::reset_for_tests();
        let path = unique_path("fread_like_requires_prototype");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4]).expect("write");
        drop(file);

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::from("like")];
        let err = unwrap_error_message(evaluate(&Value::Num(fid as f64), &args).unwrap_err());
        assert!(err.contains("expected prototype after 'like'"));

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_char_requires_precision() {
        registry::reset_for_tests();
        let path = unique_path("fread_like_char_requires_precision");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[65u8]).expect("write");
        drop(file);

        let open = fopen::evaluate(&[
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
        let err = unwrap_error_message(evaluate(&Value::Num(fid as f64), &args).unwrap_err());
        assert!(err.contains("character prototypes require"));

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_like_gpu_provider_roundtrip() {
        registry::reset_for_tests();
        let path = unique_path("fread_like_gpu_provider_roundtrip");
        let mut file = File::create(&path).expect("create");
        file.write_all(&1.5f64.to_le_bytes()).expect("write");
        drop(file);

        test_support::with_test_provider(|provider| {
            let open = fopen::evaluate(&[
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
            let eval = evaluate(&Value::Num(fid as f64), &args).expect("fread");
            match eval.data() {
                Value::GpuTensor(result) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(result.clone())).expect("gather");
                    assert_eq!(gathered.data, vec![1.5]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }

            fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        });

        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fread_wgpu_like_uploads_gpu() {
        registry::reset_for_tests();
        let path = unique_path("fread_wgpu_like_uploads_gpu");
        let mut file = File::create(&path).expect("create");
        file.write_all(&2.25f64.to_le_bytes()).expect("write");
        drop(file);

        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let gpu_proto = crate::call_builtin("gpuArray", &[Value::Tensor(proto)]).expect("gpuArray");

        let args = vec![
            Value::Num(1.0),
            Value::from("double"),
            Value::from("like"),
            gpu_proto,
        ];
        let eval = evaluate(&Value::Num(fid as f64), &args).expect("fread");
        match eval.data() {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle.clone())).expect("gather");
                assert_eq!(gathered.data, vec![2.25]);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_skip_bytes() {
        registry::reset_for_tests();
        let path = unique_path("fread_skip_bytes");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3, 4, 5, 6]).expect("write");
        drop(file);

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(3.0), Value::from("uint8"), Value::Num(1.0)];
        let eval = evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 3);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 3.0, 5.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_big_endian_machine_format() {
        registry::reset_for_tests();
        let path = unique_path("fread_big_endian");
        let mut file = File::create(&path).expect("create");
        file.write_all(&[0x01, 0x02, 0x03, 0x04]).expect("write");
        drop(file);

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
            Value::from("ieee-be"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![Value::Num(2.0), Value::from("uint16")];
        let eval = evaluate(&Value::Num(fid as f64), &args).expect("fread");
        assert_eq!(eval.count(), 2);
        match eval.data() {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![258.0, 772.0]);
                assert_eq!(t.shape, vec![2, 1]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fread_invalid_fid_errors() {
        registry::reset_for_tests();
        let err = unwrap_error_message(evaluate(&Value::Num(9999.0), &Vec::new()).unwrap_err());
        assert!(err.contains("Invalid file identifier"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
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
