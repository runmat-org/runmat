//! MATLAB-compatible `fwrite` builtin for RunMat.
use std::io::{Seek, SeekFrom, Write};

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry;
use crate::gather_if_needed;
use runmat_filesystem::File;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fwrite",
        builtin_path = "crate::builtins::io::filetext::fwrite"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fwrite"
category: "io/filetext"
keywords: ["fwrite", "binary write", "io", "precision", "machine format", "skip"]
summary: "Write binary data to a file identifier with MATLAB-compatible precision, skip, and machine-format semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/fwrite.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. When data or arguments live on the GPU, RunMat gathers them first; providers do not expose file-I/O hooks."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 4
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::filetext::fwrite::tests"
  integration:
    - "builtins::io::filetext::fwrite::tests::fwrite_double_precision_writes_native_endian"
    - "builtins::io::filetext::fwrite::tests::fwrite_gpu_tensor_gathers_before_write"
    - "builtins::io::filetext::fwrite::tests::fwrite_wgpu_tensor_roundtrip"
    - "builtins::io::filetext::fwrite::tests::fwrite_invalid_precision_errors"
    - "builtins::io::filetext::fwrite::tests::fwrite_negative_skip_errors"
---

# What does the `fwrite` function do in MATLAB / RunMat?
`fwrite` writes binary data to a file identifier obtained from `fopen`. It mirrors MATLAB's handling
of precision strings, skip values, and machine-format overrides so existing MATLAB scripts can save data
without modification. The builtin accepts numeric tensors, logical data, and character arrays; the bytes are
emitted in column-major order to match MATLAB storage.

## How does the `fwrite` function behave in MATLAB / RunMat?
- `count = fwrite(fid, A)` converts `A` to unsigned 8-bit integers and writes one byte per element.
- `count = fwrite(fid, A, precision)` converts `A` to the requested precision before writing. Supported
  precisions are `double`, `single`, `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`,
  `int64`, and `char`. Shorthand aliases such as `uchar`, `byte`, and `real*8` are also recognised.
- `count = fwrite(fid, A, precision, skip)` skips `skip` bytes after writing each element. RunMat applies the
  skip with a file seek, which produces sparse regions when the target position moves beyond the current end.
- `count = fwrite(fid, A, precision, skip, machinefmt)` overrides the byte ordering used for the conversion.
  Supported machine formats are `'native'`, `'ieee-le'`, and `'ieee-be'`. When omitted, the builtin honours the
  format recorded by `fopen`.
- Column-major ordering matches MATLAB semantics: tensors and character arrays write their first column
  completely before advancing to the next column. Scalars and vectors behave as 1-by-N matrices.
- The return value `count` is the number of elements written, not the number of bytes. A zero-length input
  produces `count == 0`.
- RunMat executes `fwrite` entirely on the host. When the data resides on a GPU (`gpuArray`), RunMat gathers the
  tensor to host memory before writing; providers do not currently implement device-side file I/O.

## `fwrite` Function GPU Execution Behaviour
`fwrite` never launches GPU kernels. If any input value (file identifier, data, or optional arguments) is backed
by a GPU tensor, RunMat gathers the value to host memory before performing the write. This mirrors MATLAB's own
behaviour when working with `gpuArray` objects: data is moved to the CPU for I/O. When a provider is available,
the gather occurs via the provider's `download` path; otherwise the builtin emits an informative error.

## GPU residency in RunMat (Do I need `gpuArray`?)
You rarely need to move data with `gpuArray` purely for `fwrite`. RunMat keeps tensors on the GPU while compute
stays in fused expressions, but explicit file I/O always happens on the host. If your data already lives on the
device, `fwrite` performs an automatic gather, writes the bytes, and leaves residency unchanged for the rest of
the program. You can still call `gpuArray` manually when porting MATLAB code verbatim—the builtin will gather it
for you automatically.

## Examples of using the `fwrite` function in MATLAB / RunMat

### Write unsigned bytes with the default precision
```matlab
fid = fopen('bytes.bin', 'w+b');
count = fwrite(fid, [1 2 3 255]);
fclose(fid);
```
Expected output:
```matlab
count = 4
```

### Write double-precision values
```matlab
fid = fopen('values.bin', 'w+b');
data = [1.5 -2.25 42.0];
count = fwrite(fid, data, 'double');
fclose(fid);
```
Expected output:
```matlab
count = 3
```
The file contains three IEEE 754 doubles in the machine format recorded by `fopen`.

### Write 16-bit integers using big-endian byte ordering
```matlab
fid = fopen('sensor.be', 'w+b', 'ieee-be');
fwrite(fid, [258 772], 'uint16');
fclose(fid);
```
Expected output:
```matlab
count = 2
```
The bytes on disk follow big-endian ordering (`01 02 03 04` for the values above).

### Insert padding bytes between samples
```matlab
fid = fopen('spaced.bin', 'w+b');
fwrite(fid, [10 20 30], 'uint8', 1);   % skip one byte between elements
fclose(fid);
```
Expected output:
```matlab
count = 3
```
The file layout is `0A 00 14 00 1E`, leaving a zero byte between each stored value.

### Write character data without manual conversions
```matlab
fid = fopen('greeting.txt', 'w+b');
fwrite(fid, 'RunMat!', 'char');
fclose(fid);
```
Expected output:
```matlab
count = 7
```
Passing text uses the character codes (UTF-16 code units truncated to 8 bits) and writes them sequentially.

### Gather GPU data before writing
```matlab
fid = fopen('gpu.bin', 'w+b');
G = gpuArray([1 2 3 4]);
count = fwrite(fid, G, 'uint16');
fclose(fid);
```
Expected output when a GPU provider is active:
```matlab
count = 4
```
RunMat gathers `G` to the host before conversion. When no provider is registered, `fwrite` raises an error
stating that GPU values cannot be gathered.

## FAQ

### What precisions does `fwrite` support?
RunMat recognises the commonly used MATLAB precisions: `double`, `single`, `uint8`, `int8`, `uint16`, `int16`,
`uint32`, `int32`, `uint64`, `int64`, and `char`, along with their documented aliases (`real*8`, `uchar`, etc.).
The `precision => output` forms are accepted when both sides match; differing output classes are not implemented yet.

### How are values converted before writing?
Numeric inputs are converted to the requested precision using MATLAB-style rounding (to the nearest integer) with
saturation to the target range. Logical inputs map `true` to 1 and `false` to 0. Character inputs use their Unicode
scalar values.

### What does the return value represent?
`fwrite` returns the number of elements successfully written, not the total number of bytes. Multiply by the element
size when you need to know the byte count.

### Does `skip` insert bytes into the file?
`skip` seeks forward after each element is written. When the seek lands beyond the current end of file, the OS
creates a sparse region (holes are zero-filled on most platforms). Use `skip = 0` (the default) to write densely.

### How do machine formats affect the output?
The machine format controls byte ordering for multi-byte precisions. `'native'` uses the host endianness, `'ieee-le'`
forces little-endian ordering, and `'ieee-be'` forces big-endian ordering regardless of the host.

### Can I write directly to standard output?
Not yet. File identifiers 0, 1, and 2 (stdin, stdout, stderr) are reserved and raise a descriptive error. Use
`fopen` to create a file handle before calling `fwrite`.

### Are GPU tensors supported?
Yes. RunMat gathers GPU tensors to host memory before writing. The gather relies on the active provider; if no
provider is registered, an informative error is raised.

### Do string arrays insert newline characters?
RunMat joins string-array elements using newline (`'\n'`) separators before writing. This mirrors how MATLAB flattens
string arrays to character data for binary I/O.

### What happens with `NaN` or infinite values?
`NaN` values map to zero for integer precisions and remain `NaN` for floating-point precisions. Infinite values
saturate to the min/max integer representable by the target precision.

## See Also
[fopen](./fopen), [fclose](./fclose), [fread](./fread), [fileread](./fileread), [filewrite](./filewrite)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/io/filetext/fwrite.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/fwrite.rs)
- Found a behavioural mismatch? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

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

#[runtime_builtin(
    name = "fwrite",
    category = "io/filetext",
    summary = "Write binary data to a file identifier.",
    keywords = "fwrite,file,io,binary,precision",
    accel = "cpu",
    builtin_path = "crate::builtins::io::filetext::fwrite"
)]
fn fwrite_builtin(fid: Value, data: Value, rest: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&fid, &data, &rest)?;
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
pub fn evaluate(
    fid_value: &Value,
    data_value: &Value,
    rest: &[Value],
) -> Result<FwriteEval, String> {
    let fid_host = gather_value(fid_value)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err("fwrite: file identifier must be non-negative".to_string());
    }
    if fid < 3 {
        return Err("fwrite: standard input/output identifiers are not supported yet".to_string());
    }

    let info = registry::info_for(fid).ok_or_else(|| {
        "fwrite: Invalid file identifier. Use fopen to generate a valid file ID.".to_string()
    })?;
    let handle = registry::take_handle(fid).ok_or_else(|| {
        "fwrite: Invalid file identifier. Use fopen to generate a valid file ID.".to_string()
    })?;

    let mut file = handle
        .lock()
        .map_err(|_| "fwrite: failed to lock file handle (poisoned mutex)".to_string())?;

    let data_host = gather_value(data_value)?;
    let rest_host = gather_args(rest)?;
    let (precision_arg, skip_arg, machine_arg) = classify_arguments(&rest_host)?;

    let precision_spec = parse_precision(precision_arg)?;
    let skip_bytes = parse_skip(skip_arg)?;
    let machine_format = parse_machine_format(machine_arg, &info.machinefmt)?;

    let elements = flatten_elements(&data_host)?;
    let count = write_elements(
        &mut *file,
        &elements,
        precision_spec,
        skip_bytes,
        machine_format,
    )?;
    Ok(FwriteEval::new(count))
}

fn gather_value(value: &Value) -> Result<Value, String> {
    gather_if_needed(value).map_err(|e| format!("fwrite: {e}"))
}

fn gather_args(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed(value).map_err(|e| format!("fwrite: {e}"))?);
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
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::Tensor;
    use runmat_filesystem::{self as fs, File};
    use runmat_time::system_time_now;
    use std::io::Read;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    #[test]
    fn fwrite_default_uint8_bytes() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_uint8");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0, 2.0, 255.0], vec![3, 1]).unwrap();
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &Vec::new()).expect("fwrite");
        assert_eq!(eval.count(), 3);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let bytes = fs::read(&path).expect("read");
        assert_eq!(bytes, vec![1u8, 2, 255]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fwrite_double_precision_writes_native_endian() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_double");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.5, -2.25], vec![2, 1]).unwrap();
        let args = vec![Value::from("double")];
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 2);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let bytes = fs::read(&path).expect("read");
        let expected: Vec<u8> = if cfg!(target_endian = "little") {
            [1.5f64.to_le_bytes(), (-2.25f64).to_le_bytes()].concat()
        } else {
            [1.5f64.to_be_bytes(), (-2.25f64).to_be_bytes()].concat()
        };
        assert_eq!(bytes, expected);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fwrite_big_endian_uint16() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_be");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
            Value::from("ieee-be"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![258.0, 772.0], vec![2, 1]).unwrap();
        let args = vec![Value::from("uint16")];
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 2);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let bytes = fs::read(&path).expect("read");
        assert_eq!(bytes, vec![0x01, 0x02, 0x03, 0x04]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fwrite_skip_inserts_padding() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_skip");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("uint8"), Value::Num(1.0)];
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 3);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let bytes = fs::read(&path).expect("read");
        assert_eq!(bytes, vec![10u8, 0, 20, 0, 30]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fwrite_gpu_tensor_gathers_before_write() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_gpu");

        test_support::with_test_provider(|provider| {
            registry::reset_for_tests();
            let open = fopen::evaluate(&[
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
            let eval = evaluate(&Value::Num(fid as f64), &Value::GpuTensor(handle), &args)
                .expect("fwrite");
            assert_eq!(eval.count(), 4);

            fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
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
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fwrite_invalid_precision_errors() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_invalid_precision");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let args = vec![Value::from("bogus-class")];
        let err = evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).unwrap_err();
        assert!(err.contains("unsupported precision"));
        let _ = fclose::evaluate(&[Value::Num(fid as f64)]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fwrite_negative_skip_errors() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_negative_skip");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![10.0], vec![1, 1]).unwrap();
        let args = vec![Value::from("uint8"), Value::Num(-1.0)];
        let err = evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).unwrap_err();
        assert!(err.contains("skip value must be non-negative"));
        let _ = fclose::evaluate(&[Value::Num(fid as f64)]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn fwrite_wgpu_tensor_roundtrip() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_wgpu_roundtrip");
        let open = fopen::evaluate(&[
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
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::GpuTensor(handle), &args).expect("fwrite");
        assert_eq!(eval.count(), 3);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

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
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fwrite_invalid_identifier_errors() {
        registry::reset_for_tests();
        let err = evaluate(&Value::Num(-1.0), &Value::Num(1.0), &Vec::new()).unwrap_err();
        assert!(err.contains("file identifier must be non-negative"));
    }

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
