//! MATLAB-compatible `jsonencode` builtin for serialising RunMat values to JSON text.

use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;

use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, IntValue, LogicalArray, ObjectInstance, StringArray,
    StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

const OPTION_NAME_ERROR: &str = "jsonencode: option names must be character vectors or strings";
const OPTION_VALUE_ERROR: &str = "jsonencode: option value must be scalar logical or numeric";
const INF_NAN_ERROR: &str = "jsonencode: ConvertInfAndNaN must be true to encode NaN or Inf values";
const UNSUPPORTED_TYPE_ERROR: &str =
    "jsonencode: unsupported input type; expected numeric, logical, string, struct, cell, or object data";

#[allow(clippy::too_many_lines)]
#[runmat_macros::register_doc_text(
    name = "jsonencode",
    builtin_path = "crate::builtins::io::json::jsonencode"
)]
pub const DOC_MD: &str = r#"---
title: "jsonencode"
category: "io/json"
keywords: ["jsonencode", "json", "serialization", "struct to json", "pretty print json", "gpu gather"]
summary: "Serialize MATLAB values to UTF-8 JSON text with MATLAB-compatible defaults."
references:
  - https://www.mathworks.com/help/matlab/ref/jsonencode.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "jsonencode gathers GPU-resident data to host memory before serialisation and executes entirely on the CPU."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::json::jsonencode::tests"
  integration:
    - "builtins::io::json::jsonencode::tests::jsonencode_struct_round_trip"
    - "builtins::io::json::jsonencode::tests::jsonencode_gpu_tensor_gathers_host_data"
    - "builtins::io::json::jsonencode::tests::jsonencode_struct_options_enable_pretty_print"
    - "builtins::io::json::jsonencode::tests::jsonencode_convert_inf_and_nan_controls_null_output"
    - "builtins::io::json::jsonencode::tests::jsonencode_char_array_zero_rows_is_empty_array"
    - "builtins::io::json::jsonencode::tests::jsonencode_gpu_tensor_wgpu_gathers_host_data"
---

# What does the `jsonencode` function do in MATLAB / RunMat?
`jsonencode` converts MATLAB values into UTF-8 JSON text. The builtin mirrors MATLAB defaults:
scalars become numbers or strings, matrices turn into JSON arrays, structs map to JSON objects,
and `NaN`/`Inf` values encode as `null` unless you disable the conversion.

## How does the `jsonencode` function behave in MATLAB / RunMat?
- Returns a 1×N character array containing UTF-8 encoded JSON text.
- Numeric and logical arrays become JSON arrays, preserving MATLAB column-major ordering.
- Scalars encode as bare numbers/strings rather than single-element arrays.
- Struct scalars become JSON objects; struct arrays become JSON arrays of objects.
- Cell arrays map to JSON arrays, with nested arrays when the cell is 2-D.
- String arrays and char arrays become JSON strings (1 element) or arrays of strings (multiple rows).
- By default, `NaN`, `Inf`, and `-Inf` values encode as `null`. Set `'ConvertInfAndNaN'` to `false`
  to raise an error instead.
- Pretty printing is disabled by default; enable it with the `'PrettyPrint'` option.
- Inputs that reside on the GPU are gathered back to host memory automatically.

## jsonencode Options
| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `PrettyPrint` | logical | `false` | Emit indented, multi-line JSON output. |
| `ConvertInfAndNaN` | logical | `true` | Convert `NaN`, `Inf`, `-Inf` to `null`. Set `false` to raise an error when these values are encountered. |

## Examples of using the `jsonencode` function in MATLAB / RunMat

### Converting a MATLAB struct to JSON
```matlab
person = struct('name', 'Ada', 'age', 37);
encoded = jsonencode(person);
```
Expected output:
```matlab
encoded = '{"age":37,"name":"Ada"}'
```

### Serialising a matrix with pretty printing
```matlab
A = magic(3);
encoded = jsonencode(A, 'PrettyPrint', true);
```
Expected output:
```matlab
encoded =
'[
    [8,1,6],
    [3,5,7],
    [4,9,2]
]'
```

### Encoding nested cell arrays
```matlab
C = {struct('task','encode','ok',true), {'nested', 42}};
encoded = jsonencode(C);
```
Expected output:
```matlab
encoded = '[{"ok":true,"task":"encode"},["nested",42]]'
```

### Handling NaN and Inf values
```matlab
data = [1 NaN Inf];
encoded = jsonencode(data);
```
Expected output:
```matlab
encoded = '[1,null,null]'
```

### Rejecting NaN when ConvertInfAndNaN is false
```matlab
try
    jsonencode(data, 'ConvertInfAndNaN', false);
catch err
    disp(err.message);
end
```
Expected output:
```matlab
jsonencode: ConvertInfAndNaN must be true to encode NaN or Inf values
```

### Serialising GPU-resident tensors
```matlab
G = gpuArray(eye(2));
encoded = jsonencode(G);
```
Expected output:
```matlab
encoded = '[[1,0],[0,1]]'
```

## `jsonencode` Function GPU Execution Behaviour
`jsonencode` never launches GPU kernels. When the input contains `gpuArray` data, RunMat gathers
those values back to host memory via the active acceleration provider and then serialises on the CPU.
If no provider is registered, the builtin propagates the same gather error used by other residency
sinks (`gather: no acceleration provider registered`).

## GPU residency in RunMat (Do I need `gpuArray`?)
For most workflows you do not need to call `gpuArray` explicitly before using `jsonencode`. The
auto-offload planner and fusion system keep track of residency, so any GPU-backed tensors that flow
into `jsonencode` are gathered automatically as part of this sink operation. If you prefer to control
residency manually—or need MATLAB parity—you can still wrap data with `gpuArray` and call `gather`
explicitly before serialising.

## FAQ

### What MATLAB types does `jsonencode` support?
Numeric, logical, string, char, struct, cell, and table-like structs are supported. Unsupported types
such as function handles or opaque objects raise an error.

### Why are my field names sorted alphabetically?
RunMat sorts struct field names to produce deterministic JSON (matching MATLAB when fields are stored
as scalar structs).

### How are complex numbers encoded?
Complex scalars become objects with `real` and `imag` fields. Complex arrays become arrays of those
objects, mirroring MATLAB.

### Does `jsonencode` return a character array or string?
It returns a row character array (`char`) for MATLAB compatibility. Use `string(jsonencode(x))` if
you prefer string scalars.

### Can I pretty-print nested structures?
Yes. Pass `'PrettyPrint', true` to `jsonencode`. Indentation uses four spaces per nesting level, just
like MATLAB's pretty-print mode.

### How are empty arrays encoded?
Empty numeric, logical, char, and string arrays become `[]`. Empty structs become `{}` if scalar, or
`[]` if they are empty arrays of structs.

### Does `jsonencode` preserve MATLAB column-major ordering?
Yes. Arrays are emitted in MATLAB's logical row/column order, so reshaping on decode reproduces the
original layout.

### What happens when ConvertInfAndNaN is false?
Encountering `NaN`, `Inf`, or `-Inf` raises `jsonencode: ConvertInfAndNaN must be true to encode NaN or Inf values`.

### How do I control the newline style?
`jsonencode` always emits `\n` (LF) line endings when `PrettyPrint` is enabled, regardless of platform,
matching MATLAB's behaviour.

### Are Unicode characters escaped?
Printable Unicode characters are emitted verbatim. Control characters and quotes are escaped using
standard JSON escape sequences.

## See Also
[jsondecode](./jsondecode), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `jsonencode` function is available at: [`crates/runmat-runtime/src/builtins/io/json/jsonencode.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/json/jsonencode.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::json::jsonencode")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "jsonencode",
    op_kind: GpuOpKind::Custom("serialization"),
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
        "Serialization sink that gathers GPU data to host memory before emitting UTF-8 JSON text.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::json::jsonencode")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "jsonencode",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "jsonencode is a residency sink and never participates in fusion planning.",
};

#[derive(Debug, Clone)]
struct JsonEncodeOptions {
    pretty_print: bool,
    convert_inf_and_nan: bool,
}

impl Default for JsonEncodeOptions {
    fn default() -> Self {
        Self {
            pretty_print: false,
            convert_inf_and_nan: true,
        }
    }
}

#[derive(Debug, Clone)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

#[derive(Debug, Clone)]
enum JsonNumber {
    Float(f64),
    I64(i64),
    U64(u64),
}

#[runtime_builtin(
    name = "jsonencode",
    category = "io/json",
    summary = "Serialize MATLAB values to UTF-8 JSON text.",
    keywords = "jsonencode,json,serialization,struct,gpu",
    accel = "cpu",
    builtin_path = "crate::builtins::io::json::jsonencode"
)]
fn jsonencode_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let host_value = gather_if_needed(&value)?;
    let gathered_args: Vec<Value> = rest
        .iter()
        .map(gather_if_needed)
        .collect::<Result<_, _>>()?;

    let options = parse_options(&gathered_args)?;
    let json_value = value_to_json(&host_value, &options)?;
    let json_string = render_json(&json_value, &options);

    Ok(Value::CharArray(CharArray::new_row(&json_string)))
}

fn parse_options(args: &[Value]) -> Result<JsonEncodeOptions, String> {
    let mut options = JsonEncodeOptions::default();
    if args.is_empty() {
        return Ok(options);
    }

    if args.len() == 1 {
        if let Value::Struct(struct_value) = &args[0] {
            apply_struct_options(struct_value, &mut options)?;
            return Ok(options);
        }
        return Err("jsonencode: expected name/value pairs or options struct".to_string());
    }

    if !args.len().is_multiple_of(2) {
        return Err("jsonencode: name/value pairs must come in pairs".to_string());
    }

    let mut idx = 0usize;
    while idx < args.len() {
        let name = option_name(&args[idx])?;
        let value = &args[idx + 1];
        apply_option(&name, value, &mut options)?;
        idx += 2;
    }

    Ok(options)
}

fn apply_struct_options(
    struct_value: &StructValue,
    options: &mut JsonEncodeOptions,
) -> Result<(), String> {
    for (key, value) in &struct_value.fields {
        apply_option(key, value, options)?;
    }
    Ok(())
}

fn option_name(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(OPTION_NAME_ERROR.to_string()),
    }
}

fn apply_option(
    raw_name: &str,
    value: &Value,
    options: &mut JsonEncodeOptions,
) -> Result<(), String> {
    let lowered = raw_name.to_ascii_lowercase();
    match lowered.as_str() {
        "prettyprint" => {
            options.pretty_print = coerce_bool(value)?;
            Ok(())
        }
        "convertinfandnan" => {
            options.convert_inf_and_nan = coerce_bool(value)?;
            Ok(())
        }
        other => Err(format!("jsonencode: unknown option '{}'", other)),
    }
}

fn coerce_bool(value: &Value) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(i.to_i64() != 0),
        Value::Num(n) => bool_from_f64(*n),
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                bool_from_f64(t.data[0])
            } else {
                Err(OPTION_VALUE_ERROR.to_string())
            }
        }
        Value::LogicalArray(la) => match la.data.len() {
            1 => Ok(la.data[0] != 0),
            _ => Err(OPTION_VALUE_ERROR.to_string()),
        },
        Value::CharArray(ca) if ca.rows == 1 => {
            parse_bool_string(&ca.data.iter().collect::<String>())
        }
        Value::String(s) => parse_bool_string(s),
        Value::StringArray(sa) if sa.data.len() == 1 => parse_bool_string(&sa.data[0]),
        _ => Err(OPTION_VALUE_ERROR.to_string()),
    }
}

fn bool_from_f64(value: f64) -> Result<bool, String> {
    if value.is_finite() {
        Ok(value != 0.0)
    } else {
        Err(OPTION_VALUE_ERROR.to_string())
    }
}

fn parse_bool_string(text: &str) -> Result<bool, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(OPTION_VALUE_ERROR.to_string()),
    }
}

fn value_to_json(value: &Value, options: &JsonEncodeOptions) -> Result<JsonValue, String> {
    match value {
        Value::Num(n) => number_to_json(*n, options),
        Value::Int(i) => Ok(JsonValue::Number(int_to_number(i))),
        Value::Bool(b) => Ok(JsonValue::Bool(*b)),
        Value::LogicalArray(logical) => logical_array_to_json(logical, options),
        Value::Tensor(tensor) => tensor_to_json(tensor, options),
        Value::Complex(re, im) => complex_scalar_to_json(*re, *im, options),
        Value::ComplexTensor(ct) => complex_tensor_to_json(ct, options),
        Value::String(s) => Ok(JsonValue::String(s.clone())),
        Value::StringArray(sa) => string_array_to_json(sa, options),
        Value::CharArray(ca) => char_array_to_json(ca, options),
        Value::Struct(sv) => struct_to_json(sv, options),
        Value::Cell(ca) => cell_array_to_json(ca, options),
        Value::Object(obj) => object_to_json(obj, options),
        Value::GpuTensor(_) => {
            Err("jsonencode: unexpected gpuArray handle after gather pass".to_string())
        }
        Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(UNSUPPORTED_TYPE_ERROR.to_string()),
    }
}

fn int_to_number(value: &IntValue) -> JsonNumber {
    match value {
        IntValue::I8(v) => JsonNumber::I64(*v as i64),
        IntValue::I16(v) => JsonNumber::I64(*v as i64),
        IntValue::I32(v) => JsonNumber::I64(*v as i64),
        IntValue::I64(v) => JsonNumber::I64(*v),
        IntValue::U8(v) => JsonNumber::U64(*v as u64),
        IntValue::U16(v) => JsonNumber::U64(*v as u64),
        IntValue::U32(v) => JsonNumber::U64(*v as u64),
        IntValue::U64(v) => JsonNumber::U64(*v),
    }
}

fn number_to_json(value: f64, options: &JsonEncodeOptions) -> Result<JsonValue, String> {
    if !value.is_finite() {
        if options.convert_inf_and_nan {
            return Ok(JsonValue::Null);
        }
        return Err(INF_NAN_ERROR.to_string());
    }
    Ok(JsonValue::Number(JsonNumber::Float(value)))
}

fn logical_array_to_json(
    logical: &LogicalArray,
    _options: &JsonEncodeOptions,
) -> Result<JsonValue, String> {
    let keep_dims = compute_keep_dims(&logical.shape, true);
    if logical.shape.is_empty() || logical.data.is_empty() {
        return Ok(JsonValue::Array(Vec::new()));
    }
    if keep_dims.is_empty() {
        let first = logical.data.first().copied().unwrap_or(0) != 0;
        return Ok(JsonValue::Bool(first));
    }
    build_strided_array(&logical.shape, &keep_dims, |offset| {
        Ok(JsonValue::Bool(logical.data[offset] != 0))
    })
}

fn tensor_to_json(tensor: &Tensor, options: &JsonEncodeOptions) -> Result<JsonValue, String> {
    if tensor.data.is_empty() {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let keep_dims = compute_keep_dims(&tensor.shape, true);
    if keep_dims.is_empty() {
        return number_to_json(tensor.data[0], options);
    }
    build_strided_array(&tensor.shape, &keep_dims, |offset| {
        number_to_json(tensor.data[offset], options)
    })
}

fn complex_scalar_to_json(
    real: f64,
    imag: f64,
    options: &JsonEncodeOptions,
) -> Result<JsonValue, String> {
    let real_json = number_to_json(real, options)?;
    let imag_json = number_to_json(imag, options)?;
    Ok(JsonValue::Object(vec![
        ("real".to_string(), real_json),
        ("imag".to_string(), imag_json),
    ]))
}

fn complex_tensor_to_json(
    ct: &ComplexTensor,
    options: &JsonEncodeOptions,
) -> Result<JsonValue, String> {
    if ct.data.is_empty() {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let keep_dims = compute_keep_dims(&ct.shape, true);
    if keep_dims.is_empty() {
        let (re, im) = ct.data[0];
        return complex_scalar_to_json(re, im, options);
    }
    build_strided_array(&ct.shape, &keep_dims, |offset| {
        let (re, im) = ct.data[offset];
        complex_scalar_to_json(re, im, options)
    })
}

fn string_array_to_json(
    sa: &StringArray,
    _options: &JsonEncodeOptions,
) -> Result<JsonValue, String> {
    if sa.data.is_empty() {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let keep_dims = compute_keep_dims(&sa.shape, true);
    if keep_dims.is_empty() {
        return Ok(JsonValue::String(sa.data[0].clone()));
    }
    build_strided_array(&sa.shape, &keep_dims, |offset| {
        Ok(JsonValue::String(sa.data[offset].clone()))
    })
}

fn char_array_to_json(ca: &CharArray, _options: &JsonEncodeOptions) -> Result<JsonValue, String> {
    if ca.rows == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }

    if ca.cols == 0 {
        if ca.rows == 1 {
            return Ok(JsonValue::String(String::new()));
        }
        let mut rows = Vec::with_capacity(ca.rows);
        for _ in 0..ca.rows {
            rows.push(JsonValue::String(String::new()));
        }
        return Ok(JsonValue::Array(rows));
    }

    if ca.rows == 1 {
        return Ok(JsonValue::String(ca.data.iter().collect()));
    }

    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row_string = String::with_capacity(ca.cols);
        for c in 0..ca.cols {
            row_string.push(ca.data[r * ca.cols + c]);
        }
        rows.push(JsonValue::String(row_string));
    }
    Ok(JsonValue::Array(rows))
}

fn struct_to_json(sv: &StructValue, options: &JsonEncodeOptions) -> Result<JsonValue, String> {
    if sv.fields.is_empty() {
        return Ok(JsonValue::Object(Vec::new()));
    }
    let mut map = BTreeMap::new();
    for (key, value) in &sv.fields {
        map.insert(key.clone(), value_to_json(value, options)?);
    }
    Ok(JsonValue::Object(map.into_iter().collect()))
}

fn object_to_json(obj: &ObjectInstance, options: &JsonEncodeOptions) -> Result<JsonValue, String> {
    let mut map = BTreeMap::new();
    for (key, value) in &obj.properties {
        map.insert(key.clone(), value_to_json(value, options)?);
    }
    Ok(JsonValue::Object(map.into_iter().collect()))
}

fn cell_array_to_json(ca: &CellArray, options: &JsonEncodeOptions) -> Result<JsonValue, String> {
    if ca.rows == 0 || ca.cols == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }

    if ca.rows == 1 && ca.cols == 1 {
        let value = ca.get(0, 0).map_err(|e| format!("jsonencode: {e}"))?;
        return Ok(JsonValue::Array(vec![value_to_json(&value, options)?]));
    }

    if ca.rows == 1 {
        let mut row = Vec::with_capacity(ca.cols);
        for c in 0..ca.cols {
            let element = ca.get(0, c).map_err(|e| format!("jsonencode: {e}"))?;
            row.push(value_to_json(&element, options)?);
        }
        return Ok(JsonValue::Array(row));
    }

    if ca.cols == 1 {
        let mut column = Vec::with_capacity(ca.rows);
        for r in 0..ca.rows {
            let element = ca.get(r, 0).map_err(|e| format!("jsonencode: {e}"))?;
            column.push(value_to_json(&element, options)?);
        }
        return Ok(JsonValue::Array(column));
    }

    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row = Vec::with_capacity(ca.cols);
        for c in 0..ca.cols {
            let element = ca.get(r, c).map_err(|e| format!("jsonencode: {e}"))?;
            row.push(value_to_json(&element, options)?);
        }
        rows.push(JsonValue::Array(row));
    }
    Ok(JsonValue::Array(rows))
}

fn compute_keep_dims(shape: &[usize], drop_singletons: bool) -> Vec<usize> {
    let mut keep = Vec::new();
    for (idx, &size) in shape.iter().enumerate() {
        if size != 1 || !drop_singletons {
            keep.push(idx);
        }
    }
    keep
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut acc = 1usize;
    for &size in shape {
        strides.push(acc);
        acc = acc.saturating_mul(size.max(1));
    }
    strides
}

fn build_strided_array<F>(
    shape: &[usize],
    keep_dims: &[usize],
    mut fetch: F,
) -> Result<JsonValue, String>
where
    F: FnMut(usize) -> Result<JsonValue, String>,
{
    if keep_dims.is_empty() {
        return fetch(0);
    }
    if keep_dims.iter().any(|&idx| shape[idx] == 0) {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let strides = compute_strides(shape);
    let dims: Vec<usize> = keep_dims.iter().map(|&idx| shape[idx]).collect();
    build_nd_array(&dims, |indices| {
        let mut offset = 0usize;
        for (value, dim_idx) in indices.iter().zip(keep_dims.iter()) {
            offset += value * strides[*dim_idx];
        }
        fetch(offset)
    })
}

fn build_nd_array<F>(dims: &[usize], mut fetch: F) -> Result<JsonValue, String>
where
    F: FnMut(&[usize]) -> Result<JsonValue, String>,
{
    if dims.is_empty() {
        return fetch(&[]);
    }
    if dims[0] == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let mut indices = vec![0usize; dims.len()];
    build_nd_array_recursive(dims, 0, &mut indices, &mut fetch)
}

fn build_nd_array_recursive<F>(
    dims: &[usize],
    level: usize,
    indices: &mut [usize],
    fetch: &mut F,
) -> Result<JsonValue, String>
where
    F: FnMut(&[usize]) -> Result<JsonValue, String>,
{
    let size = dims[level];
    if size == 0 {
        return Ok(JsonValue::Array(Vec::new()));
    }
    if level + 1 == dims.len() {
        let mut items = Vec::with_capacity(size);
        for i in 0..size {
            indices[level] = i;
            items.push(fetch(indices)?);
        }
        return Ok(JsonValue::Array(items));
    }
    let mut items = Vec::with_capacity(size);
    for i in 0..size {
        indices[level] = i;
        items.push(build_nd_array_recursive(dims, level + 1, indices, fetch)?);
    }
    Ok(JsonValue::Array(items))
}

fn render_json(value: &JsonValue, options: &JsonEncodeOptions) -> String {
    let mut writer = JsonWriter::new(options.pretty_print);
    writer.write_value(value);
    writer.finish()
}

struct JsonWriter {
    output: String,
    pretty: bool,
    indent: usize,
}

impl JsonWriter {
    fn new(pretty: bool) -> Self {
        Self {
            output: String::new(),
            pretty,
            indent: 0,
        }
    }

    fn finish(self) -> String {
        self.output
    }

    fn write_value(&mut self, value: &JsonValue) {
        match value {
            JsonValue::Null => self.output.push_str("null"),
            JsonValue::Bool(true) => self.output.push_str("true"),
            JsonValue::Bool(false) => self.output.push_str("false"),
            JsonValue::Number(number) => self.write_number(number),
            JsonValue::String(text) => {
                self.output.push('"');
                self.output.push_str(&escape_json_string(text));
                self.output.push('"');
            }
            JsonValue::Array(items) => self.write_array(items),
            JsonValue::Object(fields) => self.write_object(fields),
        }
    }

    fn write_number(&mut self, number: &JsonNumber) {
        match number {
            JsonNumber::Float(f) => {
                if f.is_nan() || !f.is_finite() {
                    self.output.push_str("null");
                } else {
                    self.output.push_str(&format_number(*f));
                }
            }
            JsonNumber::I64(i) => {
                let _ = write!(self.output, "{i}");
            }
            JsonNumber::U64(u) => {
                let _ = write!(self.output, "{u}");
            }
        }
    }

    fn write_array(&mut self, items: &[JsonValue]) {
        if items.is_empty() {
            self.output.push_str("[]");
            return;
        }
        let inline = if self.pretty {
            items.iter().all(|item| {
                matches!(
                    item,
                    JsonValue::Null
                        | JsonValue::Bool(_)
                        | JsonValue::Number(_)
                        | JsonValue::String(_)
                )
            })
        } else {
            false
        };
        if inline {
            self.output.push('[');
            for (index, item) in items.iter().enumerate() {
                self.write_value(item);
                if index + 1 < items.len() {
                    self.output.push(',');
                }
            }
            self.output.push(']');
            return;
        }
        self.output.push('[');
        if self.pretty {
            self.output.push('\n');
            self.indent += 1;
        }
        for (index, item) in items.iter().enumerate() {
            if self.pretty {
                self.write_indent();
            }
            self.write_value(item);
            if index + 1 < items.len() {
                if self.pretty {
                    self.output.push_str(",\n");
                } else {
                    self.output.push(',');
                }
            }
        }
        if self.pretty {
            self.output.push('\n');
            if self.indent > 0 {
                self.indent -= 1;
            }
            self.write_indent();
        }
        self.output.push(']');
    }

    fn write_object(&mut self, fields: &[(String, JsonValue)]) {
        if fields.is_empty() {
            self.output.push_str("{}");
            return;
        }
        self.output.push('{');
        if self.pretty {
            self.output.push('\n');
            self.indent += 1;
        }
        for (index, (key, value)) in fields.iter().enumerate() {
            if self.pretty {
                self.write_indent();
            }
            self.output.push('"');
            self.output.push_str(&escape_json_string(key));
            self.output.push('"');
            if self.pretty {
                self.output.push_str(": ");
            } else {
                self.output.push(':');
            }
            self.write_value(value);
            if index + 1 < fields.len() {
                if self.pretty {
                    self.output.push_str(",\n");
                } else {
                    self.output.push(',');
                }
            }
        }
        if self.pretty {
            self.output.push('\n');
            if self.indent > 0 {
                self.indent -= 1;
            }
            self.write_indent();
        }
        self.output.push('}');
    }

    fn write_indent(&mut self) {
        if self.pretty {
            for _ in 0..self.indent {
                self.output.push_str("    ");
            }
        }
    }
}

fn escape_json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0C}' => escaped.push_str("\\f"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(escaped, "\\u{:04X}", c as u32);
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        // Display integer-like doubles without decimal point
        format!("{:.0}", value)
    } else {
        format!("{}", value)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, StructValue, Tensor,
    };

    fn as_string(value: Value) -> String {
        match value {
            Value::CharArray(ca) => ca.data.iter().collect(),
            Value::String(s) => s,
            other => panic!("expected char array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_scalar_double() {
        let encoded = jsonencode_builtin(Value::Num(5.0), Vec::new()).expect("jsonencode");
        assert_eq!(as_string(encoded), "5");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_matrix_pretty_print() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).expect("tensor");
        let args = vec![Value::from("PrettyPrint"), Value::Bool(true)];
        let encoded = jsonencode_builtin(Value::Tensor(tensor), args).expect("jsonencode");
        let expected = "[\n    [1,2,3],\n    [4,5,6]\n]";
        assert_eq!(as_string(encoded), expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_struct_round_trip() {
        let mut fields = StructValue::new();
        fields
            .fields
            .insert("name".to_string(), Value::from("RunMat"));
        fields
            .fields
            .insert("year".to_string(), Value::Int(IntValue::I32(2025)));
        let encoded = jsonencode_builtin(Value::Struct(fields), Vec::new()).expect("jsonencode");
        assert_eq!(as_string(encoded), "{\"name\":\"RunMat\",\"year\":2025}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_struct_options_enable_pretty_print() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).expect("tensor");
        let mut opts = StructValue::new();
        opts.fields
            .insert("PrettyPrint".to_string(), Value::Bool(true));
        let encoded = jsonencode_builtin(Value::Tensor(tensor), vec![Value::Struct(opts)])
            .expect("jsonencode");
        let expected = "[\n    [1,2],\n    [4,5]\n]";
        assert_eq!(as_string(encoded), expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_options_accept_scalar_tensor_bool() {
        let tensor_value = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
        let args = vec![Value::from("PrettyPrint"), Value::Tensor(tensor_value)];
        let encoded = jsonencode_builtin(Value::Num(42.0), args).expect("jsonencode");
        assert_eq!(as_string(encoded), "42");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_options_reject_non_scalar_tensor_bool() {
        let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
        let err = jsonencode_builtin(
            Value::Num(1.0),
            vec![Value::from("PrettyPrint"), Value::Tensor(tensor)],
        )
        .expect_err("expected failure");
        assert_eq!(err, OPTION_VALUE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_options_accept_scalar_logical_array() {
        let logical = LogicalArray::new(vec![1], vec![1]).expect("logical");
        let args = vec![Value::from("PrettyPrint"), Value::LogicalArray(logical)];
        let encoded = jsonencode_builtin(Value::Num(7.0), args).expect("jsonencode");
        assert_eq!(as_string(encoded), "7");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_convert_inf_and_nan_controls_null_output() {
        let tensor = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).expect("tensor");
        let encoded =
            jsonencode_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("jsonencode");
        assert_eq!(as_string(encoded), "[1,null]");

        let err = jsonencode_builtin(
            Value::Tensor(tensor),
            vec![Value::from("ConvertInfAndNaN"), Value::Bool(false)],
        )
        .expect_err("expected failure");
        assert_eq!(err, INF_NAN_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_cell_array() {
        let elements = vec![Value::from(1.0), Value::from("two")];
        let cell = CellArray::new(elements, 1, 2).expect("cell");
        let encoded = jsonencode_builtin(Value::Cell(cell), Vec::new()).expect("jsonencode");
        assert_eq!(as_string(encoded), "[1,\"two\"]");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_char_array_zero_rows_is_empty_array() {
        let chars = CharArray::new(Vec::new(), 0, 3).expect("char array");
        let encoded = jsonencode_builtin(Value::CharArray(chars), Vec::new()).expect("jsonencode");
        assert_eq!(as_string(encoded), "[]");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_char_array_empty_strings_per_row() {
        let chars = CharArray::new(Vec::new(), 2, 0).expect("char array");
        let encoded = jsonencode_builtin(Value::CharArray(chars), Vec::new()).expect("jsonencode");
        let encoded_str = as_string(encoded);
        assert_eq!(encoded_str, "[\"\",\"\"]");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_string_array_matrix() {
        let sa = StringArray::new(vec!["alpha".to_string(), "beta".to_string()], vec![2, 1])
            .expect("string array");
        let encoded = jsonencode_builtin(Value::StringArray(sa), Vec::new()).expect("jsonencode");
        assert_eq!(as_string(encoded), "[\"alpha\",\"beta\"]");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_complex_tensor_outputs_objects() {
        let ct = ComplexTensor::new(vec![(1.0, 2.0), (3.5, -4.0)], vec![2, 1]).expect("complex");
        let encoded = jsonencode_builtin(Value::ComplexTensor(ct), Vec::new()).expect("jsonencode");
        assert_eq!(
            as_string(encoded),
            "[{\"real\":1,\"imag\":2},{\"real\":3.5,\"imag\":-4}]"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsonencode_gpu_tensor_gathers_host_data() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let encoded =
                jsonencode_builtin(Value::GpuTensor(handle), Vec::new()).expect("jsonencode");
            assert_eq!(as_string(encoded), "[[1,0],[0,1]]");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn jsonencode_gpu_tensor_wgpu_gathers_host_data() {
        let ensure = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider();
        let Some(_) = ensure.ok().flatten() else {
            // No WGPU device available on this host; skip.
            return;
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let encoded = jsonencode_builtin(Value::GpuTensor(handle), Vec::new()).expect("jsonencode");
        assert_eq!(as_string(encoded), "[1,2,3]");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let examples = test_support::doc_examples(DOC_MD);
        assert!(!examples.is_empty());
    }
}
