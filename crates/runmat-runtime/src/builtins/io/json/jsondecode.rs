//! MATLAB-compatible `jsondecode` builtin for deserialising JSON text into RunMat values.

use once_cell::sync::Lazy;
use runmat_builtins::{
    CellArray, CharArray, LogicalArray, StringArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use serde_json::Value as JsonValue;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{gather_if_needed, build_runtime_error, BuiltinResult, RuntimeError};

const INPUT_TYPE_ERROR: &str = "jsondecode: JSON text must be a character vector or string scalar";
const PARSE_ERROR_PREFIX: &str = "jsondecode: invalid JSON text";

#[allow(clippy::too_many_lines)]
#[runmat_macros::register_doc_text(
    name = "jsondecode",
    builtin_path = "crate::builtins::io::json::jsondecode"
)]
pub const DOC_MD: &str = r#"---
title: "jsondecode"
category: "io/json"
keywords: ["jsondecode", "json", "parse json", "struct from json", "gpu gather"]
summary: "Parse UTF-8 JSON text into MATLAB-compatible RunMat values."
references:
  - https://www.mathworks.com/help/matlab/ref/jsondecode.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "jsondecode gathers GPU-resident text operands to host memory before parsing and executes entirely on the CPU."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::json::jsondecode::tests"
  integration:
    - "builtins::io::json::jsondecode::tests::jsondecode_matrix_to_tensor"
    - "builtins::io::json::jsondecode::tests::jsondecode_object_to_struct"
    - "builtins::io::json::jsondecode::tests::jsondecode_string_array"
    - "builtins::io::json::jsondecode::tests::jsondecode_mixed_array_returns_cell"
    - "builtins::io::json::jsondecode::tests::jsondecode_rectangular_cell_array_preserves_layout"
    - "builtins::io::json::jsondecode::tests::jsondecode_array_of_objects_returns_cell"
    - "builtins::io::json::jsondecode::tests::jsondecode_null_returns_empty_double"
    - "builtins::io::json::jsondecode::tests::jsondecode_round_trip_with_jsonencode"
    - "builtins::io::json::jsondecode::tests::jsondecode_doc_examples_present"
---

# What does the `jsondecode` function do in MATLAB / RunMat?
`jsondecode` converts UTF-8 JSON text into MATLAB-compatible data. Numbers become doubles, logicals
become `logical`, character vectors become char arrays, objects become structs, and arrays expand to
numeric/logical/string arrays when rectangular or cell arrays when heterogeneous.

## How does the `jsondecode` function behave in MATLAB / RunMat?
- Accepts a character vector or string scalar containing UTF-8 JSON data; leading and trailing whitespace is ignored.
- JSON numbers decode to double-precision scalars or dense tensors that follow MATLAB's column-major layout.
- JSON booleans decode to logical scalars or logical arrays with the same shape as the source JSON array.
- JSON strings decode to char row vectors; homogeneous JSON arrays of strings become string arrays with matching dimensions.
- JSON objects decode to scalar structs whose field names match the JSON keys; arrays of objects become cell arrays of structs that preserve element order.
- JSON arrays decode to numeric, logical, or string arrays when every element shares the same type and the array is perfectly rectangular. Rectangular heterogeneous arrays (including those that mix `null` with other types) become cell arrays that preserve the original row/column layout, while irregular arrays fall back to 1-by-N cell vectors that keep element order.
- JSON `null` decodes to the empty double `[]`. When it appears inside a heterogeneous array it becomes a cell element containing `[]`.
- Invalid JSON raises `jsondecode: invalid JSON text (...)` with the underlying parser message.

## Examples of using the `jsondecode` function in MATLAB / RunMat

### Parsing a scalar number
```matlab
value = jsondecode("42");
```
Expected output:
```matlab
value = 42
```

### Decoding a matrix of doubles
```matlab
text = "[[1,2,3],[4,5,6]]";
matrix = jsondecode(text);
```
Expected output:
```matlab
matrix =
     1     2     3
     4     5     6
```

### Converting a JSON object into a struct
```matlab
person = jsondecode('{"name":"Ada","age":37}');
disp(person.name);
disp(person.age);
```
Expected output:
```matlab
Ada
    37
```

### Decoding an array of objects into a cell array
```matlab
text = '[{"name":"Ada","role":"Researcher"},{"name":"Grace","role":"Engineer"}]';
people = jsondecode(text);
people{1}.name
people{2}.role
```
Expected output:
```matlab
people =
  1x2 cell array
    {1x1 struct}    {1x1 struct}

ans = 'Ada'
ans = 'Engineer'
```

### Handling heterogeneous arrays with cell arrays
```matlab
text = '["RunMat", 42, true]';
data = jsondecode(text);
```
Expected output:
```matlab
data =
  1x3 cell array
    {"RunMat"}    {[42]}    {[1]}
```

### Working with null and empty JSON values
```matlab
value = jsondecode("null");
disp(value);
isempty(value)
```
Expected output:
```matlab
[]
ans = logical 1
```

## `jsondecode` Function GPU Execution Behaviour
`jsondecode` does not execute on the GPU. Because the builtin registers as `accel = "sink"` and uses
`ResidencyPolicy::GatherImmediately`, any `gpuArray` input is synchronously gathered to host memory
through the active provider before parsing. Providers do not expose specialised hooks for JSON parsing,
so all work stays on the CPU.

## GPU residency in RunMat (Do I need `gpuArray`?)
You rarely need to manage residency manually. The auto-offload planner gathers GPU-resident text for
`jsondecode` automatically. For MATLAB compatibility, you can still call `gather` yourself, but it is
not required, and fusion graphs terminate at `jsondecode` so downstream results always live on the host.

## FAQ

### What encodings does `jsondecode` support?
JSON text must be UTF-8. RunMat preserves surrogate pairs and Unicode code points exactly as provided.

### How are JSON objects represented?
As scalar structs. Array-of-object JSON values become cell arrays of structs to preserve ordering.

### What happens with JSON numbers that overflow double?
Values must fit in the IEEE-754 double range. When a literal exceeds that range, the parser reports
`number out of range`, which surfaces as a `jsondecode: invalid JSON text (...)` error.

### Does `jsondecode` support comments or trailing commas?
No. The JSON must adhere to RFC 8259. The builtin reports an error when encountering comments or
other non-standard extensions.

### How are JSON arrays of strings represented?
Rectangular arrays of strings become `string` arrays. Mixed content falls back to cell arrays.

### How does `jsondecode` treat `null`?
`null` decodes to the empty double `[]`. When it appears inside an otherwise homogeneous numeric array,
the array is promoted to a cell array so that the `[]` value can be preserved alongside other elements.

### Can I decode deeply nested arrays?
Yes. Rectangular arrays of numbers, logicals, or strings produce dense tensors/string arrays with the
appropriate dimensionality. Irregular arrays return nested cell arrays.

### What if the input is not valid JSON?
The builtin raises `jsondecode: invalid JSON text (...)` with the parser error message.

### Does whitespace or newlines matter?
Leading and trailing whitespace is ignored, and embedded newlines are permitted as long as the text
remains valid JSON.

## See Also
[jsonencode](./jsonencode), [struct](./struct), [cell](./cell), [fileread](./fileread)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::json::jsondecode")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "jsondecode",
    op_kind: GpuOpKind::Custom("parse"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "No GPU kernels: jsondecode gathers gpuArray input to host memory before parsing the JSON text.",
};

fn jsondecode_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("jsondecode")
        .build()
}

fn jsondecode_flow_with_context(err: RuntimeError) -> RuntimeError {
    build_runtime_error(err.message().to_string())
        .with_builtin("jsondecode")
        .with_source(err)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::json::jsondecode")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "jsondecode",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "jsondecode is a residency sink; it always runs on the CPU and breaks fusion graphs.",
};

#[runtime_builtin(
    name = "jsondecode",
    category = "io/json",
    summary = "Parse UTF-8 JSON text into MATLAB-compatible RunMat values.",
    keywords = "jsondecode,json,parse json,struct,gpu",
    accel = "sink",
    builtin_path = "crate::builtins::io::json::jsondecode"
)]
fn jsondecode_builtin(text: Value) -> crate::BuiltinResult<Value> {
    let gathered = gather_if_needed(&text).map_err(jsondecode_flow_with_context)?;
    let source = extract_text(gathered)?;
    let parsed: JsonValue = serde_json::from_str(&source).map_err(|err| {
        build_runtime_error(format!("{PARSE_ERROR_PREFIX} ({err})"))
            .with_builtin("jsondecode")
            .with_source(err)
            .build()
    })?;
    value_from_json(&parsed)
}

pub(crate) fn decode_json_text(text: &str) -> BuiltinResult<Value> {
    let parsed: JsonValue = serde_json::from_str(text).map_err(|err| {
        build_runtime_error(format!("{PARSE_ERROR_PREFIX} ({err})"))
            .with_builtin("jsondecode")
            .with_source(err)
            .build()
    })?;
    value_from_json(&parsed)
}

fn extract_text(value: Value) -> BuiltinResult<String> {
    match value {
        Value::CharArray(array) => {
            if array.rows > 1 {
                return Err(jsondecode_error(INPUT_TYPE_ERROR));
            }
            Ok(array.data.into_iter().collect::<String>())
        }
        Value::String(s) => Ok(s),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(jsondecode_error(INPUT_TYPE_ERROR))
            }
        }
        _other => Err(jsondecode_error(INPUT_TYPE_ERROR)),
    }
}

fn value_from_json(value: &JsonValue) -> BuiltinResult<Value> {
    match value {
        JsonValue::Null => empty_double(),
        JsonValue::Bool(b) => Ok(Value::Bool(*b)),
        JsonValue::Number(num) => parse_json_number(num).map(Value::Num),
        JsonValue::String(s) => {
            let char_array = CharArray::new_row(s);
            Ok(Value::CharArray(char_array))
        }
        JsonValue::Array(arr) => decode_json_array(arr),
        JsonValue::Object(map) => decode_json_object(map),
    }
}

fn decode_json_object(map: &serde_json::Map<String, JsonValue>) -> BuiltinResult<Value> {
    let mut struct_value = StructValue::new();
    for (key, val) in map {
        struct_value
            .fields
            .insert(key.clone(), value_from_json(val)?);
    }
    Ok(Value::Struct(struct_value))
}

fn parse_json_number(number: &serde_json::Number) -> BuiltinResult<f64> {
    if let Some(value) = number.as_f64() {
        return Ok(value);
    }
    let text = number.to_string();
    match text.parse::<f64>() {
        Ok(value) => {
            if value.is_nan() {
                Err(jsondecode_error(format!(
                    "{PARSE_ERROR_PREFIX}: unsupported numeric literal ({text})"
                )))
            } else {
                Ok(value)
            }
        }
        Err(_) => {
            let display = if text.len() > 64 {
                format!("{}...", &text[..64])
            } else {
                text
            };
            Err(jsondecode_error(format!(
                "{PARSE_ERROR_PREFIX}: numeric literal out of range ({display})"
            )))
        }
    }
}

fn decode_json_array(values: &[JsonValue]) -> BuiltinResult<Value> {
    if values.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| jsondecode_error(format!("jsondecode: {e}")))?;
        return Ok(Value::Tensor(tensor));
    }

    if let Some(numeric) = parse_numeric_array(values)? {
        let tensor =
            Tensor::new(numeric.data, numeric.shape).map_err(|e| jsondecode_error(format!("jsondecode: {e}")))?;
        return Ok(Value::Tensor(tensor));
    }

    if let Some(logical) = parse_logical_array(values) {
        let array = LogicalArray::new(logical.data, logical.shape)
            .map_err(|e| jsondecode_error(format!("jsondecode: {e}")))?;
        return Ok(Value::LogicalArray(array));
    }

    if let Some(strings) = parse_string_array(values) {
        let array = StringArray::new(strings.data, strings.shape)
            .map_err(|e| jsondecode_error(format!("jsondecode: {e}")))?;
        return Ok(Value::StringArray(array));
    }

    if let Some(cell) = parse_rectangular_cell_array(values)? {
        return Ok(cell);
    }

    let mut elements = Vec::with_capacity(values.len());
    for element in values {
        elements.push(value_from_json(element)?);
    }
    cell_row(elements)
}

fn empty_double() -> BuiltinResult<Value> {
    static EMPTY_DOUBLE: Lazy<Option<Value>> = Lazy::new(|| {
        Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .ok()
    });
    if let Some(value) = EMPTY_DOUBLE.as_ref() {
        return Ok(value.clone());
    }
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| jsondecode_error(format!("jsondecode: {e}")))
}

fn cell_matrix(elements: Vec<Value>, rows: usize, cols: usize) -> BuiltinResult<Value> {
    let cell =
        CellArray::new(elements, rows, cols).map_err(|e| jsondecode_error(format!("jsondecode: {e}")))?;
    Ok(Value::Cell(cell))
}

fn cell_row(elements: Vec<Value>) -> BuiltinResult<Value> {
    let cols = elements.len();
    cell_matrix(elements, 1, cols)
}


struct NumericTensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

fn parse_numeric_array(values: &[JsonValue]) -> BuiltinResult<Option<NumericTensor>> {
    if values.is_empty() {
        return Ok(None);
    }

    if values.iter().all(|v| v.is_number()) {
        let mut data = Vec::with_capacity(values.len());
        for value in values {
            if let JsonValue::Number(number) = value {
                data.push(parse_json_number(number)?);
            }
        }
        return Ok(Some(NumericTensor {
            data,
            shape: vec![values.len()],
        }));
    }

    if values.iter().all(|v| v.is_array()) {
        let mut children = Vec::with_capacity(values.len());
        for value in values {
            let Some(child_values) = value.as_array() else {
                return Ok(None);
            };
            let Some(child) = parse_numeric_array(child_values)? else {
                return Ok(None);
            };
            children.push(child);
        }
        if children.is_empty() {
            return Ok(None);
        }
        let first_shape = children[0].shape.clone();
        if !children.iter().all(|child| child.shape == first_shape) {
            return Ok(None);
        }
        let mut shape = Vec::with_capacity(first_shape.len() + 1);
        shape.push(children.len());
        shape.extend(first_shape.clone());

        let total: usize = shape.iter().product();
        let rows = shape[0];
        if rows == 0 {
            return Ok(None);
        }
        let inner = total / rows;
        let mut data = vec![0.0; total];
        for (row, child) in children.into_iter().enumerate() {
            if child.data.len() != inner {
                return Ok(None);
            }
            for (idx, value) in child.data.into_iter().enumerate() {
                let offset = row + rows * idx;
                data[offset] = value;
            }
        }
        return Ok(Some(NumericTensor { data, shape }));
    }

    Ok(None)
}

struct LogicalTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
}

fn parse_logical_array(values: &[JsonValue]) -> Option<LogicalTensor> {
    if values.is_empty() {
        return None;
    }

    if values.iter().all(|v| v.is_boolean()) {
        let mut data = Vec::with_capacity(values.len());
        for value in values {
            data.push(if value.as_bool()? { 1 } else { 0 });
        }
        return Some(LogicalTensor {
            data,
            shape: vec![values.len()],
        });
    }

    if values.iter().all(|v| v.is_array()) {
        let mut children = Vec::with_capacity(values.len());
        for value in values {
            let child = parse_logical_array(value.as_array()?)?;
            children.push(child);
        }
        if children.is_empty() {
            return None;
        }
        let first_shape = children[0].shape.clone();
        if !children.iter().all(|child| child.shape == first_shape) {
            return None;
        }
        let mut shape = Vec::with_capacity(first_shape.len() + 1);
        shape.push(children.len());
        shape.extend(first_shape.clone());

        let total: usize = shape.iter().product();
        let rows = shape[0];
        if rows == 0 {
            return None;
        }
        let inner = total / rows;
        let mut data = vec![0u8; total];
        for (row, child) in children.into_iter().enumerate() {
            if child.data.len() != inner {
                return None;
            }
            for (idx, value) in child.data.into_iter().enumerate() {
                let offset = row + rows * idx;
                data[offset] = value;
            }
        }
        return Some(LogicalTensor { data, shape });
    }

    None
}

struct StringTensor {
    data: Vec<String>,
    shape: Vec<usize>,
}

fn parse_string_array(values: &[JsonValue]) -> Option<StringTensor> {
    if values.is_empty() {
        return None;
    }

    if values.iter().all(|v| v.is_string()) {
        let mut data = Vec::with_capacity(values.len());
        for value in values {
            data.push(value.as_str()?.to_string());
        }
        return Some(StringTensor {
            data,
            shape: vec![values.len()],
        });
    }

    if values.iter().all(|v| v.is_array()) {
        let mut children = Vec::with_capacity(values.len());
        for value in values {
            let child = parse_string_array(value.as_array()?)?;
            children.push(child);
        }
        if children.is_empty() {
            return None;
        }
        let first_shape = children[0].shape.clone();
        if !children.iter().all(|child| child.shape == first_shape) {
            return None;
        }
        let mut shape = Vec::with_capacity(first_shape.len() + 1);
        shape.push(children.len());
        shape.extend(first_shape.clone());

        let total: usize = shape.iter().product();
        let rows = shape[0];
        if rows == 0 {
            return None;
        }
        let inner = total / rows;
        let mut data = vec![String::new(); total];
        for (row, mut child) in children.into_iter().enumerate() {
            if child.data.len() != inner {
                return None;
            }
            for (idx, value) in child.data.drain(..).enumerate() {
                let offset = row + rows * idx;
                data[offset] = value;
            }
        }
        return Some(StringTensor { data, shape });
    }

    None
}

fn parse_rectangular_cell_array(values: &[JsonValue]) -> BuiltinResult<Option<Value>> {
    if values.is_empty() || !values.iter().all(|v| v.is_array()) {
        return Ok(None);
    }

    let mut expected_len: Option<usize> = None;
    for value in values {
        let arr = value.as_array().ok_or_else(|| {
            // Should be unreachable due to the `all(|v| v.is_array())` guard above.
            jsondecode_error("jsondecode: inconsistent array state")
        })?;
        match expected_len {
            Some(len) if arr.len() != len => return Ok(None),
            None => expected_len = Some(arr.len()),
            _ => {}
        }
    }

    let cols = expected_len.unwrap_or(0);
    let rows = values.len();

    let mut elements = Vec::with_capacity(rows.saturating_mul(cols));
    if cols > 0 {
        for value in values {
            let arr = value.as_array().expect("validated array value");
            for element in arr {
                elements.push(value_from_json(element)?);
            }
        }
    }

    let cell = cell_matrix(elements, rows, cols)?;
    Ok(Some(cell))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{IntValue, Tensor};

    use crate::builtins::common::test_support;
    use crate::RuntimeError;

    fn char_row(text: &str) -> Value {
        Value::CharArray(CharArray::new_row(text))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_scalar_number() {
        let result = jsondecode_builtin(char_row("42")).expect("jsondecode");
        assert_eq!(result, Value::Num(42.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_boolean_array() {
        let result = jsondecode_builtin(char_row("[true,false,true]")).expect("jsondecode");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![3]);
                assert_eq!(array.data, vec![1, 0, 1]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_matrix_to_tensor() {
        let result = jsondecode_builtin(char_row("[[1,2,3],[4,5,6]]")).expect("jsondecode matrix");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 3]);
                assert_eq!(tensor.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_numeric_tensor_3d() {
        let json = "[[[1,2],[3,4]],[[5,6],[7,8]]]";
        let result = jsondecode_builtin(char_row(json)).expect("jsondecode 3d tensor");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2, 2]);
                assert_eq!(tensor.data, vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]);
            }
            other => panic!("expected tensor, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_numeric_singleton_array_retains_tensor() {
        let result =
            jsondecode_builtin(char_row("[42]")).expect("jsondecode singleton numeric array");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1]);
                assert_eq!(tensor.rows(), 1);
                assert_eq!(tensor.cols(), 1);
                assert_eq!(tensor.data, vec![42.0]);
            }
            other => panic!("expected tensor, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_object_to_struct() {
        let result = jsondecode_builtin(char_row("{\"name\":\"RunMat\",\"year\":2025}"))
            .expect("jsondecode struct");
        match result {
            Value::Struct(struct_value) => {
                assert_eq!(
                    struct_value.fields.get("name"),
                    Some(&Value::CharArray(CharArray::new_row("RunMat")))
                );
                match struct_value.fields.get("year") {
                    Some(Value::Num(year)) => assert_eq!(*year, 2025.0),
                    Some(Value::Int(IntValue::I32(year))) => assert_eq!(*year, 2025),
                    other => panic!("unexpected year field {other:?}"),
                }
            }
            other => panic!("expected struct, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_string_array() {
        let result = jsondecode_builtin(char_row("[\"alpha\",\"beta\",\"gamma\"]"))
            .expect("jsondecode string array");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3]);
                assert_eq!(array.rows, 1);
                assert_eq!(array.cols, 3);
                assert_eq!(
                    array.data,
                    vec![
                        String::from("alpha"),
                        String::from("beta"),
                        String::from("gamma"),
                    ]
                );
            }
            other => panic!("expected string array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_mixed_array_returns_cell() {
        let result =
            jsondecode_builtin(char_row("[\"RunMat\",42,true]")).expect("jsondecode mixed");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 3);
                let first = cell.get(0, 0).expect("cell");
                assert_eq!(first, Value::CharArray(CharArray::new_row("RunMat")));
                let second = cell.get(0, 1).expect("cell");
                assert_eq!(second, Value::Num(42.0));
                let third = cell.get(0, 2).expect("cell");
                assert_eq!(third, Value::Bool(true));
            }
            other => panic!("expected cell array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_rectangular_cell_array_preserves_layout() {
        let text = "[[1,true],[false,null]]";
        let result =
            jsondecode_builtin(char_row(text)).expect("jsondecode rectangular heterogeneous array");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 2);
                assert_eq!(cell.get(0, 0).unwrap(), Value::Num(1.0));
                assert_eq!(cell.get(0, 1).unwrap(), Value::Bool(true));
                assert_eq!(cell.get(1, 0).unwrap(), Value::Bool(false));
                match cell.get(1, 1).unwrap() {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![0, 0]);
                        assert!(t.data.is_empty());
                    }
                    other => panic!("expected empty tensor, got {:?}", other),
                }
            }
            other => panic!("expected cell array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_array_of_objects_returns_cell() {
        let text = "[{\"id\":1,\"name\":\"Ada\"},{\"id\":2,\"name\":\"Charles\"}]";
        let result = jsondecode_builtin(char_row(text)).expect("jsondecode object array");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 2);

                let first = cell.get(0, 0).expect("first struct");
                match first {
                    Value::Struct(struct_value) => {
                        assert_eq!(struct_value.fields.get("id"), Some(&Value::Num(1.0)));
                        assert_eq!(
                            struct_value.fields.get("name"),
                            Some(&Value::CharArray(CharArray::new_row("Ada")))
                        );
                    }
                    other => panic!("expected struct, got {:?}", other),
                }

                let second = cell.get(0, 1).expect("second struct");
                match second {
                    Value::Struct(struct_value) => {
                        assert_eq!(struct_value.fields.get("id"), Some(&Value::Num(2.0)));
                        assert_eq!(
                            struct_value.fields.get("name"),
                            Some(&Value::CharArray(CharArray::new_row("Charles")))
                        );
                    }
                    other => panic!("expected struct, got {:?}", other),
                }
            }
            other => panic!("expected cell array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_null_returns_empty_double() {
        let result = jsondecode_builtin(char_row("null")).expect("jsondecode null");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![0, 0]);
                assert!(tensor.data.is_empty());
            }
            other => panic!("expected empty tensor, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_invalid_text_reports_error() {
        let err = jsondecode_builtin(char_row("{not json}")).expect_err("expected failure");
        let err = error_message(err);
        assert!(
            err.starts_with(PARSE_ERROR_PREFIX),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_rejects_multirow_char_input() {
        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
        let err = jsondecode_builtin(Value::CharArray(chars)).expect_err("expected type error");
        assert_eq!(error_message(err), INPUT_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_accepts_string_input() {
        let result = jsondecode_builtin(Value::String("[1,2]".to_string())).expect("jsondecode");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2]);
                assert_eq!(tensor.data, vec![1.0, 2.0]);
            }
            other => panic!("expected tensor, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_accepts_string_array_scalar_input() {
        let array = StringArray::new(vec!["[1,2]".to_string()], vec![1, 1]).expect("string scalar");
        let result = jsondecode_builtin(Value::StringArray(array)).expect("jsondecode");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2]);
                assert_eq!(tensor.data, vec![1.0, 2.0]);
            }
            other => panic!("expected tensor, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn jsondecode_round_trip_with_jsonencode() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor");
        let encoded =
            crate::call_builtin("jsonencode", &[Value::Tensor(tensor.clone())]).expect("encode");
        let decoded = jsondecode_builtin(encoded).expect("decode");
        match decoded {
            Value::Tensor(result) => {
                assert_eq!(result.data.len(), tensor.data.len());
                assert_eq!(result.data, tensor.data);
            }
            other => panic!("expected tensor, got {:?}", other),
        }
    }
}
