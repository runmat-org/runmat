//! MATLAB-compatible `str2double` builtin with GPU-aware semantics for RunMat.

use std::borrow::Cow;

use runmat_builtins::{CellArray, CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::gather_if_needed;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "str2double",
        builtin_path = "crate::builtins::strings::core::str2double"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "str2double"
category: "strings/core"
keywords: ["str2double", "string to double", "text conversion", "numeric parsing", "gpu"]
summary: "Convert strings, character arrays, or cell arrays of text into double-precision numbers with MATLAB-compatible rules."
references:
  - https://www.mathworks.com/help/matlab/ref/str2double.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs on the CPU. When inputs reference GPU data, RunMat gathers them before parsing so results match MATLAB exactly."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::str2double::tests"
  integration: "builtins::strings::core::str2double::tests::str2double_cell_array_of_text"
---

# What does the `str2double` function do in MATLAB / RunMat?
`str2double` converts text representations of numbers into double-precision values. It accepts string
scalars, string arrays, character vectors, character arrays, and cell arrays of character vectors.
Each element is parsed independently; values that cannot be interpreted as real scalars become `NaN`.

## How does the `str2double` function behave in MATLAB / RunMat?
- Leading and trailing whitespace is ignored, as are padding spaces that MATLAB inserts in character arrays.
- Text that contains a single finite real number returns that number. Text with additional characters,
  embedded operators, or multiple values results in `NaN`.
- Scientific notation with `e`, `E`, `d`, or `D` exponents is supported (`"1.2e3"`, `"4.5D-6"`, etc.).
- `"Inf"`, `"Infinity"`, and `"NaN"` (any letter case, with optional sign on `Inf`) map to IEEE special values.
- Missing string scalars (displayed as `<missing>`) convert to `NaN`, matching MATLAB behaviour.
- Character arrays return a column vector whose length equals the number of rows; cell arrays preserve their shape.

## `str2double` Function GPU Execution Behaviour
`str2double` executes entirely on the CPU. If any argument is backed by a GPU buffer (for example, a cell array
that still wraps GPU-resident character data), RunMat gathers the values first, parses the text on the host,
and returns CPU-resident doubles. Providers do not need custom kernels for this builtin.

## Examples of using the `str2double` function in MATLAB / RunMat

### Convert a string scalar into a double
```matlab
value = str2double("3.14159");
```
Expected output:
```matlab
value = 3.14159
```

### Convert every element of a string array
```matlab
temps = ["12.5" "19.8" "not-a-number"];
data = str2double(temps);
```
Expected output:
```matlab
data = 1×3
   12.5000   19.8000       NaN
```

### Parse scientific notation text
```matlab
result = str2double("6.022e23");
```
Expected output:
```matlab
result = 6.0220e+23
```

### Handle engineering exponents written with `D`
```matlab
cap = str2double("4.7D-9");
```
Expected output:
```matlab
cap = 4.7000e-09
```

### Convert a character array one row at a time
```matlab
chars = ['42   '; '  100'];
numbers = str2double(chars);
```
Expected output:
```matlab
numbers = 2×1
    42
   100
```

### Work with cell arrays of character vectors
```matlab
C = {'3.14', 'NaN', '-Inf'};
values = str2double(C);
```
Expected output:
```matlab
values = 1×3
    3.1400      NaN      -Inf
```

### Detect invalid numeric text
```matlab
status = str2double("error42");
```
Expected output:
```matlab
status = NaN
```

### Recognise special values `Inf` and `NaN`
```matlab
special = str2double(["Inf"; "-Infinity"; "NaN"]);
```
Expected output:
```matlab
special = 3×1
     Inf
    -Inf
     NaN
```

## FAQ

### What input types does `str2double` accept?
String scalars, string arrays, character vectors, character arrays, and cell arrays of character vectors or
string scalars are supported. Other types raise an error so that mismatched inputs are caught early.

### How are invalid or empty strings handled?
Invalid text—including empty strings, whitespace-only rows, or strings with extra characters—converts to `NaN`.
This matches MATLAB, which uses `NaN` as a sentinel for failed conversions.

### Does `str2double` evaluate arithmetic expressions?
No. Unlike `str2num`, `str2double` never calls the evaluator. Text such as `"1+2"` or `"sqrt(2)"` yields `NaN`
instead of executing the expression, keeping the builtin safe for untrusted input.

### Can `str2double` parse complex numbers?
No. Complex text like `"3+4i"` returns `NaN`. Use `str2num` when you need MATLAB to interpret complex literals.

### Are engineering exponents with `D` supported?
Yes. Exponents that use `d` or `D` are rewritten to `e` automatically, so `"1.0D3"` converts to `1000`.

### How does `str2double` treat missing strings?
Missing strings produced with `string(missing)` display as `<missing>` and convert to `NaN`. You can detect them
with `ismissing` before conversion if you need special handling.

### Does locale affect parsing?
`str2double` honours digits, decimal points, and exponent letters only. Locale-specific grouping separators such as
commas are not accepted, mirroring MATLAB's behaviour.

### Will the result stay on the GPU when I pass gpuArray inputs?
No. The builtin gathers GPU-backed inputs to the host, parses them, and keeps the numeric result in host memory.
Wrap the result with `gpuArray(...)` if you need to move it back to the device.

## See Also
`str2num`, `double`, `string`, `str2int`

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/core/str2double.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/str2double.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::str2double")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "str2double",
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
    notes: "Parses text on the CPU; GPU-resident inputs are gathered before conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::str2double")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "str2double",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Conversion builtin; not eligible for fusion and materialises host-side doubles.",
};

const ARG_TYPE_ERROR: &str =
    "str2double: input must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "str2double: cell array elements must be character vectors or string scalars";

#[runtime_builtin(
    name = "str2double",
    category = "strings/core",
    summary = "Convert strings, character arrays, or cell arrays of text into doubles.",
    keywords = "str2double,string to double,text conversion,gpu",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::str2double"
)]
fn str2double_builtin(value: Value) -> Result<Value, String> {
    let gathered = gather_if_needed(&value).map_err(|e| format!("str2double: {e}"))?;
    match gathered {
        Value::String(text) => Ok(Value::Num(parse_numeric_scalar(&text))),
        Value::StringArray(array) => str2double_string_array(array),
        Value::CharArray(array) => str2double_char_array(array),
        Value::Cell(cell) => str2double_cell_array(cell),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

fn str2double_string_array(array: StringArray) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let mut values = Vec::with_capacity(data.len());
    for text in &data {
        values.push(parse_numeric_scalar(text));
    }
    let tensor = Tensor::new(values, shape).map_err(|e| format!("str2double: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn str2double_char_array(array: CharArray) -> Result<Value, String> {
    let rows = array.rows;
    let cols = array.cols;
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_text: String = array.data[start..end].iter().collect();
        values.push(parse_numeric_scalar(&row_text));
    }
    let tensor = Tensor::new(values, vec![rows, 1]).map_err(|e| format!("str2double: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn str2double_cell_array(cell: CellArray) -> Result<Value, String> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut values = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let idx = row * cols + col;
            let element: &Value = &data[idx];
            let numeric = match element {
                Value::String(text) => parse_numeric_scalar(text),
                Value::StringArray(sa) if sa.data.len() == 1 => parse_numeric_scalar(&sa.data[0]),
                Value::CharArray(char_vec) if char_vec.rows == 1 => {
                    let row_text: String = char_vec.data.iter().collect();
                    parse_numeric_scalar(&row_text)
                }
                Value::CharArray(_) => return Err(CELL_ELEMENT_ERROR.to_string()),
                _ => return Err(CELL_ELEMENT_ERROR.to_string()),
            };
            values.push(numeric);
        }
    }
    let tensor = Tensor::new(values, vec![rows, cols]).map_err(|e| format!("str2double: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn parse_numeric_scalar(text: &str) -> f64 {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return f64::NAN;
    }

    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "nan" => return f64::NAN,
        "inf" | "+inf" | "infinity" | "+infinity" => return f64::INFINITY,
        "-inf" | "-infinity" => return f64::NEG_INFINITY,
        _ => {}
    }

    let normalized: Cow<'_, str> = if trimmed.chars().any(|c| c == 'd' || c == 'D') {
        Cow::Owned(
            trimmed
                .chars()
                .map(|c| if c == 'd' || c == 'D' { 'e' } else { c })
                .collect(),
        )
    } else {
        Cow::Borrowed(trimmed)
    };

    normalized.parse::<f64>().unwrap_or(f64::NAN)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn str2double_string_scalar() {
        let result = str2double_builtin(Value::String("42.5".into())).expect("str2double");
        assert_eq!(result, Value::Num(42.5));
    }

    #[test]
    fn str2double_string_scalar_invalid_returns_nan() {
        let result = str2double_builtin(Value::String("abc".into())).expect("str2double");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn str2double_string_array_preserves_shape() {
        let array =
            StringArray::new(vec!["1".into(), " 2.5 ".into(), "foo".into()], vec![3, 1]).unwrap();
        let result = str2double_builtin(Value::StringArray(array)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 1]);
                assert_eq!(tensor.data[0], 1.0);
                assert_eq!(tensor.data[1], 2.5);
                assert!(tensor.data[2].is_nan());
            }
            Value::Num(_) => panic!("expected tensor"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn str2double_char_array_multiple_rows() {
        let data: Vec<char> = vec!['4', '2', ' ', ' ', '1', '0', '0', ' '];
        let array = CharArray::new(data, 2, 4).unwrap();
        let result = str2double_builtin(Value::CharArray(array)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data[0], 42.0);
                assert_eq!(tensor.data[1], 100.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn str2double_char_array_empty_rows() {
        let array = CharArray::new(Vec::new(), 0, 0).unwrap();
        let result = str2double_builtin(Value::CharArray(array)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![0, 1]);
                assert_eq!(tensor.data.len(), 0);
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    #[allow(
        clippy::approx_constant,
        reason = "Test ensures literal 3.14 text stays 3.14, not π"
    )]
    fn str2double_cell_array_of_text() {
        let cell = CellArray::new(
            vec![
                Value::String("3.14".into()),
                Value::CharArray(CharArray::new_row("NaN")),
                Value::String("-Inf".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = str2double_builtin(Value::Cell(cell)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data[0], 3.14);
                assert!(tensor.data[1].is_nan());
                assert_eq!(tensor.data[2], f64::NEG_INFINITY);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn str2double_cell_array_invalid_element_errors() {
        let cell = CellArray::new(vec![Value::Num(5.0)], 1, 1).unwrap();
        let err = str2double_builtin(Value::Cell(cell)).unwrap_err();
        assert!(
            err.contains("str2double"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn str2double_supports_d_exponent() {
        let result = str2double_builtin(Value::String("1.5D3".into())).expect("str2double");
        match result {
            Value::Num(v) => assert_eq!(v, 1500.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn str2double_recognises_infinity_forms() {
        let array = StringArray::new(
            vec!["Inf".into(), "-Infinity".into(), "+inf".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = str2double_builtin(Value::StringArray(array)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.data[0], f64::INFINITY);
                assert_eq!(tensor.data[1], f64::NEG_INFINITY);
                assert_eq!(tensor.data[2], f64::INFINITY);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
