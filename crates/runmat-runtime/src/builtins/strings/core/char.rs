//! MATLAB-compatible `char` builtin with GPU-aware conversion semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg(all(test, feature = "wgpu"))]
use crate::accel_provider;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "char",
        builtin_path = "crate::builtins::strings::core::char"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "char"
category: "strings/core"
keywords: ["char", "character array", "string conversion", "padding", "gpu"]
summary: "Convert text, numeric codes, and cell contents into MATLAB-style character arrays."
references:
  - https://www.mathworks.com/help/matlab/ref/char.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Conversion always runs on the CPU; GPU tensors are gathered to host memory before building the character array."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::char::tests"
  integration: "builtins::strings::core::char::tests::char_gpu_tensor_round_trip"
---

# What does the `char` function do in MATLAB / RunMat?
`char` converts its inputs into a character array. Numeric inputs are interpreted as Unicode code
points, text inputs become rows of characters, and cell elements or string scalars are padded with
spaces when necessary so every row in the result has the same width.

## How does the `char` function behave in MATLAB / RunMat?
- `char(x)` with no arguments returns a `0×0` character array.
- Numeric arrays must be real integers. The output character array has the same shape (up to two
  dimensions) as the numeric input.
- String scalars and character vectors become individual rows. Rows are padded on the right with
  spaces to match the longest row.
- String arrays with one or two dimensions contribute one row per element using MATLAB's
  column-major ordering.
- Cell arrays must contain character vectors or string scalars. Each element becomes exactly one
  row in the result.
- Inputs may be mixed and are vertically concatenated in the order they appear.
- Complex inputs are unsupported and raise MATLAB-compatible errors.

## `char` Function GPU Execution Behaviour
`char` gathers GPU tensors back to host memory using the active RunMat Accelerate provider before
performing any conversion. The resulting character array always resides in host memory; providers
do not need to supply specialised kernels.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` manually for `char`. The runtime recognises that this
builtin materialises text on the host, gathers GPU tensors automatically, and keeps the character
array in CPU memory. Wrap the result in `gpuArray(char(...))` only when you explicitly want the
characters back on the device for subsequent GPU pipelines.

## Examples of using the `char` function in MATLAB / RunMat

### Converting a string scalar to a character row
```matlab
name = char("RunMat");
```
Expected output:
```matlab
name =
    'RunMat'
```

### Building a character matrix from multiple rows
```matlab
rows = char("alpha", "beta");
```
Expected output:
```matlab
rows =
    'alpha'
    'beta '
```

### Transforming numeric codes to characters
```matlab
codes = [77 65 84 76 65 66];
letters = char(codes);
```
Expected output:
```matlab
letters =
    'MATLAB'
```

### Padding a string array into a character matrix
```matlab
animals = ["cat"; "giraffe"];
C = char(animals);
```
Expected output:
```matlab
C =
    'cat   '
    'giraffe'
```

### Creating rows from a cell array of character vectors
```matlab
dirs = {'north', 'east', 'west'};
chart = char(dirs);
```
Expected output:
```matlab
chart =
    'north'
    'east '
    'west '
```

### Converting GPU-resident codes back to text
```matlab
G = gpuArray([82 85 78 77 65 84]);
label = char(G);
```
Expected output:
```matlab
label =
    'RUNMAT'
```
RunMat downloads the numeric data from the GPU before constructing the character array.

## FAQ

### Does `char` accept numeric arrays with more than two dimensions?
No. Numeric inputs must be scalars, vectors, or two-dimensional matrices. Higher-dimensional arrays
raise an error so MATLAB's behaviour is preserved.

### How are rows padded when lengths differ?
Each row is right-padded with space characters so every row in the result has the same width as the
longest row that was produced.

### Can I convert cell arrays that contain empty text?
Yes. Empty strings or character vectors become rows with zero columns; they still participate in
padding when combined with longer rows.

### What happens if a numeric value is not an integer?
The builtin rejects non-integer numeric values. Use `round`, `floor`, or `uint32` beforehand if you
need to convert floating-point values into valid code points.

### Are code points above the Basic Multilingual Plane supported?
Yes. Any integer that represents a valid Unicode scalar value (`0..0x10FFFF`, excluding surrogates)
is accepted and converted to the corresponding character.

### Can `char` convert complex numbers?
No. Complex values are not supported because MATLAB also rejects them. Convert the data to real
values before calling `char`.

### Does `char` keep characters on the GPU?
No. After conversion the result is a CPU-resident character array. Use `gpuArray(char(...))` if you
need to move the result back to the device.

## See Also
[string](./string), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `char` function is available at: [`crates/runmat-runtime/src/builtins/strings/core/char.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/char.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::char")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "char",
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
    notes:
        "Conversion always runs on the CPU; GPU tensors are gathered before building the result.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::char")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "char",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Character materialisation runs outside of fusion; results always live on the host.",
};

#[runtime_builtin(
    name = "char",
    category = "strings/core",
    summary = "Convert numeric codes, strings, and cell contents into a character array.",
    keywords = "char,character,string,gpu",
    accel = "conversion",
    builtin_path = "crate::builtins::strings::core::char"
)]
fn char_builtin(rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        let empty = CharArray::new(Vec::new(), 0, 0).map_err(|e| format!("char: {e}"))?;
        return Ok(Value::CharArray(empty));
    }

    let mut rows: Vec<Vec<char>> = Vec::new();
    let mut max_width = 0usize;

    for arg in rest {
        let gathered = gather_if_needed(&arg)?;
        let mut produced = value_to_char_rows(&gathered)?;
        for row in &produced {
            if row.len() > max_width {
                max_width = row.len();
            }
        }
        rows.append(&mut produced);
    }

    if rows.is_empty() {
        let empty = CharArray::new(Vec::new(), 0, 0).map_err(|e| format!("char: {e}"))?;
        return Ok(Value::CharArray(empty));
    }

    let cols = max_width;
    let total_rows = rows.len();
    let mut data = vec![' '; total_rows * cols];
    for (row_idx, row) in rows.into_iter().enumerate() {
        for (col_idx, ch) in row.into_iter().enumerate() {
            if col_idx < cols {
                data[row_idx * cols + col_idx] = ch;
            }
        }
    }

    let array = CharArray::new(data, total_rows, cols).map_err(|e| format!("char: {e}"))?;
    Ok(Value::CharArray(array))
}

fn value_to_char_rows(value: &Value) -> Result<Vec<Vec<char>>, String> {
    match value {
        Value::CharArray(ca) => Ok(char_array_rows(ca)),
        Value::String(s) => Ok(vec![s.chars().collect()]),
        Value::StringArray(sa) => string_array_rows(sa),
        Value::Num(n) => Ok(vec![vec![number_to_char(*n)?]]),
        Value::Int(i) => {
            let as_double = i.to_f64();
            Ok(vec![vec![number_to_char(as_double)?]])
        }
        Value::Bool(b) => {
            let code = if *b { 1.0 } else { 0.0 };
            Ok(vec![vec![number_to_char(code)?]])
        }
        Value::Tensor(t) => tensor_rows(t),
        Value::LogicalArray(la) => logical_rows(la),
        Value::Cell(ca) => cell_rows(ca),
        Value::GpuTensor(_) => Err("char: expected host data after gather".to_string()),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("char: complex inputs are not supported".to_string())
        }
        Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(format!("char: unsupported input type {:?}", value)),
    }
}

fn char_array_rows(ca: &CharArray) -> Vec<Vec<char>> {
    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row = Vec::with_capacity(ca.cols);
        for c in 0..ca.cols {
            row.push(ca.data[r * ca.cols + c]);
        }
        rows.push(row);
    }
    rows
}

fn string_array_rows(sa: &StringArray) -> Result<Vec<Vec<char>>, String> {
    ensure_two_dimensional(&sa.shape, "char")?;
    if sa.data.is_empty() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::with_capacity(sa.data.len());
    let rows_count = sa.rows();
    let cols_count = sa.cols();
    if rows_count == 0 || cols_count == 0 {
        return Ok(Vec::new());
    }
    for c in 0..cols_count {
        for r in 0..rows_count {
            let idx = r + c * rows_count;
            rows.push(sa.data[idx].chars().collect());
        }
    }
    Ok(rows)
}

fn tensor_rows(t: &Tensor) -> Result<Vec<Vec<char>>, String> {
    ensure_two_dimensional(&t.shape, "char")?;
    let (rows, cols) = infer_rows_cols(&t.shape, t.data.len());
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            if cols == 0 {
                continue;
            }
            let idx = r + c * rows;
            let value = t.data[idx];
            row.push(number_to_char(value)?);
        }
        out.push(row);
    }
    Ok(out)
}

fn logical_rows(la: &LogicalArray) -> Result<Vec<Vec<char>>, String> {
    ensure_two_dimensional(&la.shape, "char")?;
    let (rows, cols) = infer_rows_cols(&la.shape, la.data.len());
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            if cols == 0 {
                continue;
            }
            let idx = r + c * rows;
            let code = if la.data[idx] != 0 { 1.0 } else { 0.0 };
            row.push(number_to_char(code)?);
        }
        out.push(row);
    }
    Ok(out)
}

fn cell_rows(ca: &CellArray) -> Result<Vec<Vec<char>>, String> {
    let mut rows = Vec::with_capacity(ca.data.len());
    for ptr in &ca.data {
        let element = (**ptr).clone();
        let mut converted = value_to_char_rows(&element)?;
        match converted.len() {
            0 => rows.push(Vec::new()),
            1 => rows.push(converted.remove(0)),
            _ => {
                return Err(
                    "char: cell elements must be character vectors or string scalars".to_string(),
                )
            }
        }
    }
    Ok(rows)
}

fn number_to_char(value: f64) -> Result<char, String> {
    if !value.is_finite() {
        return Err("char: numeric inputs must be finite".to_string());
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        return Err(format!(
            "char: numeric inputs must be integers in the Unicode range (got {value})"
        ));
    }
    if rounded < 0.0 {
        return Err(format!(
            "char: negative code points are invalid (got {rounded})"
        ));
    }
    if rounded > 0x10FFFF as f64 {
        return Err(format!(
            "char: code point {} exceeds Unicode range",
            rounded as u64
        ));
    }
    let code = rounded as u32;
    char::from_u32(code).ok_or_else(|| format!("char: invalid code point {code}"))
}

fn ensure_two_dimensional(shape: &[usize], context: &str) -> Result<(), String> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape.iter().skip(2).all(|&d| d == 1) {
        return Ok(());
    }
    Err(format!("{context}: inputs must be 2-D"))
}

fn infer_rows_cols(shape: &[usize], len: usize) -> (usize, usize) {
    match shape.len() {
        0 => {
            if len == 0 {
                (0, 0)
            } else {
                (1, 1)
            }
        }
        1 => (1, shape[0]),
        2 => (shape[0], shape[1]),
        _ => {
            let rows = shape[0];
            let cols = if shape.len() > 1 { shape[1] } else { 1 };
            (rows, cols)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::StringArray;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_no_arguments_returns_empty() {
        let result = char_builtin(Vec::new()).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 0);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_string_scalar() {
        let value = Value::String("RunMat".to_string());
        let result = char_builtin(vec![value]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                assert_eq!(ca.data, "RunMat".chars().collect::<Vec<_>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_numeric_tensor() {
        let tensor =
            Tensor::new(vec![82.0, 85.0, 78.0, 77.0, 65.0, 84.0], vec![1, 6]).expect("tensor");
        let result = char_builtin(vec![Value::Tensor(tensor)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                assert_eq!(ca.data, "RUNMAT".chars().collect::<Vec<_>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_string_array_with_padding() {
        let data = vec!["cat".to_string(), "giraffe".to_string()];
        let sa = StringArray::new(data, vec![2, 1]).expect("string array");
        let result = char_builtin(vec![Value::StringArray(sa)]).expect("char from string array");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 7);
                assert_eq!(
                    ca.data,
                    vec!['c', 'a', 't', ' ', ' ', ' ', ' ', 'g', 'i', 'r', 'a', 'f', 'f', 'e']
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_from_cell_array_of_strings() {
        let cell = CellArray::new(
            vec![
                Value::from("north"),
                Value::from("east"),
                Value::from("west"),
            ],
            3,
            1,
        )
        .expect("cell array");
        let result = char_builtin(vec![Value::Cell(cell)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 3);
                assert_eq!(ca.cols, 5);
                assert_eq!(
                    ca.data,
                    vec!['n', 'o', 'r', 't', 'h', 'e', 'a', 's', 't', ' ', 'w', 'e', 's', 't', ' ']
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_numeric_and_text_arguments_concatenate() {
        let text = Value::String("hi".to_string());
        let codes = Tensor::new(vec![65.0, 66.0], vec![1, 2]).expect("tensor");
        let result = char_builtin(vec![text, Value::Tensor(codes)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 2);
                assert_eq!(ca.data, vec!['h', 'i', 'A', 'B']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_gpu_tensor_round_trip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![82.0, 85.0, 78.0], vec![1, 3]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = char_builtin(vec![Value::GpuTensor(handle)]).expect("char");
            match result {
                Value::CharArray(ca) => {
                    assert_eq!(ca.rows, 1);
                    assert_eq!(ca.cols, 3);
                    assert_eq!(ca.data, vec!['R', 'U', 'N']);
                }
                other => panic!("expected char array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_non_integer_numeric() {
        let err = char_builtin(vec![Value::Num(65.5)]).expect_err("non-integer numeric");
        assert!(err.contains("integers"), "unexpected error message: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_high_dimension_tensor() {
        let tensor =
            Tensor::new(vec![65.0, 66.0], vec![1, 1, 2]).expect("tensor construction failed");
        let err = char_builtin(vec![Value::Tensor(tensor)]).expect_err("should reject >2D tensor");
        assert!(err.contains("2-D"), "expected dimension error, got {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_string_array_column_major_order() {
        let data = vec![
            "c0r0".to_string(),
            "c0r1".to_string(),
            "c1r0".to_string(),
            "c1r1".to_string(),
        ];
        let sa = StringArray::new(data, vec![2, 2]).expect("string array");
        let result = char_builtin(vec![Value::StringArray(sa)]).expect("char");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 4);
                assert_eq!(ca.cols, 4);
                assert_eq!(ca.data, "c0r0c0r1c1r0c1r1".chars().collect::<Vec<char>>());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_high_dimension_string_array() {
        let sa = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 1, 2])
            .expect("string array");
        let err =
            char_builtin(vec![Value::StringArray(sa)]).expect_err("should reject >2D string array");
        assert!(err.contains("2-D"), "expected dimension error, got {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_rejects_complex_input() {
        let err = char_builtin(vec![Value::Complex(1.0, 2.0)]).expect_err("complex input");
        assert!(
            err.contains("complex"),
            "expected complex error message, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn char_wgpu_numeric_codes_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let tensor = Tensor::new(vec![82.0, 85.0, 78.0], vec![1, 3]).unwrap();
        let cpu = char_builtin(vec![Value::Tensor(tensor.clone())]).expect("char cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = accel_provider::maybe_provider(__runmat_accel_context_char_builtin)
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu = char_builtin(vec![Value::GpuTensor(handle)]).expect("char gpu");

        match (cpu, gpu) {
            (Value::CharArray(expected), Value::CharArray(actual)) => {
                assert_eq!(actual, expected);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }
}
