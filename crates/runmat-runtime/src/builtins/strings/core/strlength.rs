//! MATLAB-compatible `strlength` builtin for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::common::is_missing_string;
use crate::{build_runtime_error, gather_if_needed, BuiltinResult, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "strlength",
        builtin_path = "crate::builtins::strings::core::strlength"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "strlength"
category: "strings/core"
keywords: ["strlength", "string length", "character count", "text analytics", "cell array"]
summary: "Return the number of characters in each element of a string array, character array, or cell array of character vectors."
references:
  - https://www.mathworks.com/help/matlab/ref/strlength.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Executes on the CPU; if any argument lives on the GPU, the runtime gathers it before computing lengths to keep semantics identical to MATLAB."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::strlength::tests"
  integration: "builtins::strings::core::strlength::tests::strlength_cell_array_of_char_vectors"
---

# What does the `strlength` function do in MATLAB / RunMat?
`strlength(str)` counts how many characters appear in each element of text inputs. It works with string
arrays, character vectors, character arrays, and cell arrays of character vectors, returning a `double`
array that mirrors the input shape.

## How does the `strlength` function behave in MATLAB / RunMat?
- String arrays return a numeric array of the same size; string scalars yield a scalar `double`.
- Character arrays report the number of characters per row and ignore padding that MATLAB inserts to keep rows the same width.
- Character vectors stored in cells contribute one scalar per cell element; the output array matches the cell array shape.
- Missing string scalars (for example values created with `string(missing)`) yield `NaN`. RunMat displays these entries as `<missing>` in the console just like MATLAB.
- Empty text inputs produce zeros-sized numeric outputs that match MATLAB's dimension rules.

## `strlength` Function GPU Execution Behaviour
`strlength` is a metadata query and always executes on the CPU. If a text container references data that
originated on the GPU (for example, a cell array that still wraps GPU-resident numeric intermediates), RunMat
gathers that data before measuring lengths. Providers do not require custom kernels for this builtin.

## Examples of using the `strlength` function in MATLAB / RunMat

### Measure Characters In A String Scalar
```matlab
len = strlength("RunMat");
```
Expected output:
```matlab
len = 6
```

### Count Characters Across A String Array
```matlab
labels = ["North" "South" "East" "West"];
counts = strlength(labels);
```
Expected output:
```matlab
counts = 1×4
    5    5    4    4
```

### Compute Lengths For Each Row Of A Character Array
```matlab
names = char("cat", "giraffe");
row_counts = strlength(names);
```
Expected output:
```matlab
row_counts = 2×1
     3
     7
```

### Handle Empty And Blank Strings
```matlab
mixed = ["", "   "];
len = strlength(mixed);
```
Expected output:
```matlab
len = 1×2
     0     3
```

### Get Lengths From A Cell Array Of Character Vectors
```matlab
C = {'red', 'green', 'blue'};
L = strlength(C);
```
Expected output:
```matlab
L = 1×3
     3     5     4
```

### Treat Missing Strings As NaN
```matlab
values = string(["alpha" "beta" "gamma"]);
values(2) = string(missing);  % Displays as <missing> when printed
lengths = strlength(values);
```
Expected output:
```matlab
lengths = 1×3
    5   NaN    5
```

## FAQ

### What numeric type does `strlength` return?
`strlength` always returns doubles, even when all lengths are whole numbers. MATLAB uses doubles for most numeric results, and RunMat follows the same rule.

### Why are padded spaces in character arrays ignored?
When MATLAB builds a character array from rows of different lengths, it pads shorter rows with spaces. Those padding characters are not part of the logical content, so `strlength` removes them before counting. Explicit trailing spaces that you type in a single character vector remain part of the count.

### How are missing string values handled?
Missing string scalars display as `<missing>` and produce `NaN` lengths. Use `ismissing` or `fillmissing` if you need to substitute a default length.

### Can I call `strlength` with numeric data?
No. `strlength` only accepts string arrays, character vectors/arrays, or cell arrays of character vectors. Numeric inputs raise an error—use `num2str` first if you need to convert numbers to text.

### Does `strlength` support multibyte Unicode characters?
Yes. Each Unicode scalar value counts as one character, so emoji or accented letters contribute a length of one. Surrogate pairs are treated as a single character, matching MATLAB's behaviour.

### Will `strlength` ever execute on the GPU?
No. The builtin inspects metadata and operates on host strings. If your data already lives on the GPU, RunMat gathers it automatically before computing lengths so results match MATLAB exactly.

## See Also
`string`, `char`, `strtrim`, `length`, `size`

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/core/strlength.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strlength.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::strlength")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strlength",
    op_kind: GpuOpKind::Custom("string-metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Measures string lengths on the CPU; any GPU-resident inputs are gathered before evaluation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::strlength")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strlength",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Metadata-only builtin; not eligible for fusion and never emits GPU kernels.",
};

const ARG_TYPE_ERROR: &str =
    "strlength: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "strlength: cell array elements must be character vectors or string scalars";

fn strlength_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("strlength")
        .build()
}

fn remap_strlength_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, "strlength")
}

#[runtime_builtin(
    name = "strlength",
    category = "strings/core",
    summary = "Count characters in string arrays, character arrays, or cell arrays of character vectors.",
    keywords = "strlength,string length,text,count,characters",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::strlength"
)]
fn strlength_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let gathered = gather_if_needed(&value).map_err(remap_strlength_flow)?;
    match gathered {
        Value::StringArray(array) => strlength_string_array(array),
        Value::String(text) => Ok(Value::Num(string_scalar_length(&text))),
        Value::CharArray(array) => strlength_char_array(array),
        Value::Cell(cell) => strlength_cell_array(cell),
        _ => Err(strlength_flow(ARG_TYPE_ERROR)),
    }
}

fn strlength_string_array(array: StringArray) -> BuiltinResult<Value> {
    let StringArray { data, shape, .. } = array;
    let mut lengths = Vec::with_capacity(data.len());
    for text in &data {
        lengths.push(string_scalar_length(text));
    }
    let tensor =
        Tensor::new(lengths, shape).map_err(|e| strlength_flow(format!("strlength: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn strlength_char_array(array: CharArray) -> BuiltinResult<Value> {
    let rows = array.rows;
    let mut lengths = Vec::with_capacity(rows);
    for row in 0..rows {
        let length = if array.rows <= 1 {
            array.cols
        } else {
            trimmed_row_length(&array, row)
        } as f64;
        lengths.push(length);
    }
    let tensor = Tensor::new(lengths, vec![rows, 1])
        .map_err(|e| strlength_flow(format!("strlength: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn strlength_cell_array(cell: CellArray) -> BuiltinResult<Value> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut lengths = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let idx = row * cols + col;
            let value: &Value = &data[idx];
            let length = match value {
                Value::String(text) => string_scalar_length(text),
                Value::StringArray(sa) if sa.data.len() == 1 => string_scalar_length(&sa.data[0]),
                Value::CharArray(char_vec) if char_vec.rows == 1 => char_vec.cols as f64,
                Value::CharArray(_) => return Err(strlength_flow(CELL_ELEMENT_ERROR)),
                _ => return Err(strlength_flow(CELL_ELEMENT_ERROR)),
            };
            lengths.push(length);
        }
    }
    let tensor = Tensor::new(lengths, vec![rows, cols])
        .map_err(|e| strlength_flow(format!("strlength: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn string_scalar_length(text: &str) -> f64 {
    if is_missing_string(text) {
        f64::NAN
    } else {
        text.chars().count() as f64
    }
}

fn trimmed_row_length(array: &CharArray, row: usize) -> usize {
    let cols = array.cols;
    let mut end = cols;
    while end > 0 {
        let ch = array.data[row * cols + end - 1];
        if ch == ' ' {
            end -= 1;
        } else {
            break;
        }
    }
    end
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_string_scalar() {
        let result = strlength_builtin(Value::String("RunMat".into())).expect("strlength");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_string_array_with_missing() {
        let array = StringArray::new(vec!["alpha".into(), "<missing>".into()], vec![2, 1]).unwrap();
        let result = strlength_builtin(Value::StringArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data.len(), 2);
                assert_eq!(tensor.data[0], 5.0);
                assert!(tensor.data[1].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_array_multiple_rows() {
        let data: Vec<char> = vec!['c', 'a', 't', ' ', ' ', 'h', 'o', 'r', 's', 'e'];
        let array = CharArray::new(data, 2, 5).unwrap();
        let result = strlength_builtin(Value::CharArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_vector_retains_explicit_spaces() {
        let data: Vec<char> = "hi   ".chars().collect();
        let array = CharArray::new(data, 1, 5).unwrap();
        let result = strlength_builtin(Value::CharArray(array)).expect("strlength");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_cell_array_of_char_vectors() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("red")),
                Value::CharArray(CharArray::new_row("green")),
            ],
            1,
            2,
        )
        .unwrap();
        let result = strlength_builtin(Value::Cell(cell)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_cell_array_with_string_scalars() {
        let cell = CellArray::new(
            vec![
                Value::String("alpha".into()),
                Value::String("beta".into()),
                Value::String("<missing>".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = strlength_builtin(Value::Cell(cell)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data.len(), 3);
                assert_eq!(tensor.data[0], 5.0);
                assert_eq!(tensor.data[1], 4.0);
                assert!(tensor.data[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_string_array_preserves_shape() {
        let array = StringArray::new(
            vec!["ab".into(), "c".into(), "def".into(), "".into()],
            vec![2, 2],
        )
        .unwrap();
        let result = strlength_builtin(Value::StringArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_eq!(tensor.data, vec![2.0, 1.0, 3.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_array_trims_padding() {
        let data: Vec<char> = vec!['d', 'o', 'g', ' ', ' ', 'h', 'o', 'r', 's', 'e'];
        let array = CharArray::new(data, 2, 5).unwrap();
        let result = strlength_builtin(Value::CharArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_errors_on_invalid_input() {
        let err = error_message(strlength_builtin(Value::Num(1.0)).unwrap_err());
        assert_eq!(err, ARG_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_rejects_cell_with_invalid_element() {
        let cell = CellArray::new(
            vec![Value::CharArray(CharArray::new_row("ok")), Value::Num(5.0)],
            1,
            2,
        )
        .unwrap();
        let err = error_message(strlength_builtin(Value::Cell(cell)).unwrap_err());
        assert_eq!(err, CELL_ELEMENT_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
