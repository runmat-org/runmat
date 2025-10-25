//! MATLAB-compatible `strtrim` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{gather_if_needed, make_cell, register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r###"---
title: "strtrim"
category: "strings/transform"
keywords: ["strtrim", "trim whitespace", "leading spaces", "trailing spaces", "character arrays", "string arrays"]
summary: "Remove leading and trailing whitespace from strings, character arrays, and cell arrays."
references:
  - https://www.mathworks.com/help/matlab/ref/strtrim.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Executes on the CPU; GPU-resident inputs are gathered automatically before trimming."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::strtrim::tests"
  integration: "builtins::strings::transform::strtrim::tests::strtrim_cell_array_mixed_content"
---

# What does the `strtrim` function do in MATLAB / RunMat?
`strtrim(text)` removes leading and trailing whitespace characters from `text`. The input can be a
string scalar, string array, character array, or a cell array of character vectors, mirroring MATLAB
behaviour. Internal whitespace is preserved exactly as provided.

## How does the `strtrim` function behave in MATLAB / RunMat?
- Whitespace is defined via MATLAB's `isspace`, so spaces, tabs, newlines, and other Unicode
  whitespace code points are removed from both ends of each element.
- String scalars and arrays keep their type and shape. Missing string scalars (`<missing>`) remain
  missing and are returned unchanged.
- Character arrays are trimmed row by row. The result keeps the original number of rows and shrinks
  the column count to the longest trimmed row, padding shorter rows with spaces so the output stays
  rectangular.
- Cell arrays must contain string scalars or character vectors. Results preserve the original cell
  layout with each element trimmed.
- Numeric, logical, or structured inputs raise MATLAB-compatible type errors.

## `strtrim` Function GPU Execution Behaviour
`strtrim` runs on the CPU. When the input (or any nested element) resides on the GPU, RunMat gathers
it to host memory before trimming so the output matches MATLAB exactly. Providers do not need to
implement device kernels for this builtin today.

## GPU residency in RunMat (Do I need `gpuArray`?)
You do not need to call `gpuArray` or `gather` manually. RunMat automatically gathers any GPU-resident
text data before applying `strtrim`, so the builtin behaves the same regardless of where the data lives.

## Examples of using the `strtrim` function in MATLAB / RunMat

### Trim Leading And Trailing Spaces From A String Scalar
```matlab
name = "   RunMat   ";
clean = strtrim(name);
```
Expected output:
```matlab
clean = "RunMat"
```

### Remove Extra Whitespace From Each Element Of A String Array
```matlab
labels = ["  Alpha  "; "Beta   "; "   Gamma"];
trimmed = strtrim(labels);
```
Expected output:
```matlab
trimmed = 3×1 string
    "Alpha"
    "Beta"
    "Gamma"
```

### Trim Character Array Rows While Preserving Shape
```matlab
animals = char('  cat   ', 'dog', ' cow ');
result = strtrim(animals);
```
Expected output:
```matlab
result =

  3×3 char array

    'cat'
    'dog'
    'cow'
```

### Trim Tabs And Newlines Alongside Spaces
```matlab
text = "\tMetrics " + newline;
clean = strtrim(text);
```
Expected output:
```matlab
clean = "Metrics"
```

### Trim Each Element Of A Cell Array Of Character Vectors
```matlab
pieces = {'  GPU  ', " Accelerate", 'RunMat   '};
out = strtrim(pieces);
```
Expected output:
```matlab
out = 1×3 cell array
    {'GPU'}    {"Accelerate"}    {'RunMat'}
```

### Preserve Missing String Scalars
```matlab
vals = [" ok "; "<missing>"; " trimmed "];
trimmed = strtrim(vals);
```
Expected output:
```matlab
trimmed = 1×3 string
    "ok"
    <missing>
    "trimmed"
```

## FAQ

### Does `strtrim` modify internal whitespace?
No. Only leading and trailing whitespace is removed; interior spacing remains intact.

### Which characters count as whitespace?
`strtrim` removes code points that MATLAB's `isspace` recognises, including spaces, tabs, newlines,
carriage returns, and many Unicode space separators.

### How are character arrays resized?
Each row is trimmed independently. The output keeps the same number of rows and shrinks the width to
match the longest trimmed row, padding shorter rows with spaces if necessary.

### What happens to missing strings?
Missing string scalars (`string(missing)`) remain `<missing>` exactly as in MATLAB.

### Can I pass numeric or logical arrays to `strtrim`?
No. Passing non-text inputs raises a MATLAB-compatible error indicating that text input is required.

### How does `strtrim` differ from `strip`?
`strtrim` always removes leading and trailing whitespace. `strip` is newer and adds options for custom
characters and directional trimming; use it when you need finer control.

## See Also
[strip](./strip), [upper](./upper), [lower](./lower)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/strtrim.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strtrim.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"###;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strtrim",
    op_kind: GpuOpKind::Custom("string-transform"),
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
        "Executes on the CPU; GPU-resident inputs are gathered to host memory before trimming whitespace.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strtrim",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion and always gathers GPU inputs.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("strtrim", DOC_MD);

const ARG_TYPE_ERROR: &str =
    "strtrim: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "strtrim: cell array elements must be string scalars or character vectors";

#[runtime_builtin(
    name = "strtrim",
    category = "strings/transform",
    summary = "Remove leading and trailing whitespace from strings, character arrays, and cell arrays.",
    keywords = "strtrim,trim,whitespace,strings,character array,text",
    accel = "sink"
)]
fn strtrim_builtin(value: Value) -> Result<Value, String> {
    let gathered = gather_if_needed(&value).map_err(|e| format!("strtrim: {e}"))?;
    match gathered {
        Value::String(text) => Ok(Value::String(trim_string(text))),
        Value::StringArray(array) => strtrim_string_array(array),
        Value::CharArray(array) => strtrim_char_array(array),
        Value::Cell(cell) => strtrim_cell_array(cell),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

fn trim_string(text: String) -> String {
    if is_missing_string(&text) {
        text
    } else {
        trim_whitespace(&text)
    }
}

fn strtrim_string_array(array: StringArray) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let trimmed = data.into_iter().map(trim_string).collect::<Vec<_>>();
    let out = StringArray::new(trimmed, shape).map_err(|e| format!("strtrim: {e}"))?;
    Ok(Value::StringArray(out))
}

fn strtrim_char_array(array: CharArray) -> Result<Value, String> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut trimmed_rows: Vec<Vec<char>> = Vec::with_capacity(rows);
    let mut target_cols: usize = 0;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row);
        let trimmed = trim_whitespace(&text);
        let chars: Vec<char> = trimmed.chars().collect();
        target_cols = target_cols.max(chars.len());
        trimmed_rows.push(chars);
    }

    let mut new_data: Vec<char> = Vec::with_capacity(rows * target_cols);
    for mut chars in trimmed_rows {
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        new_data.extend(chars);
    }

    CharArray::new(new_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| format!("strtrim: {e}"))
}

fn strtrim_cell_array(cell: CellArray) -> Result<Value, String> {
    let CellArray { data, rows, cols } = cell;
    let mut trimmed_values = Vec::with_capacity(rows * cols);
    for idx in 0..data.len() {
        let trimmed = strtrim_cell_element(&data[idx])?;
        trimmed_values.push(trimmed);
    }
    make_cell(trimmed_values, rows, cols).map_err(|e| format!("strtrim: {e}"))
}

fn strtrim_cell_element(value: &Value) -> Result<Value, String> {
    match gather_if_needed(value).map_err(|e| format!("strtrim: {e}"))? {
        Value::String(text) => Ok(Value::String(trim_string(text))),
        Value::StringArray(sa) if sa.data.len() == 1 => {
            let text = sa.data.into_iter().next().unwrap();
            Ok(Value::String(trim_string(text)))
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                return Ok(Value::CharArray(ca));
            }
            let source = char_row_to_string_slice(&ca.data, ca.cols, 0);
            let trimmed = trim_whitespace(&source);
            let chars: Vec<char> = trimmed.chars().collect();
            let cols = chars.len();
            CharArray::new(chars, ca.rows, cols)
                .map(Value::CharArray)
                .map_err(|e| format!("strtrim: {e}"))
        }
        Value::CharArray(_) => Err(CELL_ELEMENT_ERROR.to_string()),
        _ => Err(CELL_ELEMENT_ERROR.to_string()),
    }
}

fn trim_whitespace(text: &str) -> String {
    let trimmed = text.trim_matches(|c: char| c.is_whitespace());
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "doc_export")]
    use crate::builtins::common::test_support;

    #[test]
    fn strtrim_string_scalar_trims_whitespace() {
        let result =
            strtrim_builtin(Value::String("  RunMat  ".into())).expect("strtrim string scalar");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[test]
    fn strtrim_string_array_preserves_shape() {
        let array = StringArray::new(
            vec![
                " one ".into(),
                "<missing>".into(),
                "two".into(),
                " three ".into(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let result = strtrim_builtin(Value::StringArray(array)).expect("strtrim string array");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("one"),
                        String::from("<missing>"),
                        String::from("two"),
                        String::from("three")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strtrim_char_array_multiple_rows() {
        let data: Vec<char> = "  cat  ".chars().chain(" dog   ".chars()).collect();
        let array = CharArray::new(data, 2, 7).unwrap();
        let result = strtrim_builtin(Value::CharArray(array)).expect("strtrim char array");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 3);
                assert_eq!(ca.data, vec!['c', 'a', 't', 'd', 'o', 'g']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn strtrim_char_array_all_whitespace_yields_zero_width() {
        let array = CharArray::new("   ".chars().collect(), 1, 3).unwrap();
        let result = strtrim_builtin(Value::CharArray(array)).expect("strtrim char whitespace");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }
    }

    #[test]
    fn strtrim_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("  GPU  ")),
                Value::String(" Accelerate ".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = strtrim_builtin(Value::Cell(cell)).expect("strtrim cell array");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("GPU")));
                assert_eq!(second, Value::String("Accelerate".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn strtrim_preserves_missing_strings() {
        let result =
            strtrim_builtin(Value::String("<missing>".into())).expect("strtrim missing string");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[test]
    fn strtrim_handles_tabs_and_newlines() {
        let input = Value::String("\tMetrics \n".into());
        let result = strtrim_builtin(input).expect("strtrim tab/newline");
        assert_eq!(result, Value::String("Metrics".into()));
    }

    #[test]
    fn strtrim_trims_unicode_whitespace() {
        let input = Value::String("\u{00A0}RunMat\u{2003}".into());
        let result = strtrim_builtin(input).expect("strtrim unicode whitespace");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[test]
    fn strtrim_char_array_zero_rows_stable() {
        let array = CharArray::new(Vec::new(), 0, 0).unwrap();
        let result = strtrim_builtin(Value::CharArray(array.clone())).expect("strtrim 0x0 char");
        assert_eq!(result, Value::CharArray(array));
    }

    #[test]
    fn strtrim_cell_array_accepts_string_scalar() {
        let scalar = StringArray::new(vec![" padded ".into()], vec![1, 1]).unwrap();
        let cell = CellArray::new(vec![Value::StringArray(scalar)], 1, 1).unwrap();
        let trimmed = strtrim_builtin(Value::Cell(cell)).expect("strtrim cell string scalar");
        match trimmed {
            Value::Cell(out) => {
                let value = out.get(0, 0).expect("cell element");
                assert_eq!(value, Value::String("padded".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn strtrim_cell_array_rejects_non_text() {
        let cell = CellArray::new(vec![Value::Num(5.0)], 1, 1).unwrap();
        let err = strtrim_builtin(Value::Cell(cell)).expect_err("strtrim cell non-text");
        assert!(err.contains("cell array elements"));
    }

    #[test]
    fn strtrim_errors_on_invalid_input() {
        let err = strtrim_builtin(Value::Num(1.0)).unwrap_err();
        assert!(err.contains("strtrim"));
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
