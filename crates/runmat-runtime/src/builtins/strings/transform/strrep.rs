//! MATLAB-compatible `strrep` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{gather_if_needed, make_cell_with_shape};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "strrep",
        builtin_path = "crate::builtins::strings::transform::strrep"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "strrep"
category: "strings/transform"
keywords: ["strrep", "string replace", "character array replace", "text replacement", "substring replace"]
summary: "Replace substring occurrences in strings, character arrays, and cell arrays while mirroring MATLAB semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/strrep.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs on the CPU. RunMat gathers GPU-resident text inputs before performing replacements."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::strrep::tests"
  integration: "builtins::strings::transform::strrep::tests::strrep_cell_array_char_vectors, builtins::strings::transform::strrep::tests::strrep_wgpu_provider_fallback"
---

# What does the `strrep` function do in MATLAB / RunMat?
`strrep(str, old, new)` replaces every non-overlapping instance of the substring `old` that appears in
`str` with the text provided in `new`. The builtin accepts string scalars, string arrays, character arrays,
and cell arrays of character vectors, matching MATLAB behaviour exactly.

## How does the `strrep` function behave in MATLAB / RunMat?
- String scalars remain strings. Missing string values (`<missing>`) propagate unchanged.
- String arrays are processed element-wise while preserving their full shape and orientation.
- Character arrays are handled row by row. Rows expand or shrink as needed and are padded with spaces so
  the result stays a rectangular char array, just like MATLAB.
- Cell arrays must contain character vectors or string scalars. The result is a cell array of identical size
  where each element has had the replacement applied.
- The `old` and `new` arguments must be string scalars or character vectors of the same data type.
- `old` can be empty. In that case, `strrep` inserts `new` before the first character, between existing
  characters, and after the final character.

## `strrep` Function GPU Execution Behaviour
RunMat treats text replacement as a CPU-first workflow:

1. The builtin is registered as an Accelerate *sink*, so the planner gathers any GPU-resident inputs
   (string arrays, char arrays, or cell contents) back to host memory before work begins.
2. Replacements are computed entirely on the CPU, mirroring MATLAB’s behaviour and avoiding GPU/device
   divergence in string handling.
3. Results are returned as host values (string array, char array, or cell array). Residency is never pushed
   back to the GPU, keeping semantics deterministic regardless of the active provider.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `strrep` registers as a sink with RunMat Accelerate, so the fusion planner never keeps its inputs or
outputs on the GPU. Even if you start with GPU data, the runtime gathers it automatically—manual `gpuArray`
or `gather` calls are unnecessary.

## Examples of using the `strrep` function in MATLAB / RunMat

### Replacing a word inside a string scalar
```matlab
txt = "RunMat turbo mode";
result = strrep(txt, "turbo", "accelerate");
```
Expected output:
```matlab
result = "RunMat accelerate mode"
```

### Updating every element of a string array
```matlab
labels = ["GPU planner", "CPU planner"];
updated = strrep(labels, "planner", "pipeline");
```
Expected output:
```matlab
updated = 2×1 string
    "GPU pipeline"
    "CPU pipeline"
```

### Preserving rectangular shape in character arrays
```matlab
chars = char("alpha", "beta ");
out = strrep(chars, "a", "A");
```
Expected output:
```matlab
out =

  2×5 char array

    'AlphA'
    'betA '
```

### Applying replacements inside a cell array of character vectors
```matlab
C = {'Kernel Fusion', 'GPU Planner'};
renamed = strrep(C, ' ', '_');
```
Expected output:
```matlab
renamed = 1×2 cell array
    {'Kernel_Fusion'}    {'GPU_Planner'}
```

### Inserting text with an empty search pattern
```matlab
stub = "abc";
expanded = strrep(stub, "", "-");
```
Expected output:
```matlab
expanded = "-a-b-c-"
```

### Leaving missing string values untouched
```matlab
vals = ["RunMat", "<missing>", "Accelerate"];
out = strrep(vals, "RunMat", "RUNMAT");
```
Expected output:
```matlab
out = 1×3 string
    "RUNMAT"    <missing>    "Accelerate"
```

### Replacing substrings gathered from GPU inputs
```matlab
g = gpuArray("Turbine");
host = strrep(g, "bine", "bo");
```
Expected output:
```matlab
host = "Turbo"
```

## FAQ

### Which input types does `strrep` accept?
String scalars, string arrays, character vectors, character arrays, and cell arrays of character vectors.
The `old` and `new` arguments must be string scalars or character vectors of the same data type.

### Does `strrep` support multiple search terms at once?
No. Use the newer `replace` builtin if you need to substitute several search terms in a single call.

### How does `strrep` handle missing strings?
Missing string scalars remain `<missing>` and are returned unchanged, even when the search pattern matches
ordinary text.

### Will rows of a character array stay aligned?
Yes. Each row is replaced individually, then padded with spaces so that the overall array stays rectangular,
matching MATLAB exactly.

### What happens when `old` is empty?
RunMat mirrors MATLAB: `new` is inserted before the first character, between every existing character, and
after the last character.

### Does `strrep` run on the GPU?
Not today. The builtin gathers GPU-resident data to host memory automatically before performing the
replacement logic.

### Can I mix strings and character vectors for `old` and `new`?
No. MATLAB requires `old` and `new` to share the same data type. RunMat enforces the same rule and raises a
descriptive error when they differ.

### How do I replace text stored inside cell arrays?
`strrep` traverses the cell array, applying the replacement to each character vector or string scalar element
and returning a cell array of the same shape.

## See Also
[replace](./replace), [regexprep](./regexprep), [string](./string), [char](./char), [join](./join)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strrep")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strrep",
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
    notes: "Executes on the CPU; GPU-resident inputs are gathered before replacements are applied.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::strrep")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strrep",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; marked as a sink so fusion skips GPU residency.",
};

const ARGUMENT_TYPE_ERROR: &str =
    "strrep: first argument must be a string array, character array, or cell array of character vectors";
const PATTERN_TYPE_ERROR: &str = "strrep: old and new must be string scalars or character vectors";
const PATTERN_MISMATCH_ERROR: &str = "strrep: old and new must be the same data type";
const CELL_ELEMENT_ERROR: &str =
    "strrep: cell array elements must be string scalars or character vectors";

#[derive(Clone, Copy, PartialEq, Eq)]
enum PatternKind {
    String,
    Char,
}

#[runtime_builtin(
    name = "strrep",
    category = "strings/transform",
    summary = "Replace substring occurrences with MATLAB-compatible semantics.",
    keywords = "strrep,replace,strings,character array,text",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::strrep"
)]
fn strrep_builtin(str_value: Value, old_value: Value, new_value: Value) -> Result<Value, String> {
    let gathered_str = gather_if_needed(&str_value).map_err(|e| format!("strrep: {e}"))?;
    let gathered_old = gather_if_needed(&old_value).map_err(|e| format!("strrep: {e}"))?;
    let gathered_new = gather_if_needed(&new_value).map_err(|e| format!("strrep: {e}"))?;

    let (old_text, old_kind) = parse_pattern(gathered_old)?;
    let (new_text, new_kind) = parse_pattern(gathered_new)?;
    if old_kind != new_kind {
        return Err(PATTERN_MISMATCH_ERROR.to_string());
    }

    match gathered_str {
        Value::String(text) => Ok(Value::String(strrep_string_value(
            text, &old_text, &new_text,
        ))),
        Value::StringArray(array) => strrep_string_array(array, &old_text, &new_text),
        Value::CharArray(array) => strrep_char_array(array, &old_text, &new_text),
        Value::Cell(cell) => strrep_cell_array(cell, &old_text, &new_text),
        _ => Err(ARGUMENT_TYPE_ERROR.to_string()),
    }
}

fn parse_pattern(value: Value) -> Result<(String, PatternKind), String> {
    match value {
        Value::String(text) => Ok((text, PatternKind::String)),
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                Ok((array.data[0].clone(), PatternKind::String))
            } else {
                Err(PATTERN_TYPE_ERROR.to_string())
            }
        }
        Value::CharArray(array) => {
            if array.rows <= 1 {
                let text = if array.rows == 0 {
                    String::new()
                } else {
                    char_row_to_string_slice(&array.data, array.cols, 0)
                };
                Ok((text, PatternKind::Char))
            } else {
                Err(PATTERN_TYPE_ERROR.to_string())
            }
        }
        _ => Err(PATTERN_TYPE_ERROR.to_string()),
    }
}

fn strrep_string_value(text: String, old: &str, new: &str) -> String {
    if is_missing_string(&text) {
        text
    } else {
        text.replace(old, new)
    }
}

fn strrep_string_array(array: StringArray, old: &str, new: &str) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let replaced = data
        .into_iter()
        .map(|text| strrep_string_value(text, old, new))
        .collect::<Vec<_>>();
    let rebuilt = StringArray::new(replaced, shape).map_err(|e| format!("strrep: {e}"))?;
    Ok(Value::StringArray(rebuilt))
}

fn strrep_char_array(array: CharArray, old: &str, new: &str) -> Result<Value, String> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 || cols == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut replaced_rows = Vec::with_capacity(rows);
    let mut target_cols = 0usize;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row);
        let replaced = text.replace(old, new);
        target_cols = target_cols.max(replaced.chars().count());
        replaced_rows.push(replaced);
    }

    let mut new_data = Vec::with_capacity(rows * target_cols);
    for row_text in replaced_rows {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        new_data.extend(chars);
    }

    CharArray::new(new_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| format!("strrep: {e}"))
}

fn strrep_cell_array(cell: CellArray, old: &str, new: &str) -> Result<Value, String> {
    let CellArray { data, shape, .. } = cell;
    let mut replaced = Vec::with_capacity(data.len());
    for ptr in &data {
        replaced.push(strrep_cell_element(ptr, old, new)?);
    }
    make_cell_with_shape(replaced, shape).map_err(|e| format!("strrep: {e}"))
}

fn strrep_cell_element(value: &Value, old: &str, new: &str) -> Result<Value, String> {
    match value {
        Value::String(text) => Ok(Value::String(strrep_string_value(text.clone(), old, new))),
        Value::StringArray(array) => strrep_string_array(array.clone(), old, new),
        Value::CharArray(array) => strrep_char_array(array.clone(), old, new),
        _ => Err(CELL_ELEMENT_ERROR.to_string()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn strrep_string_scalar_basic() {
        let result = strrep_builtin(
            Value::String("RunMat Ignite".into()),
            Value::String("Ignite".into()),
            Value::String("Interpreter".into()),
        )
        .expect("strrep");
        assert_eq!(result, Value::String("RunMat Interpreter".into()));
    }

    #[test]
    fn strrep_string_array_preserves_missing() {
        let array = StringArray::new(
            vec![
                String::from("gpu"),
                String::from("<missing>"),
                String::from("planner"),
            ],
            vec![3, 1],
        )
        .unwrap();
        let result = strrep_builtin(
            Value::StringArray(array),
            Value::String("gpu".into()),
            Value::String("GPU".into()),
        )
        .expect("strrep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("GPU"),
                        String::from("<missing>"),
                        String::from("planner")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strrep_string_array_with_char_pattern() {
        let array = StringArray::new(
            vec![String::from("alpha"), String::from("beta")],
            vec![2, 1],
        )
        .unwrap();
        let result = strrep_builtin(
            Value::StringArray(array),
            Value::CharArray(CharArray::new_row("a")),
            Value::CharArray(CharArray::new_row("A")),
        )
        .expect("strrep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(sa.data, vec![String::from("AlphA"), String::from("betA")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strrep_char_array_padding() {
        let chars = CharArray::new(vec!['R', 'u', 'n', ' ', 'M', 'a', 't'], 1, 7).unwrap();
        let result = strrep_builtin(
            Value::CharArray(chars),
            Value::String(" ".into()),
            Value::String("_".into()),
        )
        .expect("strrep");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 7);
                let expected: Vec<char> = "Run_Mat".chars().collect();
                assert_eq!(out.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn strrep_char_array_shrinks_rows_pad_with_spaces() {
        let mut data: Vec<char> = "alpha".chars().collect();
        data.extend("beta ".chars());
        let array = CharArray::new(data, 2, 5).unwrap();
        let result = strrep_builtin(
            Value::CharArray(array),
            Value::String("a".into()),
            Value::String("".into()),
        )
        .expect("strrep");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 4);
                let expected: Vec<char> = vec!['l', 'p', 'h', ' ', 'b', 'e', 't', ' '];
                assert_eq!(out.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn strrep_cell_array_char_vectors() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("Kernel Fusion")),
                Value::CharArray(CharArray::new_row("GPU Planner")),
            ],
            1,
            2,
        )
        .unwrap();
        let result = strrep_builtin(
            Value::Cell(cell),
            Value::String(" ".into()),
            Value::String("_".into()),
        )
        .expect("strrep");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 2);
                assert_eq!(
                    out.get(0, 0).unwrap(),
                    Value::CharArray(CharArray::new_row("Kernel_Fusion"))
                );
                assert_eq!(
                    out.get(0, 1).unwrap(),
                    Value::CharArray(CharArray::new_row("GPU_Planner"))
                );
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn strrep_cell_array_string_scalars() {
        let cell = CellArray::new(
            vec![
                Value::String("Planner".into()),
                Value::String("Profiler".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = strrep_builtin(
            Value::Cell(cell),
            Value::String("er".into()),
            Value::String("ER".into()),
        )
        .expect("strrep");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 2);
                assert_eq!(out.get(0, 0).unwrap(), Value::String("PlannER".into()));
                assert_eq!(out.get(0, 1).unwrap(), Value::String("ProfilER".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn strrep_cell_array_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = strrep_builtin(
            Value::Cell(cell),
            Value::String("1".into()),
            Value::String("one".into()),
        )
        .expect_err("expected cell element error");
        assert!(err.contains("cell array elements"));
    }

    #[test]
    fn strrep_cell_array_char_matrix_element() {
        let mut chars: Vec<char> = "alpha".chars().collect();
        chars.extend("beta ".chars());
        let element = CharArray::new(chars, 2, 5).unwrap();
        let cell = CellArray::new(vec![Value::CharArray(element)], 1, 1).unwrap();
        let result = strrep_builtin(
            Value::Cell(cell),
            Value::String("a".into()),
            Value::String("A".into()),
        )
        .expect("strrep");
        match result {
            Value::Cell(out) => {
                let nested = out.get(0, 0).unwrap();
                match nested {
                    Value::CharArray(ca) => {
                        assert_eq!(ca.rows, 2);
                        assert_eq!(ca.cols, 5);
                        let expected: Vec<char> =
                            vec!['A', 'l', 'p', 'h', 'A', 'b', 'e', 't', 'A', ' '];
                        assert_eq!(ca.data, expected);
                    }
                    other => panic!("expected char array element, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn strrep_cell_array_string_arrays() {
        let element = StringArray::new(vec!["alpha".into(), "beta".into()], vec![1, 2]).unwrap();
        let cell = CellArray::new(vec![Value::StringArray(element)], 1, 1).unwrap();
        let result = strrep_builtin(
            Value::Cell(cell),
            Value::String("a".into()),
            Value::String("A".into()),
        )
        .expect("strrep");
        match result {
            Value::Cell(out) => {
                let nested = out.get(0, 0).unwrap();
                match nested {
                    Value::StringArray(sa) => {
                        assert_eq!(sa.shape, vec![1, 2]);
                        assert_eq!(sa.data, vec![String::from("AlphA"), String::from("betA")]);
                    }
                    other => panic!("expected string array element, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn strrep_empty_pattern_inserts_replacement() {
        let result = strrep_builtin(
            Value::String("abc".into()),
            Value::String("".into()),
            Value::String("-".into()),
        )
        .expect("strrep");
        assert_eq!(result, Value::String("-a-b-c-".into()));
    }

    #[test]
    fn strrep_type_mismatch_errors() {
        let err = strrep_builtin(
            Value::String("abc".into()),
            Value::String("a".into()),
            Value::CharArray(CharArray::new_row("x")),
        )
        .expect_err("expected type mismatch");
        assert!(err.contains("same data type"));
    }

    #[test]
    fn strrep_invalid_pattern_type_errors() {
        let err = strrep_builtin(
            Value::String("abc".into()),
            Value::Num(1.0),
            Value::String("x".into()),
        )
        .expect_err("expected pattern error");
        assert!(err.contains("string scalars or character vectors"));
    }

    #[test]
    fn strrep_first_argument_type_error() {
        let err = strrep_builtin(
            Value::Num(42.0),
            Value::String("a".into()),
            Value::String("b".into()),
        )
        .expect_err("expected argument type error");
        assert!(err.contains("first argument"));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn strrep_wgpu_provider_fallback() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            // Unable to initialize the provider in this environment; skip.
            return;
        }
        let result = strrep_builtin(
            Value::String("Turbine Engine".into()),
            Value::String("Engine".into()),
            Value::String("JIT".into()),
        )
        .expect("strrep");
        assert_eq!(result, Value::String("Turbine JIT".into()));
    }

    #[test]
    fn doc_examples_smoke() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
