//! MATLAB-compatible `replace` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{gather_if_needed, make_cell};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "replace",
        wasm_path = "crate::builtins::strings::transform::replace"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "replace"
category: "strings/transform"
keywords: ["replace", "substring replace", "string replace", "strrep", "text replace", "character array replace"]
summary: "Replace substring occurrences in strings, character arrays, and cell arrays."
references:
  - https://www.mathworks.com/help/matlab/ref/replace.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs on the CPU. RunMat gathers GPU-resident text before performing replacements."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::replace::tests"
  integration: "builtins::strings::transform::replace::tests::replace_cell_array_mixed_content"
---

# What does the `replace` function do in MATLAB / RunMat?
`replace(str, old, new)` substitutes every occurrence of `old` found in `str` with `new`. The builtin
accepts string scalars, string arrays, character arrays, and cell arrays of character vectors or strings,
matching MATLAB semantics. Multiple search terms are supported—each matched entry is replaced by its
corresponding replacement text.

## How does the `replace` function behave in MATLAB / RunMat?
- String scalars remain strings. Missing string scalars (`<missing>`) propagate unchanged.
- String arrays are processed element wise while preserving shape.
- Character arrays are handled row by row. Rows expand or shrink as needed and are padded with spaces so
  the result remains a rectangular char array, just like MATLAB.
- Cell arrays must contain string scalars or character vectors. The result is a cell array with the same
  size and element types mirrored after replacement.
- The `old` and `new` inputs can be string scalars, string arrays, character arrays, or cell arrays of
  character vectors / strings. `new` must be a scalar or match the number of search terms in `old`.
- Non-text inputs (numeric, logical, structs, GPU tensors, etc.) produce MATLAB-compatible errors.

## `replace` Function GPU Execution Behaviour
`replace` executes on the CPU. The builtin registers as an Accelerate sink, so the fusion planner never
attempts to keep results on the device. When any argument is GPU-resident, RunMat gathers it to host memory
before performing replacements. Providers do not need special kernels for this builtin, and GPU-resident
results are returned on the host.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `replace` automatically gathers GPU inputs back to the host when necessary. Because it is marked with a
gather-immediately residency policy, both inputs and outputs live on the CPU, so you never have to move
text values manually. This mirrors MATLAB behaviour where string manipulation runs on the CPU.

## Examples of using the `replace` function in MATLAB / RunMat

### Replace all instances of a word in a string
```matlab
txt = "RunMat accelerates MATLAB code";
result = replace(txt, "RunMat", "RunMat Accelerate");
```
Expected output:
```matlab
result = "RunMat Accelerate accelerates MATLAB code"
```

### Replace multiple terms in a string array
```matlab
labels = ["GPU pipeline"; "CPU pipeline"];
result = replace(labels, ["GPU", "CPU"], ["Device", "Host"]);
```
Expected output:
```matlab
result = 2×1 string
    "Device pipeline"
    "Host pipeline"
```

### Replace substrings in a character array while preserving padding
```matlab
chars = char("alpha", "beta ");
out = replace(chars, "a", "A");
```
Expected output:
```matlab
out =

  2×5 char array

    'AlphA'
    'betA '
```

### Replace text within a cell array of character vectors
```matlab
C = {'Kernel Fusion', 'GPU Planner'};
updated = replace(C, {'Kernel', 'GPU'}, {'Shader', 'Device'});
```
Expected output:
```matlab
updated = 1×2 cell array
    {'Shader Fusion'}    {'Device Planner'}
```

### Remove substrings by replacing with empty text
```matlab
paths = ["runmat/bin", "runmat/lib"];
clean = replace(paths, "runmat/", "");
```
Expected output:
```matlab
clean = 1×2 string
    "bin"    "lib"
```

### Replace using scalar replacement for multiple search terms
```matlab
message = "OpenCL or CUDA or Vulkan";
unified = replace(message, ["OpenCL", "CUDA", "Vulkan"], "GPU backend");
```
Expected output:
```matlab
unified = "GPU backend or GPU backend or GPU backend"
```

### Replace text stored inside a cell array of strings
```matlab
cells = { "Snapshot", "Ignition Interpreter" };
renamed = replace(cells, " ", "_");
```
Expected output:
```matlab
renamed = 1×2 cell array
    {"Snapshot"}    {"Ignition_Interpreter"}
```

### Preserve missing strings during replacement
```matlab
vals = ["runmat", "<missing>", "accelerate"];
out = replace(vals, "runmat", "RunMat");
```
Expected output:
```matlab
out = 1×3 string
    "RunMat"    <missing>    "accelerate"
```

## FAQ

### What sizes are allowed for `old` and `new` inputs?
`old` must contain at least one search term. `new` may be a scalar or contain the same number of elements
as `old`. Otherwise, `replace` raises a size-mismatch error matching MATLAB behaviour.

### Does `replace` modify the original input?
No. The builtin returns a new value with substitutions applied. The original inputs are left untouched.

### How are character arrays padded after replacement?
Each row is expanded or truncated according to the longest resulting row. Shorter rows are padded with space
characters so the output remains a proper char matrix.

### How are missing strings handled?
Missing string scalars (`<missing>`) propagate unchanged. Replacements never convert a missing value into a
non-missing string.

### Can I replace with an empty string?
Yes. Provide `""` (empty string) or `''` as the replacement to remove matched substrings entirely.

### Does `replace` support overlapping matches?
Replacements are non-overlapping and proceed from left to right, matching MATLAB’s behaviour for `replace`.

### How does `replace` behave with GPU data?
RunMat gathers GPU-resident inputs to host memory before performing replacements. The resulting value is
returned on the host. Providers do not need to implement a GPU kernel for this builtin.

## See Also
[regexprep](../../regex/regexprep),
[string](../core/string),
[char](../core/char),
[strtrim](../transform/strtrim),
[strip](../transform/strip)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/replace.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/replace.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::strings::transform::replace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "replace",
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
        "Executes on the CPU; GPU-resident inputs are gathered to host memory prior to replacement.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::strings::transform::replace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "replace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "String manipulation builtin; not eligible for fusion plans and always gathers GPU inputs.",
};

const ARG_TYPE_ERROR: &str =
    "replace: first argument must be a string array, character array, or cell array of character vectors";
const PATTERN_TYPE_ERROR: &str =
    "replace: second argument must be a string array, character array, or cell array of character vectors";
const REPLACEMENT_TYPE_ERROR: &str =
    "replace: third argument must be a string array, character array, or cell array of character vectors";
const EMPTY_PATTERN_ERROR: &str =
    "replace: second argument must contain at least one search string";
const EMPTY_REPLACEMENT_ERROR: &str =
    "replace: third argument must contain at least one replacement string";
const SIZE_MISMATCH_ERROR: &str =
    "replace: replacement array must be a scalar or match the number of search strings";
const CELL_ELEMENT_ERROR: &str =
    "replace: cell array elements must be string scalars or character vectors";

#[runtime_builtin(
    name = "replace",
    category = "strings/transform",
    summary = "Replace substring occurrences in strings, character arrays, and cell arrays.",
    keywords = "replace,strrep,strings,character array,text",
    accel = "sink",
    wasm_path = "crate::builtins::strings::transform::replace"
)]
fn replace_builtin(text: Value, old: Value, new: Value) -> Result<Value, String> {
    let text = gather_if_needed(&text).map_err(|e| format!("replace: {e}"))?;
    let old = gather_if_needed(&old).map_err(|e| format!("replace: {e}"))?;
    let new = gather_if_needed(&new).map_err(|e| format!("replace: {e}"))?;

    let spec = ReplacementSpec::from_values(&old, &new)?;

    match text {
        Value::String(s) => Ok(Value::String(replace_string_scalar(s, &spec))),
        Value::StringArray(sa) => replace_string_array(sa, &spec),
        Value::CharArray(ca) => replace_char_array(ca, &spec),
        Value::Cell(cell) => replace_cell_array(cell, &spec),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

fn replace_string_scalar(text: String, spec: &ReplacementSpec) -> String {
    if is_missing_string(&text) {
        text
    } else {
        spec.apply(&text)
    }
}

fn replace_string_array(array: StringArray, spec: &ReplacementSpec) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let mut replaced = Vec::with_capacity(data.len());
    for entry in data {
        if is_missing_string(&entry) {
            replaced.push(entry);
        } else {
            replaced.push(spec.apply(&entry));
        }
    }
    let result = StringArray::new(replaced, shape).map_err(|e| format!("replace: {e}"))?;
    Ok(Value::StringArray(result))
}

fn replace_char_array(array: CharArray, spec: &ReplacementSpec) -> Result<Value, String> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut replaced_rows = Vec::with_capacity(rows);
    let mut target_cols = 0usize;
    for row in 0..rows {
        let slice = char_row_to_string_slice(&data, cols, row);
        let replaced = spec.apply(&slice);
        let len = replaced.chars().count();
        target_cols = target_cols.max(len);
        replaced_rows.push(replaced);
    }

    let mut flattened = Vec::with_capacity(rows * target_cols);
    for row_text in replaced_rows {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        flattened.extend(chars);
    }

    CharArray::new(flattened, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| format!("replace: {e}"))
}

fn replace_cell_array(cell: CellArray, spec: &ReplacementSpec) -> Result<Value, String> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut replaced = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let value = replace_cell_element(&data[idx], spec)?;
            replaced.push(value);
        }
    }
    make_cell(replaced, rows, cols).map_err(|e| format!("replace: {e}"))
}

fn replace_cell_element(value: &Value, spec: &ReplacementSpec) -> Result<Value, String> {
    match value {
        Value::String(text) => Ok(Value::String(replace_string_scalar(text.clone(), spec))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Value::String(replace_string_scalar(
            sa.data[0].clone(),
            spec,
        ))),
        Value::CharArray(ca) if ca.rows <= 1 => replace_char_array(ca.clone(), spec),
        Value::CharArray(_) => Err(CELL_ELEMENT_ERROR.to_string()),
        _ => Err(CELL_ELEMENT_ERROR.to_string()),
    }
}

fn extract_pattern_list(value: &Value) -> Result<Vec<String>, String> {
    extract_text_list(value, PATTERN_TYPE_ERROR)
}

fn extract_replacement_list(value: &Value) -> Result<Vec<String>, String> {
    extract_text_list(value, REPLACEMENT_TYPE_ERROR)
}

fn extract_text_list(value: &Value, type_error: &str) -> Result<Vec<String>, String> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) => {
            let CharArray { data, rows, cols } = array.clone();
            if rows == 0 {
                Ok(Vec::new())
            } else {
                let mut entries = Vec::with_capacity(rows);
                for row in 0..rows {
                    entries.push(char_row_to_string_slice(&data, cols, row));
                }
                Ok(entries)
            }
        }
        Value::Cell(cell) => {
            let CellArray { data, .. } = cell.clone();
            let mut entries = Vec::with_capacity(data.len());
            for element in data {
                match &*element {
                    Value::String(text) => entries.push(text.clone()),
                    Value::StringArray(sa) if sa.data.len() == 1 => {
                        entries.push(sa.data[0].clone());
                    }
                    Value::CharArray(ca) if ca.rows <= 1 => {
                        if ca.rows == 0 {
                            entries.push(String::new());
                        } else {
                            entries.push(char_row_to_string_slice(&ca.data, ca.cols, 0));
                        }
                    }
                    Value::CharArray(_) => return Err(CELL_ELEMENT_ERROR.to_string()),
                    _ => return Err(CELL_ELEMENT_ERROR.to_string()),
                }
            }
            Ok(entries)
        }
        _ => Err(type_error.to_string()),
    }
}

struct ReplacementSpec {
    pairs: Vec<(String, String)>,
}

impl ReplacementSpec {
    fn from_values(old: &Value, new: &Value) -> Result<Self, String> {
        let patterns = extract_pattern_list(old)?;
        if patterns.is_empty() {
            return Err(EMPTY_PATTERN_ERROR.to_string());
        }

        let replacements = extract_replacement_list(new)?;
        if replacements.is_empty() {
            return Err(EMPTY_REPLACEMENT_ERROR.to_string());
        }

        let pairs = if replacements.len() == patterns.len() {
            patterns.into_iter().zip(replacements).collect::<Vec<_>>()
        } else if replacements.len() == 1 {
            let replacement = replacements[0].clone();
            patterns
                .into_iter()
                .map(|pattern| (pattern, replacement.clone()))
                .collect::<Vec<_>>()
        } else {
            return Err(SIZE_MISMATCH_ERROR.to_string());
        };

        Ok(Self { pairs })
    }

    fn apply(&self, input: &str) -> String {
        let mut current = input.to_string();
        for (pattern, replacement) in &self.pairs {
            if pattern.is_empty() && replacement.is_empty() {
                continue;
            }
            if pattern == replacement {
                continue;
            }
            current = current.replace(pattern, replacement);
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn replace_string_scalar_single_term() {
        let result = replace_builtin(
            Value::String("RunMat runtime".into()),
            Value::String("runtime".into()),
            Value::String("engine".into()),
        )
        .expect("replace");
        assert_eq!(result, Value::String("RunMat engine".into()));
    }

    #[test]
    fn replace_string_array_multiple_terms() {
        let strings = StringArray::new(
            vec!["gpu".into(), "cpu".into(), "<missing>".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = replace_builtin(
            Value::StringArray(strings),
            Value::StringArray(
                StringArray::new(vec!["gpu".into(), "cpu".into()], vec![2, 1]).unwrap(),
            ),
            Value::String("device".into()),
        )
        .expect("replace");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("device"),
                        String::from("device"),
                        String::from("<missing>")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn replace_char_array_adjusts_width() {
        let chars = CharArray::new("matrix".chars().collect(), 1, 6).unwrap();
        let result = replace_builtin(
            Value::CharArray(chars),
            Value::String("matrix".into()),
            Value::String("tensor".into()),
        )
        .expect("replace");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 6);
                let expected: Vec<char> = "tensor".chars().collect();
                assert_eq!(out.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn replace_char_array_handles_padding() {
        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let result = replace_builtin(
            Value::CharArray(chars),
            Value::String("b".into()),
            Value::String("beta".into()),
        )
        .expect("replace");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 5);
                let expected: Vec<char> = vec!['a', 'b', 'e', 't', 'a', 'c', 'd', ' ', ' ', ' '];
                assert_eq!(out.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn replace_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("Kernel Planner")),
                Value::String("GPU Fusion".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = replace_builtin(
            Value::Cell(cell),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("Kernel".into()), Value::String("GPU".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["Shader".into(), "Device".into()], vec![1, 2]).unwrap(),
            ),
        )
        .expect("replace");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(
                    first,
                    Value::CharArray(CharArray::new_row("Shader Planner"))
                );
                assert_eq!(second, Value::String("Device Fusion".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn replace_errors_on_invalid_first_argument() {
        let err = replace_builtin(
            Value::Num(1.0),
            Value::String("a".into()),
            Value::String("b".into()),
        )
        .unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
    }

    #[test]
    fn replace_errors_on_invalid_pattern_type() {
        let err = replace_builtin(
            Value::String("abc".into()),
            Value::Num(1.0),
            Value::String("x".into()),
        )
        .unwrap_err();
        assert_eq!(err, PATTERN_TYPE_ERROR);
    }

    #[test]
    fn replace_errors_on_size_mismatch() {
        let err = replace_builtin(
            Value::String("abc".into()),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap()),
            Value::StringArray(
                StringArray::new(vec!["x".into(), "y".into(), "z".into()], vec![3, 1]).unwrap(),
            ),
        )
        .unwrap_err();
        assert_eq!(err, SIZE_MISMATCH_ERROR);
    }

    #[test]
    fn replace_preserves_missing_string() {
        let result = replace_builtin(
            Value::String("<missing>".into()),
            Value::String("missing".into()),
            Value::String("value".into()),
        )
        .expect("replace");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
