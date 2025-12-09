//! MATLAB-compatible `erase` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{gather_if_needed, make_cell_with_shape};

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "erase")]
pub const DOC_MD: &str = r#"---
title: "erase"
category: "strings/transform"
keywords: ["erase", "remove substring", "delete text", "string manipulation", "character array"]
summary: "Remove substring occurrences from strings, character arrays, and cell arrays with MATLAB-compatible semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/erase.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs on the CPU; RunMat gathers GPU-resident text before removing substrings."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::erase::tests::erase_string_array_shape_mismatch_applies_all_patterns"
  integration: "builtins::strings::transform::erase::tests::erase_cell_array_mixed_content"
---

# What does the `erase` function do in MATLAB / RunMat?
`erase(text, pattern)` removes every occurrence of `pattern` from `text`. The builtin accepts string
scalars, string arrays, character arrays, and cell arrays of character vectors or string scalars,
mirroring MATLAB behaviour. When `pattern` is an array, `erase` removes every occurrence of each
pattern entry; the `text` and `pattern` arguments do not need to be the same size.

## How does the `erase` function behave in MATLAB / RunMat?
- String inputs stay as strings. Missing string scalars (`<missing>`) propagate unchanged.
- String arrays preserve their size and orientation. Each element has every supplied pattern removed.
- Character arrays are processed row by row. Rows shrink as characters are removed and are padded with
  spaces so the result remains a rectangular char array.
- Cell arrays must contain string scalars or character vectors. The result is a cell array with the same
  shape whose elements reflect the removed substrings.
- The `pattern` input can be a string scalar, string array, character array, or cell array of character
  vectors/string scalars. Provide either a scalar pattern or a list; an empty list leaves `text` unchanged.
- Pattern values are treated literally—no regular expressions are used. Use [`replace`](./replace) or the
  regex builtins for pattern-based removal.

## `erase` Function GPU Execution Behaviour
`erase` executes on the CPU. When any argument is GPU-resident, RunMat gathers it to host memory before
removing substrings. Outputs are returned on the host as well. Providers do not need to implement device
kernels for this builtin, and the fusion planner treats it as a sink to avoid keeping text on the GPU.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `erase` automatically gathers GPU inputs and produces host results. You never need to move text to or
from the GPU manually for this builtin, and `gpuArray` inputs are handled transparently.

## Examples of using the `erase` function in MATLAB / RunMat

### Remove a single word from a string scalar
```matlab
txt = "RunMat accelerates MATLAB code";
clean = erase(txt, "accelerates ");
```
Expected output:
```matlab
clean = "RunMat MATLAB code"
```

### Remove multiple substrings from each element of a string array
```matlab
labels = ["GPU pipeline"; "CPU pipeline"];
result = erase(labels, ["GPU ", "CPU "]);
```
Expected output:
```matlab
result = 2×1 string
    "pipeline"
    "pipeline"
```

### Erase characters from a character array while preserving padding
```matlab
chars = char("workspace", "snapshots");
trimmed = erase(chars, "s");
```
Expected output:
```matlab
trimmed =

  2×8 char array

    'workpace'
    'napshot '
```

### Remove substrings from a cell array of text
```matlab
C = {'Kernel Planner', "GPU Fusion"};
out = erase(C, ["Kernel ", "GPU "]);
```
Expected output:
```matlab
out = 1×2 cell array
    {'Planner'}    {"Fusion"}
```

### Provide an empty pattern list to leave the text unchanged
```matlab
data = ["alpha", "beta"];
unchanged = erase(data, string.empty);
```
Expected output:
```matlab
unchanged = 1×2 string
    "alpha"    "beta"
```

### Remove delimiters before splitting text
```matlab
path = "runmat/bin:runmat/lib";
clean = erase(path, ":");
parts = split(clean, "runmat/");
```
Expected output:
```matlab
clean = "runmat/binrunmat/lib"
parts = 1×3 string
    ""    "bin"    "lib"
```

## FAQ

### Can I remove multiple patterns at once?
Yes. Supply `pattern` as a string array or cell array. Each pattern is removed in order from every element
of the input text.

### What happens if `pattern` is empty?
An empty pattern list leaves the input unchanged. Empty string patterns are ignored because removing empty
text would have no effect.

### Does `erase` modify the original data?
No. It returns a new value with substrings removed. The input variables remain unchanged.

### How are missing string scalars handled?
They propagate unchanged. Calling `erase` on `<missing>` returns `<missing>`, matching MATLAB.

### Can `erase` operate on GPU-resident data?
Indirectly. RunMat automatically gathers GPU values to the host, performs the removal, and returns a host
result. No explicit `gpuArray` calls are required.

### How do I remove substrings using patterns or regular expressions?
Use `replace` for literal substitution or `regexprep` for regular expressions when you need pattern-based
removal rather than literal substring erasure.

## See Also
[replace](./replace), [strrep](./strrep), [split](./split), [regexprep](../regex/regexprep), [string](../core/string)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/erase.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/erase.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "erase",
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
        "Executes on the CPU; GPU-resident inputs are gathered to host memory before substrings are removed.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "erase",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "String manipulation builtin; not eligible for fusion plans and always gathers GPU inputs before execution.",
};

const ARG_TYPE_ERROR: &str =
    "erase: first argument must be a string array, character array, or cell array of character vectors";
const PATTERN_TYPE_ERROR: &str =
    "erase: second argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "erase: cell array elements must be string scalars or character vectors";

#[runtime_builtin(
    name = "erase",
    category = "strings/transform",
    summary = "Remove substring occurrences from strings, character arrays, and cell arrays.",
    keywords = "erase,remove substring,strings,character array,text",
    accel = "sink"
)]
fn erase_builtin(text: Value, pattern: Value) -> Result<Value, String> {
    let text = gather_if_needed(&text).map_err(|e| format!("erase: {e}"))?;
    let pattern = gather_if_needed(&pattern).map_err(|e| format!("erase: {e}"))?;

    let patterns = PatternList::from_value(&pattern)?;

    match text {
        Value::String(s) => Ok(Value::String(erase_string_scalar(s, &patterns))),
        Value::StringArray(sa) => erase_string_array(sa, &patterns),
        Value::CharArray(ca) => erase_char_array(ca, &patterns),
        Value::Cell(cell) => erase_cell_array(cell, &patterns),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

struct PatternList {
    entries: Vec<String>,
}

impl PatternList {
    fn from_value(value: &Value) -> Result<Self, String> {
        let entries = match value {
            Value::String(text) => vec![text.clone()],
            Value::StringArray(array) => array.data.clone(),
            Value::CharArray(array) => {
                if array.rows == 0 {
                    Vec::new()
                } else {
                    let mut list = Vec::with_capacity(array.rows);
                    for row in 0..array.rows {
                        list.push(char_row_to_string_slice(&array.data, array.cols, row));
                    }
                    list
                }
            }
            Value::Cell(cell) => {
                let mut list = Vec::with_capacity(cell.data.len());
                for handle in &cell.data {
                    match &**handle {
                        Value::String(text) => list.push(text.clone()),
                        Value::StringArray(sa) if sa.data.len() == 1 => {
                            list.push(sa.data[0].clone());
                        }
                        Value::CharArray(ca) if ca.rows == 0 => list.push(String::new()),
                        Value::CharArray(ca) if ca.rows == 1 => {
                            list.push(char_row_to_string_slice(&ca.data, ca.cols, 0));
                        }
                        Value::CharArray(_) => return Err(CELL_ELEMENT_ERROR.to_string()),
                        _ => return Err(CELL_ELEMENT_ERROR.to_string()),
                    }
                }
                list
            }
            _ => return Err(PATTERN_TYPE_ERROR.to_string()),
        };
        Ok(Self { entries })
    }

    fn apply(&self, input: &str) -> String {
        if self.entries.is_empty() {
            return input.to_string();
        }
        let mut current = input.to_string();
        for pattern in &self.entries {
            if pattern.is_empty() {
                continue;
            }
            if current.is_empty() {
                break;
            }
            current = current.replace(pattern, "");
        }
        current
    }
}

fn erase_string_scalar(text: String, patterns: &PatternList) -> String {
    if is_missing_string(&text) {
        text
    } else {
        patterns.apply(&text)
    }
}

fn erase_string_array(array: StringArray, patterns: &PatternList) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let mut erased = Vec::with_capacity(data.len());
    for entry in data {
        if is_missing_string(&entry) {
            erased.push(entry);
        } else {
            erased.push(patterns.apply(&entry));
        }
    }
    StringArray::new(erased, shape)
        .map(Value::StringArray)
        .map_err(|e| format!("erase: {e}"))
}

fn erase_char_array(array: CharArray, patterns: &PatternList) -> Result<Value, String> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut processed: Vec<String> = Vec::with_capacity(rows);
    let mut target_cols = 0usize;
    for row in 0..rows {
        let slice = char_row_to_string_slice(&data, cols, row);
        let erased = patterns.apply(&slice);
        let len = erased.chars().count();
        if len > target_cols {
            target_cols = len;
        }
        processed.push(erased);
    }

    let mut flattened: Vec<char> = Vec::with_capacity(rows * target_cols);
    for row_text in processed {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        flattened.extend(chars);
    }

    CharArray::new(flattened, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| format!("erase: {e}"))
}

fn erase_cell_array(cell: CellArray, patterns: &PatternList) -> Result<Value, String> {
    let shape = cell.shape.clone();
    let mut values = Vec::with_capacity(cell.data.len());
    for handle in &cell.data {
        values.push(erase_cell_element(handle, patterns)?);
    }
    make_cell_with_shape(values, shape).map_err(|e| format!("erase: {e}"))
}

fn erase_cell_element(value: &Value, patterns: &PatternList) -> Result<Value, String> {
    match value {
        Value::String(text) => Ok(Value::String(erase_string_scalar(text.clone(), patterns))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Value::String(erase_string_scalar(
            sa.data[0].clone(),
            patterns,
        ))),
        Value::CharArray(ca) if ca.rows == 0 => Ok(Value::CharArray(ca.clone())),
        Value::CharArray(ca) if ca.rows == 1 => {
            let slice = char_row_to_string_slice(&ca.data, ca.cols, 0);
            let erased = patterns.apply(&slice);
            Ok(Value::CharArray(CharArray::new_row(&erased)))
        }
        Value::CharArray(_) => Err(CELL_ELEMENT_ERROR.to_string()),
        _ => Err(CELL_ELEMENT_ERROR.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn erase_string_scalar_single_pattern() {
        let result = erase_builtin(
            Value::String("RunMat runtime".into()),
            Value::String(" runtime".into()),
        )
        .expect("erase");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[test]
    fn erase_string_array_multiple_patterns() {
        let strings = StringArray::new(
            vec!["gpu".into(), "cpu".into(), "<missing>".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = erase_builtin(
            Value::StringArray(strings),
            Value::StringArray(StringArray::new(vec!["g".into(), "c".into()], vec![2, 1]).unwrap()),
        )
        .expect("erase");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("pu"),
                        String::from("pu"),
                        String::from("<missing>")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn erase_string_array_shape_mismatch_applies_all_patterns() {
        let strings =
            StringArray::new(vec!["GPU kernel".into(), "CPU kernel".into()], vec![2, 1]).unwrap();
        let patterns = StringArray::new(vec!["GPU ".into(), "CPU ".into()], vec![1, 2]).unwrap();
        let result = erase_builtin(Value::StringArray(strings), Value::StringArray(patterns))
            .expect("erase");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(
                    sa.data,
                    vec![String::from("kernel"), String::from("kernel")]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn erase_char_array_adjusts_width() {
        let chars = CharArray::new("matrix".chars().collect(), 1, 6).unwrap();
        let result =
            erase_builtin(Value::CharArray(chars), Value::String("tr".into())).expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 4);
                let expected: Vec<char> = "maix".chars().collect();
                assert_eq!(out.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn erase_char_array_handles_full_removal() {
        let chars = CharArray::new_row("abc");
        let result = erase_builtin(Value::CharArray(chars.clone()), Value::String("abc".into()))
            .expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 0);
                assert!(out.data.is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }
    }

    #[test]
    fn erase_char_array_multiple_rows_sequential_patterns() {
        let chars = CharArray::new(
            vec![
                'G', 'P', 'U', ' ', 'p', 'i', 'p', 'e', 'l', 'i', 'n', 'e', 'C', 'P', 'U', ' ',
                'p', 'i', 'p', 'e', 'l', 'i', 'n', 'e',
            ],
            2,
            12,
        )
        .unwrap();
        let patterns = CharArray::new_row("GPU ");
        let result =
            erase_builtin(Value::CharArray(chars), Value::CharArray(patterns)).expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 12);
                let first = char_row_to_string_slice(&out.data, out.cols, 0);
                let second = char_row_to_string_slice(&out.data, out.cols, 1);
                assert_eq!(first.trim_end(), "pipeline");
                assert_eq!(second.trim_end(), "CPU pipeline");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn erase_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("Kernel Planner")),
                Value::String("GPU Fusion".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = erase_builtin(
            Value::Cell(cell),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("Kernel ".into()),
                        Value::String("GPU ".into()),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        )
        .expect("erase");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("Planner")));
                assert_eq!(second, Value::String("Fusion".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn erase_cell_array_preserves_shape() {
        let cell = CellArray::new(
            vec![
                Value::String("alpha".into()),
                Value::String("beta".into()),
                Value::String("gamma".into()),
                Value::String("delta".into()),
            ],
            2,
            2,
        )
        .unwrap();
        let patterns = StringArray::new(vec!["a".into()], vec![1, 1]).unwrap();
        let result = erase_builtin(Value::Cell(cell), Value::StringArray(patterns)).expect("erase");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.get(0, 0).unwrap(), Value::String("lph".into()));
                assert_eq!(out.get(1, 1).unwrap(), Value::String("delt".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn erase_preserves_missing_string() {
        let result = erase_builtin(
            Value::String("<missing>".into()),
            Value::String("missing".into()),
        )
        .expect("erase");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[test]
    fn erase_allows_empty_pattern_list() {
        let strings = StringArray::new(vec!["alpha".into(), "beta".into()], vec![2, 1]).unwrap();
        let pattern = StringArray::new(Vec::<String>::new(), vec![0, 0]).unwrap();
        let result = erase_builtin(
            Value::StringArray(strings.clone()),
            Value::StringArray(pattern),
        )
        .expect("erase");
        assert_eq!(result, Value::StringArray(strings));
    }

    #[test]
    fn erase_errors_on_invalid_first_argument() {
        let err = erase_builtin(Value::Num(1.0), Value::String("a".into())).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
    }

    #[test]
    fn erase_errors_on_invalid_pattern_type() {
        let err = erase_builtin(Value::String("abc".into()), Value::Num(1.0)).unwrap_err();
        assert_eq!(err, PATTERN_TYPE_ERROR);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
