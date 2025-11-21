//! MATLAB-compatible `split` builtin with GPU-aware semantics for RunMat.

use std::collections::HashSet;

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "split"
category: "strings/transform"
keywords: ["split", "string split", "text split", "delimiters", "collapse delimiters", "include delimiters"]
summary: "Split strings, character arrays, and cell arrays into substrings using delimiters."
references:
  - https://www.mathworks.com/help/matlab/ref/split.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Executes on the CPU; GPU-resident arguments are gathered to host memory prior to splitting."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::split::tests"
  integration: "builtins::strings::transform::split::tests::split_cell_array_mixed_inputs"
---

# What does the `split` function do in MATLAB / RunMat?
`split(text)` breaks text into substrings separated by delimiters. The input can be a string scalar,
string array, character array, or a cell array of character vectors—`split` mirrors MATLAB behaviour
across each of these representations. When you omit the delimiter argument, `split` collapses
whitespace runs and returns the remaining tokens as a string array.

## How does the `split` function behave in MATLAB / RunMat?
- The default delimiter is whitespace (`isspace`), and consecutive whitespace is treated as a single
  separator (equivalent to `'CollapseDelimiters', true`).
- When you supply explicit delimiters, they can be a string scalar, string array, character array
  (rows), or a cell array of character vectors. Delimiters are matched left to right and the longest
  delimiter wins when several candidates match at the same position.
- `'CollapseDelimiters'` controls whether consecutive delimiters generate empty substrings. The default
  is `false` when you specify explicit delimiters and `true` when you rely on the whitespace default.
- `'IncludeDelimiters'` inserts the matched delimiters as separate elements in the output string array.
- Outputs are string arrays. For scalar inputs, the result is a row vector. For string/character arrays,
  the first dimension matches the number of rows in the input and additional columns are appended to
  accommodate the longest token list. Missing values are padded with `<missing>`.
- Missing string scalars propagate unchanged.

## `split` Function GPU Execution Behaviour
`split` executes on the CPU. When the input or delimiter arguments reside on the GPU, RunMat gathers
them to host memory before performing the split so the results match MATLAB exactly. Providers do not
need to implement custom kernels for this builtin today.

## GPU residency in RunMat (Do I need `gpuArray`?)
String manipulation currently runs on the host. If text data lives on the GPU (for example after a
gathered computation), `split` automatically fetches it. You never need to move text explicitly before
calling this builtin.

## Examples of using the `split` function in MATLAB / RunMat

### Split A String On Whitespace
```matlab
txt = "RunMat Accelerate Planner";
pieces = split(txt);
```
Expected output:
```matlab
pieces = 1×3 string
    "RunMat"    "Accelerate"    "Planner"
```

### Split A String Using A Custom Delimiter
```matlab
csv = "alpha,beta,gamma";
tokens = split(csv, ",");
```
Expected output:
```matlab
tokens = 1×3 string
    "alpha"    "beta"    "gamma"
```

### Include Delimiters In The Output
```matlab
expr = "A+B-C";
segments = split(expr, ["+", "-"], "IncludeDelimiters", true);
```
Expected output:
```matlab
segments = 1×5 string
    "A"    "+"    "B"    "-"    "C"
```

### Preserve Empty Segments When CollapseDelimiters Is False
```matlab
values = "one,,three,";
parts = split(values, ",", "CollapseDelimiters", false);
```
Expected output:
```matlab
parts = 1×4 string
    "one"    ""    "three"    ""
```

### Split Each Row Of A Character Array
```matlab
rows = char("GPU Accelerate", "Ignition Interpreter");
result = split(rows);
```
Expected output:
```matlab
result = 2×2 string
    "GPU"          "Accelerate"
    "Ignition"     "Interpreter"
```

### Split Elements Of A Cell Array
```matlab
C = {'RunMat Snapshot'; "Fusion Planner"};
out = split(C, " ");
```
Expected output:
```matlab
out = 2×2 string
    "RunMat"    "Snapshot"
    "Fusion"    "Planner"
```

### Handle Missing String Inputs
```matlab
names = ["RunMat", "<missing>", "Accelerate Engine"];
split_names = split(names);
```
Expected output:
```matlab
split_names = 3×2 string
    "RunMat"        "<missing>"
    "<missing>"     "<missing>"
    "Accelerate"    "Engine"
```

## FAQ

### What delimiters does `split` use by default?
When you omit the second argument, `split` treats any Unicode whitespace as a delimiter and collapses
consecutive whitespace runs so they produce a single split point.

### How do explicit delimiters change the defaults?
Providing explicit delimiters switches the default for `'CollapseDelimiters'` to `false`, matching MATLAB.
You can override that behaviour with a name-value pair.

### What happens when `'IncludeDelimiters'` is `true`?
Matched delimiters are inserted between tokens in the output string array, preserving their original
order. Tokens still expand to fill rows and columns, with missing values used for padding.

### How is the output sized for string arrays?
The number of rows matches the input. Columns are added to accommodate the longest token list observed
across all elements. Shorter rows are padded with `<missing>`.

### How does `split` handle missing strings?
Missing string scalars propagate unchanged. When padding is required, `<missing>` is used so MATLAB and
RunMat stay aligned.

### Can I provide empty delimiters?
No. Empty delimiters are disallowed, matching MATLAB's input validation. Specify at least one character
per delimiter.

### Which argument types are accepted as delimiters?
You may pass string scalars, string arrays, character arrays (each row is a delimiter), or cell arrays
containing string scalars or character vectors.

## See Also
[strsplit](../../search/strsplit), [replace](./replace), [lower](./lower), [upper](./upper), [strip](./strip)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/split.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/split.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "split",
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
    notes: "Executes on the CPU; GPU-resident inputs are gathered to host memory before splitting.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "split",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion planning and always gathers GPU inputs.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("split", DOC_MD);

const ARG_TYPE_ERROR: &str =
    "split: first argument must be a string scalar, string array, character array, or cell array of character vectors";
const DELIMITER_TYPE_ERROR: &str =
    "split: delimiter input must be a string scalar, string array, character array, or cell array of character vectors";
const NAME_VALUE_PAIR_ERROR: &str = "split: name-value arguments must be supplied in pairs";
const UNKNOWN_NAME_ERROR: &str =
    "split: unrecognized name-value argument; supported names are 'CollapseDelimiters' and 'IncludeDelimiters'";
const EMPTY_DELIMITER_ERROR: &str = "split: delimiters must contain at least one character";
const CELL_ELEMENT_ERROR: &str =
    "split: cell array elements must be string scalars or character vectors";

#[runtime_builtin(
    name = "split",
    category = "strings/transform",
    summary = "Split strings, character arrays, and cell arrays into substrings using delimiters.",
    keywords = "split,strsplit,delimiter,CollapseDelimiters,IncludeDelimiters",
    accel = "sink"
)]
fn split_builtin(text: Value, rest: Vec<Value>) -> Result<Value, String> {
    let text = gather_if_needed(&text).map_err(|e| format!("split: {e}"))?;
    let mut args: Vec<Value> = Vec::with_capacity(rest.len());
    for arg in rest {
        args.push(gather_if_needed(&arg).map_err(|e| format!("split: {e}"))?);
    }

    let options = SplitOptions::parse(&args)?;
    let matrix = TextMatrix::from_value(text)?;
    matrix.into_split_result(&options)
}

#[derive(Clone)]
enum DelimiterSpec {
    Whitespace,
    Patterns(Vec<String>),
}

#[derive(Clone)]
struct SplitOptions {
    delimiters: DelimiterSpec,
    collapse_delimiters: bool,
    include_delimiters: bool,
}

impl SplitOptions {
    fn parse(args: &[Value]) -> Result<Self, String> {
        let mut index = 0usize;
        let mut delimiters = DelimiterSpec::Whitespace;

        if index < args.len() && !is_name_key(&args[index]) {
            let list = extract_delimiters(&args[index])?;
            if list.is_empty() {
                return Err(EMPTY_DELIMITER_ERROR.to_string());
            }
            let mut seen = HashSet::new();
            let mut patterns: Vec<String> = Vec::new();
            for pattern in list {
                if pattern.is_empty() {
                    return Err(EMPTY_DELIMITER_ERROR.to_string());
                }
                if seen.insert(pattern.clone()) {
                    patterns.push(pattern);
                }
            }
            patterns.sort_by_key(|pat| std::cmp::Reverse(pat.len()));
            delimiters = DelimiterSpec::Patterns(patterns);
            index += 1;
        }

        let mut collapse = match delimiters {
            DelimiterSpec::Whitespace => true,
            DelimiterSpec::Patterns(_) => false,
        };
        let mut include = false;

        while index < args.len() {
            let name = match name_key(&args[index]) {
                Some(NameKey::CollapseDelimiters) => NameKey::CollapseDelimiters,
                Some(NameKey::IncludeDelimiters) => NameKey::IncludeDelimiters,
                None => return Err(UNKNOWN_NAME_ERROR.to_string()),
            };
            index += 1;
            if index >= args.len() {
                return Err(NAME_VALUE_PAIR_ERROR.to_string());
            }
            let value = &args[index];
            index += 1;

            match name {
                NameKey::CollapseDelimiters => {
                    collapse = parse_bool(value, "CollapseDelimiters")?;
                }
                NameKey::IncludeDelimiters => {
                    include = parse_bool(value, "IncludeDelimiters")?;
                }
            }
        }

        Ok(Self {
            delimiters,
            collapse_delimiters: collapse,
            include_delimiters: include,
        })
    }
}

struct TextMatrix {
    data: Vec<String>,
    rows: usize,
    cols: usize,
}

impl TextMatrix {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::String(text) => Ok(Self {
                data: vec![text],
                rows: 1,
                cols: 1,
            }),
            Value::StringArray(array) => Ok(Self {
                data: array.data,
                rows: array.rows,
                cols: array.cols,
            }),
            Value::CharArray(array) => Self::from_char_array(array),
            Value::Cell(cell) => Self::from_cell_array(cell),
            _ => Err(ARG_TYPE_ERROR.to_string()),
        }
    }

    fn from_char_array(array: CharArray) -> Result<Self, String> {
        let CharArray { data, rows, cols } = array;
        if rows == 0 {
            return Ok(Self {
                data: Vec::new(),
                rows: 0,
                cols: 1,
            });
        }
        let mut strings = Vec::with_capacity(rows);
        for row in 0..rows {
            strings.push(char_row_to_string_slice(&data, cols, row));
        }
        Ok(Self {
            data: strings,
            rows,
            cols: 1,
        })
    }

    fn from_cell_array(cell: CellArray) -> Result<Self, String> {
        let CellArray {
            data, rows, cols, ..
        } = cell;
        let mut strings = Vec::with_capacity(data.len());
        for col in 0..cols {
            for row in 0..rows {
                let idx = row * cols + col;
                let value_ref: &Value = &data[idx];
                strings.push(
                    cell_element_to_string(value_ref)
                        .ok_or_else(|| CELL_ELEMENT_ERROR.to_string())?,
                );
            }
        }
        Ok(Self {
            data: strings,
            rows,
            cols,
        })
    }

    fn into_split_result(self, options: &SplitOptions) -> Result<Value, String> {
        let TextMatrix { data, rows, cols } = self;

        if data.is_empty() {
            let block_cols = if cols == 0 { 0 } else { 1 };
            let shape = if cols == 0 {
                vec![rows, 0]
            } else {
                vec![rows, cols * block_cols]
            };
            let array = StringArray::new(Vec::new(), shape).map_err(|e| format!("split: {e}"))?;
            return Ok(Value::StringArray(array));
        }

        let mut per_element: Vec<Vec<String>> = Vec::with_capacity(data.len());
        let mut max_tokens = 0usize;
        for text in &data {
            let tokens = split_text(text, options);
            max_tokens = max_tokens.max(tokens.len());
            per_element.push(tokens);
        }
        if max_tokens == 0 {
            max_tokens = 1;
        }
        let block_cols = max_tokens;
        let result_cols = block_cols * cols.max(1);
        let total = rows * result_cols;
        let missing = "<missing>".to_string();
        let mut output = vec![missing.clone(); total];

        for col in 0..cols.max(1) {
            for row in 0..rows {
                let element_index = if cols == 0 { row } else { row + col * rows };
                if element_index >= per_element.len() {
                    continue;
                }
                let tokens = &per_element[element_index];
                for t in 0..block_cols {
                    let out_col = if cols == 0 { t } else { col * block_cols + t };
                    let out_index = row + out_col * rows;
                    if out_index >= output.len() {
                        continue;
                    }
                    if t < tokens.len() {
                        output[out_index] = tokens[t].clone();
                    } else {
                        output[out_index] = missing.clone();
                    }
                }
            }
        }

        let shape = vec![rows, result_cols];
        let array = StringArray::new(output, shape).map_err(|e| format!("split: {e}"))?;
        Ok(Value::StringArray(array))
    }
}

fn split_text(text: &str, options: &SplitOptions) -> Vec<String> {
    if is_missing_string(text) {
        return vec![text.to_string()];
    }
    match &options.delimiters {
        DelimiterSpec::Whitespace => split_whitespace(text, options),
        DelimiterSpec::Patterns(patterns) => split_by_patterns(text, patterns, options),
    }
}

fn split_whitespace(text: &str, options: &SplitOptions) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut parts: Vec<String> = Vec::new();
    let mut idx = 0usize;
    let mut last = 0usize;
    let len = text.len();

    while idx < len {
        let ch = text[idx..].chars().next().unwrap();
        let width = ch.len_utf8();
        if !ch.is_whitespace() {
            idx += width;
            continue;
        }

        let token = &text[last..idx];
        if !token.is_empty() || !options.collapse_delimiters {
            parts.push(token.to_string());
        }

        let run_end = advance_whitespace(text, idx);
        if options.include_delimiters {
            if options.collapse_delimiters {
                parts.push(text[idx..run_end].to_string());
            } else {
                parts.push(text[idx..idx + width].to_string());
            }
        }

        if options.collapse_delimiters {
            idx = run_end;
            last = run_end;
        } else {
            idx += width;
            last = idx;
        }
    }

    let tail = &text[last..];
    if !tail.is_empty() || !options.collapse_delimiters {
        parts.push(tail.to_string());
    }
    if parts.is_empty() {
        parts.push(String::new());
    }
    parts
}

fn split_by_patterns(text: &str, patterns: &[String], options: &SplitOptions) -> Vec<String> {
    if patterns.is_empty() {
        return vec![text.to_string()];
    }

    let mut parts: Vec<String> = Vec::new();
    let mut idx = 0usize;
    let mut last = 0usize;
    while idx < text.len() {
        if let Some(pattern) = patterns
            .iter()
            .find(|candidate| text[idx..].starts_with(candidate.as_str()))
        {
            let token = &text[last..idx];
            if !token.is_empty() || !options.collapse_delimiters {
                parts.push(token.to_string());
            }

            let pat_len = pattern.len();
            if options.collapse_delimiters {
                let mut run_end = idx + pat_len;
                while run_end < text.len() {
                    if let Some(next) = patterns
                        .iter()
                        .find(|candidate| text[run_end..].starts_with(candidate.as_str()))
                    {
                        let len = next.len();
                        if len == 0 {
                            break;
                        }
                        run_end += len;
                    } else {
                        break;
                    }
                }
                if options.include_delimiters {
                    parts.push(text[idx..run_end].to_string());
                }
                idx = run_end;
                last = run_end;
            } else {
                if options.include_delimiters {
                    parts.push(text[idx..idx + pat_len].to_string());
                }
                idx += pat_len;
                last = idx;
            }

            continue;
        }
        let ch = text[idx..].chars().next().unwrap();
        idx += ch.len_utf8();
    }
    let tail = &text[last..];
    if !tail.is_empty() || !options.collapse_delimiters {
        parts.push(tail.to_string());
    }
    if parts.is_empty() {
        parts.push(String::new());
    }
    parts
}

fn advance_whitespace(text: &str, mut start: usize) -> usize {
    while start < text.len() {
        let ch = text[start..].chars().next().unwrap();
        if !ch.is_whitespace() {
            break;
        }
        start += ch.len_utf8();
    }
    start
}

fn extract_delimiters(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) => {
            if array.rows == 0 {
                return Ok(Vec::new());
            }
            let mut entries = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                entries.push(char_row_to_string_slice(&array.data, array.cols, row));
            }
            Ok(entries)
        }
        Value::Cell(cell) => {
            let mut entries = Vec::with_capacity(cell.data.len());
            for element in &cell.data {
                entries.push(
                    cell_element_to_string(element)
                        .ok_or_else(|| CELL_ELEMENT_ERROR.to_string())?,
                );
            }
            Ok(entries)
        }
        _ => Err(DELIMITER_TYPE_ERROR.to_string()),
    }
}

fn cell_element_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&array.data, array.cols, 0))
            }
        }
        _ => None,
    }
}

fn value_to_scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows <= 1 => {
            if array.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&array.data, array.cols, 0))
            }
        }
        Value::Cell(cell) if cell.data.len() == 1 => cell_element_to_string(&cell.data[0]),
        _ => None,
    }
}

fn parse_bool(value: &Value, name: &str) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(i.to_i64() != 0),
        Value::Num(n) => Ok(*n != 0.0),
        Value::LogicalArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0] != 0)
            } else {
                Err(format!(
                    "split: value for '{}' must be logical true or false",
                    name
                ))
            }
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                Ok(tensor.data[0] != 0.0)
            } else {
                Err(format!(
                    "split: value for '{}' must be logical true or false",
                    name
                ))
            }
        }
        _ => {
            if let Some(text) = value_to_scalar_string(value) {
                let lowered = text.trim().to_ascii_lowercase();
                match lowered.as_str() {
                    "true" | "on" | "yes" => Ok(true),
                    "false" | "off" | "no" => Ok(false),
                    _ => Err(format!(
                        "split: value for '{}' must be logical true or false",
                        name
                    )),
                }
            } else {
                Err(format!(
                    "split: value for '{}' must be logical true or false",
                    name
                ))
            }
        }
    }
}

#[derive(PartialEq, Eq)]
enum NameKey {
    CollapseDelimiters,
    IncludeDelimiters,
}

fn is_name_key(value: &Value) -> bool {
    name_key(value).is_some()
}

fn name_key(value: &Value) -> Option<NameKey> {
    value_to_scalar_string(value).and_then(|text| {
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "collapsedelimiters" => Some(NameKey::CollapseDelimiters),
            "includedelimiters" => Some(NameKey::IncludeDelimiters),
            _ => None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "doc_export")]
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, LogicalArray, Tensor};

    #[test]
    fn split_string_whitespace_default() {
        let input = Value::String("RunMat Accelerate Planner".to_string());
        let result = split_builtin(input, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(
                    array.data,
                    vec![
                        "RunMat".to_string(),
                        "Accelerate".to_string(),
                        "Planner".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_string_custom_delimiter() {
        let input = Value::String("alpha,beta,gamma".to_string());
        let args = vec![Value::String(",".to_string())];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(
                    array.data,
                    vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_include_delimiters_true() {
        let input = Value::String("A+B-C".to_string());
        let args = vec![
            Value::StringArray(
                StringArray::new(vec!["+".to_string(), "-".to_string()], vec![1, 2]).unwrap(),
            ),
            Value::String("IncludeDelimiters".to_string()),
            Value::Bool(true),
        ];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 5]);
                assert_eq!(
                    array.data,
                    vec![
                        "A".to_string(),
                        "+".to_string(),
                        "B".to_string(),
                        "-".to_string(),
                        "C".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_include_delimiters_whitespace_collapse_default() {
        let input = Value::String("A  B".to_string());
        let args = vec![
            Value::String("IncludeDelimiters".to_string()),
            Value::Bool(true),
        ];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(
                    array.data,
                    vec!["A".to_string(), "  ".to_string(), "B".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_patterns_include_delimiters_collapse_true() {
        let input = Value::String("a,,b".to_string());
        let args = vec![
            Value::String(",".to_string()),
            Value::String("IncludeDelimiters".to_string()),
            Value::Bool(true),
            Value::String("CollapseDelimiters".to_string()),
            Value::Bool(true),
        ];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(
                    array.data,
                    vec!["a".to_string(), ",,".to_string(), "b".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_collapse_false_preserves_empty_segments() {
        let input = Value::String("one,,three,".to_string());
        let args = vec![
            Value::String(",".to_string()),
            Value::String("CollapseDelimiters".to_string()),
            Value::Bool(false),
        ];
        let result = split_builtin(input, args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 4]);
                assert_eq!(
                    array.data,
                    vec![
                        "one".to_string(),
                        "".to_string(),
                        "three".to_string(),
                        "".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_character_array_rows() {
        let mut row1: Vec<char> = "GPU Accelerate".chars().collect();
        let mut row2: Vec<char> = "Ignition Engine".chars().collect();
        let width = row1.len().max(row2.len());
        row1.resize(width, ' ');
        row2.resize(width, ' ');
        let mut data = row1;
        data.extend(row2);
        let char_array = CharArray::new(data, 2, width).unwrap();
        let input = Value::CharArray(char_array);
        let result = split_builtin(input, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(
                    array.data,
                    vec![
                        "GPU".to_string(),
                        "Ignition".to_string(),
                        "Accelerate".to_string(),
                        "Engine".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_string_array_multiple_columns() {
        let data = vec![
            "RunMat Core".to_string(),
            "Ignition Interpreter".to_string(),
            "Accelerate Engine".to_string(),
            "<missing>".to_string(),
        ];
        let array = StringArray::new(data, vec![2, 2]).unwrap();
        let input = Value::StringArray(array);
        let result = split_builtin(input, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 4]);
                assert_eq!(
                    array.data,
                    vec![
                        "RunMat".to_string(),
                        "Ignition".to_string(),
                        "Core".to_string(),
                        "Interpreter".to_string(),
                        "Accelerate".to_string(),
                        "<missing>".to_string(),
                        "Engine".to_string(),
                        "<missing>".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_cell_array_outputs_string_array() {
        let values = vec![
            Value::String("RunMat Snapshot".to_string()),
            Value::String("Fusion Planner".to_string()),
        ];
        let cell = crate::make_cell(values, 2, 1).expect("cell");
        let result = split_builtin(cell, vec![Value::String(" ".to_string())]).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(
                    array.data,
                    vec![
                        "RunMat".to_string(),
                        "Fusion".to_string(),
                        "Snapshot".to_string(),
                        "Planner".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_cell_array_multiple_columns() {
        let values = vec![
            Value::String("alpha beta".to_string()),
            Value::String("gamma".to_string()),
            Value::String("delta epsilon".to_string()),
            Value::String("<missing>".to_string()),
        ];
        let cell = crate::make_cell(values, 2, 2).expect("cell");
        let result = split_builtin(cell, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 4]);
                assert_eq!(
                    array.data,
                    vec![
                        "alpha".to_string(),
                        "delta".to_string(),
                        "beta".to_string(),
                        "epsilon".to_string(),
                        "gamma".to_string(),
                        "<missing>".to_string(),
                        "<missing>".to_string(),
                        "<missing>".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_missing_string_propagates() {
        let input = Value::String("<missing>".to_string());
        let result = split_builtin(input, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 1]);
                assert_eq!(array.data, vec!["<missing>".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_invalid_name_value_pair_errors() {
        let input = Value::String("abc".to_string());
        let args = vec![Value::String("CollapseDelimiters".to_string())];
        let err = split_builtin(input, args).unwrap_err();
        assert!(err.contains("name-value"));
    }

    #[test]
    fn split_invalid_text_argument_errors() {
        let err = split_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(err.contains("first argument"));
    }

    #[test]
    fn split_invalid_delimiter_type_errors() {
        let err =
            split_builtin(Value::String("abc".to_string()), vec![Value::Num(1.0)]).unwrap_err();
        assert!(err.contains("delimiter input"));
    }

    #[test]
    fn split_empty_delimiter_errors() {
        let err = split_builtin(
            Value::String("abc".to_string()),
            vec![Value::String(String::new())],
        )
        .unwrap_err();
        assert!(err.contains("at least one character"));
    }

    #[test]
    fn split_unknown_name_argument_errors() {
        let err = split_builtin(
            Value::String("abc".to_string()),
            vec![
                Value::String("UnknownOption".to_string()),
                Value::Bool(true),
            ],
        )
        .unwrap_err();
        assert!(err.contains("unrecognized"));
    }

    #[test]
    fn split_collapse_delimiters_accepts_logical_array() {
        let logical = LogicalArray::new(vec![1u8], vec![1]).unwrap();
        let args = vec![
            Value::String(",".to_string()),
            Value::String("CollapseDelimiters".to_string()),
            Value::LogicalArray(logical),
        ];
        let result = split_builtin(Value::String("a,,b".to_string()), args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 2]);
                assert_eq!(array.data, vec!["a".to_string(), "b".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_include_delimiters_accepts_tensor_scalar() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let args = vec![
            Value::String(",".to_string()),
            Value::String("IncludeDelimiters".to_string()),
            Value::Tensor(tensor),
        ];
        let result = split_builtin(Value::String("a,b".to_string()), args).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(
                    array.data,
                    vec!["a".to_string(), ",".to_string(), "b".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn split_cell_array_mixed_inputs() {
        let handles: Vec<_> = vec![
            runmat_gc::gc_allocate(Value::String("alpha beta".to_string())).unwrap(),
            runmat_gc::gc_allocate(Value::CharArray(
                CharArray::new("gamma".chars().collect(), 1, 5).unwrap(),
            ))
            .unwrap(),
        ];
        let cell =
            Value::Cell(CellArray::new_handles(handles, 1, 2).expect("cell array construction"));
        let result = split_builtin(cell, Vec::new()).expect("split");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 4]);
                assert_eq!(
                    array.data,
                    vec![
                        "alpha".to_string(),
                        "beta".to_string(),
                        "gamma".to_string(),
                        "<missing>".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
