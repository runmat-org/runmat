//! MATLAB-compatible `pad` builtin with GPU-aware semantics for RunMat.

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
        name = "pad",
        builtin_path = "crate::builtins::strings::transform::pad"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "pad"
category: "strings/transform"
keywords: ["pad", "pad string", "left pad", "right pad", "center text", "character arrays"]
summary: "Pad strings, character arrays, and cell arrays to a target length using MATLAB-compatible options."
references:
  - https://www.mathworks.com/help/matlab/ref/pad.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Executes on the CPU; GPU-resident inputs are gathered before padding so behaviour matches MATLAB."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::pad::tests"
  integration: "builtins::strings::transform::pad::tests::pad_cell_array_mixed_content"
---

# What does the `pad` function do in MATLAB / RunMat?
`pad` adds characters to the beginning, end, or both sides of strings so that each element reaches a
specified length. It mirrors MATLAB semantics for string arrays, character arrays, and cell arrays of
character vectors, including direction keywords, default space padding, and optional custom characters.

## How does the `pad` function behave in MATLAB / RunMat?
- Without a target length, `pad` extends each element to match the longest text in the input.
- Providing a numeric target guarantees a minimum length; existing text that already meets or exceeds
  the target is returned unchanged.
- Direction keywords (`'left'`, `'right'`, `'both'`) are case-insensitive; `'right'` is the default.
  When an odd number of pad characters is required for `'both'`, the extra character is appended to the end.
- `padChar` must be a single character (string scalar or 1×1 char array). The default is a space.
- Character arrays remain rectangular. Each row is padded independently and then widened with spaces so
  the array keeps MATLAB’s column-major layout.
- Cell arrays preserve their structure. Elements must be string scalars or 1×N character vectors and are
  padded while keeping their original type.
- Missing strings (`string(missing)`) and empty character vectors pass through unchanged, preserving metadata.

## `pad` Function GPU Execution Behaviour
`pad` always executes on the CPU. When an argument (or a value nested inside a cell array) lives on the GPU,
RunMat gathers it, performs the padding step, and produces a host result or re-wraps the padded value inside
the cell. No provider hooks exist yet for string padding, so providers and fusion planners treat `pad` as a
sink that terminates device residency.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. Text data in RunMat lives on the host today. If text happens to originate from a GPU computation,
`pad` automatically gathers it before padding, so you never have to manage residency manually for this
builtin.

## Examples of using the `pad` function in MATLAB / RunMat

### Pad Strings To A Common Width
```matlab
labels = ["GPU"; "Accelerate"; "RunMat"];
aligned = pad(labels);
```
Expected output:
```matlab
aligned =
  3×1 string
    "GPU       "
    "Accelerate"
    "RunMat    "
```

### Pad Strings On The Left With Zeros
```matlab
ids = ["42"; "7"; "512"];
zero_padded = pad(ids, 4, 'left', '0');
```
Expected output:
```matlab
zero_padded =
  3×1 string
    "0042"
    "0007"
    "0512"
```

### Center Text With Both-Sided Padding
```matlab
titles = ["core"; "planner"];
centered = pad(titles, 10, 'both', '*');
```
Expected output:
```matlab
centered =
  2×1 string
    "***core***"
    "*planner**"
```

### Pad Character Array Rows
```matlab
chars = char("GPU", "RunMat");
out = pad(chars, 8);
```
Expected output:
```matlab
out =

  2×8 char array

    'GPU     '
    'RunMat  '
```

### Pad A Cell Array Of Character Vectors
```matlab
C = {'solver', "planner", 'jit'};
cell_out = pad(C, 'right', '.');
```
Expected output:
```matlab
cell_out = 1×3 cell array
    {'solver.'}    {"planner"}    {'jit....'}
```

### Leave Missing Strings Unchanged
```matlab
values = ["RunMat", "<missing>", "GPU"];
kept = pad(values, 8);
```
Expected output:
```matlab
kept =
  1×3 string
    "RunMat  "    <missing>    "GPU     "
```

## FAQ

### What inputs does `pad` accept?
String scalars, string arrays, character arrays, and cell arrays containing string scalars or character
vectors. Other types raise MATLAB-compatible errors.

### How are direction keywords interpreted?
`'left'`, `'right'`, and `'both'` are supported (case-insensitive). `'right'` is the default. With `'both'`,
extra characters are added to the end when an odd number of padding characters is required.

### Can I shorten text with `pad`?
No. When the existing text is already longer than the requested target length, it is returned unchanged.

### What happens when I supply a custom padding character?
The character must be length one. RunMat repeats it as many times as needed in the specified direction.

### Do missing strings get padded?
Missing strings (`<missing>`) are passed through untouched so downstream code that checks for missing
values continues to work.

### How are cell array elements returned?
Each cell retains its type: string scalars remain strings and character vectors remain 1×N character
arrays after padding.

### Does `pad` change the orientation of row or column string arrays?
No. The shape of the input array is preserved exactly; only element lengths change.

### Will `pad` run on the GPU in the future?
Possibly, but today it always gathers to the CPU. Providers may add device-side implementations later,
and the behaviour documented here will remain the reference.

## See Also
[strip](./strip), [strcat](./strcat), [lower](./lower), [upper](./upper), [compose](./compose)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/pad.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/pad.rs)
- Found an issue? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::pad")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pad",
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
    notes: "Executes on the CPU; GPU-resident inputs are gathered before padding to preserve MATLAB semantics.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::pad")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pad",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; always gathers inputs and is not eligible for fusion.",
};

const ARG_TYPE_ERROR: &str =
    "pad: first argument must be a string array, character array, or cell array of character vectors";
const LENGTH_ERROR: &str = "pad: target length must be a non-negative integer scalar";
const DIRECTION_ERROR: &str = "pad: direction must be 'left', 'right', or 'both'";
const PAD_CHAR_ERROR: &str =
    "pad: padding character must be a string scalar or character vector containing one character";
const CELL_ELEMENT_ERROR: &str =
    "pad: cell array elements must be string scalars or character vectors";
const ARGUMENT_CONFIG_ERROR: &str = "pad: unable to interpret input arguments";

#[derive(Clone, Copy, Eq, PartialEq)]
enum PadDirection {
    Left,
    Right,
    Both,
}

#[derive(Clone, Copy)]
enum PadTarget {
    Auto,
    Length(usize),
}

#[derive(Clone, Copy)]
struct PadOptions {
    target: PadTarget,
    direction: PadDirection,
    pad_char: char,
}

impl Default for PadOptions {
    fn default() -> Self {
        Self {
            target: PadTarget::Auto,
            direction: PadDirection::Right,
            pad_char: ' ',
        }
    }
}

impl PadOptions {
    fn base_target(&self, auto_target: usize) -> usize {
        match self.target {
            PadTarget::Auto => auto_target,
            PadTarget::Length(len) => len,
        }
    }
}

#[runtime_builtin(
    name = "pad",
    category = "strings/transform",
    summary = "Pad strings, character arrays, and cell arrays to a target length.",
    keywords = "pad,align,strings,character array",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::pad"
)]
fn pad_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let options = parse_arguments(&rest)?;
    let gathered = gather_if_needed(&value).map_err(|e| format!("pad: {e}"))?;
    match gathered {
        Value::String(text) => pad_string(text, options),
        Value::StringArray(array) => pad_string_array(array, options),
        Value::CharArray(array) => pad_char_array(array, options),
        Value::Cell(cell) => pad_cell_array(cell, options),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

fn pad_string(text: String, options: PadOptions) -> Result<Value, String> {
    if is_missing_string(&text) {
        return Ok(Value::String(text));
    }
    let char_count = string_length(&text);
    let base_target = options.base_target(char_count);
    let target_len = element_target_length(&options, base_target, char_count);
    let padded = apply_padding_owned(text, char_count, target_len, &options);
    Ok(Value::String(padded))
}

fn pad_string_array(array: StringArray, options: PadOptions) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let mut auto_len: usize = 0;
    if matches!(options.target, PadTarget::Auto) {
        for text in &data {
            if !is_missing_string(text) {
                auto_len = auto_len.max(string_length(text));
            }
        }
    }
    let base_target = options.base_target(auto_len);
    let mut padded: Vec<String> = Vec::with_capacity(data.len());
    for text in data.into_iter() {
        if is_missing_string(&text) {
            padded.push(text);
            continue;
        }
        let char_count = string_length(&text);
        let target_len = element_target_length(&options, base_target, char_count);
        let new_text = apply_padding_owned(text, char_count, target_len, &options);
        padded.push(new_text);
    }
    let result = StringArray::new(padded, shape).map_err(|e| format!("pad: {e}"))?;
    Ok(Value::StringArray(result))
}

fn pad_char_array(array: CharArray, options: PadOptions) -> Result<Value, String> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut rows_text: Vec<String> = Vec::with_capacity(rows);
    let mut auto_len = 0usize;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row);
        auto_len = auto_len.max(string_length(&text));
        rows_text.push(text);
    }

    let base_target = options.base_target(auto_len);
    let mut padded_rows: Vec<String> = Vec::with_capacity(rows);
    let mut final_cols: usize = 0;
    for row_text in rows_text.into_iter() {
        let char_count = string_length(&row_text);
        let target_len = element_target_length(&options, base_target, char_count);
        let padded = apply_padding_owned(row_text, char_count, target_len, &options);
        final_cols = final_cols.max(string_length(&padded));
        padded_rows.push(padded);
    }

    let mut new_data: Vec<char> = Vec::with_capacity(rows * final_cols);
    for row_text in padded_rows.into_iter() {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < final_cols {
            chars.resize(final_cols, ' ');
        }
        new_data.extend(chars.into_iter());
    }

    CharArray::new(new_data, rows, final_cols)
        .map(Value::CharArray)
        .map_err(|e| format!("pad: {e}"))
}

fn pad_cell_array(cell: CellArray, options: PadOptions) -> Result<Value, String> {
    let rows = cell.rows;
    let cols = cell.cols;
    let total = rows * cols;
    let mut items: Vec<CellItem> = Vec::with_capacity(total);
    let mut auto_len = 0usize;

    for idx in 0..total {
        let value = &cell.data[idx];
        let gathered = gather_if_needed(value).map_err(|e| format!("pad: {e}"))?;
        let item = match gathered {
            Value::String(text) => {
                let is_missing = is_missing_string(&text);
                let len = if is_missing { 0 } else { string_length(&text) };
                if !is_missing {
                    auto_len = auto_len.max(len);
                }
                CellItem {
                    kind: CellKind::String,
                    text,
                    char_count: len,
                    is_missing,
                }
            }
            Value::StringArray(sa) if sa.data.len() == 1 => {
                let text = sa.data.into_iter().next().unwrap_or_default();
                let is_missing = is_missing_string(&text);
                let len = if is_missing { 0 } else { string_length(&text) };
                if !is_missing {
                    auto_len = auto_len.max(len);
                }
                CellItem {
                    kind: CellKind::String,
                    text,
                    char_count: len,
                    is_missing,
                }
            }
            Value::CharArray(ca) if ca.rows <= 1 => {
                let text = if ca.rows == 0 {
                    String::new()
                } else {
                    char_row_to_string_slice(&ca.data, ca.cols, 0)
                };
                let len = string_length(&text);
                auto_len = auto_len.max(len);
                CellItem {
                    kind: CellKind::Char { rows: ca.rows },
                    text,
                    char_count: len,
                    is_missing: false,
                }
            }
            Value::CharArray(_) => return Err(CELL_ELEMENT_ERROR.to_string()),
            _ => return Err(CELL_ELEMENT_ERROR.to_string()),
        };
        items.push(item);
    }

    let base_target = options.base_target(auto_len);
    let mut results: Vec<Value> = Vec::with_capacity(total);
    for item in items.into_iter() {
        if item.is_missing {
            results.push(Value::String(item.text));
            continue;
        }
        let target_len = element_target_length(&options, base_target, item.char_count);
        let padded = apply_padding_owned(item.text, item.char_count, target_len, &options);
        match item.kind {
            CellKind::String => results.push(Value::String(padded)),
            CellKind::Char { rows } => {
                let chars: Vec<char> = padded.chars().collect();
                let cols = chars.len();
                let array = CharArray::new(chars, rows, cols).map_err(|e| format!("pad: {e}"))?;
                results.push(Value::CharArray(array));
            }
        }
    }

    make_cell(results, rows, cols).map_err(|e| format!("pad: {e}"))
}

#[derive(Clone)]
struct CellItem {
    kind: CellKind,
    text: String,
    char_count: usize,
    is_missing: bool,
}

#[derive(Clone)]
enum CellKind {
    String,
    Char { rows: usize },
}

fn parse_arguments(args: &[Value]) -> Result<PadOptions, String> {
    let mut options = PadOptions::default();
    match args.len() {
        0 => Ok(options),
        1 => {
            if let Some(length) = parse_length(&args[0])? {
                options.target = PadTarget::Length(length);
                return Ok(options);
            }
            if let Some(direction) = try_parse_direction(&args[0], false)? {
                options.direction = direction;
                return Ok(options);
            }
            let pad_char = parse_pad_char(&args[0])?;
            options.pad_char = pad_char;
            Ok(options)
        }
        2 => {
            if let Some(length) = parse_length(&args[0])? {
                options.target = PadTarget::Length(length);
                if let Some(direction) = try_parse_direction(&args[1], false)? {
                    options.direction = direction;
                } else {
                    match parse_pad_char(&args[1]) {
                        Ok(pad_char) => options.pad_char = pad_char,
                        Err(_) => return Err(DIRECTION_ERROR.to_string()),
                    }
                }
                Ok(options)
            } else if let Some(direction) = try_parse_direction(&args[0], false)? {
                options.direction = direction;
                let pad_char = parse_pad_char(&args[1])?;
                options.pad_char = pad_char;
                Ok(options)
            } else {
                Err(ARGUMENT_CONFIG_ERROR.to_string())
            }
        }
        3 => {
            let length = parse_length(&args[0])?.ok_or_else(|| LENGTH_ERROR.to_string())?;
            let direction =
                try_parse_direction(&args[1], true)?.ok_or_else(|| DIRECTION_ERROR.to_string())?;
            let pad_char = parse_pad_char(&args[2])?;
            options.target = PadTarget::Length(length);
            options.direction = direction;
            options.pad_char = pad_char;
            Ok(options)
        }
        _ => Err("pad: too many input arguments".to_string()),
    }
}

fn parse_length(value: &Value) -> Result<Option<usize>, String> {
    match value {
        Value::Num(n) => {
            if !n.is_finite() || *n < 0.0 {
                return Err(LENGTH_ERROR.to_string());
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(LENGTH_ERROR.to_string());
            }
            Ok(Some(*n as usize))
        }
        Value::Int(i) => {
            let val = i.to_i64();
            if val < 0 {
                return Err(LENGTH_ERROR.to_string());
            }
            Ok(Some(val as usize))
        }
        _ => Ok(None),
    }
}

fn try_parse_direction(value: &Value, strict: bool) -> Result<Option<PadDirection>, String> {
    let Some(text) = value_to_single_string(value) else {
        return if strict {
            Err(DIRECTION_ERROR.to_string())
        } else {
            Ok(None)
        };
    };
    let lowered = text.trim().to_ascii_lowercase();
    if lowered.is_empty() {
        return if strict {
            Err(DIRECTION_ERROR.to_string())
        } else {
            Ok(None)
        };
    }
    let direction = match lowered.as_str() {
        "left" => PadDirection::Left,
        "right" => PadDirection::Right,
        "both" => PadDirection::Both,
        _ => {
            return if strict {
                Err(DIRECTION_ERROR.to_string())
            } else {
                Ok(None)
            };
        }
    };
    Ok(Some(direction))
}

fn parse_pad_char(value: &Value) -> Result<char, String> {
    let text = value_to_single_string(value).ok_or_else(|| PAD_CHAR_ERROR.to_string())?;
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return Err(PAD_CHAR_ERROR.to_string());
    };
    if chars.next().is_some() {
        return Err(PAD_CHAR_ERROR.to_string());
    }
    Ok(first)
}

fn value_to_single_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Some(sa.data[0].clone())
            } else {
                None
            }
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&ca.data, ca.cols, 0))
            }
        }
        _ => None,
    }
}

fn string_length(text: &str) -> usize {
    text.chars().count()
}

fn element_target_length(options: &PadOptions, base_target: usize, current_len: usize) -> usize {
    match options.target {
        PadTarget::Auto => base_target.max(current_len),
        PadTarget::Length(_) => base_target.max(current_len),
    }
}

fn apply_padding_owned(
    text: String,
    current_len: usize,
    target_len: usize,
    options: &PadOptions,
) -> String {
    if current_len >= target_len {
        return text;
    }
    let delta = target_len - current_len;
    let (left_pad, right_pad) = match options.direction {
        PadDirection::Left => (delta, 0),
        PadDirection::Right => (0, delta),
        PadDirection::Both => {
            let left = delta / 2;
            (left, delta - left)
        }
    };
    let mut result = String::with_capacity(text.len() + delta * options.pad_char.len_utf8());
    for _ in 0..left_pad {
        result.push(options.pad_char);
    }
    result.push_str(&text);
    for _ in 0..right_pad {
        result.push(options.pad_char);
    }
    result
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_length_right() {
        let result = pad_builtin(Value::String("GPU".into()), vec![Value::Num(5.0)]).expect("pad");
        assert_eq!(result, Value::String("GPU  ".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_left_with_custom_char() {
        let result = pad_builtin(
            Value::String("42".into()),
            vec![
                Value::Num(4.0),
                Value::String("left".into()),
                Value::String("0".into()),
            ],
        )
        .expect("pad");
        assert_eq!(result, Value::String("0042".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_both_with_odd_count() {
        let result = pad_builtin(
            Value::String("core".into()),
            vec![
                Value::Num(9.0),
                Value::String("both".into()),
                Value::String("*".into()),
            ],
        )
        .expect("pad");
        assert_eq!(result, Value::String("**core***".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_auto_uses_longest_element() {
        let strings =
            StringArray::new(vec!["GPU".into(), "Accelerate".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(Value::StringArray(strings), Vec::new()).expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "GPU       ");
                assert_eq!(sa.data[1], "Accelerate");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_pad_character_only() {
        let strings = StringArray::new(vec!["A".into(), "Run".into()], vec![2, 1]).unwrap();
        let result =
            pad_builtin(Value::StringArray(strings), vec![Value::String("*".into())]).expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "A**");
                assert_eq!(sa.data[1], "Run");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_length_with_pad_character() {
        let strings = StringArray::new(vec!["7".into(), "512".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(
            Value::StringArray(strings),
            vec![Value::Num(4.0), Value::String("0".into())],
        )
        .expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "7000");
                assert_eq!(sa.data[1], "5120");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_direction_only() {
        let strings =
            StringArray::new(vec!["Mary".into(), "Elizabeth".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(
            Value::StringArray(strings),
            vec![Value::String("left".into())],
        )
        .expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "     Mary");
                assert_eq!(sa.data[1], "Elizabeth");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_single_string_pad_character_only_leaves_length() {
        let result =
            pad_builtin(Value::String("GPU".into()), vec![Value::String("-".into())]).expect("pad");
        assert_eq!(result, Value::String("GPU".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_char_array_resizes_columns() {
        let chars: Vec<char> = "GPUrun".chars().collect();
        let array = CharArray::new(chars, 2, 3).unwrap();
        let result = pad_builtin(Value::CharArray(array), vec![Value::Num(5.0)]).expect("pad");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 5);
                let expected: Vec<char> = "GPU  run  ".chars().collect();
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::String("solver".into()),
                Value::CharArray(CharArray::new_row("jit")),
                Value::String("planner".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = pad_builtin(
            Value::Cell(cell),
            vec![Value::String("right".into()), Value::String(".".into())],
        )
        .expect("pad");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 3);
                assert_eq!(out.get(0, 0).unwrap(), Value::String("solver.".into()));
                assert_eq!(
                    out.get(0, 1).unwrap(),
                    Value::CharArray(CharArray::new_row("jit...."))
                );
                assert_eq!(out.get(0, 2).unwrap(), Value::String("planner".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_preserves_missing_string() {
        let result =
            pad_builtin(Value::String("<missing>".into()), vec![Value::Num(8.0)]).expect("pad");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_input_type() {
        let err = pad_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_negative_length() {
        let err = pad_builtin(Value::String("data".into()), vec![Value::Num(-1.0)]).unwrap_err();
        assert_eq!(err, LENGTH_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_direction() {
        let err = pad_builtin(
            Value::String("data".into()),
            vec![Value::Num(6.0), Value::String("around".into())],
        )
        .unwrap_err();
        assert_eq!(err, DIRECTION_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_pad_character() {
        let err = pad_builtin(
            Value::String("data".into()),
            vec![Value::String("left".into()), Value::String("##".into())],
        )
        .unwrap_err();
        assert_eq!(err, PAD_CHAR_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pad_works_with_wgpu_provider_active() {
        test_support::with_test_provider(|_| {
            let result =
                pad_builtin(Value::String("GPU".into()), vec![Value::Num(6.0)]).expect("pad");
            assert_eq!(result, Value::String("GPU   ".into()));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
