//! MATLAB-compatible `strip` builtin with GPU-aware semantics for RunMat.

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
        name = "strip",
        builtin_path = "crate::builtins::strings::transform::strip"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r###"---
title: "strip"
category: "strings/transform"
keywords: ["strip", "trim", "whitespace", "leading characters", "trailing characters", "character arrays"]
summary: "Remove leading and trailing characters from strings, character arrays, and cell arrays."
references:
  - https://www.mathworks.com/help/matlab/ref/strip.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Executes on the CPU; GPU-resident inputs are gathered before trimming to match MATLAB semantics."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::strip::tests"
  integration: "builtins::strings::transform::strip::tests::strip_cell_array_mixed_content"
---

# What does the `strip` function do in MATLAB / RunMat?
`strip(text)` removes consecutive whitespace characters from the beginning and end of `text`. The
input can be a string scalar, string array, character array, or a cell array of character vectors,
mirroring MATLAB behaviour. Optional arguments let you control which side to trim (`'left'`,
`'right'`, or `'both'`) and provide custom characters to remove instead of whitespace.

## How does the `strip` function behave in MATLAB / RunMat?
- By default, `strip` removes leading and trailing whitespace determined by `isspace`.
- Direction keywords are case-insensitive. `'left'`/`'leading'` trim the beginning, `'right'`/`'trailing'`
  trim the end, and `'both'` removes characters on both sides.
- Provide a second argument containing characters to remove to strip those characters instead of
  whitespace. Supply a scalar string/char vector to apply the same rule to every element or a string /
  cell array matching the input size to specify element-wise character sets.
- Missing string scalars remain `<missing>`.
- Character arrays shrink or retain their width to match the longest stripped row; shorter rows are
  padded with spaces so the output stays rectangular.
- Cell arrays must contain string scalars or character vectors. Results preserve the original cell
  layout with trimmed elements.

## `strip` Function GPU Execution Behaviour
`strip` executes on the CPU. When the input or any nested element resides on the GPU, RunMat gathers
those values to host memory before trimming so the results match MATLAB exactly. Providers do not need
to implement device kernels for this builtin today.

## GPU residency in RunMat (Do I need `gpuArray`?)
Text data typically lives on the host. If you deliberately store text on the GPU (for example, by
keeping character code points in device buffers), RunMat gathers them automatically when `strip` runs.
You do not need to call `gpuArray` or `gather` manually for this builtin.

## Examples of using the `strip` function in MATLAB / RunMat

### Remove Leading And Trailing Spaces From A String Scalar
```matlab
name = "   RunMat   ";
clean = strip(name);
```
Expected output:
```matlab
clean = "RunMat"
```

### Trim Only The Right Side Of Each String
```matlab
labels = ["   Alpha   "; "   Beta     "; "   Gamma    "];
right_stripped = strip(labels, 'right');
```
Expected output:
```matlab
right_stripped = 3×1 string
    "   Alpha"
    "   Beta"
    "   Gamma"
```

### Remove Leading Zeros While Preserving Trailing Digits
```matlab
codes = ["00095"; "00137"; "00420"];
numeric = strip(codes, 'left', '0');
```
Expected output:
```matlab
numeric = 3×1 string
    "95"
    "137"
    "420"
```

### Strip Character Arrays And Preserve Rectangular Shape
```matlab
animals = char("   cat  ", " dog", "cow   ");
trimmed = strip(animals);
```
Expected output:
```matlab
trimmed =

  3×4 char array

    'cat '
    'dog '
    'cow '
```

### Supply Per-Element Characters To Remove
```matlab
metrics = ["##pass##", "--warn--", "**fail**"];
per_char = ["#"; "-"; "*"];
normalized = strip(metrics, 'both', per_char);
```
Expected output:
```matlab
normalized = 3×1 string
    "pass"
    "warn"
    "fail"
```

### Trim Cell Array Elements With Mixed Types
```matlab
pieces = {'  GPU  ', "   Accelerate", 'RunMat   '};
out = strip(pieces);
```
Expected output:
```matlab
out = 1×3 cell array
    {'GPU'}    {"Accelerate"}    {'RunMat'}
```

## FAQ

### Which direction keywords are supported?
`'left'` and `'leading'` trim the beginning of the text, `'right'` and `'trailing'` trim the end, and
`'both'` (the default) trims both sides.

### How do I remove characters other than whitespace?
Provide a second argument containing the characters to remove, for example `strip(str, "xyz")` removes
any leading or trailing `x`, `y`, or `z` characters. Combine it with a direction argument to control
which side is affected.

### Can I specify different characters for each element?
Yes. Pass a string array or cell array of character vectors that matches the size of the input. Each
element is trimmed using the corresponding character set.

### What happens to missing strings?
Missing string scalars (`string(missing)`) remain `<missing>` exactly as in MATLAB.

### Does `strip` change the shape of character arrays?
Only the width can change. `strip` keeps the same number of rows and pads shorter rows with spaces so
the array stays rectangular.

### Will `strip` run on the GPU?
Not currently. RunMat gathers GPU-resident inputs automatically and performs trimming on the CPU to
maintain MATLAB compatibility.

## See Also
[lower](./lower), [upper](./upper), [string](./string), [char](./char), [compose](./compose)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/strip.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strip.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"###;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strip")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strip",
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
        "Executes on the CPU; GPU-resident inputs are gathered to host memory before trimming characters.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::strip")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strip",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion and always gathers GPU inputs.",
};

const ARG_TYPE_ERROR: &str =
    "strip: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "strip: cell array elements must be string scalars or character vectors";
const DIRECTION_ERROR: &str = "strip: direction must be 'left', 'right', or 'both'";
const CHARACTERS_ERROR: &str =
    "strip: characters to remove must be a string array, character vector, or cell array of character vectors";
const SIZE_MISMATCH_ERROR: &str =
    "strip: stripCharacters must be the same size as the input when supplying multiple values";

#[derive(Clone, Copy, Eq, PartialEq)]
enum StripDirection {
    Both,
    Left,
    Right,
}

enum PatternSpec {
    Default,
    Scalar(Vec<char>),
    PerElement(Vec<Vec<char>>),
}

enum PatternRef<'a> {
    Default,
    Custom(&'a [char]),
}

#[derive(Clone)]
struct PatternExpectation {
    len: usize,
    shape: Option<Vec<usize>>,
}

impl PatternExpectation {
    fn scalar() -> Self {
        Self {
            len: 1,
            shape: None,
        }
    }

    fn with_len(len: usize) -> Self {
        Self { len, shape: None }
    }

    fn with_shape(len: usize, shape: &[usize]) -> Self {
        Self {
            len,
            shape: Some(shape.to_vec()),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn shape(&self) -> Option<&[usize]> {
        self.shape.as_deref()
    }
}

impl PatternSpec {
    fn pattern_for_index(&self, idx: usize) -> PatternRef<'_> {
        match self {
            PatternSpec::Default => PatternRef::Default,
            PatternSpec::Scalar(chars) => PatternRef::Custom(chars),
            PatternSpec::PerElement(patterns) => patterns
                .get(idx)
                .map(|chars| PatternRef::Custom(chars))
                .unwrap_or(PatternRef::Default),
        }
    }
}

#[runtime_builtin(
    name = "strip",
    category = "strings/transform",
    summary = "Remove leading and trailing characters from strings, character arrays, and cell arrays.",
    keywords = "strip,trim,strings,character array,text",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::strip"
)]
fn strip_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let gathered = gather_if_needed(&value).map_err(|e| format!("strip: {e}"))?;
    match gathered {
        Value::String(text) => strip_string(text, &rest),
        Value::StringArray(array) => strip_string_array(array, &rest),
        Value::CharArray(array) => strip_char_array(array, &rest),
        Value::Cell(cell) => strip_cell_array(cell, &rest),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

fn strip_string(text: String, args: &[Value]) -> Result<Value, String> {
    if is_missing_string(&text) {
        return Ok(Value::String(text));
    }
    let expectation = PatternExpectation::scalar();
    let (direction, pattern_spec) = parse_arguments(args, &expectation)?;
    let stripped = strip_text(&text, direction, pattern_spec.pattern_for_index(0));
    Ok(Value::String(stripped))
}

fn strip_string_array(array: StringArray, args: &[Value]) -> Result<Value, String> {
    let expected_len = array.data.len();
    let expectation = PatternExpectation::with_shape(expected_len, &array.shape);
    let (direction, pattern_spec) = parse_arguments(args, &expectation)?;
    let StringArray { data, shape, .. } = array;
    let mut stripped: Vec<String> = Vec::with_capacity(expected_len);
    for (idx, text) in data.into_iter().enumerate() {
        if is_missing_string(&text) {
            stripped.push(text);
        } else {
            let pattern = pattern_spec.pattern_for_index(idx);
            stripped.push(strip_text(&text, direction, pattern));
        }
    }
    let result = StringArray::new(stripped, shape).map_err(|e| format!("strip: {e}"))?;
    Ok(Value::StringArray(result))
}

fn strip_char_array(array: CharArray, args: &[Value]) -> Result<Value, String> {
    let CharArray { data, rows, cols } = array;
    let expectation = PatternExpectation::with_len(rows);
    let (direction, pattern_spec) = parse_arguments(args, &expectation)?;

    if rows == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut stripped_rows: Vec<String> = Vec::with_capacity(rows);
    let mut target_cols: usize = 0;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row);
        let pattern = pattern_spec.pattern_for_index(row);
        let stripped = strip_text(&text, direction, pattern);
        let len = stripped.chars().count();
        target_cols = target_cols.max(len);
        stripped_rows.push(stripped);
    }

    let mut new_data: Vec<char> = Vec::with_capacity(rows * target_cols);
    for row_text in stripped_rows {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < target_cols {
            chars.resize(target_cols, ' ');
        }
        new_data.extend(chars.into_iter());
    }

    CharArray::new(new_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| format!("strip: {e}"))
}

fn strip_cell_array(cell: CellArray, args: &[Value]) -> Result<Value, String> {
    let rows = cell.rows;
    let cols = cell.cols;
    let dims = [rows, cols];
    let expectation = PatternExpectation::with_shape(rows * cols, &dims);
    let (direction, pattern_spec) = parse_arguments(args, &expectation)?;
    let total = rows * cols;
    let mut stripped_values: Vec<Value> = Vec::with_capacity(total);
    for idx in 0..total {
        let value = &cell.data[idx];
        let pattern = pattern_spec.pattern_for_index(idx);
        let stripped = strip_cell_element(value, direction, pattern)?;
        stripped_values.push(stripped);
    }
    make_cell(stripped_values, rows, cols).map_err(|e| format!("strip: {e}"))
}

fn strip_cell_element(
    value: &Value,
    direction: StripDirection,
    pattern: PatternRef<'_>,
) -> Result<Value, String> {
    let gathered = gather_if_needed(value).map_err(|e| format!("strip: {e}"))?;
    match gathered {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                let stripped = strip_text(&text, direction, pattern);
                Ok(Value::String(stripped))
            }
        }
        Value::StringArray(sa) if sa.data.len() == 1 => {
            let text = sa.data.into_iter().next().unwrap();
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                let stripped = strip_text(&text, direction, pattern);
                Ok(Value::String(stripped))
            }
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            let source = if ca.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&ca.data, ca.cols, 0)
            };
            let stripped = strip_text(&source, direction, pattern);
            let len = stripped.chars().count();
            let data: Vec<char> = stripped.chars().collect();
            let rows = ca.rows;
            let cols = if rows == 0 { ca.cols } else { len };
            CharArray::new(data, rows, cols)
                .map(Value::CharArray)
                .map_err(|e| format!("strip: {e}"))
        }
        Value::CharArray(_) => Err(CELL_ELEMENT_ERROR.to_string()),
        _ => Err(CELL_ELEMENT_ERROR.to_string()),
    }
}

fn parse_arguments(
    args: &[Value],
    expectation: &PatternExpectation,
) -> Result<(StripDirection, PatternSpec), String> {
    match args.len() {
        0 => Ok((StripDirection::Both, PatternSpec::Default)),
        1 => {
            if let Some(direction) = try_parse_direction(&args[0], false)? {
                Ok((direction, PatternSpec::Default))
            } else {
                let pattern = parse_pattern(&args[0], expectation)?;
                Ok((StripDirection::Both, pattern))
            }
        }
        2 => {
            let direction = match try_parse_direction(&args[0], true)? {
                Some(dir) => dir,
                None => return Err(DIRECTION_ERROR.to_string()),
            };
            let pattern = parse_pattern(&args[1], expectation)?;
            Ok((direction, pattern))
        }
        _ => Err("strip: too many input arguments".to_string()),
    }
}

fn try_parse_direction(value: &Value, strict: bool) -> Result<Option<StripDirection>, String> {
    let Some(text) = value_to_single_string(value) else {
        return Ok(None);
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return if strict {
            Err(DIRECTION_ERROR.to_string())
        } else {
            Ok(None)
        };
    }
    let lowered = trimmed.to_ascii_lowercase();
    let direction = match lowered.as_str() {
        "both" => Some(StripDirection::Both),
        "left" | "leading" => Some(StripDirection::Left),
        "right" | "trailing" => Some(StripDirection::Right),
        _ => {
            if strict {
                return Err(DIRECTION_ERROR.to_string());
            }
            None
        }
    };
    Ok(direction)
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
        Value::CharArray(ca) => {
            if ca.rows <= 1 {
                Some(char_row_to_string_slice(&ca.data, ca.cols, 0))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn parse_pattern(value: &Value, expectation: &PatternExpectation) -> Result<PatternSpec, String> {
    let expected_len = expectation.len();
    match value {
        Value::String(text) => Ok(PatternSpec::Scalar(text.chars().collect())),
        Value::StringArray(sa) => {
            if sa.data.len() <= 1 {
                if let Some(first) = sa.data.first() {
                    Ok(PatternSpec::Scalar(first.chars().collect()))
                } else {
                    Ok(PatternSpec::Scalar(Vec::new()))
                }
            } else if sa.data.len() == expected_len {
                if let Some(shape) = expectation.shape() {
                    if sa.shape != shape {
                        return Err(SIZE_MISMATCH_ERROR.to_string());
                    }
                }
                let mut patterns = Vec::with_capacity(sa.data.len());
                for text in &sa.data {
                    patterns.push(text.chars().collect());
                }
                Ok(PatternSpec::PerElement(patterns))
            } else {
                Err(SIZE_MISMATCH_ERROR.to_string())
            }
        }
        Value::CharArray(ca) => {
            if ca.rows <= 1 {
                if ca.rows == 0 {
                    Ok(PatternSpec::Scalar(Vec::new()))
                } else {
                    let chars = char_row_to_string_slice(&ca.data, ca.cols, 0);
                    Ok(PatternSpec::Scalar(chars.chars().collect()))
                }
            } else if ca.rows == expected_len {
                let mut patterns = Vec::with_capacity(ca.rows);
                for row in 0..ca.rows {
                    let text = char_row_to_string_slice(&ca.data, ca.cols, row);
                    patterns.push(text.chars().collect());
                }
                Ok(PatternSpec::PerElement(patterns))
            } else {
                Err(SIZE_MISMATCH_ERROR.to_string())
            }
        }
        Value::Cell(cell) => parse_pattern_cell(cell, expectation),
        _ => Err(CHARACTERS_ERROR.to_string()),
    }
}

fn parse_pattern_cell(
    cell: &CellArray,
    expectation: &PatternExpectation,
) -> Result<PatternSpec, String> {
    let len = cell.rows * cell.cols;
    if len == 0 {
        return Ok(PatternSpec::Scalar(Vec::new()));
    }
    if len == 1 {
        let chars = pattern_chars_from_value(&cell.data[0])?;
        return Ok(PatternSpec::Scalar(chars));
    }
    if len != expectation.len() {
        return Err(SIZE_MISMATCH_ERROR.to_string());
    }
    if let Some(shape) = expectation.shape() {
        match shape.len() {
            0 => {}
            1 => {
                if cell.rows != shape[0] || cell.cols != 1 {
                    return Err(SIZE_MISMATCH_ERROR.to_string());
                }
            }
            _ => {
                if cell.rows != shape[0] || cell.cols != shape[1] {
                    return Err(SIZE_MISMATCH_ERROR.to_string());
                }
            }
        }
    }
    let mut patterns = Vec::with_capacity(len);
    for value in &cell.data {
        patterns.push(pattern_chars_from_value(value)?);
    }
    Ok(PatternSpec::PerElement(patterns))
}

fn pattern_chars_from_value(value: &Value) -> Result<Vec<char>, String> {
    let gathered = gather_if_needed(value).map_err(|e| format!("strip: {e}"))?;
    match gathered {
        Value::String(text) => Ok(text.chars().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].chars().collect()),
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                Ok(Vec::new())
            } else {
                let text = char_row_to_string_slice(&ca.data, ca.cols, 0);
                Ok(text.chars().collect())
            }
        }
        Value::CharArray(_) => Err(CHARACTERS_ERROR.to_string()),
        _ => Err(CHARACTERS_ERROR.to_string()),
    }
}

fn strip_text(text: &str, direction: StripDirection, pattern: PatternRef<'_>) -> String {
    match pattern {
        PatternRef::Default => strip_text_with_predicate(text, direction, char::is_whitespace),
        PatternRef::Custom(chars) => {
            strip_text_with_predicate(text, direction, |c| chars.contains(&c))
        }
    }
}

fn strip_text_with_predicate<F>(text: &str, direction: StripDirection, mut predicate: F) -> String
where
    F: FnMut(char) -> bool,
{
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    let mut start = 0usize;
    let mut end = chars.len();

    if direction != StripDirection::Right {
        while start < end && predicate(chars[start]) {
            start += 1;
        }
    }

    if direction != StripDirection::Left {
        while end > start && predicate(chars[end - 1]) {
            end -= 1;
        }
    }

    chars[start..end].iter().collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn strip_string_scalar_default() {
        let result = strip_builtin(Value::String("  RunMat  ".into()), Vec::new()).expect("strip");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[test]
    fn strip_string_scalar_direction() {
        let result = strip_builtin(
            Value::String("...data".into()),
            vec![Value::String("left".into()), Value::String(".".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
    }

    #[test]
    fn strip_string_scalar_custom_characters() {
        let result = strip_builtin(
            Value::String("00052".into()),
            vec![Value::String("left".into()), Value::String("0".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("52".into()));
    }

    #[test]
    fn strip_string_scalar_pattern_only() {
        let result = strip_builtin(
            Value::String("xxaccelerationxx".into()),
            vec![Value::String("x".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("acceleration".into()));
    }

    #[test]
    fn strip_empty_pattern_returns_original() {
        let result = strip_builtin(
            Value::String("abc".into()),
            vec![Value::String(String::new())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("abc".into()));
    }

    #[test]
    fn strip_supports_leading_synonym() {
        let result = strip_builtin(
            Value::String("   data".into()),
            vec![Value::String("leading".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
    }

    #[test]
    fn strip_supports_trailing_synonym() {
        let result = strip_builtin(
            Value::String("data   ".into()),
            vec![Value::String("trailing".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
    }

    #[test]
    fn strip_string_array_per_element_characters() {
        let strings = StringArray::new(
            vec!["##ok##".into(), "--warn--".into(), "**fail**".into()],
            vec![3, 1],
        )
        .unwrap();
        let chars = CharArray::new(vec!['#', '#', '-', '-', '*', '*'], 3, 2).unwrap();
        let result = strip_builtin(
            Value::StringArray(strings),
            vec![Value::String("both".into()), Value::CharArray(chars)],
        )
        .expect("strip");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(
                    sa.data,
                    vec![
                        String::from("ok"),
                        String::from("warn"),
                        String::from("fail")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strip_string_array_cell_pattern_per_element() {
        let strings =
            StringArray::new(vec!["__pass__".into(), "--warn--".into()], vec![2, 1]).unwrap();
        let patterns = CellArray::new(
            vec![Value::String("_".into()), Value::String("-".into())],
            2,
            1,
        )
        .unwrap();
        let result =
            strip_builtin(Value::StringArray(strings), vec![Value::Cell(patterns)]).expect("strip");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec![String::from("pass"), String::from("warn")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strip_string_array_preserves_missing() {
        let strings =
            StringArray::new(vec!["   data   ".into(), "<missing>".into()], vec![2, 1]).unwrap();
        let result = strip_builtin(Value::StringArray(strings), Vec::new()).expect("strip");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "data");
                assert_eq!(sa.data[1], "<missing>");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strip_char_array_shrinks_width() {
        let source = "  cat  dog  ";
        let chars: Vec<char> = source.chars().collect();
        let array = CharArray::new(chars, 1, source.chars().count()).unwrap();
        let result = strip_builtin(Value::CharArray(array), Vec::new()).expect("strip");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 8);
                let expected: Vec<char> = "cat  dog".chars().collect();
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn strip_char_array_supports_trailing_direction() {
        let array = CharArray::new_row("gpu   ");
        let result = strip_builtin(
            Value::CharArray(array),
            vec![Value::String("trailing".into())],
        )
        .expect("strip");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                let expected: Vec<char> = "gpu".chars().collect();
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn strip_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("  GPU ")),
                Value::String("   Accelerate".into()),
                Value::String("RunMat   ".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = strip_builtin(Value::Cell(cell), Vec::new()).expect("strip");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 3);
                assert_eq!(
                    out.get(0, 0).unwrap(),
                    Value::CharArray(CharArray::new_row("GPU"))
                );
                assert_eq!(out.get(0, 1).unwrap(), Value::String("Accelerate".into()));
                assert_eq!(out.get(0, 2).unwrap(), Value::String("RunMat".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn strip_preserves_missing_string() {
        let result = strip_builtin(Value::String("<missing>".into()), Vec::new()).expect("strip");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[test]
    fn strip_errors_on_invalid_input() {
        let err = strip_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
    }

    #[test]
    fn strip_errors_on_invalid_pattern_type() {
        let err = strip_builtin(Value::String("abc".into()), vec![Value::Num(1.0)]).unwrap_err();
        assert_eq!(err, CHARACTERS_ERROR);
    }

    #[test]
    fn strip_errors_on_invalid_direction() {
        let err = strip_builtin(
            Value::String("abc".into()),
            vec![Value::String("sideways".into()), Value::String("a".into())],
        )
        .unwrap_err();
        assert_eq!(err, DIRECTION_ERROR);
    }

    #[test]
    fn strip_errors_on_pattern_size_mismatch() {
        let strings = StringArray::new(vec!["one".into(), "two".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["x".into(), "y".into(), "z".into()], vec![3, 1]).unwrap();
        let err = strip_builtin(
            Value::StringArray(strings),
            vec![Value::StringArray(pattern)],
        )
        .unwrap_err();
        assert_eq!(err, SIZE_MISMATCH_ERROR);
    }

    #[test]
    fn strip_errors_on_pattern_shape_mismatch() {
        let strings = StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap();
        let pattern = StringArray::new(vec!["x".into(), "y".into()], vec![2, 1]).unwrap();
        let err = strip_builtin(
            Value::StringArray(strings),
            vec![Value::StringArray(pattern)],
        )
        .unwrap_err();
        assert_eq!(err, SIZE_MISMATCH_ERROR);
    }

    #[test]
    fn strip_errors_on_cell_pattern_shape_mismatch() {
        let strings = StringArray::new(vec!["aa".into(), "bb".into()], vec![1, 2]).unwrap();
        let cell_pattern = CellArray::new(
            vec![Value::String("a".into()), Value::String("b".into())],
            2,
            1,
        )
        .unwrap();
        let err = strip_builtin(Value::StringArray(strings), vec![Value::Cell(cell_pattern)])
            .unwrap_err();
        assert_eq!(err, SIZE_MISMATCH_ERROR);
    }

    #[test]
    fn strip_errors_on_too_many_arguments() {
        let err = strip_builtin(
            Value::String("abc".into()),
            vec![
                Value::String("both".into()),
                Value::String("a".into()),
                Value::String("b".into()),
            ],
        )
        .unwrap_err();
        assert_eq!(err, "strip: too many input arguments");
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn strip_gpu_tensor_errors() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let host_data = [1.0f64, 2.0];
        let host_shape = [2usize, 1usize];
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &host_data,
                shape: &host_shape,
            })
            .expect("upload");
        let err = strip_builtin(Value::GpuTensor(handle.clone()), Vec::new()).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
        provider.free(&handle).ok();
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
