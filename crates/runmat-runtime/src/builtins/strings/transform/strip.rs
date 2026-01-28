//! MATLAB-compatible `strip` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

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

const BUILTIN_NAME: &str = "strip";
const ARG_TYPE_ERROR: &str =
    "strip: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "strip: cell array elements must be string scalars or character vectors";
const DIRECTION_ERROR: &str = "strip: direction must be 'left', 'right', or 'both'";
const CHARACTERS_ERROR: &str =
    "strip: characters to remove must be a string array, character vector, or cell array of character vectors";
const SIZE_MISMATCH_ERROR: &str =
    "strip: stripCharacters must be the same size as the input when supplying multiple values";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

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
async fn strip_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_if_needed_async(&value).await.map_err(map_flow)?;
    match gathered {
        Value::String(text) => strip_string(text, &rest).await,
        Value::StringArray(array) => strip_string_array(array, &rest).await,
        Value::CharArray(array) => strip_char_array(array, &rest).await,
        Value::Cell(cell) => strip_cell_array(cell, &rest).await,
        _ => Err(runtime_error_for(ARG_TYPE_ERROR)),
    }
}

async fn strip_string(text: String, args: &[Value]) -> BuiltinResult<Value> {
    if is_missing_string(&text) {
        return Ok(Value::String(text));
    }
    let expectation = PatternExpectation::scalar();
    let (direction, pattern_spec) = parse_arguments(args, &expectation).await?;
    let stripped = strip_text(&text, direction, pattern_spec.pattern_for_index(0));
    Ok(Value::String(stripped))
}

async fn strip_string_array(array: StringArray, args: &[Value]) -> BuiltinResult<Value> {
    let expected_len = array.data.len();
    let expectation = PatternExpectation::with_shape(expected_len, &array.shape);
    let (direction, pattern_spec) = parse_arguments(args, &expectation).await?;
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
    let result = StringArray::new(stripped, shape)
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
    Ok(Value::StringArray(result))
}

async fn strip_char_array(array: CharArray, args: &[Value]) -> BuiltinResult<Value> {
    let CharArray { data, rows, cols } = array;
    let expectation = PatternExpectation::with_len(rows);
    let (direction, pattern_spec) = parse_arguments(args, &expectation).await?;

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
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
}

async fn strip_cell_array(cell: CellArray, args: &[Value]) -> BuiltinResult<Value> {
    let rows = cell.rows;
    let cols = cell.cols;
    let dims = [rows, cols];
    let expectation = PatternExpectation::with_shape(rows * cols, &dims);
    let (direction, pattern_spec) = parse_arguments(args, &expectation).await?;
    let total = rows * cols;
    let mut stripped_values: Vec<Value> = Vec::with_capacity(total);
    for idx in 0..total {
        let value = &cell.data[idx];
        let pattern = pattern_spec.pattern_for_index(idx);
        let stripped = strip_cell_element(value, direction, pattern).await?;
        stripped_values.push(stripped);
    }
    make_cell(stripped_values, rows, cols)
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
}

async fn strip_cell_element(
    value: &Value,
    direction: StripDirection,
    pattern: PatternRef<'_>,
) -> BuiltinResult<Value> {
    let gathered = gather_if_needed_async(value).await.map_err(map_flow)?;
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
                .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
        }
        Value::CharArray(_) => Err(runtime_error_for(CELL_ELEMENT_ERROR)),
        _ => Err(runtime_error_for(CELL_ELEMENT_ERROR)),
    }
}

async fn parse_arguments(
    args: &[Value],
    expectation: &PatternExpectation,
) -> BuiltinResult<(StripDirection, PatternSpec)> {
    match args.len() {
        0 => Ok((StripDirection::Both, PatternSpec::Default)),
        1 => {
            if let Some(direction) = try_parse_direction(&args[0], false)? {
                Ok((direction, PatternSpec::Default))
            } else {
                let pattern = parse_pattern(&args[0], expectation).await?;
                Ok((StripDirection::Both, pattern))
            }
        }
        2 => {
            let direction = match try_parse_direction(&args[0], true)? {
                Some(dir) => dir,
                None => return Err(runtime_error_for(DIRECTION_ERROR)),
            };
            let pattern = parse_pattern(&args[1], expectation).await?;
            Ok((direction, pattern))
        }
        _ => Err(runtime_error_for("strip: too many input arguments")),
    }
}

fn try_parse_direction(value: &Value, strict: bool) -> BuiltinResult<Option<StripDirection>> {
    let Some(text) = value_to_single_string(value) else {
        return Ok(None);
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return if strict {
            Err(runtime_error_for(DIRECTION_ERROR))
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
                return Err(runtime_error_for(DIRECTION_ERROR));
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

async fn parse_pattern(
    value: &Value,
    expectation: &PatternExpectation,
) -> BuiltinResult<PatternSpec> {
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
                        return Err(runtime_error_for(SIZE_MISMATCH_ERROR));
                    }
                }
                let mut patterns = Vec::with_capacity(sa.data.len());
                for text in &sa.data {
                    patterns.push(text.chars().collect());
                }
                Ok(PatternSpec::PerElement(patterns))
            } else {
                Err(runtime_error_for(SIZE_MISMATCH_ERROR))
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
                Err(runtime_error_for(SIZE_MISMATCH_ERROR))
            }
        }
        Value::Cell(cell) => parse_pattern_cell(cell, expectation).await,
        _ => Err(runtime_error_for(CHARACTERS_ERROR)),
    }
}

async fn parse_pattern_cell(
    cell: &CellArray,
    expectation: &PatternExpectation,
) -> BuiltinResult<PatternSpec> {
    let len = cell.rows * cell.cols;
    if len == 0 {
        return Ok(PatternSpec::Scalar(Vec::new()));
    }
    if len == 1 {
        let chars = pattern_chars_from_value(&cell.data[0]).await?;
        return Ok(PatternSpec::Scalar(chars));
    }
    if len != expectation.len() {
        return Err(runtime_error_for(SIZE_MISMATCH_ERROR));
    }
    if let Some(shape) = expectation.shape() {
        match shape.len() {
            0 => {}
            1 => {
                if cell.rows != shape[0] || cell.cols != 1 {
                    return Err(runtime_error_for(SIZE_MISMATCH_ERROR));
                }
            }
            _ => {
                if cell.rows != shape[0] || cell.cols != shape[1] {
                    return Err(runtime_error_for(SIZE_MISMATCH_ERROR));
                }
            }
        }
    }
    let mut patterns = Vec::with_capacity(len);
    for value in &cell.data {
        patterns.push(pattern_chars_from_value(value).await?);
    }
    Ok(PatternSpec::PerElement(patterns))
}

async fn pattern_chars_from_value(value: &Value) -> BuiltinResult<Vec<char>> {
    let gathered = gather_if_needed_async(value).await.map_err(map_flow)?;
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
        Value::CharArray(_) => Err(runtime_error_for(CHARACTERS_ERROR)),
        _ => Err(runtime_error_for(CHARACTERS_ERROR)),
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

    fn run_strip(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(strip_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_scalar_default() {
        let result = run_strip(Value::String("  RunMat  ".into()), Vec::new()).expect("strip");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_scalar_direction() {
        let result = run_strip(
            Value::String("...data".into()),
            vec![Value::String("left".into()), Value::String(".".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_scalar_custom_characters() {
        let result = run_strip(
            Value::String("00052".into()),
            vec![Value::String("left".into()), Value::String("0".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("52".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_scalar_pattern_only() {
        let result = run_strip(
            Value::String("xxaccelerationxx".into()),
            vec![Value::String("x".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("acceleration".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_empty_pattern_returns_original() {
        let result = run_strip(
            Value::String("abc".into()),
            vec![Value::String(String::new())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("abc".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_supports_leading_synonym() {
        let result = run_strip(
            Value::String("   data".into()),
            vec![Value::String("leading".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_supports_trailing_synonym() {
        let result = run_strip(
            Value::String("data   ".into()),
            vec![Value::String("trailing".into())],
        )
        .expect("strip");
        assert_eq!(result, Value::String("data".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_array_per_element_characters() {
        let strings = StringArray::new(
            vec!["##ok##".into(), "--warn--".into(), "**fail**".into()],
            vec![3, 1],
        )
        .unwrap();
        let chars = CharArray::new(vec!['#', '#', '-', '-', '*', '*'], 3, 2).unwrap();
        let result = run_strip(
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
            run_strip(Value::StringArray(strings), vec![Value::Cell(patterns)]).expect("strip");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec![String::from("pass"), String::from("warn")]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_string_array_preserves_missing() {
        let strings =
            StringArray::new(vec!["   data   ".into(), "<missing>".into()], vec![2, 1]).unwrap();
        let result = run_strip(Value::StringArray(strings), Vec::new()).expect("strip");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "data");
                assert_eq!(sa.data[1], "<missing>");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_char_array_shrinks_width() {
        let source = "  cat  dog  ";
        let chars: Vec<char> = source.chars().collect();
        let array = CharArray::new(chars, 1, source.chars().count()).unwrap();
        let result = run_strip(Value::CharArray(array), Vec::new()).expect("strip");
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_char_array_supports_trailing_direction() {
        let array = CharArray::new_row("gpu   ");
        let result = run_strip(
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let result = run_strip(Value::Cell(cell), Vec::new()).expect("strip");
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_preserves_missing_string() {
        let result = run_strip(Value::String("<missing>".into()), Vec::new()).expect("strip");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_invalid_input() {
        let err = run_strip(Value::Num(1.0), Vec::new()).unwrap_err();
        assert_eq!(err.to_string(), ARG_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_invalid_pattern_type() {
        let err = run_strip(Value::String("abc".into()), vec![Value::Num(1.0)]).unwrap_err();
        assert_eq!(err.to_string(), CHARACTERS_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_invalid_direction() {
        let err = run_strip(
            Value::String("abc".into()),
            vec![Value::String("sideways".into()), Value::String("a".into())],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), DIRECTION_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_pattern_size_mismatch() {
        let strings = StringArray::new(vec!["one".into(), "two".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["x".into(), "y".into(), "z".into()], vec![3, 1]).unwrap();
        let err = run_strip(
            Value::StringArray(strings),
            vec![Value::StringArray(pattern)],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), SIZE_MISMATCH_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_pattern_shape_mismatch() {
        let strings = StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap();
        let pattern = StringArray::new(vec!["x".into(), "y".into()], vec![2, 1]).unwrap();
        let err = run_strip(
            Value::StringArray(strings),
            vec![Value::StringArray(pattern)],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), SIZE_MISMATCH_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_cell_pattern_shape_mismatch() {
        let strings = StringArray::new(vec!["aa".into(), "bb".into()], vec![1, 2]).unwrap();
        let cell_pattern = CellArray::new(
            vec![Value::String("a".into()), Value::String("b".into())],
            2,
            1,
        )
        .unwrap();
        let err =
            run_strip(Value::StringArray(strings), vec![Value::Cell(cell_pattern)]).unwrap_err();
        assert_eq!(err.to_string(), SIZE_MISMATCH_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strip_errors_on_too_many_arguments() {
        let err = run_strip(
            Value::String("abc".into()),
            vec![
                Value::String("both".into()),
                Value::String("a".into()),
                Value::String("b".into()),
            ],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "strip: too many input arguments");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let err = run_strip(Value::GpuTensor(handle.clone()), Vec::new()).unwrap_err();
        assert_eq!(err.to_string(), ARG_TYPE_ERROR);
        provider.free(&handle).ok();
    }
}
