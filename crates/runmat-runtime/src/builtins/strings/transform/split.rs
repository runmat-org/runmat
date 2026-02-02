//! MATLAB-compatible `split` builtin with GPU-aware semantics for RunMat.

use std::collections::HashSet;

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::split")]
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::split")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "split",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion planning and always gathers GPU inputs.",
};

const BUILTIN_NAME: &str = "split";
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

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "split",
    category = "strings/transform",
    summary = "Split strings, character arrays, and cell arrays into substrings using delimiters.",
    keywords = "split,strsplit,delimiter,CollapseDelimiters,IncludeDelimiters",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::split"
)]
async fn split_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let mut args: Vec<Value> = Vec::with_capacity(rest.len());
    for arg in rest {
        args.push(gather_if_needed_async(&arg).await.map_err(map_flow)?);
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
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut index = 0usize;
        let mut delimiters = DelimiterSpec::Whitespace;

        if index < args.len() && !is_name_key(&args[index]) {
            let list = extract_delimiters(&args[index])?;
            if list.is_empty() {
                return Err(runtime_error_for(EMPTY_DELIMITER_ERROR));
            }
            let mut seen = HashSet::new();
            let mut patterns: Vec<String> = Vec::new();
            for pattern in list {
                if pattern.is_empty() {
                    return Err(runtime_error_for(EMPTY_DELIMITER_ERROR));
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
                None => return Err(runtime_error_for(UNKNOWN_NAME_ERROR)),
            };
            index += 1;
            if index >= args.len() {
                return Err(runtime_error_for(NAME_VALUE_PAIR_ERROR));
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
    fn from_value(value: Value) -> BuiltinResult<Self> {
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
            _ => Err(runtime_error_for(ARG_TYPE_ERROR)),
        }
    }

    fn from_char_array(array: CharArray) -> BuiltinResult<Self> {
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

    fn from_cell_array(cell: CellArray) -> BuiltinResult<Self> {
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
                        .ok_or_else(|| runtime_error_for(CELL_ELEMENT_ERROR))?,
                );
            }
        }
        Ok(Self {
            data: strings,
            rows,
            cols,
        })
    }

    fn into_split_result(self, options: &SplitOptions) -> BuiltinResult<Value> {
        let TextMatrix { data, rows, cols } = self;

        if data.is_empty() {
            let block_cols = if cols == 0 { 0 } else { 1 };
            let shape = if cols == 0 {
                vec![rows, 0]
            } else {
                vec![rows, cols * block_cols]
            };
            let array = StringArray::new(Vec::new(), shape)
                .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
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
        let array = StringArray::new(output, shape)
            .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
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

fn extract_delimiters(value: &Value) -> BuiltinResult<Vec<String>> {
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
                        .ok_or_else(|| runtime_error_for(CELL_ELEMENT_ERROR))?,
                );
            }
            Ok(entries)
        }
        _ => Err(runtime_error_for(DELIMITER_TYPE_ERROR)),
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

fn parse_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(i.to_i64() != 0),
        Value::Num(n) => Ok(*n != 0.0),
        Value::LogicalArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0] != 0)
            } else {
                Err(runtime_error_for(format!(
                    "{BUILTIN_NAME}: value for '{}' must be logical true or false",
                    name
                )))
            }
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                Ok(tensor.data[0] != 0.0)
            } else {
                Err(runtime_error_for(format!(
                    "{BUILTIN_NAME}: value for '{}' must be logical true or false",
                    name
                )))
            }
        }
        _ => {
            if let Some(text) = value_to_scalar_string(value) {
                let lowered = text.trim().to_ascii_lowercase();
                match lowered.as_str() {
                    "true" | "on" | "yes" => Ok(true),
                    "false" | "off" | "no" => Ok(false),
                    _ => Err(runtime_error_for(format!(
                        "{BUILTIN_NAME}: value for '{}' must be logical true or false",
                        name
                    ))),
                }
            } else {
                Err(runtime_error_for(format!(
                    "{BUILTIN_NAME}: value for '{}' must be logical true or false",
                    name
                )))
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
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{CellArray, LogicalArray, Tensor};

    fn split_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::split_builtin(text, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_invalid_name_value_pair_errors() {
        let input = Value::String("abc".to_string());
        let args = vec![Value::String("CollapseDelimiters".to_string())];
        let err = split_builtin(input, args).unwrap_err();
        assert!(err.to_string().contains("name-value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_invalid_text_argument_errors() {
        let err = split_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("first argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_invalid_delimiter_type_errors() {
        let err =
            split_builtin(Value::String("abc".to_string()), vec![Value::Num(1.0)]).unwrap_err();
        assert!(err.to_string().contains("delimiter input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn split_empty_delimiter_errors() {
        let err = split_builtin(
            Value::String("abc".to_string()),
            vec![Value::String(String::new())],
        )
        .unwrap_err();
        assert!(err.to_string().contains("at least one character"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        assert!(err.to_string().contains("unrecognized"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
}
