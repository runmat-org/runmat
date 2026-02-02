//! MATLAB-compatible `replace` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::replace")]
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

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::transform::replace"
)]
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

const BUILTIN_NAME: &str = "replace";
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

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "replace",
    category = "strings/transform",
    summary = "Replace substring occurrences in strings, character arrays, and cell arrays.",
    keywords = "replace,strrep,strings,character array,text",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::replace"
)]
async fn replace_builtin(text: Value, old: Value, new: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let old = gather_if_needed_async(&old).await.map_err(map_flow)?;
    let new = gather_if_needed_async(&new).await.map_err(map_flow)?;

    let spec = ReplacementSpec::from_values(&old, &new)?;

    match text {
        Value::String(s) => Ok(Value::String(replace_string_scalar(s, &spec))),
        Value::StringArray(sa) => replace_string_array(sa, &spec),
        Value::CharArray(ca) => replace_char_array(ca, &spec),
        Value::Cell(cell) => replace_cell_array(cell, &spec),
        _ => Err(runtime_error_for(ARG_TYPE_ERROR)),
    }
}

fn replace_string_scalar(text: String, spec: &ReplacementSpec) -> String {
    if is_missing_string(&text) {
        text
    } else {
        spec.apply(&text)
    }
}

fn replace_string_array(array: StringArray, spec: &ReplacementSpec) -> BuiltinResult<Value> {
    let StringArray { data, shape, .. } = array;
    let mut replaced = Vec::with_capacity(data.len());
    for entry in data {
        if is_missing_string(&entry) {
            replaced.push(entry);
        } else {
            replaced.push(spec.apply(&entry));
        }
    }
    let result = StringArray::new(replaced, shape)
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
    Ok(Value::StringArray(result))
}

fn replace_char_array(array: CharArray, spec: &ReplacementSpec) -> BuiltinResult<Value> {
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
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
}

fn replace_cell_array(cell: CellArray, spec: &ReplacementSpec) -> BuiltinResult<Value> {
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
    make_cell(replaced, rows, cols).map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
}

fn replace_cell_element(value: &Value, spec: &ReplacementSpec) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::String(replace_string_scalar(text.clone(), spec))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Value::String(replace_string_scalar(
            sa.data[0].clone(),
            spec,
        ))),
        Value::CharArray(ca) if ca.rows <= 1 => replace_char_array(ca.clone(), spec),
        Value::CharArray(_) => Err(runtime_error_for(CELL_ELEMENT_ERROR)),
        _ => Err(runtime_error_for(CELL_ELEMENT_ERROR)),
    }
}

fn extract_pattern_list(value: &Value) -> BuiltinResult<Vec<String>> {
    extract_text_list(value, PATTERN_TYPE_ERROR)
}

fn extract_replacement_list(value: &Value) -> BuiltinResult<Vec<String>> {
    extract_text_list(value, REPLACEMENT_TYPE_ERROR)
}

fn extract_text_list(value: &Value, type_error: &str) -> BuiltinResult<Vec<String>> {
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
                    Value::CharArray(_) => return Err(runtime_error_for(CELL_ELEMENT_ERROR)),
                    _ => return Err(runtime_error_for(CELL_ELEMENT_ERROR)),
                }
            }
            Ok(entries)
        }
        _ => Err(runtime_error_for(type_error)),
    }
}

struct ReplacementSpec {
    pairs: Vec<(String, String)>,
}

impl ReplacementSpec {
    fn from_values(old: &Value, new: &Value) -> BuiltinResult<Self> {
        let patterns = extract_pattern_list(old)?;
        if patterns.is_empty() {
            return Err(runtime_error_for(EMPTY_PATTERN_ERROR));
        }

        let replacements = extract_replacement_list(new)?;
        if replacements.is_empty() {
            return Err(runtime_error_for(EMPTY_REPLACEMENT_ERROR));
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
            return Err(runtime_error_for(SIZE_MISMATCH_ERROR));
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
pub(crate) mod tests {
    use super::*;

    fn replace_builtin(text: Value, old: Value, new: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::replace_builtin(text, old, new))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_errors_on_invalid_first_argument() {
        let err = replace_builtin(
            Value::Num(1.0),
            Value::String("a".into()),
            Value::String("b".into()),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), ARG_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_errors_on_invalid_pattern_type() {
        let err = replace_builtin(
            Value::String("abc".into()),
            Value::Num(1.0),
            Value::String("x".into()),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), PATTERN_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        assert_eq!(err.to_string(), SIZE_MISMATCH_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
}
