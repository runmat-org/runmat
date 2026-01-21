//! MATLAB-compatible `strtrim` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{gather_if_needed, make_cell};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strtrim")]
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

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::transform::strtrim"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strtrim",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; not eligible for fusion and always gathers GPU inputs.",
};

const ARG_TYPE_ERROR: &str =
    "strtrim: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "strtrim: cell array elements must be string scalars or character vectors";

#[runtime_builtin(
    name = "strtrim",
    category = "strings/transform",
    summary = "Remove leading and trailing whitespace from strings, character arrays, and cell arrays.",
    keywords = "strtrim,trim,whitespace,strings,character array,text",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::strtrim"
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
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut trimmed_values = Vec::with_capacity(rows * cols);
    for value in &data {
        let trimmed = strtrim_cell_element(value)?;
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
pub(crate) mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_string_scalar_trims_whitespace() {
        let result =
            strtrim_builtin(Value::String("  RunMat  ".into())).expect("strtrim string scalar");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_preserves_missing_strings() {
        let result =
            strtrim_builtin(Value::String("<missing>".into())).expect("strtrim missing string");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_handles_tabs_and_newlines() {
        let input = Value::String("\tMetrics \n".into());
        let result = strtrim_builtin(input).expect("strtrim tab/newline");
        assert_eq!(result, Value::String("Metrics".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_trims_unicode_whitespace() {
        let input = Value::String("\u{00A0}RunMat\u{2003}".into());
        let result = strtrim_builtin(input).expect("strtrim unicode whitespace");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_char_array_zero_rows_stable() {
        let array = CharArray::new(Vec::new(), 0, 0).unwrap();
        let result = strtrim_builtin(Value::CharArray(array.clone())).expect("strtrim 0x0 char");
        assert_eq!(result, Value::CharArray(array));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_cell_array_rejects_non_text() {
        let cell = CellArray::new(vec![Value::Num(5.0)], 1, 1).unwrap();
        let err = strtrim_builtin(Value::Cell(cell)).expect_err("strtrim cell non-text");
        assert!(err.contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_errors_on_invalid_input() {
        let err = strtrim_builtin(Value::Num(1.0)).unwrap_err();
        assert!(err.contains("strtrim"));
    }
}
