//! MATLAB-compatible `str2double` builtin with GPU-aware semantics for RunMat.

use std::borrow::Cow;

use runmat_builtins::{CellArray, CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::str2double")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "str2double",
    op_kind: GpuOpKind::Custom("conversion"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Parses text on the CPU; GPU-resident inputs are gathered before conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::str2double")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "str2double",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Conversion builtin; not eligible for fusion and materialises host-side doubles.",
};

const ARG_TYPE_ERROR: &str =
    "str2double: input must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "str2double: cell array elements must be character vectors or string scalars";

fn str2double_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("str2double")
        .build()
}

fn remap_str2double_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, "str2double")
}

#[runtime_builtin(
    name = "str2double",
    category = "strings/core",
    summary = "Convert strings, character arrays, or cell arrays of text into doubles.",
    keywords = "str2double,string to double,text conversion,gpu",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::str2double"
)]
async fn str2double_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(remap_str2double_flow)?;
    match gathered {
        Value::String(text) => Ok(Value::Num(parse_numeric_scalar(&text))),
        Value::StringArray(array) => str2double_string_array(array),
        Value::CharArray(array) => str2double_char_array(array),
        Value::Cell(cell) => str2double_cell_array(cell),
        _ => Err(str2double_flow(ARG_TYPE_ERROR)),
    }
}

fn str2double_string_array(array: StringArray) -> BuiltinResult<Value> {
    let StringArray { data, shape, .. } = array;
    let mut values = Vec::with_capacity(data.len());
    for text in &data {
        values.push(parse_numeric_scalar(text));
    }
    let tensor =
        Tensor::new(values, shape).map_err(|e| str2double_flow(format!("str2double: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn str2double_char_array(array: CharArray) -> BuiltinResult<Value> {
    let rows = array.rows;
    let cols = array.cols;
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_text: String = array.data[start..end].iter().collect();
        values.push(parse_numeric_scalar(&row_text));
    }
    let tensor = Tensor::new(values, vec![rows, 1])
        .map_err(|e| str2double_flow(format!("str2double: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn str2double_cell_array(cell: CellArray) -> BuiltinResult<Value> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut values = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let idx = row * cols + col;
            let element: &Value = &data[idx];
            let numeric = match element {
                Value::String(text) => parse_numeric_scalar(text),
                Value::StringArray(sa) if sa.data.len() == 1 => parse_numeric_scalar(&sa.data[0]),
                Value::CharArray(char_vec) if char_vec.rows == 1 => {
                    let row_text: String = char_vec.data.iter().collect();
                    parse_numeric_scalar(&row_text)
                }
                Value::CharArray(_) => return Err(str2double_flow(CELL_ELEMENT_ERROR)),
                _ => return Err(str2double_flow(CELL_ELEMENT_ERROR)),
            };
            values.push(numeric);
        }
    }
    let tensor = Tensor::new(values, vec![rows, cols])
        .map_err(|e| str2double_flow(format!("str2double: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn parse_numeric_scalar(text: &str) -> f64 {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return f64::NAN;
    }

    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "nan" => return f64::NAN,
        "inf" | "+inf" | "infinity" | "+infinity" => return f64::INFINITY,
        "-inf" | "-infinity" => return f64::NEG_INFINITY,
        _ => {}
    }

    let normalized: Cow<'_, str> = if trimmed.chars().any(|c| c == 'd' || c == 'D') {
        Cow::Owned(
            trimmed
                .chars()
                .map(|c| if c == 'd' || c == 'D' { 'e' } else { c })
                .collect(),
        )
    } else {
        Cow::Borrowed(trimmed)
    };

    normalized.parse::<f64>().unwrap_or(f64::NAN)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn str2double_builtin(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::str2double_builtin(value))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn str2double_string_scalar() {
        let result = str2double_builtin(Value::String("42.5".into())).expect("str2double");
        assert_eq!(result, Value::Num(42.5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn str2double_string_scalar_invalid_returns_nan() {
        let result = str2double_builtin(Value::String("abc".into())).expect("str2double");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn str2double_string_array_preserves_shape() {
        let array =
            StringArray::new(vec!["1".into(), " 2.5 ".into(), "foo".into()], vec![3, 1]).unwrap();
        let result = str2double_builtin(Value::StringArray(array)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 1]);
                assert_eq!(tensor.data[0], 1.0);
                assert_eq!(tensor.data[1], 2.5);
                assert!(tensor.data[2].is_nan());
            }
            Value::Num(_) => panic!("expected tensor"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn str2double_char_array_multiple_rows() {
        let data: Vec<char> = vec!['4', '2', ' ', ' ', '1', '0', '0', ' '];
        let array = CharArray::new(data, 2, 4).unwrap();
        let result = str2double_builtin(Value::CharArray(array)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data[0], 42.0);
                assert_eq!(tensor.data[1], 100.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn str2double_char_array_empty_rows() {
        let array = CharArray::new(Vec::new(), 0, 0).unwrap();
        let result = str2double_builtin(Value::CharArray(array)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![0, 1]);
                assert_eq!(tensor.data.len(), 0);
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[allow(
        clippy::approx_constant,
        reason = "Test ensures literal 3.14 text stays 3.14, not π"
    )]
    fn str2double_cell_array_of_text() {
        let cell = CellArray::new(
            vec![
                Value::String("3.14".into()),
                Value::CharArray(CharArray::new_row("NaN")),
                Value::String("-Inf".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = str2double_builtin(Value::Cell(cell)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data[0], 3.14);
                assert!(tensor.data[1].is_nan());
                assert_eq!(tensor.data[2], f64::NEG_INFINITY);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn str2double_cell_array_invalid_element_errors() {
        let cell = CellArray::new(vec![Value::Num(5.0)], 1, 1).unwrap();
        let err = error_message(str2double_builtin(Value::Cell(cell)).unwrap_err());
        assert!(
            err.contains("str2double"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn str2double_supports_d_exponent() {
        let result = str2double_builtin(Value::String("1.5D3".into())).expect("str2double");
        match result {
            Value::Num(v) => assert_eq!(v, 1500.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn str2double_recognises_infinity_forms() {
        let array = StringArray::new(
            vec!["Inf".into(), "-Infinity".into(), "+inf".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = str2double_builtin(Value::StringArray(array)).expect("str2double");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.data[0], f64::INFINITY);
                assert_eq!(tensor.data[1], f64::NEG_INFINITY);
                assert_eq!(tensor.data[2], f64::INFINITY);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
