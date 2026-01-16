//! MATLAB-compatible `strlength` builtin for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::common::is_missing_string;
use crate::gather_if_needed;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::strlength")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strlength",
    op_kind: GpuOpKind::Custom("string-metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Measures string lengths on the CPU; any GPU-resident inputs are gathered before evaluation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::strlength")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strlength",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Metadata-only builtin; not eligible for fusion and never emits GPU kernels.",
};

const ARG_TYPE_ERROR: &str =
    "strlength: first argument must be a string array, character array, or cell array of character vectors";
const CELL_ELEMENT_ERROR: &str =
    "strlength: cell array elements must be character vectors or string scalars";

#[runtime_builtin(
    name = "strlength",
    category = "strings/core",
    summary = "Count characters in string arrays, character arrays, or cell arrays of character vectors.",
    keywords = "strlength,string length,text,count,characters",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::strlength"
)]
fn strlength_builtin(value: Value) -> Result<Value, String> {
    let gathered = gather_if_needed(&value).map_err(|e| format!("strlength: {e}"))?;
    match gathered {
        Value::StringArray(array) => strlength_string_array(array),
        Value::String(text) => Ok(Value::Num(string_scalar_length(&text))),
        Value::CharArray(array) => strlength_char_array(array),
        Value::Cell(cell) => strlength_cell_array(cell),
        _ => Err(ARG_TYPE_ERROR.to_string()),
    }
}

fn strlength_string_array(array: StringArray) -> Result<Value, String> {
    let StringArray { data, shape, .. } = array;
    let mut lengths = Vec::with_capacity(data.len());
    for text in &data {
        lengths.push(string_scalar_length(text));
    }
    let tensor = Tensor::new(lengths, shape).map_err(|e| format!("strlength: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn strlength_char_array(array: CharArray) -> Result<Value, String> {
    let rows = array.rows;
    let mut lengths = Vec::with_capacity(rows);
    for row in 0..rows {
        let length = if array.rows <= 1 {
            array.cols
        } else {
            trimmed_row_length(&array, row)
        } as f64;
        lengths.push(length);
    }
    let tensor = Tensor::new(lengths, vec![rows, 1]).map_err(|e| format!("strlength: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn strlength_cell_array(cell: CellArray) -> Result<Value, String> {
    let CellArray {
        data, rows, cols, ..
    } = cell;
    let mut lengths = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let idx = row * cols + col;
            let value: &Value = &data[idx];
            let length = match value {
                Value::String(text) => string_scalar_length(text),
                Value::StringArray(sa) if sa.data.len() == 1 => string_scalar_length(&sa.data[0]),
                Value::CharArray(char_vec) if char_vec.rows == 1 => char_vec.cols as f64,
                Value::CharArray(_) => return Err(CELL_ELEMENT_ERROR.to_string()),
                _ => return Err(CELL_ELEMENT_ERROR.to_string()),
            };
            lengths.push(length);
        }
    }
    let tensor = Tensor::new(lengths, vec![rows, cols]).map_err(|e| format!("strlength: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn string_scalar_length(text: &str) -> f64 {
    if is_missing_string(text) {
        f64::NAN
    } else {
        text.chars().count() as f64
    }
}

fn trimmed_row_length(array: &CharArray, row: usize) -> usize {
    let cols = array.cols;
    let mut end = cols;
    while end > 0 {
        let ch = array.data[row * cols + end - 1];
        if ch == ' ' {
            end -= 1;
        } else {
            break;
        }
    }
    end
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_string_scalar() {
        let result = strlength_builtin(Value::String("RunMat".into())).expect("strlength");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_string_array_with_missing() {
        let array = StringArray::new(vec!["alpha".into(), "<missing>".into()], vec![2, 1]).unwrap();
        let result = strlength_builtin(Value::StringArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data.len(), 2);
                assert_eq!(tensor.data[0], 5.0);
                assert!(tensor.data[1].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_array_multiple_rows() {
        let data: Vec<char> = vec!['c', 'a', 't', ' ', ' ', 'h', 'o', 'r', 's', 'e'];
        let array = CharArray::new(data, 2, 5).unwrap();
        let result = strlength_builtin(Value::CharArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_vector_retains_explicit_spaces() {
        let data: Vec<char> = "hi   ".chars().collect();
        let array = CharArray::new(data, 1, 5).unwrap();
        let result = strlength_builtin(Value::CharArray(array)).expect("strlength");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_cell_array_of_char_vectors() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("red")),
                Value::CharArray(CharArray::new_row("green")),
            ],
            1,
            2,
        )
        .unwrap();
        let result = strlength_builtin(Value::Cell(cell)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_cell_array_with_string_scalars() {
        let cell = CellArray::new(
            vec![
                Value::String("alpha".into()),
                Value::String("beta".into()),
                Value::String("<missing>".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = strlength_builtin(Value::Cell(cell)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data.len(), 3);
                assert_eq!(tensor.data[0], 5.0);
                assert_eq!(tensor.data[1], 4.0);
                assert!(tensor.data[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_string_array_preserves_shape() {
        let array = StringArray::new(
            vec!["ab".into(), "c".into(), "def".into(), "".into()],
            vec![2, 2],
        )
        .unwrap();
        let result = strlength_builtin(Value::StringArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_eq!(tensor.data, vec![2.0, 1.0, 3.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_array_trims_padding() {
        let data: Vec<char> = vec!['d', 'o', 'g', ' ', ' ', 'h', 'o', 'r', 's', 'e'];
        let array = CharArray::new(data, 2, 5).unwrap();
        let result = strlength_builtin(Value::CharArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_errors_on_invalid_input() {
        let err = strlength_builtin(Value::Num(1.0)).unwrap_err();
        assert_eq!(err, ARG_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_rejects_cell_with_invalid_element() {
        let cell = CellArray::new(
            vec![Value::CharArray(CharArray::new_row("ok")), Value::Num(5.0)],
            1,
            2,
        )
        .unwrap();
        let err = strlength_builtin(Value::Cell(cell)).unwrap_err();
        assert_eq!(err, CELL_ELEMENT_ERROR);
    }
}
