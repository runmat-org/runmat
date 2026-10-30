//! MATLAB-compatible `strlength` builtin for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, STRLENGTH_DESCRIPTOR, STRLENGTH_ERROR_INTERNAL,
    STRLENGTH_ERROR_INVALID_CELL_ELEMENT, STRLENGTH_ERROR_INVALID_INPUT,
    STRLENGTH_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor;
use crate::builtins::strings::common::contains_numeric_or_resident_text_input;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "strlength";

fn strlength_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strlength_error_with_message(error.message, error)
}

fn strlength_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_strlength_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "strlength",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::strlength"
)]
async fn strlength_builtin(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &STRLENGTH_DESCRIPTOR,
        &STRLENGTH_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if contains_numeric_or_resident_text_input(&value) {
        return Err(strlength_error(&STRLENGTH_ERROR_INVALID_INPUT));
    }
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(remap_strlength_flow)?;
    match gathered {
        Value::StringArray(array) => strlength_string_array(array),
        Value::String(text) => Ok(Value::Num(string_scalar_length(text.as_str()))),
        Value::CharArray(array) => strlength_char_array(array),
        Value::Cell(cell) => strlength_cell_array(cell),
        _ => Err(strlength_error(&STRLENGTH_ERROR_INVALID_INPUT)),
    }
}

fn strlength_string_array(array: StringArray) -> BuiltinResult<Value> {
    let (data, shape, _, _) = array.into_parts();
    let mut lengths = Vec::with_capacity(data.len());
    for text in &data {
        lengths.push(string_scalar_length(text.as_str()));
    }
    let tensor =
        Tensor::new(lengths, shape).map_err(|_| strlength_error(&STRLENGTH_ERROR_INTERNAL))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn strlength_char_array(array: CharArray) -> BuiltinResult<Value> {
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
    let tensor = Tensor::new(lengths, vec![rows, 1])
        .map_err(|_| strlength_error(&STRLENGTH_ERROR_INTERNAL))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn strlength_cell_array(cell: CellArray) -> BuiltinResult<Value> {
    let (data, _, rows, cols) = cell.into_parts();
    let mut lengths = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let idx = row * cols + col;
            let value: &Value = &data[idx];
            let length = match value {
                Value::String(text) => string_scalar_length(text.as_str()),
                Value::StringArray(sa) if sa.data().len() == 1 => {
                    string_scalar_length(sa.data()[0].as_str())
                }
                Value::CharArray(char_vec) if char_vec.rows == 1 => char_vec.cols as f64,
                Value::CharArray(_) => {
                    return Err(strlength_error(&STRLENGTH_ERROR_INVALID_CELL_ELEMENT));
                }
                _ => return Err(strlength_error(&STRLENGTH_ERROR_INVALID_CELL_ELEMENT)),
            };
            lengths.push(length);
        }
    }
    let tensor = Tensor::new(lengths, vec![rows, cols])
        .map_err(|_| strlength_error(&STRLENGTH_ERROR_INTERNAL))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn string_scalar_length(text: Option<&str>) -> f64 {
    match text {
        None => f64::NAN,
        Some(text) => text.chars().count() as f64,
    }
}

fn trimmed_row_length(array: &CharArray, row: usize) -> usize {
    let cols = array.cols;
    let mut end = cols;
    while end > 0 {
        let ch = array.data()[row * cols + end - 1];
        if ch.raw() == u16::from(b' ') {
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

    fn strlength_builtin(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::strlength_builtin(value))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_rejects_excess_outputs_before_measuring() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error =
            strlength_builtin(Value::String("text".into())).expect_err("fixed output bound");
        assert_eq!(
            error.identifier(),
            STRLENGTH_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_string_scalar() {
        let result = strlength_builtin(Value::String("RunMat".into())).expect("strlength");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_string_array_with_missing() {
        let array = StringArray::from_elements_unbounded(
            vec!["alpha".into(), runmat_value::StringElement::Missing],
            vec![2, 1],
        )
        .unwrap();
        let result = strlength_builtin(Value::StringArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[2, 1]);
                assert_eq!(tensor.materialize_f64().len(), 2);
                assert_eq!(tensor.materialize_f64()[0], 5.0);
                assert!(tensor.materialize_f64()[1].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_array_multiple_rows() {
        let data: Vec<char> = vec!['c', 'a', 't', ' ', ' ', 'h', 'o', 'r', 's', 'e'];
        let array = CharArray::from_unicode_scalars(data, 2, 5).unwrap();
        let result = strlength_builtin(Value::CharArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[2, 1]);
                assert_eq!(tensor.materialize_f64(), vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_vector_retains_explicit_spaces() {
        let data: Vec<char> = "hi   ".chars().collect();
        let array = CharArray::from_unicode_scalars(data, 1, 5).unwrap();
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
                assert_eq!(tensor.shape(), &[1, 2]);
                assert_eq!(tensor.materialize_f64(), vec![3.0, 5.0]);
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
                Value::String(runmat_value::StringElement::Missing),
            ],
            1,
            3,
        )
        .unwrap();
        let result = strlength_builtin(Value::Cell(cell)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 3]);
                assert_eq!(tensor.materialize_f64().len(), 3);
                assert_eq!(tensor.materialize_f64()[0], 5.0);
                assert_eq!(tensor.materialize_f64()[1], 4.0);
                assert!(tensor.materialize_f64()[2].is_nan());
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
                assert_eq!(tensor.shape(), &[2, 2]);
                assert_eq!(tensor.materialize_f64(), vec![2.0, 1.0, 3.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_char_array_trims_padding() {
        let data: Vec<char> = vec!['d', 'o', 'g', ' ', ' ', 'h', 'o', 'r', 's', 'e'];
        let array = CharArray::from_unicode_scalars(data, 2, 5).unwrap();
        let result = strlength_builtin(Value::CharArray(array)).expect("strlength");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[2, 1]);
                assert_eq!(tensor.materialize_f64(), vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strlength_errors_on_invalid_input() {
        let err = error_message(strlength_builtin(Value::Num(1.0)).unwrap_err());
        assert_eq!(err, STRLENGTH_ERROR_INVALID_INPUT.message);
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
        let err = error_message(strlength_builtin(Value::Cell(cell)).unwrap_err());
        assert_eq!(err, STRLENGTH_ERROR_INVALID_CELL_ELEMENT.message);
    }
}
