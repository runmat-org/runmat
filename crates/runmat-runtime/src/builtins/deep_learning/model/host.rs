//! Output metadata and host parameter layout adaptation.
use super::*;

pub(super) fn dlarray_format(value: &Value) -> String {
    match value {
        Value::Object(object) if object.class_name().is(super::DLARRAY_CLASS) => object
            .properties()
            .get("Format")
            .and_then(|value| match value {
                Value::String(text) => Some(text.to_string()),
                _ => None,
            })
            .unwrap_or_default(),
        _ => String::new(),
    }
}

pub(super) fn elu_alpha(layer: &ObjectInstance, function: &'static str) -> BuiltinResult<f64> {
    Ok(layer
        .properties()
        .get("Alpha")
        .map(|value| super::super::numeric_scalar(value, function, "Alpha"))
        .transpose()?
        .unwrap_or(1.0))
}

fn host_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn transpose_2d(tensor: &Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    if tensor.shape().len() > 2 {
        return Err(deep_learning_error(
            function,
            "predict: ObservationsIn='columns' requires a 2-D matrix",
        ));
    }
    let (rows, cols) = host_matrix_dimensions(tensor);
    let mut indices = vec![0usize; tensor::tensor_element_len(tensor)];
    for row in 0..rows {
        for col in 0..cols {
            indices[col + row * cols] = row + col * rows;
        }
    }
    let storage = tensor
        .clone()
        .into_numeric_storage()
        .map_err(|err| deep_learning_error(function, err))?
        .reorder(&indices)
        .map_err(|err| deep_learning_error(function, err))?;
    Tensor::from_numeric_storage(storage, vec![cols, rows])
        .map_err(|err| deep_learning_error(function, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn host_transpose_preserves_dense_rank_one_matrix_geometry() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let transposed = transpose_2d(&row, "predict").expect("rank-one row transpose");
        assert_eq!(transposed.shape(), [3, 1]);
        assert_eq!(transposed.materialize_f64(), vec![1.0, 2.0, 3.0]);

        let scalar = Tensor::new(vec![1.0], Vec::new()).unwrap();
        assert_eq!(host_matrix_dimensions(&scalar), (0, 0));
        assert!(transpose_2d(&scalar, "predict").is_err());
    }
}
