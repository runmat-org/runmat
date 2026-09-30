//! Output metadata and host parameter layout adaptation.
use super::*;

pub(super) fn dlarray_format(value: &Value) -> String {
    match value {
        Value::Object(object) if object.class_name.is(super::DLARRAY_CLASS) => object
            .properties
            .get("Format")
            .and_then(|value| match value {
                Value::String(text) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default(),
        _ => String::new(),
    }
}

pub(super) fn elu_alpha(layer: &ObjectInstance, function: &'static str) -> BuiltinResult<f64> {
    Ok(layer
        .properties
        .get("Alpha")
        .map(|value| super::super::numeric_scalar(value, function, "Alpha"))
        .transpose()?
        .unwrap_or(1.0))
}

pub(super) fn transpose_2d(tensor: &Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    if tensor.shape.len() > 2 {
        return Err(deep_learning_error(
            function,
            "predict: ObservationsIn='columns' requires a 2-D matrix",
        ));
    }
    let mut indices = vec![0usize; tensor::tensor_element_len(tensor)];
    for row in 0..tensor.rows {
        for col in 0..tensor.cols {
            indices[col + row * tensor.cols] = row + col * tensor.rows;
        }
    }
    let storage = tensor
        .clone()
        .into_numeric_storage()
        .map_err(|err| deep_learning_error(function, err))?
        .reorder(&indices)
        .map_err(|err| deep_learning_error(function, err))?;
    Tensor::from_numeric_storage(storage, vec![tensor.cols, tensor.rows])
        .map_err(|err| deep_learning_error(function, err))
}
