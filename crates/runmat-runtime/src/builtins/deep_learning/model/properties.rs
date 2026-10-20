//! Borrowed numerical layer metadata and host parameter decoding.
use super::*;
#[cfg(test)]
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;

pub(in crate::builtins) fn feature_input_width(
    object: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<usize> {
    let values = object
        .properties()
        .get("InputSize")
        .ok_or_else(|| deep_learning_error(function, "featureInputLayer is missing InputSize"))
        .and_then(|value| numeric_vector(value, function, "InputSize"))?;
    if values.len() > 1 {
        return Err(deep_learning_error(
            function,
            "featureInputLayer with vector InputSize is metadata-only; this forward path requires one feature dimension",
        ));
    }
    let Some(width) = values.first() else {
        return Err(deep_learning_error(
            function,
            "featureInputLayer InputSize must not be empty",
        ));
    };
    Ok(*width)
}

pub(in crate::builtins) fn positive_property_usize(
    object: &ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<usize> {
    let values = object.properties().get(name).ok_or_else(|| {
        deep_learning_error(function, format!("{function}: layer is missing {name}"))
    })?;
    match positive_usize(values, function, name) {
        Ok(value) => Ok(value),
        Err(_) => Err(deep_learning_error(
            function,
            format!("{function}: {name} must be a positive integer scalar"),
        )),
    }
}

pub(in crate::builtins) fn tensor_property(
    object: &ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let value = tensor_property_value(object, name, function)?.clone();
    crate::builtins::common::tensor::value_into_tensor_for(function, value)
        .map_err(|err| deep_learning_error(function, format!("{function}: {err}")))
        .and_then(|tensor| normalize_numeric_tensor(tensor, function))
}

/// Borrow the shape of a consumed numeric parameter without materializing its
/// host payload or downloading a resident buffer. Callers perform this pure
/// admission before any selective preparation of Weights and Bias.
pub(in crate::builtins) fn tensor_property_shape<'a>(
    object: &'a ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<&'a [usize]> {
    let value = tensor_property_value(object, name, function)?;
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape()),
        Value::LogicalArray(array) => Ok(&array.shape),
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            Ok(handle.shape())
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(&[1, 1]),
        other => Err(deep_learning_error(
            function,
            format!("{function}: {name} must have real numeric array storage, got {other:?}"),
        )),
    }
}

pub(in crate::builtins) fn tensor_property_value<'a>(
    object: &'a ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<&'a Value> {
    let mut value = object.properties().get(name).ok_or_else(|| {
        deep_learning_error(function, format!("{function}: layer is missing {name}"))
    })?;
    if let Value::Object(dlarray) = value {
        if dlarray.class_name().is(super::DLARRAY_CLASS) {
            value = dlarray.properties().get("Data").ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: {name} dlarray is missing Data"),
                )
            })?;
        }
    }
    Ok(value)
}

/// Preflight the exact fully connected dimensions using only borrowed
/// structural metadata. Neither unrelated network properties nor parameter
/// payloads are traversed.
pub(in crate::builtins) fn fully_connected_output_width(
    layer: &ObjectInstance,
    input_width: usize,
    function: &'static str,
) -> BuiltinResult<usize> {
    let weights = tensor_property_shape(layer, "Weights", function)?;
    let (rows, cols) = match weights {
        [width] => (1, *width),
        [rows, cols] => (*rows, *cols),
        _ => {
            return Err(deep_learning_error(
                function,
                format!(
                    "{function}: fullyConnectedLayer Weights must be outputSize-by-{input_width}"
                ),
            ));
        }
    };
    if cols != input_width {
        return Err(deep_learning_error(
            function,
            format!("{function}: fullyConnectedLayer Weights must be outputSize-by-{input_width}"),
        ));
    }
    if positive_property_usize(layer, "OutputSize", function)? != rows {
        return Err(deep_learning_error(
            function,
            format!("{function}: fullyConnectedLayer OutputSize must match Weights rows"),
        ));
    }
    let bias = tensor_property_shape(layer, "Bias", function)?;
    if bias
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        != Some(rows)
    {
        return Err(deep_learning_error(
            function,
            format!("{function}: fullyConnectedLayer Bias must have {rows} elements"),
        ));
    }
    Ok(rows)
}

pub(super) fn normalize_numeric_tensor(
    tensor: Tensor,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    tensor::integer_tensor_to_f64(tensor).map_err(|err| deep_learning_error(function, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn dense_property_admission_preserves_rank_one_weights_and_rank_zero_bias() {
        let mut layer = ObjectInstance::new(
            runmat_builtins::DeepLearningLayerKind::FullyConnected
                .class()
                .owned(),
        );
        layer
            .replace_dl_property("OutputSize", Value::Num(1.))
            .expect("fixture property");
        layer
            .replace_dl_property(
                "Weights",
                Value::Tensor(Tensor::from_f32(vec![1., 2.], vec![2]).unwrap()),
            )
            .expect("fixture property");
        layer
            .replace_dl_property(
                "Bias",
                Value::Tensor(Tensor::from_f32(vec![0.], Vec::new()).unwrap()),
            )
            .expect("fixture property");
        assert_eq!(
            tensor_property_shape(&layer, "Weights", "forward").unwrap(),
            &[2]
        );
        assert!(tensor_property_shape(&layer, "Bias", "forward")
            .unwrap()
            .is_empty());
        assert_eq!(
            fully_connected_output_width(&layer, 2, "forward").unwrap(),
            1
        );
        layer
            .replace_dl_property(
                "Weights",
                Value::Tensor(Tensor::from_f32(vec![1.], Vec::new()).unwrap()),
            )
            .expect("fixture property");
        assert!(fully_connected_output_width(&layer, 1, "forward").is_err());
    }
}
