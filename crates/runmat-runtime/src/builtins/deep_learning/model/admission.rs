//! Borrowed admission and selective host preparation before provider effects.
use super::*;

pub(super) fn gpu_backed_dlarray_handle(value: &Value) -> Option<&GpuTensorHandle> {
    let Value::Object(object) = value else {
        return None;
    };
    if !object.class_name.is(super::DLARRAY_CLASS) {
        return None;
    }
    match object.properties.get("Data") {
        Some(Value::GpuTensor(handle)) => Some(handle),
        _ => None,
    }
}

pub(super) fn forward_data(value: &Value) -> &Value {
    if let Value::Object(object) = value {
        if object.class_name.is(super::DLARRAY_CLASS) {
            return object.properties.get("Data").unwrap_or(value);
        }
    }
    value
}

/// Prepare only the consumed numeric payload. Host values move unchanged, and
/// unrelated dlarray metadata is never materialized by this execution path.
pub(super) async fn prepare_input_for_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => crate::gather_if_needed_async(&Value::GpuTensor(handle)).await,
        Value::Object(mut object) if object.class_name.is(super::DLARRAY_CLASS) => {
            if let Some(data @ Value::GpuTensor(_)) = object.properties.get_mut("Data") {
                *data = crate::gather_if_needed_async(data).await?;
            }
            Ok(Value::Object(object))
        }
        other => Ok(other),
    }
}

pub(super) fn network_consumed_integer(network: &ObjectInstance) -> BuiltinResult<bool> {
    let layers = network
        .properties
        .get("Layers")
        .map(|value| layer_slice(value, "forward"))
        .transpose()?
        .unwrap_or(&[]);
    for layer in layers {
        let Value::Object(layer) = layer else {
            continue;
        };
        if !layer.class_name.is(super::FULLY_CONNECTED_LAYER_CLASS) {
            continue;
        }
        for name in ["Weights", "Bias"] {
            let value = tensor_property_value(layer, name, "forward")?;
            if crate::builtins::common::validation::value_has_native_integer_class(value) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn forward_input_shape(value: &Value) -> BuiltinResult<&[usize]> {
    match forward_data(value) {
        Value::Tensor(tensor) => Ok(&tensor.shape),
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real =>
        {
            Ok(handle.shape())
        }
        Value::GpuTensor(_) if gpu_backed_dlarray_handle(value).is_some() => {
            Err(unsupported_error(
                "forward",
                "forward: provider-resident forward currently requires a real dlarray input",
            ))
        }
        Value::GpuTensor(_) => Err(deep_learning_error(
            "forward",
            "forward: input must contain real numeric or logical array data",
        )),
        Value::LogicalArray(array) => Ok(&array.shape),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(&[1, 1]),
        other => Err(deep_learning_error(
            "forward",
            format!("forward: input must contain numeric or logical array data, got {other:?}"),
        )),
    }
}

pub(super) fn preflight_forward_network(
    network: &ObjectInstance,
    input: &Value,
    function: &'static str,
) -> BuiltinResult<()> {
    let shape = forward_input_shape(input)?;
    if shape.len() > 2 {
        return Err(deep_learning_error(
            function,
            format!("{function}: input must be a 2-D numeric matrix"),
        ));
    }
    let layers = network
        .properties
        .get("Layers")
        .map(|value| layer_slice(value, function))
        .transpose()?
        .unwrap_or(&[]);
    validate_forward_layers(layers, function)?;
    let mut features = match shape {
        [width] => *width,
        [_, width] => *width,
        _ => 0,
    };
    let mut saw_input = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            unreachable!("validated layer collection contains only objects")
        };
        match ForwardLayerKind::from_identity(&layer.class_name) {
            Some(ForwardLayerKind::FeatureInput) if !saw_input => {
                let expected = feature_input_width(layer, function)?;
                if features != expected {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: input has {features} features, but featureInputLayer expects {expected}"),
                    ));
                }
                saw_input = true;
            }
            Some(ForwardLayerKind::FullyConnected) if saw_input => {
                features = fully_connected_output_width(layer, features, function)?;
            }
            Some(ForwardLayerKind::Elu) if saw_input => {
                let _ = elu_alpha(layer, function)?;
            }
            Some(_) if saw_input => {}
            _ => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: network must start with a supported input layer"),
                ));
            }
        }
    }
    Ok(())
}

pub(super) async fn prepare_network_for_host(
    network: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<ObjectInstance> {
    let layers = network
        .properties
        .get("Layers")
        .map(|value| layer_slice(value, function))
        .transpose()?
        .unwrap_or(&[]);
    let mut prepared = Vec::with_capacity(layers.len());
    for layer in layers {
        let Value::Object(layer) = layer else {
            unreachable!("preflight admitted only layer objects")
        };
        let mut layer = layer.clone();
        if layer.class_name.is(super::FULLY_CONNECTED_LAYER_CLASS) {
            for name in ["Weights", "Bias"] {
                let value = layer
                    .properties
                    .remove(name)
                    .expect("preflight admitted parameter");
                layer
                    .properties
                    .insert(name.into(), prepare_input_for_host(value).await?);
            }
        }
        prepared.push(Value::Object(layer));
    }
    let count = prepared.len();
    let mut result = ObjectInstance::new(network.class_name.clone());
    result.properties.insert(
        "Layers".into(),
        Value::Cell(
            CellArray::new(prepared, count, 1).map_err(|err| deep_learning_error(function, err))?,
        ),
    );
    Ok(result)
}

pub(super) fn require_network_object<'a>(
    value: &'a Value,
    function: &'static str,
) -> BuiltinResult<&'a ObjectInstance> {
    match value {
        Value::Object(object) if is_deep_learning_network_object(object) => Ok(object),
        other => Err(deep_learning_error(
            function,
            format!("{function}: expected a dlnetwork or trained network object, got {other:?}"),
        )),
    }
}
