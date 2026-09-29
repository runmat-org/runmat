use super::*;

pub(super) fn record_activation(
    input: &Value,
    output: Tensor,
    activation: ActivationKind,
) -> BuiltinResult<Value> {
    let payload = Payload::parse(input.clone(), "forward")?;
    let node = if tape_is_active() {
        payload
            .node
            .map(|id| {
                let kind = match activation {
                    ActivationKind::Relu => NodeKind::Relu {
                        input: id,
                        input_value: payload.tensor.clone(),
                    },
                    ActivationKind::Elu { alpha } => NodeKind::Elu {
                        input: id,
                        input_value: payload.tensor.clone(),
                        alpha,
                    },
                    ActivationKind::Softmax => NodeKind::Softmax {
                        input: id,
                        output_value: output.clone(),
                    },
                };
                push_node(output.clone(), kind)
            })
            .transpose()?
    } else {
        None
    };
    Ok(wrap_result(&payload, output, node))
}

pub(super) enum ActivationKind {
    Relu,
    Elu { alpha: f64 },
    Softmax,
}

pub(in crate::builtins::deep_learning) fn record_network_forward(
    network: &ObjectInstance,
    input: &Value,
    function: &'static str,
) -> BuiltinResult<Option<Value>> {
    if !tape_is_active() {
        return Ok(None);
    }
    let layers = network
        .properties
        .get("Layers")
        .cloned()
        .map(|value| super::layers_from_value(value, function))
        .transpose()?
        .unwrap_or_default();
    model::validate_forward_layers(&layers, function)?;
    let mut current = input.clone();
    let mut saw_input = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        current = match ForwardLayerKind::from_identity(&layer.class_name) {
            Some(ForwardLayerKind::FeatureInput) => {
                let tensor = host_tensor_from_value(&current, function, "input")?;
                let expected = model::feature_input_width(&layer, function)?;
                if tensor.cols != expected {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: input has {} features, but featureInputLayer expects {expected}",
                            tensor.cols
                        ),
                    ));
                }
                saw_input = true;
                current
            }
            Some(ForwardLayerKind::FullyConnected) => {
                record_fully_connected(&current, &layer, function)?
            }
            Some(ForwardLayerKind::Relu) => {
                let input_tensor = host_tensor_from_value(&current, function, "input")?;
                let output = map_tensor(input_tensor, |value| value.max(0.0), function)?;
                record_activation(&current, output, ActivationKind::Relu)?
            }
            Some(ForwardLayerKind::Elu) => {
                let input_tensor = host_tensor_from_value(&current, function, "input")?;
                let alpha = layer
                    .properties
                    .get("Alpha")
                    .map(|value| super::numeric_values(value, function, "Alpha"))
                    .transpose()?
                    .and_then(|values| values.first().copied())
                    .unwrap_or(1.0);
                let output = map_tensor(
                    input_tensor,
                    |value| {
                        if value > 0.0 {
                            value
                        } else {
                            alpha * (value.exp() - 1.0)
                        }
                    },
                    function,
                )?;
                record_activation(&current, output, ActivationKind::Elu { alpha })?
            }
            Some(ForwardLayerKind::Softmax) => {
                let input_tensor = host_tensor_from_value(&current, function, "input")?;
                let output = softmax_rows(input_tensor, function)?;
                record_activation(&current, output, ActivationKind::Softmax)?
            }
            Some(kind) if kind.is_output() => current,
            _ => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: unsupported layer type '{}'", layer.class_name),
                ));
            }
        };
    }
    if !saw_input {
        return Err(deep_learning_error(
            function,
            format!("{function}: network must start with a supported input layer"),
        ));
    }
    Ok(Some(current))
}

fn record_fully_connected(
    input: &Value,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Value> {
    let input_payload = Payload::parse(input.clone(), function)?;
    let weights_value = layer.properties.get("Weights").cloned().ok_or_else(|| {
        deep_learning_error(function, format!("{function}: layer is missing Weights"))
    })?;
    let bias_value = layer.properties.get("Bias").cloned().ok_or_else(|| {
        deep_learning_error(function, format!("{function}: layer is missing Bias"))
    })?;
    let weights = Payload::parse(weights_value, function)?;
    let bias = Payload::parse(bias_value, function)?;
    if weights.tensor.shape.len() > 2 || weights.tensor.cols != input_payload.tensor.cols {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Weights must be outputSize-by-{}",
                input_payload.tensor.cols
            ),
        ));
    }
    if tensor::tensor_element_len(&bias.tensor) != weights.tensor.rows {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Bias must have {} elements",
                weights.tensor.rows
            ),
        ));
    }
    let input_values = floating_tensor_values(&input_payload.tensor, function)?;
    let weight_values = floating_tensor_values(&weights.tensor, function)?;
    let bias_values = floating_tensor_values(&bias.tensor, function)?;
    let mut out = vec![0.0; input_payload.tensor.rows * weights.tensor.rows];
    for row in 0..input_payload.tensor.rows {
        for out_col in 0..weights.tensor.rows {
            let mut acc = bias_values[out_col];
            for feature in 0..input_payload.tensor.cols {
                acc += input_values[row + feature * input_payload.tensor.rows]
                    * weight_values[out_col + feature * weights.tensor.rows];
            }
            out[row + out_col * input_payload.tensor.rows] = acc;
        }
    }
    let output = floating_tensor_from_f64(
        out,
        vec![input_payload.tensor.rows, weights.tensor.rows],
        input_payload.dtype,
        function,
    )?;
    let node = if input_payload.node.is_some() || weights.node.is_some() || bias.node.is_some() {
        Some(push_node(
            output.clone(),
            NodeKind::FullyConnected {
                input: input_payload.node,
                weights: weights.node,
                bias: bias.node,
                input_value: input_payload.tensor.clone(),
                weights_value: weights.tensor.clone(),
            },
        )?)
    } else {
        None
    };
    Ok(wrap_result(&input_payload, output, node))
}

fn map_tensor(
    input: Tensor,
    f: impl Fn(f64) -> f64,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let dtype = input.numeric_dtype();
    let shape = input.shape.clone();
    let data = floating_tensor_values(&input, function)?
        .into_iter()
        .map(f)
        .collect();
    floating_tensor_from_f64(data, shape, dtype, function)
}

fn softmax_rows(input: Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    let dtype = input.numeric_dtype();
    let input_values = floating_tensor_values(&input, function)?;
    let mut out = vec![0.0; input_values.len()];
    for row in 0..input.rows {
        let max_value = (0..input.cols)
            .map(|col| input_values[row + col * input.rows])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0;
        for col in 0..input.cols {
            let value = (input_values[row + col * input.rows] - max_value).exp();
            out[row + col * input.rows] = value;
            denom += value;
        }
        if !denom.is_finite() || denom <= 0.0 {
            return Err(deep_learning_error(
                function,
                format!("{function}: softmax produced invalid normalization"),
            ));
        }
        for col in 0..input.cols {
            out[row + col * input.rows] /= denom;
        }
    }
    floating_tensor_from_f64(out, input.shape, dtype, function)
}
