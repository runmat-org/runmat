use super::*;

pub(super) fn loss_and_gradient(
    layers: &[Value],
    x: &Matrix,
    targets: &Task,
    function: &'static str,
) -> BuiltinResult<(f64, Matrix, Vec<LayerCache>)> {
    let (prediction, caches) = forward_training(layers, x, function)?;
    match targets {
        Task::Regression(target) => {
            if prediction.rows != target.rows || prediction.cols != target.cols {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: regression target size must match network output size {:?}",
                        [prediction.rows, prediction.cols]
                    ),
                ));
            }
            let mut loss = 0.0;
            let mut grad = Matrix::zeros(prediction.rows, prediction.cols);
            for idx in 0..prediction.data.len() {
                let diff = prediction.data[idx] - target.data[idx];
                loss += 0.5 * diff * diff;
                grad.data[idx] = diff / prediction.rows as f64;
            }
            Ok((loss / prediction.rows as f64, grad, caches))
        }
        Task::Classification(target) => {
            if prediction.rows != target.one_hot.rows || prediction.cols != target.one_hot.cols {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: classification target size must match network output size {:?}",
                        [prediction.rows, prediction.cols]
                    ),
                ));
            }
            let has_softmax = layers.iter().any(|layer| {
                matches!(
                    layer,
                    Value::Object(object) if object.class_name.is(super::super::SOFTMAX_LAYER_CLASS)
                )
            });
            let probabilities = if has_softmax {
                prediction.clone()
            } else {
                softmax(&prediction, function)?
            };
            let eps = 1.0e-12;
            let mut loss = 0.0;
            let mut grad = Matrix::zeros(probabilities.rows, probabilities.cols);
            if has_softmax {
                for idx in 0..probabilities.data.len() {
                    let probability = probabilities.data[idx].clamp(eps, 1.0);
                    let target_value = target.one_hot.data[idx];
                    loss -= target_value * probability.ln();
                    grad.data[idx] = if target_value == 0.0 {
                        0.0
                    } else {
                        -target_value / probability / probabilities.rows as f64
                    };
                }
            } else {
                for idx in 0..probabilities.data.len() {
                    let probability = probabilities.data[idx].clamp(eps, 1.0);
                    let target_value = target.one_hot.data[idx];
                    loss -= target_value * probability.ln();
                    grad.data[idx] =
                        (probabilities.data[idx] - target_value) / probabilities.rows as f64;
                }
            }
            Ok((loss / probabilities.rows as f64, grad, caches))
        }
    }
}

enum LayerCache {
    FullyConnected {
        layer_index: usize,
        input: Matrix,
        weights: Matrix,
    },
    Relu {
        input: Matrix,
    },
    Elu {
        input: Matrix,
        alpha: f64,
    },
    Softmax {
        output: Matrix,
    },
}

fn forward_training(
    layers: &[Value],
    input: &Matrix,
    function: &'static str,
) -> BuiltinResult<(Matrix, Vec<LayerCache>)> {
    let mut current = input.clone();
    let mut caches = Vec::new();
    let mut saw_input = false;
    for (layer_index, layer) in layers.iter().enumerate() {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        current = match ForwardLayerKind::from_identity(&layer.class_name) {
            Some(ForwardLayerKind::FeatureInput) => {
                let expected = model::feature_input_width(layer, function)?;
                if current.cols != expected {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: input has {} features, but featureInputLayer expects {expected}",
                            current.cols
                        ),
                    ));
                }
                saw_input = true;
                current
            }
            Some(ForwardLayerKind::FullyConnected) => {
                let weights = Matrix::from_tensor(
                    model::tensor_property(layer, "Weights", function)?,
                    function,
                    "Weights",
                )?;
                let bias = Matrix::from_tensor(
                    model::tensor_property(layer, "Bias", function)?,
                    function,
                    "Bias",
                )?;
                let output = fully_connected(&current, &weights, &bias, function)?;
                caches.push(LayerCache::FullyConnected {
                    layer_index,
                    input: current,
                    weights,
                });
                output
            }
            Some(ForwardLayerKind::Relu) => {
                let previous = current;
                let output = map_matrix(&previous, |value| value.max(0.0));
                caches.push(LayerCache::Relu { input: previous });
                output
            }
            Some(ForwardLayerKind::Elu) => {
                let alpha = layer
                    .properties
                    .get("Alpha")
                    .map(|value| super::super::numeric_scalar(value, function, "Alpha"))
                    .transpose()?
                    .unwrap_or(1.0);
                let previous = current;
                let output = map_matrix(&previous, |value| {
                    if value > 0.0 {
                        value
                    } else {
                        alpha * (value.exp() - 1.0)
                    }
                });
                caches.push(LayerCache::Elu {
                    input: previous,
                    alpha,
                });
                output
            }
            Some(ForwardLayerKind::Softmax) => {
                let output = softmax(&current, function)?;
                caches.push(LayerCache::Softmax {
                    output: output.clone(),
                });
                output
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
    Ok((current, caches))
}

fn fully_connected(
    input: &Matrix,
    weights: &Matrix,
    bias: &Matrix,
    function: &'static str,
) -> BuiltinResult<Matrix> {
    if weights.cols != input.cols {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Weights must be outputSize-by-{}",
                input.cols
            ),
        ));
    }
    if bias.data.len() != weights.rows {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Bias must have {} elements",
                weights.rows
            ),
        ));
    }
    let mut out = Matrix::zeros(input.rows, weights.rows);
    for row in 0..input.rows {
        for out_col in 0..weights.rows {
            let mut acc = bias.data[out_col];
            for feature in 0..input.cols {
                acc += input.get(row, feature) * weights.get(out_col, feature);
            }
            out.set(row, out_col, acc);
        }
    }
    Ok(out)
}

fn map_matrix(input: &Matrix, f: impl Fn(f64) -> f64) -> Matrix {
    Matrix {
        data: input.data.iter().map(|value| f(*value)).collect(),
        rows: input.rows,
        cols: input.cols,
    }
}

fn softmax(input: &Matrix, function: &'static str) -> BuiltinResult<Matrix> {
    let mut out = Matrix::zeros(input.rows, input.cols);
    for row in 0..input.rows {
        let max_value = (0..input.cols)
            .map(|col| input.get(row, col))
            .fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0;
        for col in 0..input.cols {
            let value = (input.get(row, col) - max_value).exp();
            out.set(row, col, value);
            denom += value;
        }
        if !denom.is_finite() || denom <= 0.0 {
            return Err(deep_learning_error(
                function,
                format!("{function}: softmax produced invalid normalization"),
            ));
        }
        for col in 0..input.cols {
            out.set(row, col, out.get(row, col) / denom);
        }
    }
    Ok(out)
}

pub(super) fn backward_update(
    layers: &mut [Value],
    caches: Vec<LayerCache>,
    mut grad: Matrix,
    config: &TrainingConfig,
    state: &mut OptimizerState,
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    for cache in caches.into_iter().rev() {
        grad = match cache {
            LayerCache::Softmax { output } => softmax_backward(&output, &grad),
            LayerCache::Relu { input } => relu_backward(&input, &grad),
            LayerCache::Elu { input, alpha } => elu_backward(&input, &grad, alpha),
            LayerCache::FullyConnected {
                layer_index,
                input,
                weights,
            } => {
                let (grad_input, grad_weights, grad_bias) =
                    fully_connected_backward(&input, &weights, &grad);
                update_fully_connected_layer(
                    &mut layers[layer_index],
                    grad_weights,
                    grad_bias,
                    config,
                    state,
                    iteration,
                    layer_index,
                    function,
                )?;
                grad_input
            }
        };
    }
    Ok(())
}

fn softmax_backward(output: &Matrix, grad: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(output.rows, output.cols);
    for row in 0..output.rows {
        let mut dot = 0.0;
        for col in 0..output.cols {
            dot += grad.get(row, col) * output.get(row, col);
        }
        for col in 0..output.cols {
            out.set(row, col, output.get(row, col) * (grad.get(row, col) - dot));
        }
    }
    out
}

fn relu_backward(input: &Matrix, grad: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(input.rows, input.cols);
    for idx in 0..input.data.len() {
        out.data[idx] = if input.data[idx] > 0.0 {
            grad.data[idx]
        } else {
            0.0
        };
    }
    out
}

fn elu_backward(input: &Matrix, grad: &Matrix, alpha: f64) -> Matrix {
    let mut out = Matrix::zeros(input.rows, input.cols);
    for idx in 0..input.data.len() {
        out.data[idx] = if input.data[idx] > 0.0 {
            grad.data[idx]
        } else {
            grad.data[idx] * alpha * input.data[idx].exp()
        };
    }
    out
}

fn fully_connected_backward(
    input: &Matrix,
    weights: &Matrix,
    grad: &Matrix,
) -> (Matrix, Matrix, Matrix) {
    let mut grad_input = Matrix::zeros(input.rows, input.cols);
    let mut grad_weights = Matrix::zeros(weights.rows, weights.cols);
    let mut grad_bias = Matrix::zeros(weights.rows, 1);

    for row in 0..input.rows {
        for out_col in 0..weights.rows {
            let g = grad.get(row, out_col);
            grad_bias.data[out_col] += g;
            for feature in 0..input.cols {
                grad_weights.data[out_col + feature * weights.rows] += input.get(row, feature) * g;
                grad_input.data[row + feature * input.rows] += g * weights.get(out_col, feature);
            }
        }
    }
    (grad_input, grad_weights, grad_bias)
}

fn update_fully_connected_layer(
    layer: &mut Value,
    grad_weights: Matrix,
    grad_bias: Matrix,
    config: &TrainingConfig,
    state: &mut OptimizerState,
    iteration: usize,
    layer_index: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    let Value::Object(object) = layer else {
        return Err(deep_learning_error(
            function,
            format!("{function}: network layers must be layer objects"),
        ));
    };
    let mut weights = Matrix::from_tensor(
        model::tensor_property(object, "Weights", function)?,
        function,
        "Weights",
    )?;
    let mut bias = Matrix::from_tensor(
        model::tensor_property(object, "Bias", function)?,
        function,
        "Bias",
    )?;
    state.update_parameter(
        &mut weights.data,
        &grad_weights.data,
        config,
        iteration,
        layer_index,
        "Weights",
        function,
    )?;
    state.update_parameter(
        &mut bias.data,
        &grad_bias.data,
        config,
        iteration,
        layer_index,
        "Bias",
        function,
    )?;
    object.properties.insert(
        "Weights".to_string(),
        Value::Tensor(weights.into_tensor(function)?),
    );
    object.properties.insert(
        "Bias".to_string(),
        Value::Tensor(bias.into_tensor(function)?),
    );
    Ok(())
}
