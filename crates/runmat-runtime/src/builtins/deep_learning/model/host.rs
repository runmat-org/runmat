//! Host forward and predict kernels and numerical representation boundaries.
use super::*;

pub(crate) fn predict_deep_learning_object(
    object: ObjectInstance,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    let options = parse_predict_options(rest)?;
    let tensor = input_tensor(xnew, "predict")?;
    let input = match options.observations_in {
        ObservationsIn::Rows => tensor,
        ObservationsIn::Columns => transpose_2d(&tensor, "predict")?,
    };
    let scores = evaluate_network(&object, input, "predict")?;
    Ok(vec![Value::Tensor(scores)])
}

pub(super) fn input_tensor(value: Value, function: &'static str) -> BuiltinResult<Tensor> {
    let tensor = match value {
        Value::Object(mut object) if object.class_name.is(super::DLARRAY_CLASS) => {
            let data = object.properties.remove("Data").ok_or_else(|| {
                deep_learning_error(function, format!("{function}: dlarray is missing Data"))
            })?;
            input_tensor(data, function)
        }
        Value::GpuTensor(handle) => {
            let tensor = futures::executor::block_on(
                crate::builtins::common::gpu_helpers::gather_tensor_async(&handle),
            )?;
            Ok(tensor)
        }
        other => crate::builtins::common::tensor::value_into_tensor_for(function, other)
            .map_err(|err| deep_learning_error(function, format!("{function}: {err}"))),
    }?;
    normalize_numeric_tensor(tensor, function)
}

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

pub(super) fn evaluate_network(
    object: &ObjectInstance,
    input: Tensor,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let layers = object
        .properties
        .get("Layers")
        .map(|value| layer_slice(value, function))
        .transpose()?
        .unwrap_or(&[]);
    validate_forward_layers(layers, function)?;
    let mut current = require_2d(input, function, "input")?;
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
                let expected = feature_input_width(layer, function)?;
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
                fully_connected_forward(current, layer, function)?
            }
            Some(ForwardLayerKind::Relu) => map_tensor(current, |value| value.max(0.0), function)?,
            Some(ForwardLayerKind::Elu) => elu_forward(current, layer, function)?,
            Some(ForwardLayerKind::Softmax) => softmax_rows(current, function)?,
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
    Ok(current)
}

fn fully_connected_forward(
    input: Tensor,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let weights = tensor_property(layer, "Weights", function)?;
    if weights.shape.len() > 2 || weights.cols != input.cols {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Weights must be outputSize-by-{}",
                input.cols
            ),
        ));
    }
    let bias = tensor_property(layer, "Bias", function)?;
    if tensor::tensor_element_len(&bias) != weights.rows {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Bias must have {} elements",
                weights.rows
            ),
        ));
    }
    let dtype = input.numeric_dtype();
    let input_values = floating_tensor_values(&input, function)?;
    let weight_values = floating_tensor_values(&weights, function)?;
    let bias_values = floating_tensor_values(&bias, function)?;
    let mut out = vec![0.0; input.rows * weights.rows];
    for row in 0..input.rows {
        for out_col in 0..weights.rows {
            let mut acc = bias_values[out_col];
            for feature in 0..input.cols {
                acc += input_values[row + feature * input.rows]
                    * weight_values[out_col + feature * weights.rows];
            }
            out[row + out_col * input.rows] = acc;
        }
    }
    floating_tensor_from_f64(out, vec![input.rows, weights.rows], dtype, function)
}

fn elu_forward(
    input: Tensor,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let alpha = elu_alpha(layer, function)?;
    map_tensor(
        input,
        |value| {
            if value > 0.0 {
                value
            } else {
                alpha * (value.exp() - 1.0)
            }
        },
        function,
    )
}

pub(super) fn elu_alpha(layer: &ObjectInstance, function: &'static str) -> BuiltinResult<f64> {
    Ok(layer
        .properties
        .get("Alpha")
        .map(|value| super::super::numeric_scalar(value, function, "Alpha"))
        .transpose()?
        .unwrap_or(1.0))
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

fn require_2d(tensor: Tensor, function: &'static str, label: &str) -> BuiltinResult<Tensor> {
    if tensor.shape.len() > 2 {
        return Err(deep_learning_error(
            function,
            format!("{function}: {label} must be a 2-D numeric matrix"),
        ));
    }
    Ok(tensor)
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

fn floating_tensor_values(tensor: &Tensor, function: &'static str) -> BuiltinResult<Vec<f64>> {
    match tensor
        .clone()
        .into_numeric_storage()
        .map_err(|err| deep_learning_error(function, err))?
    {
        NumericStorage::F64(values) => Ok(values),
        NumericStorage::F32(values) => Ok(values.into_iter().map(f64::from).collect()),
        storage => Err(deep_learning_error(
            function,
            format!(
                "{function}: model forward requires double or single tensors, got {}",
                storage.class_name()
            ),
        )),
    }
}

pub(super) fn floating_tensor_from_f64(
    data: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    match dtype {
        NumericDType::F64 | NumericDType::F32 => Tensor::new_with_dtype(data, shape, dtype)
            .map_err(|err| deep_learning_error(function, err)),
        dtype => Err(deep_learning_error(
            function,
            format!(
                "{function}: model forward cannot construct {} output",
                dtype.class_name()
            ),
        )),
    }
}

#[derive(Clone, Copy)]
enum ObservationsIn {
    Rows,
    Columns,
}

struct PredictOptions {
    observations_in: ObservationsIn,
}

fn parse_predict_options(values: Vec<Value>) -> BuiltinResult<PredictOptions> {
    if !values.len().is_multiple_of(2) {
        return Err(deep_learning_error(
            "predict",
            "predict: name-value options must be paired",
        ));
    }
    let mut options = PredictOptions {
        observations_in: ObservationsIn::Rows,
    };
    let mut idx = 0usize;
    while idx < values.len() {
        let name = scalar_text(&values[idx], "predict")?.to_ascii_lowercase();
        match name.as_str() {
            "observationsin" => {
                let value = scalar_text(&values[idx + 1], "predict")?;
                options.observations_in = match value.to_ascii_lowercase().as_str() {
                    "rows" => ObservationsIn::Rows,
                    "columns" => ObservationsIn::Columns,
                    other => {
                        return Err(deep_learning_error(
                            "predict",
                            format!("predict: unsupported ObservationsIn '{other}'"),
                        ));
                    }
                };
            }
            other => {
                return Err(deep_learning_error(
                    "predict",
                    format!("predict: unknown option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    Ok(options)
}
