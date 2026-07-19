use runmat_builtins::{CellArray, ObjectInstance, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::BuiltinResult;

use super::{
    any_type, deep_learning_error, gather_args, layer_names, layers_from_value, numeric_values,
    object, parse_name_values, scalar_text, string_array, tensor_value,
};

pub(crate) const DLNETWORK_CLASS: &str = "dlnetwork";
pub(crate) const SERIES_NETWORK_CLASS: &str = "SeriesNetwork";
pub(crate) const DAG_NETWORK_CLASS: &str = "DAGNetwork";

#[runtime_builtin(
    name = "dlnetwork",
    category = "deep_learning",
    summary = "Create a Deep Learning network compatibility object.",
    keywords = "dlnetwork,deep learning,network,forward,predict",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::model"
)]
pub(super) async fn dlnetwork_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args(args).await?;
    let Some((layers_value, rest)) = args.split_first() else {
        return Err(deep_learning_error(
            "dlnetwork",
            "dlnetwork: expected a layer collection or layerGraph object",
        ));
    };
    let options = parse_name_values(rest.to_vec(), "dlnetwork")?;
    let mut layers = layers_from_network_value(layers_value.clone(), "dlnetwork")?;
    validate_forward_layers(&layers, "dlnetwork")?;
    let initialize = parse_initialize_option(&options)?;
    if initialize {
        initialise_fully_connected_layers(&mut layers, "dlnetwork")?;
    }
    network_object(DLNETWORK_CLASS, layers, options, "dlnetwork")
}

#[runtime_builtin(
    name = "forward",
    category = "deep_learning",
    summary = "Run a supported Deep Learning network forward pass.",
    keywords = "forward,dlnetwork,deep learning,predict",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::model"
)]
pub(super) async fn forward_builtin(
    network: Value,
    input: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(deep_learning_error(
            "forward",
            "forward: name-value options are not supported for RunMat network forward execution",
        ));
    }
    let network = crate::gather_if_needed_async(&network).await?;
    let input = crate::gather_if_needed_async(&input).await?;
    let wrap_dlarray = matches!(&input, Value::Object(object) if object.class_name == "dlarray");
    let format = dlarray_format(&input);
    let tensor = input_tensor(input, "forward")?;
    let network_object = require_network_object(network, "forward")?;
    let output = evaluate_network(&network_object, tensor, "forward")?;
    if wrap_dlarray {
        Ok(object(
            "dlarray",
            vec![
                ("Data", Value::Tensor(output)),
                ("Format", Value::String(format.clone())),
                ("Labels", Value::String(format)),
            ],
        ))
    } else {
        Ok(Value::Tensor(output))
    }
}

pub(crate) fn is_deep_learning_network_object(object: &ObjectInstance) -> bool {
    matches!(
        object.class_name.as_str(),
        DLNETWORK_CLASS | SERIES_NETWORK_CLASS | DAG_NETWORK_CLASS
    )
}

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

pub(super) fn layers_from_network_value(
    value: Value,
    function: &'static str,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Object(object)
            if object.class_name == "nnet.cnn.LayerGraph"
                || is_deep_learning_network_object(&object) =>
        {
            Ok(object
                .properties
                .get("Layers")
                .cloned()
                .map(|value| layers_from_value(value, function))
                .transpose()?
                .unwrap_or_default())
        }
        other => layers_from_value(other, function),
    }
}

pub(super) fn network_object(
    class_name: &str,
    layers: Vec<Value>,
    mut options: std::collections::BTreeMap<String, Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let names = layer_names(&layers, function)?;
    let layer_count = layers.len();
    let input_names = names
        .first()
        .cloned()
        .map(|name| vec![name])
        .unwrap_or_default();
    let output_names = names
        .last()
        .cloned()
        .map(|name| vec![name])
        .unwrap_or_default();
    let layers_cell = CellArray::new(layers.clone(), layer_count, 1)
        .map(Value::Cell)
        .map_err(|err| deep_learning_error(function, err))?;
    let mut properties = vec![
        ("Layers".to_string(), layers_cell),
        (
            "LayerNames".to_string(),
            string_array(names.clone(), vec![layer_count, 1], function)?,
        ),
        (
            "InputNames".to_string(),
            string_array(
                input_names,
                vec![1, names.first().map_or(0, |_| 1)],
                function,
            )?,
        ),
        (
            "OutputNames".to_string(),
            string_array(
                output_names,
                vec![1, names.last().map_or(0, |_| 1)],
                function,
            )?,
        ),
        ("Connections".to_string(), Value::Struct(StructValue::new())),
        (
            "Learnables".to_string(),
            learnables_struct(&layers, function)?,
        ),
        ("State".to_string(), Value::Struct(StructValue::new())),
        ("Initialized".to_string(), Value::Bool(true)),
        (
            "SupportedExecution".to_string(),
            Value::String("sequential-feedforward".into()),
        ),
    ];
    if let Some(value) = remove_case_insensitive_option(&mut options, "Initialize") {
        let initialize = logical_scalar(&value, function, "Initialize")?;
        properties.push(("Initialized".to_string(), Value::Bool(initialize)));
        properties.push(("Initialize".to_string(), Value::Bool(initialize)));
    }
    for (name, value) in options {
        properties.push((canonical_network_option(&name), value));
    }
    Ok(object(class_name, properties))
}

fn canonical_network_option(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "initialize" => "Initialize",
        "inputnames" => "InputNames",
        "outputnames" => "OutputNames",
        other => other,
    }
    .to_string()
}

fn parse_initialize_option(
    options: &std::collections::BTreeMap<String, Value>,
) -> BuiltinResult<bool> {
    let Some(value) = get_case_insensitive_option(options, "Initialize") else {
        return Ok(true);
    };
    logical_scalar(value, "dlnetwork", "Initialize")
}

fn get_case_insensitive_option<'a>(
    options: &'a std::collections::BTreeMap<String, Value>,
    name: &str,
) -> Option<&'a Value> {
    options
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

fn remove_case_insensitive_option(
    options: &mut std::collections::BTreeMap<String, Value>,
    name: &str,
) -> Option<Value> {
    let key = options
        .keys()
        .find(|key| key.eq_ignore_ascii_case(name))
        .cloned()?;
    options.remove(&key)
}

fn logical_scalar(value: &Value, function: &'static str, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) => {
            let n = i.to_f64();
            if n == 0.0 || n == 1.0 {
                Ok(n != 0.0)
            } else {
                Err(deep_learning_error(
                    function,
                    format!("{function}: {label} must be logical scalar true or false"),
                ))
            }
        }
        Value::Tensor(t) if t.data.len() == 1 && (t.data[0] == 0.0 || t.data[0] == 1.0) => {
            Ok(t.data[0] != 0.0)
        }
        other => Err(deep_learning_error(
            function,
            format!("{function}: {label} must be logical scalar true or false, got {other:?}"),
        )),
    }
}

fn learnables_struct(layers: &[Value], function: &'static str) -> BuiltinResult<Value> {
    let mut layer_names = Vec::new();
    let mut parameter_names = Vec::new();
    let mut values = Vec::new();
    for layer in layers {
        let Value::Object(object) = layer else {
            continue;
        };
        if object.class_name != "nnet.cnn.layer.FullyConnectedLayer" {
            continue;
        }
        let name = layer_name(object);
        for parameter in ["Weights", "Bias"] {
            if let Some(value) = object.properties.get(parameter).cloned() {
                layer_names.push(Value::String(name.clone()));
                parameter_names.push(Value::String(parameter.to_string()));
                values.push(value);
            }
        }
    }
    let rows = values.len();
    let mut st = StructValue::new();
    st.insert(
        "Layer",
        CellArray::new(layer_names, rows, 1)
            .map(Value::Cell)
            .map_err(|err| deep_learning_error(function, err))?,
    );
    st.insert(
        "Parameter",
        CellArray::new(parameter_names, rows, 1)
            .map(Value::Cell)
            .map_err(|err| deep_learning_error(function, err))?,
    );
    st.insert(
        "Value",
        CellArray::new(values, rows, 1)
            .map(Value::Cell)
            .map_err(|err| deep_learning_error(function, err))?,
    );
    Ok(Value::Struct(st))
}

pub(super) fn initialise_fully_connected_layers(
    layers: &mut [Value],
    function: &'static str,
) -> BuiltinResult<()> {
    let mut current_width = None;
    for layer in layers {
        let Value::Object(object) = layer else {
            continue;
        };
        match object.class_name.as_str() {
            "nnet.cnn.layer.FeatureInputLayer" => {
                current_width = Some(feature_input_width(object, function)?);
            }
            "nnet.cnn.layer.FullyConnectedLayer" => {
                let input_width = current_width.ok_or_else(|| {
                    deep_learning_error(
                        function,
                        "dlnetwork: fullyConnectedLayer requires a preceding featureInputLayer or fullyConnectedLayer",
                    )
                })?;
                let output_size = positive_property_usize(object, "OutputSize", function)?;
                if !object.properties.contains_key("Weights") {
                    object.properties.insert(
                        "Weights".to_string(),
                        deterministic_weights(output_size, input_width, function)?,
                    );
                }
                if !object.properties.contains_key("Bias") {
                    object.properties.insert(
                        "Bias".to_string(),
                        tensor_value(vec![0.0; output_size], vec![output_size, 1], function)?,
                    );
                }
                current_width = Some(output_size);
            }
            "nnet.cnn.layer.ReLULayer"
            | "nnet.cnn.layer.ELULayer"
            | "nnet.cnn.layer.SoftmaxLayer"
            | "nnet.cnn.layer.ClassificationOutputLayer"
            | "nnet.cnn.layer.RegressionOutputLayer" => {}
            _ => {}
        }
    }
    Ok(())
}

fn deterministic_weights(
    output_size: usize,
    input_width: usize,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut values = Vec::with_capacity(output_size * input_width);
    let scale = (input_width as f64).sqrt().max(1.0);
    for col in 0..input_width {
        for row in 0..output_size {
            let signed = ((row + col + 1) % 5) as f64 - 2.0;
            values.push(signed / (10.0 * scale));
        }
    }
    tensor_value(values, vec![output_size, input_width], function)
}

pub(super) fn validate_forward_layers(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<()> {
    if layers.is_empty() {
        return Err(deep_learning_error(
            function,
            format!("{function}: network must contain at least one layer"),
        ));
    }
    for layer in layers {
        let Value::Object(object) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        match object.class_name.as_str() {
            "nnet.cnn.layer.FeatureInputLayer"
            | "nnet.cnn.layer.FullyConnectedLayer"
            | "nnet.cnn.layer.ReLULayer"
            | "nnet.cnn.layer.ELULayer"
            | "nnet.cnn.layer.SoftmaxLayer"
            | "nnet.cnn.layer.ClassificationOutputLayer"
            | "nnet.cnn.layer.RegressionOutputLayer" => {}
            other => {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: layer type '{other}' is not supported for RunMat forward execution"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn require_network_object(value: Value, function: &'static str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if is_deep_learning_network_object(&object) => Ok(object),
        other => Err(deep_learning_error(
            function,
            format!("{function}: expected a dlnetwork or trained network object, got {other:?}"),
        )),
    }
}

fn input_tensor(value: Value, function: &'static str) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(object) if object.class_name == "dlarray" => {
            let data = object.properties.get("Data").cloned().ok_or_else(|| {
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
    }
}

fn dlarray_format(value: &Value) -> String {
    match value {
        Value::Object(object) if object.class_name == "dlarray" => object
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

fn evaluate_network(
    object: &ObjectInstance,
    input: Tensor,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let layers = object
        .properties
        .get("Layers")
        .cloned()
        .map(|value| layers_from_value(value, function))
        .transpose()?
        .unwrap_or_default();
    validate_forward_layers(&layers, function)?;
    let mut current = require_2d(input, function, "input")?;
    let mut saw_input = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        current = match layer.class_name.as_str() {
            "nnet.cnn.layer.FeatureInputLayer" => {
                let expected = feature_input_width(&layer, function)?;
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
            "nnet.cnn.layer.FullyConnectedLayer" => {
                fully_connected_forward(current, &layer, function)?
            }
            "nnet.cnn.layer.ReLULayer" => map_tensor(current, |value| value.max(0.0), function)?,
            "nnet.cnn.layer.ELULayer" => elu_forward(current, &layer, function)?,
            "nnet.cnn.layer.SoftmaxLayer" => softmax_rows(current, function)?,
            "nnet.cnn.layer.ClassificationOutputLayer" | "nnet.cnn.layer.RegressionOutputLayer" => {
                current
            }
            other => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: unsupported layer type '{other}'"),
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
    if bias.data.len() != weights.rows {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Bias must have {} elements",
                weights.rows
            ),
        ));
    }
    let mut out = vec![0.0; input.rows * weights.rows];
    for row in 0..input.rows {
        for out_col in 0..weights.rows {
            let mut acc = bias.data[out_col];
            for feature in 0..input.cols {
                acc += input.data[row + feature * input.rows]
                    * weights.data[out_col + feature * weights.rows];
            }
            out[row + out_col * input.rows] = acc;
        }
    }
    Tensor::new(out, vec![input.rows, weights.rows])
        .map_err(|err| deep_learning_error(function, err))
}

fn elu_forward(
    input: Tensor,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let alpha = layer
        .properties
        .get("Alpha")
        .map(|value| numeric_values(value, function, "Alpha"))
        .transpose()?
        .and_then(|values| values.first().copied())
        .unwrap_or(1.0);
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

fn softmax_rows(input: Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    let mut out = vec![0.0; input.data.len()];
    for row in 0..input.rows {
        let max_value = (0..input.cols)
            .map(|col| input.data[row + col * input.rows])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0;
        for col in 0..input.cols {
            let value = (input.data[row + col * input.rows] - max_value).exp();
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
    Tensor::new(out, input.shape).map_err(|err| deep_learning_error(function, err))
}

fn map_tensor(
    input: Tensor,
    f: impl Fn(f64) -> f64,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    Tensor::new_with_dtype(
        input.data.into_iter().map(f).collect(),
        input.shape,
        input.dtype,
    )
    .map_err(|err| deep_learning_error(function, err))
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

fn transpose_2d(tensor: &Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    if tensor.shape.len() > 2 {
        return Err(deep_learning_error(
            function,
            "predict: ObservationsIn='columns' requires a 2-D matrix",
        ));
    }
    let mut out = vec![0.0; tensor.data.len()];
    for row in 0..tensor.rows {
        for col in 0..tensor.cols {
            out[col + row * tensor.cols] = tensor.data[row + col * tensor.rows];
        }
    }
    Tensor::new(out, vec![tensor.cols, tensor.rows])
        .map_err(|err| deep_learning_error(function, err))
}

pub(super) fn feature_input_width(
    object: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<usize> {
    let values = object
        .properties
        .get("InputSize")
        .ok_or_else(|| deep_learning_error(function, "featureInputLayer is missing InputSize"))
        .and_then(|value| numeric_values(value, function, "InputSize"))?;
    let Some(width) = values.first() else {
        return Err(deep_learning_error(
            function,
            "featureInputLayer InputSize must not be empty",
        ));
    };
    if !width.is_finite() || *width < 1.0 || width.fract().abs() > f64::EPSILON {
        return Err(deep_learning_error(
            function,
            "featureInputLayer InputSize must contain positive integers",
        ));
    }
    Ok(*width as usize)
}

pub(super) fn positive_property_usize(
    object: &ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<usize> {
    let values = object
        .properties
        .get(name)
        .ok_or_else(|| {
            deep_learning_error(function, format!("{function}: layer is missing {name}"))
        })
        .and_then(|value| numeric_values(value, function, name))?;
    if values.len() != 1
        || !values[0].is_finite()
        || values[0] < 1.0
        || values[0].fract().abs() > f64::EPSILON
    {
        return Err(deep_learning_error(
            function,
            format!("{function}: {name} must be a positive integer scalar"),
        ));
    }
    Ok(values[0] as usize)
}

pub(super) fn tensor_property(
    object: &ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let value = object.properties.get(name).cloned().ok_or_else(|| {
        deep_learning_error(function, format!("{function}: layer is missing {name}"))
    })?;
    crate::builtins::common::tensor::value_into_tensor_for(function, value)
        .map_err(|err| deep_learning_error(function, format!("{function}: {err}")))
}

pub(super) fn layer_name(object: &ObjectInstance) -> String {
    object
        .properties
        .get("Name")
        .and_then(|value| match value {
            Value::String(name) if !name.is_empty() => Some(name.clone()),
            _ => None,
        })
        .unwrap_or_default()
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
