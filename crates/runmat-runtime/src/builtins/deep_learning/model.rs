use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CellArray, ObjectInstance, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{builtins::common::tensor, BuiltinResult};

use super::{
    any_type, autodiff, deep_learning_error, gather_args, layer_names, layers_from_value,
    logical_scalar, numeric_values, numeric_vector, object, parse_name_values, positive_usize,
    scalar_text, string_array, tensor_value, unsupported_error,
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
    if let Some(handle) = gpu_backed_dlarray_handle(&input) {
        let network_object = require_network_object(network, "forward")?;
        let format = dlarray_format(&input);
        let output = evaluate_network_gpu(&network_object, handle, "forward").await?;
        let wrapped = object(
            "dlarray",
            vec![
                (
                    "Data",
                    crate::builtins::common::gpu_helpers::resident_gpu_value(output),
                ),
                ("Format", Value::String(format.clone())),
                ("Labels", Value::String(format)),
            ],
        );
        return autodiff::annotate_dlarray_value(wrapped);
    }
    let network = crate::gather_if_needed_async(&network).await?;
    let input = crate::gather_if_needed_async(&input).await?;
    let wrap_dlarray = matches!(&input, Value::Object(object) if object.class_name == "dlarray");
    let format = dlarray_format(&input);
    let traced_input = input.clone();
    let tensor = input_tensor(input, "forward")?;
    let network_object = require_network_object(network, "forward")?;
    if wrap_dlarray && autodiff::tape_is_active() {
        if let Some(value) =
            autodiff::record_network_forward(&network_object, &traced_input, "forward")?
        {
            return Ok(value);
        }
    }
    let output = evaluate_network(&network_object, tensor, "forward")?;
    if wrap_dlarray {
        let wrapped = object(
            "dlarray",
            vec![
                ("Data", Value::Tensor(output)),
                ("Format", Value::String(format.clone())),
                ("Labels", Value::String(format)),
            ],
        );
        autodiff::annotate_dlarray_value(wrapped)
    } else {
        Ok(Value::Tensor(output))
    }
}

fn gpu_backed_dlarray_handle(value: &Value) -> Option<&GpuTensorHandle> {
    let Value::Object(object) = value else {
        return None;
    };
    if object.class_name != "dlarray" {
        return None;
    }
    match object.properties.get("Data") {
        Some(Value::GpuTensor(handle)) => Some(handle),
        _ => None,
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

pub(super) fn learnables_struct(layers: &[Value], function: &'static str) -> BuiltinResult<Value> {
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
    let tensor = match value {
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
    }?;
    normalize_numeric_tensor(tensor, function)
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

/// Execute the subset of sequential feed-forward networks for which every
/// operation is available through the acceleration provider. The complete
/// layer sequence is validated before the first upload or dispatch so a
/// rejected layer cannot leave behind a partially executed GPU forward pass.
async fn evaluate_network_gpu(
    object: &ObjectInstance,
    input: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let provider = runmat_accelerate_api::provider_for_handle(input).ok_or_else(|| {
        unsupported_error(
            function,
            format!("{function}: no acceleration provider owns the GPU dlarray input"),
        )
    })?;
    let layers = object
        .properties
        .get("Layers")
        .cloned()
        .map(|value| layers_from_value(value, function))
        .transpose()?
        .unwrap_or_default();
    validate_gpu_forward_layers(&layers, input, function)?;

    let mut current = input.clone();
    let mut current_is_temporary = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            unreachable!("validate_gpu_forward_layers accepts layer objects only");
        };
        match layer.class_name.as_str() {
            "nnet.cnn.layer.FullyConnectedLayer" => {
                current = fully_connected_forward_gpu(
                    provider,
                    current,
                    current_is_temporary,
                    &layer,
                    function,
                )
                .await?;
                current_is_temporary = true;
            }
            "nnet.cnn.layer.ReLULayer" => {
                let output = provider.scalar_max(&current, 0.0).map_err(|err| {
                    deep_learning_error(
                        function,
                        format!("{function}: provider-resident ReLU failed: {err}"),
                    )
                })?;
                if current_is_temporary {
                    let _ = provider.free(&current);
                }
                current = output;
                current_is_temporary = true;
            }
            "nnet.cnn.layer.ELULayer" => {
                let alpha = elu_alpha(&layer, function)?;
                let output = match provider.activation_elu(&current, alpha).await {
                    Ok(handle) => handle,
                    Err(err) => {
                        if current_is_temporary {
                            let _ = provider.free(&current);
                        }
                        return Err(deep_learning_error(
                            function,
                            format!("{function}: provider-resident ELU failed: {err}"),
                        ));
                    }
                };
                if current_is_temporary {
                    let _ = provider.free(&current);
                }
                current = output;
                current_is_temporary = true;
            }
            "nnet.cnn.layer.SoftmaxLayer" => {
                let output = match provider.activation_softmax_rows(&current).await {
                    Ok(handle) => handle,
                    Err(err) => {
                        if current_is_temporary {
                            let _ = provider.free(&current);
                        }
                        return Err(deep_learning_error(
                            function,
                            format!("{function}: provider-resident Softmax failed: {err}"),
                        ));
                    }
                };
                if current_is_temporary {
                    let _ = provider.free(&current);
                }
                current = output;
                current_is_temporary = true;
            }
            _ => {}
        }
    }
    Ok(current)
}

fn validate_gpu_forward_layers(
    layers: &[Value],
    input: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<()> {
    if runmat_accelerate_api::handle_storage(input) != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Err(unsupported_error(
            function,
            format!(
                "{function}: provider-resident forward currently requires a real dlarray input"
            ),
        ));
    }
    if input.shape.len() != 2 {
        return Err(unsupported_error(
            function,
            format!("{function}: provider-resident forward currently requires a 2-D dlarray input"),
        ));
    }
    let mut features = input.shape[1];
    let mut saw_input = false;
    let mut saw_transform = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        match layer.class_name.as_str() {
            "nnet.cnn.layer.FeatureInputLayer" if !saw_input => {
                let expected = feature_input_width(layer, function)?;
                if features != expected {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: input has {features} features, but featureInputLayer expects {expected}"
                        ),
                    ));
                }
                saw_input = true;
            }
            "nnet.cnn.layer.FullyConnectedLayer" if saw_input => {
                let weights = tensor_property(layer, "Weights", function)?;
                if weights.shape.len() > 2 || weights.cols != features {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: fullyConnectedLayer Weights must be outputSize-by-{features}"
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
                features = weights.rows;
                saw_transform = true;
            }
            "nnet.cnn.layer.ReLULayer" if saw_input => saw_transform = true,
            "nnet.cnn.layer.ELULayer" if saw_input => {
                let _ = elu_alpha(layer, function)?;
                saw_transform = true;
            }
            "nnet.cnn.layer.SoftmaxLayer" if saw_input => saw_transform = true,
            "nnet.cnn.layer.ClassificationOutputLayer" | "nnet.cnn.layer.RegressionOutputLayer"
                if saw_input && saw_transform => {}
            other => {
                return Err(unsupported_error(
                    function,
                    format!(
                        "{function}: provider-resident forward does not yet support layer type '{other}'"
                    ),
                ));
            }
        }
    }
    if !saw_input {
        return Err(deep_learning_error(
            function,
            format!("{function}: network must start with a supported input layer"),
        ));
    }
    if !saw_transform {
        return Err(unsupported_error(
            function,
            format!(
                "{function}: provider-resident forward requires at least one supported transform layer"
            ),
        ));
    }
    Ok(())
}

async fn fully_connected_forward_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    current: GpuTensorHandle,
    current_is_temporary: bool,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let weights = tensor_property(layer, "Weights", function)?;
    let bias = tensor_property(layer, "Bias", function)?;
    let weights_transposed = transpose_2d(&weights, function)?;
    let bias_row = Tensor::new(bias.data, vec![1, weights.rows])
        .map_err(|err| deep_learning_error(function, err))?;
    let weights_handle = provider
        .upload(&HostTensorView {
            data: &weights_transposed.data,
            shape: &weights_transposed.shape,
        })
        .map_err(|err| {
            deep_learning_error(
                function,
                format!("{function}: GPU weight upload failed: {err}"),
            )
        })?;
    let bias_handle = match provider.upload(&HostTensorView {
        data: &bias_row.data,
        shape: &bias_row.shape,
    }) {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&weights_handle);
            if current_is_temporary {
                let _ = provider.free(&current);
            }
            return Err(deep_learning_error(
                function,
                format!("{function}: GPU bias upload failed: {err}"),
            ));
        }
    };
    let product = match provider.matmul(&current, &weights_handle).await {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&weights_handle);
            let _ = provider.free(&bias_handle);
            if current_is_temporary {
                let _ = provider.free(&current);
            }
            return Err(deep_learning_error(
                function,
                format!("{function}: provider-resident fully connected matmul failed: {err}"),
            ));
        }
    };
    let output = match provider.elem_add(&product, &bias_handle).await {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&weights_handle);
            let _ = provider.free(&bias_handle);
            let _ = provider.free(&product);
            if current_is_temporary {
                let _ = provider.free(&current);
            }
            return Err(deep_learning_error(
                function,
                format!("{function}: provider-resident fully connected bias add failed: {err}"),
            ));
        }
    };
    let _ = provider.free(&weights_handle);
    let _ = provider.free(&bias_handle);
    let _ = provider.free(&product);
    if current_is_temporary {
        let _ = provider.free(&current);
    }
    Ok(output)
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

fn elu_alpha(layer: &ObjectInstance, function: &'static str) -> BuiltinResult<f64> {
    Ok(layer
        .properties
        .get("Alpha")
        .map(|value| numeric_values(value, function, "Alpha"))
        .transpose()?
        .and_then(|values| values.first().copied())
        .unwrap_or(1.0))
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
    let values = tensor::tensor_values_f64_cow(tensor);
    let mut out = vec![0.0; values.len()];
    for row in 0..tensor.rows {
        for col in 0..tensor.cols {
            out[col + row * tensor.cols] = values[row + col * tensor.rows];
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
        .and_then(|value| numeric_vector(value, function, "InputSize"))?;
    let Some(width) = values.first() else {
        return Err(deep_learning_error(
            function,
            "featureInputLayer InputSize must not be empty",
        ));
    };
    Ok(*width)
}

pub(super) fn positive_property_usize(
    object: &ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<usize> {
    let values = object.properties.get(name).ok_or_else(|| {
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

pub(super) fn tensor_property(
    object: &ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let value = object.properties.get(name).cloned().ok_or_else(|| {
        deep_learning_error(function, format!("{function}: layer is missing {name}"))
    })?;
    let value = autodiff::dlarray_data(&value, function)?;
    crate::builtins::common::tensor::value_into_tensor_for(function, value)
        .map_err(|err| deep_learning_error(function, format!("{function}: {err}")))
        .and_then(|tensor| normalize_numeric_tensor(tensor, function))
}

fn normalize_numeric_tensor(tensor: Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    tensor::integer_tensor_to_f64(tensor).map_err(|err| deep_learning_error(function, err))
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
