use runmat_builtins::{ObjectInstance, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::BuiltinResult;

use super::{
    any_type, deep_learning_error, gather_args, model, parse_name_values, scalar_text,
    unsupported_error,
};

const EXPORT_ONNX: &str = "exportONNXNetwork";
const DEFAULT_OPSET_VERSION: i64 = 14;
const MIN_SUPPORTED_OPSET_VERSION: i64 = 13;
const MAX_SUPPORTED_OPSET_VERSION: i64 = 20;
const ONNX_DOUBLE: i32 = 11;

#[derive(Debug)]
struct ExportOptions {
    opset_version: i64,
    batch_size: Option<i64>,
    input_name: Option<String>,
    output_name: Option<String>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            opset_version: DEFAULT_OPSET_VERSION,
            batch_size: None,
            input_name: None,
            output_name: None,
        }
    }
}

#[derive(Debug)]
struct ExportGraph {
    nodes: Vec<Node>,
    initializers: Vec<Initializer>,
    input: ValueInfo,
    output: ValueInfo,
}

#[derive(Debug)]
struct Node {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: Vec<Attribute>,
}

#[derive(Debug)]
struct Initializer {
    name: String,
    dims: Vec<i64>,
    data: Vec<f64>,
}

#[derive(Debug)]
struct ValueInfo {
    name: String,
    batch_size: Option<i64>,
    features: i64,
}

#[derive(Debug)]
enum Attribute {
    Float { name: &'static str, value: f32 },
    Int { name: &'static str, value: i64 },
}

#[runtime_builtin(
    name = "exportONNXNetwork",
    category = "deep_learning",
    summary = "Export supported feed-forward deep-learning networks to ONNX.",
    keywords = "exportONNXNetwork,deep learning,onnx,export",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::onnx"
)]
pub(super) async fn export_onnx_network_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 0) {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork does not return output arguments",
        ));
    }

    let args = gather_args(args).await?;
    if args.len() < 2 {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: expected network and filename",
        ));
    }
    let network = require_network_object(&args[0])?;
    let filename = scalar_text(&args[1], EXPORT_ONNX)?;
    let options = parse_export_options(&args[2..])?;
    let graph = build_export_graph(network, &options)?;
    let bytes = encode_model(&graph, options.opset_version);
    runmat_filesystem::write_async(PathBuf::from(&filename), &bytes)
        .await
        .map_err(|err| {
            deep_learning_error(
                EXPORT_ONNX,
                format!("exportONNXNetwork: failed to write '{filename}': {err}"),
            )
        })?;
    Ok(Value::OutputList(Vec::new()))
}

fn require_network_object(value: &Value) -> BuiltinResult<&ObjectInstance> {
    match value {
        Value::Object(object) if model::is_deep_learning_network_object(object) => Ok(object),
        other => Err(deep_learning_error(
            EXPORT_ONNX,
            format!(
                "exportONNXNetwork: expected a dlnetwork or trained network object, got {other:?}"
            ),
        )),
    }
}

fn parse_export_options(args: &[Value]) -> BuiltinResult<ExportOptions> {
    let map = parse_name_values(args.to_vec(), EXPORT_ONNX)?;
    let mut options = ExportOptions::default();
    for (key, value) in map {
        match key.as_str() {
            "opsetversion" => {
                let version = integer_scalar(&value, "OpsetVersion")?;
                if !(MIN_SUPPORTED_OPSET_VERSION..=MAX_SUPPORTED_OPSET_VERSION).contains(&version) {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        format!(
                            "exportONNXNetwork: OpsetVersion must be between {MIN_SUPPORTED_OPSET_VERSION} and {MAX_SUPPORTED_OPSET_VERSION}"
                        ),
                    ));
                }
                options.opset_version = version;
            }
            "batchsize" => {
                options.batch_size = Some(integer_scalar(&value, "BatchSize")?);
            }
            "inputnames" => {
                options.input_name = Some(single_name(&value, "InputNames")?);
            }
            "outputnames" => {
                options.output_name = Some(single_name(&value, "OutputNames")?);
            }
            "verbose" => {
                let _ = logical_scalar(&value, "Verbose")?;
            }
            other => {
                return Err(deep_learning_error(
                    EXPORT_ONNX,
                    format!("exportONNXNetwork: unsupported option '{other}'"),
                ));
            }
        }
    }
    Ok(options)
}

fn integer_scalar(value: &Value, label: &str) -> BuiltinResult<i64> {
    let n = super::numeric_scalar(value, EXPORT_ONNX, label)?;
    if n.fract().abs() > f64::EPSILON || n < 1.0 || n > i64::MAX as f64 {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            format!("exportONNXNetwork: {label} must be a positive integer scalar"),
        ));
    }
    Ok(n as i64)
}

fn logical_scalar(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Tensor(t) if t.data.len() == 1 && (t.data[0] == 0.0 || t.data[0] == 1.0) => {
            Ok(t.data[0] != 0.0)
        }
        other => Err(deep_learning_error(
            EXPORT_ONNX,
            format!(
                "exportONNXNetwork: {label} must be logical scalar true or false, got {other:?}"
            ),
        )),
    }
}

fn single_name(value: &Value, label: &str) -> BuiltinResult<String> {
    let values = match value {
        Value::String(_) | Value::CharArray(_) => vec![scalar_text(value, EXPORT_ONNX)?],
        Value::StringArray(array) => array.data.clone(),
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
                names.push(scalar_text(item, EXPORT_ONNX)?);
            }
            names
        }
        other => {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                format!("exportONNXNetwork: {label} must be text, string array, or cell text, got {other:?}"),
            ));
        }
    };
    match values.as_slice() {
        [name] if !name.is_empty() => Ok(name.clone()),
        [_] => Err(deep_learning_error(
            EXPORT_ONNX,
            format!("exportONNXNetwork: {label} must not be empty"),
        )),
        _ => Err(deep_learning_error(
            EXPORT_ONNX,
            format!("exportONNXNetwork: this exporter supports exactly one {label} value"),
        )),
    }
}

fn build_export_graph(
    network: &ObjectInstance,
    options: &ExportOptions,
) -> BuiltinResult<ExportGraph> {
    let layers = network
        .properties
        .get("Layers")
        .cloned()
        .map(|value| model::layers_from_network_value(value, EXPORT_ONNX))
        .transpose()?
        .unwrap_or_default();
    model::validate_forward_layers(&layers, EXPORT_ONNX)?;
    if layers.is_empty() {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must contain at least one layer",
        ));
    }

    let layer_names = super::layer_names(&layers, EXPORT_ONNX)?;
    ensure_unique_layer_names(&layer_names)?;
    let input_name = options
        .input_name
        .clone()
        .or(network_name(network, "InputNames")?)
        .unwrap_or_else(|| layer_names[0].clone());
    let output_name = options
        .output_name
        .clone()
        .or(network_name(network, "OutputNames")?)
        .unwrap_or_else(|| "output".to_string());

    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut current_tensor = input_name.clone();
    let mut current_features = None;
    let mut saw_input = false;

    for (idx, layer_value) in layers.iter().enumerate() {
        let Value::Object(layer) = layer_value else {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                "exportONNXNetwork: network layers must be layer objects",
            ));
        };
        let layer_name = layer_names[idx].clone();
        match layer.class_name.as_str() {
            "nnet.cnn.layer.FeatureInputLayer" => {
                if idx != 0 {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        "exportONNXNetwork: featureInputLayer must be the first exported layer",
                    ));
                }
                current_features = Some(model::feature_input_width(layer, EXPORT_ONNX)? as i64);
                saw_input = true;
            }
            "nnet.cnn.layer.FullyConnectedLayer" => {
                let input_features = current_features.ok_or_else(|| {
                    deep_learning_error(
                        EXPORT_ONNX,
                        "exportONNXNetwork: fullyConnectedLayer requires a preceding featureInputLayer",
                    )
                })?;
                let weights = model::tensor_property(layer, "Weights", EXPORT_ONNX)?;
                if weights.shape.len() > 2 || weights.cols as i64 != input_features {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        format!(
                            "exportONNXNetwork: fullyConnectedLayer Weights must be outputSize-by-{input_features}"
                        ),
                    ));
                }
                let bias = model::tensor_property(layer, "Bias", EXPORT_ONNX)?;
                if bias.data.len() != weights.rows {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        format!(
                            "exportONNXNetwork: fullyConnectedLayer Bias must have {} elements",
                            weights.rows
                        ),
                    ));
                }
                let output_size =
                    model::positive_property_usize(layer, "OutputSize", EXPORT_ONNX)? as i64;
                if output_size as usize != weights.rows {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        "exportONNXNetwork: fullyConnectedLayer OutputSize must match Weights rows",
                    ));
                }
                let weight_name = format!("{layer_name}.Weights");
                let bias_name = format!("{layer_name}.Bias");
                initializers.push(Initializer {
                    name: weight_name.clone(),
                    dims: vec![weights.rows as i64, weights.cols as i64],
                    data: tensor_to_row_major_2d(&weights),
                });
                initializers.push(Initializer {
                    name: bias_name.clone(),
                    dims: vec![weights.rows as i64],
                    data: bias.data.clone(),
                });
                let next_tensor = format!("{layer_name}/Gemm");
                nodes.push(Node {
                    name: layer_name,
                    op_type: "Gemm".to_string(),
                    inputs: vec![current_tensor, weight_name, bias_name],
                    outputs: vec![next_tensor.clone()],
                    attributes: vec![Attribute::Int {
                        name: "transB",
                        value: 1,
                    }],
                });
                current_tensor = next_tensor;
                current_features = Some(output_size);
            }
            "nnet.cnn.layer.ReLULayer" => {
                let next_tensor = format!("{layer_name}/Relu");
                nodes.push(Node {
                    name: layer_name,
                    op_type: "Relu".to_string(),
                    inputs: vec![current_tensor],
                    outputs: vec![next_tensor.clone()],
                    attributes: vec![],
                });
                current_tensor = next_tensor;
            }
            "nnet.cnn.layer.ELULayer" => {
                let alpha = layer
                    .properties
                    .get("Alpha")
                    .map(|value| super::numeric_scalar(value, EXPORT_ONNX, "Alpha"))
                    .transpose()?
                    .unwrap_or(1.0);
                let next_tensor = format!("{layer_name}/Elu");
                nodes.push(Node {
                    name: layer_name,
                    op_type: "Elu".to_string(),
                    inputs: vec![current_tensor],
                    outputs: vec![next_tensor.clone()],
                    attributes: vec![Attribute::Float {
                        name: "alpha",
                        value: alpha as f32,
                    }],
                });
                current_tensor = next_tensor;
            }
            "nnet.cnn.layer.SoftmaxLayer" => {
                let next_tensor = format!("{layer_name}/Softmax");
                nodes.push(Node {
                    name: layer_name,
                    op_type: "Softmax".to_string(),
                    inputs: vec![current_tensor],
                    outputs: vec![next_tensor.clone()],
                    attributes: vec![Attribute::Int {
                        name: "axis",
                        value: 1,
                    }],
                });
                current_tensor = next_tensor;
            }
            "nnet.cnn.layer.ClassificationOutputLayer" | "nnet.cnn.layer.RegressionOutputLayer" => {
                if idx + 1 != layers.len() {
                    return Err(deep_learning_error(
                        EXPORT_ONNX,
                        "exportONNXNetwork: output layers must be terminal layers",
                    ));
                }
            }
            other => {
                return Err(unsupported_error(
                    EXPORT_ONNX,
                    format!("exportONNXNetwork: layer type '{other}' cannot be exported to ONNX"),
                ));
            }
        }
    }

    if !saw_input {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must start with a supported input layer",
        ));
    }
    let output_features = current_features.ok_or_else(|| {
        deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: exported graph must produce a numeric feature tensor",
        )
    })?;
    if current_tensor != output_name {
        nodes.push(Node {
            name: "output".to_string(),
            op_type: "Identity".to_string(),
            inputs: vec![current_tensor],
            outputs: vec![output_name.clone()],
            attributes: vec![],
        });
    }

    Ok(ExportGraph {
        nodes,
        initializers,
        input: ValueInfo {
            name: input_name,
            batch_size: options.batch_size,
            features: feature_input_width_from_first_layer(&layers)?,
        },
        output: ValueInfo {
            name: output_name,
            batch_size: options.batch_size,
            features: output_features,
        },
    })
}

fn feature_input_width_from_first_layer(layers: &[Value]) -> BuiltinResult<i64> {
    let Some(Value::Object(layer)) = layers.first() else {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must start with a featureInputLayer",
        ));
    };
    if layer.class_name != "nnet.cnn.layer.FeatureInputLayer" {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork: network must start with a featureInputLayer",
        ));
    }
    Ok(model::feature_input_width(layer, EXPORT_ONNX)? as i64)
}

fn ensure_unique_layer_names(names: &[String]) -> BuiltinResult<()> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(deep_learning_error(
                EXPORT_ONNX,
                format!("exportONNXNetwork: duplicate layer name '{name}'"),
            ));
        }
    }
    Ok(())
}

fn network_name(network: &ObjectInstance, property: &str) -> BuiltinResult<Option<String>> {
    let Some(value) = network.properties.get(property) else {
        return Ok(None);
    };
    match value {
        Value::StringArray(array) if array.data.is_empty() => Ok(None),
        Value::StringArray(array) if array.data.len() == 1 && !array.data[0].is_empty() => {
            Ok(Some(array.data[0].clone()))
        }
        Value::StringArray(_) => Err(deep_learning_error(
            EXPORT_ONNX,
            format!("exportONNXNetwork: network {property} metadata must contain exactly one non-empty name"),
        )),
        Value::String(name) if !name.is_empty() => Ok(Some(name.clone())),
        Value::String(_) => Ok(None),
        _ => Ok(None),
    }
}

fn tensor_to_row_major_2d(tensor: &Tensor) -> Vec<f64> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            out.push(tensor.data[row + col * rows]);
        }
    }
    out
}

fn encode_model(graph: &ExportGraph, opset_version: i64) -> Vec<u8> {
    let mut model = Vec::new();
    put_varint_field(&mut model, 1, 10);
    put_string_field(&mut model, 2, "runmat");
    put_string_field(&mut model, 3, env!("CARGO_PKG_VERSION"));
    put_varint_field(&mut model, 5, 1);
    put_message_field(&mut model, 7, &encode_graph(graph));
    put_message_field(&mut model, 8, &encode_opset("", opset_version));
    put_message_field(
        &mut model,
        14,
        &encode_string_pair("runmat.supported_execution", "sequential-feedforward"),
    );
    model
}

fn encode_graph(graph: &ExportGraph) -> Vec<u8> {
    let mut out = Vec::new();
    for node in &graph.nodes {
        put_message_field(&mut out, 1, &encode_node(node));
    }
    put_string_field(&mut out, 2, "RunMatExportedNetwork");
    for initializer in &graph.initializers {
        put_message_field(&mut out, 5, &encode_initializer(initializer));
    }
    put_message_field(&mut out, 11, &encode_value_info(&graph.input));
    put_message_field(&mut out, 12, &encode_value_info(&graph.output));
    out
}

fn encode_node(node: &Node) -> Vec<u8> {
    let mut out = Vec::new();
    for input in &node.inputs {
        put_string_field(&mut out, 1, input);
    }
    for output in &node.outputs {
        put_string_field(&mut out, 2, output);
    }
    put_string_field(&mut out, 3, &node.name);
    put_string_field(&mut out, 4, &node.op_type);
    for attr in &node.attributes {
        put_message_field(&mut out, 5, &encode_attribute(attr));
    }
    out
}

fn encode_attribute(attribute: &Attribute) -> Vec<u8> {
    let mut out = Vec::new();
    match attribute {
        Attribute::Float { name, value } => {
            put_string_field(&mut out, 1, name);
            put_fixed32_field(&mut out, 2, value.to_bits());
            put_varint_field(&mut out, 20, 1);
        }
        Attribute::Int { name, value } => {
            put_string_field(&mut out, 1, name);
            put_varint_field(&mut out, 3, *value as u64);
            put_varint_field(&mut out, 20, 2);
        }
    }
    out
}

fn encode_initializer(initializer: &Initializer) -> Vec<u8> {
    let mut out = Vec::new();
    for dim in &initializer.dims {
        put_varint_field(&mut out, 1, *dim as u64);
    }
    put_varint_field(&mut out, 2, ONNX_DOUBLE as u64);
    put_string_field(&mut out, 8, &initializer.name);
    let mut raw = Vec::with_capacity(initializer.data.len() * 8);
    for value in &initializer.data {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    put_bytes_field(&mut out, 9, &raw);
    out
}

fn encode_value_info(info: &ValueInfo) -> Vec<u8> {
    let mut out = Vec::new();
    put_string_field(&mut out, 1, &info.name);
    put_message_field(&mut out, 2, &encode_type(info));
    out
}

fn encode_type(info: &ValueInfo) -> Vec<u8> {
    let mut tensor = Vec::new();
    put_varint_field(&mut tensor, 1, ONNX_DOUBLE as u64);
    put_message_field(&mut tensor, 2, &encode_shape(info));
    let mut out = Vec::new();
    put_message_field(&mut out, 1, &tensor);
    out
}

fn encode_shape(info: &ValueInfo) -> Vec<u8> {
    let mut out = Vec::new();
    put_message_field(
        &mut out,
        1,
        &encode_dimension(info.batch_size, Some("batch")),
    );
    put_message_field(&mut out, 1, &encode_dimension(Some(info.features), None));
    out
}

fn encode_dimension(value: Option<i64>, param: Option<&str>) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(value) = value {
        put_varint_field(&mut out, 1, value as u64);
    } else if let Some(param) = param {
        put_string_field(&mut out, 2, param);
    }
    out
}

fn encode_opset(domain: &str, version: i64) -> Vec<u8> {
    let mut out = Vec::new();
    if !domain.is_empty() {
        put_string_field(&mut out, 1, domain);
    }
    put_varint_field(&mut out, 2, version as u64);
    out
}

fn encode_string_pair(key: &str, value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    put_string_field(&mut out, 1, key);
    put_string_field(&mut out, 2, value);
    out
}

fn put_string_field(out: &mut Vec<u8>, field: u32, value: &str) {
    put_bytes_field(out, field, value.as_bytes());
}

fn put_message_field(out: &mut Vec<u8>, field: u32, value: &[u8]) {
    put_bytes_field(out, field, value);
}

fn put_bytes_field(out: &mut Vec<u8>, field: u32, value: &[u8]) {
    put_key(out, field, 2);
    put_varint(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn put_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
    put_key(out, field, 0);
    put_varint(out, value);
}

fn put_fixed32_field(out: &mut Vec<u8>, field: u32, value: u32) {
    put_key(out, field, 5);
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_key(out: &mut Vec<u8>, field: u32, wire_type: u8) {
    put_varint(out, ((field as u64) << 3) | wire_type as u64);
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_length_delimited_field(bytes: &[u8], field: u32, payload: &[u8]) -> bool {
        let key = ((field as u64) << 3) | 2;
        let mut expected = Vec::new();
        put_varint(&mut expected, key);
        put_varint(&mut expected, payload.len() as u64);
        expected.extend_from_slice(payload);
        bytes
            .windows(expected.len())
            .any(|window| window == expected.as_slice())
    }

    #[test]
    fn initializer_name_uses_tensor_proto_name_field() {
        let initializer = Initializer {
            name: "fc.Weights".to_string(),
            dims: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let encoded = encode_initializer(&initializer);

        assert!(has_length_delimited_field(&encoded, 8, b"fc.Weights"));
        assert!(!has_length_delimited_field(&encoded, 4, b"fc.Weights"));
    }
}
