use runmat_builtins::{CellArray, StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::{gather_if_needed_async, BuiltinResult};

use super::{
    any_type, deep_learning_error, gather_args, layer_names, layers_from_value, object,
    string_array,
};

#[runtime_builtin(
    name = "layerGraph",
    category = "deep_learning",
    summary = "Create a layer graph compatibility object.",
    keywords = "layerGraph,deep learning,network,layer graph",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::graph"
)]
pub(super) async fn layer_graph_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args(args).await?;
    let layers = if args.is_empty() {
        Vec::new()
    } else if args.len() == 1 {
        layers_from_value(args[0].clone(), "layerGraph")?
    } else {
        return Err(deep_learning_error(
            "layerGraph",
            "layerGraph: expected zero inputs or one layer collection",
        ));
    };
    let names = layer_names(&layers, "layerGraph")?;
    let layer_count = layers.len();
    let layers_cell = CellArray::new(layers, layer_count, 1)
        .map(Value::Cell)
        .map_err(|err| deep_learning_error("layerGraph", err))?;
    Ok(object(
        "nnet.cnn.LayerGraph",
        vec![
            ("Layers", layers_cell),
            (
                "LayerNames",
                string_array(names, vec![layer_count, 1], "layerGraph")?,
            ),
            ("Connections", Value::Struct(StructValue::new())),
        ],
    ))
}

#[runtime_builtin(
    name = "analyzeNetwork",
    category = "deep_learning",
    summary = "Analyze network/layer metadata without opening MATLAB's UI analyzer.",
    keywords = "analyzeNetwork,deep learning,network,analyzer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::graph"
)]
pub(super) async fn analyze_network_builtin(network: Value) -> BuiltinResult<Value> {
    let network = gather_if_needed_async(&network).await?;
    let layers = match &network {
        Value::Object(object) if object.class_name == "nnet.cnn.LayerGraph" => object
            .properties
            .get("Layers")
            .cloned()
            .map(|value| layers_from_value(value, "analyzeNetwork"))
            .transpose()?
            .unwrap_or_default(),
        _ => layers_from_value(network.clone(), "analyzeNetwork")?,
    };
    let names = layer_names(&layers, "analyzeNetwork")?;
    Ok(object(
        "nnet.cnn.NetworkAnalysis",
        vec![
            ("NumLayers", Value::Num(layers.len() as f64)),
            (
                "LayerNames",
                string_array(names, vec![layers.len(), 1], "analyzeNetwork")?,
            ),
            (
                "Diagnostics",
                Value::String("RunMat metadata analysis completed without launching UI".into()),
            ),
        ],
    ))
}
