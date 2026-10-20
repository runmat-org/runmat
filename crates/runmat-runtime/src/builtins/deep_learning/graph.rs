use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, StructValue, Value};

use crate::{BuiltinResult, RuntimeError};

use super::{layer_names, layers_from_value, object, string_array};

fn graph_error(function: &'static str, message: impl Into<String>) -> RuntimeError {
    super::errors::descriptor_error(function, message, &runmat_builtins::GRAPH_INVALID_INPUT)
}

#[runtime_builtin(
    name = "layerGraph",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::graph"
)]
pub(super) async fn layer_graph_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output("layerGraph", &runmat_builtins::GRAPH_INVALID_INPUT)?;
    if args.len() > 1 {
        return Err(graph_error(
            "layerGraph",
            "layerGraph: expected zero inputs or one layer collection",
        ));
    }
    let layers = if args.is_empty() {
        Vec::new()
    } else if matches!(
        args.first(),
        Some(Value::Object(_) | Value::Cell(_) | Value::OutputList(_))
    ) {
        layers_from_value(args[0].clone(), "layerGraph")?
    } else {
        return Err(graph_error(
            "layerGraph",
            "layerGraph: layers must be a layer object or cell collection",
        ));
    };
    let names = layer_names(&layers, "layerGraph")?;
    let layer_count = layers.len();
    let layers_cell = CellArray::new(layers, layer_count, 1)
        .map(Value::Cell)
        .map_err(|err| graph_error("layerGraph", err))?;
    object(
        "nnet.cnn.LayerGraph",
        vec![
            ("Layers", layers_cell),
            (
                "LayerNames",
                string_array(names, vec![layer_count, 1], "layerGraph")?,
            ),
            ("Connections", Value::Struct(StructValue::new())),
        ],
    )
}

#[runtime_builtin(
    name = "analyzeNetwork",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::graph"
)]
pub(super) async fn analyze_network_builtin(network: Value) -> BuiltinResult<Value> {
    super::errors::admit_single_output("analyzeNetwork", &runmat_builtins::GRAPH_INVALID_INPUT)?;
    let layers = match &network {
        Value::Object(object) if object.class_name().is(super::LAYER_GRAPH_CLASS) => object
            .properties()
            .get("Layers")
            .cloned()
            .map(|value| layers_from_value(value, "analyzeNetwork"))
            .transpose()?
            .unwrap_or_default(),
        Value::Object(_) | Value::Cell(_) | Value::OutputList(_) => {
            layers_from_value(network.clone(), "analyzeNetwork")?
        }
        _ => {
            return Err(graph_error(
                "analyzeNetwork",
                "analyzeNetwork: expected a layer graph, layer object, or cell collection",
            ));
        }
    };
    let names = layer_names(&layers, "analyzeNetwork")?;
    object(
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
    )
}
