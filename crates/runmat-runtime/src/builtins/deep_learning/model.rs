//! Sequential network construction and forward execution.

use super::layer_identity::ForwardLayerKind;
use super::{
    deep_learning_error, layer_names, logical_scalar, numeric_vector, object, object_with_identity,
    positive_usize, scalar_text, string_array, tensor_value, unsupported_error, DLARRAY_CLASS,
    ELU_LAYER_CLASS, FULLY_CONNECTED_LAYER_CLASS, LAYER_GRAPH_CLASS, RELU_LAYER_CLASS,
    SOFTMAX_LAYER_CLASS,
};
use crate::{builtins::common::tensor, BuiltinResult};
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ProviderExecutionError, ProviderPrecision,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, NumericDType, ObjectInstance, StructValue, Tensor, Value};

pub(in crate::builtins::deep_learning) mod admission;
mod construction;
pub(in crate::builtins::deep_learning) mod execution;
pub(in crate::builtins::deep_learning) mod gpu;
mod host;
mod initialization;
mod output;
pub(in crate::builtins::deep_learning) mod parameters;
pub(in crate::builtins::deep_learning) mod plan;
pub(in crate::builtins::deep_learning) mod precision;
mod predict;
mod properties;
pub(in crate::builtins::deep_learning) mod state;
mod table_rows;
mod topology;

use admission::require_network_object;
use construction::{layer_slice, parse_network_options};
use host::{elu_alpha, transpose_2d};
use properties::normalize_numeric_tensor;

pub(in crate::builtins) use predict::predict_deep_learning_object;

pub(in crate::builtins) use construction::{
    is_deep_learning_network_object, layers_from_network_ref, layers_from_network_value,
    network_object,
};
pub(in crate::builtins) use initialization::initialise_layers_in_runtime;
pub(in crate::builtins) use parameters::{
    apply as apply_learnables, project as project_learnables, validate_matching_networks,
};
pub(in crate::builtins) use state::{apply as apply_state, project as project_state};

pub(crate) const fn state_storage_property() -> &'static str {
    state::STORAGE_PROPERTY
}

#[runtime_builtin(
    name = "get.Learnables",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::model"
)]
pub(super) async fn get_learnables_builtin(network: Value) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(deep_learning_error(
            "get.Learnables",
            "get.Learnables: at most one output is supported",
        ));
    }
    let Value::Object(network) = network else {
        return Err(deep_learning_error(
            "get.Learnables",
            "get.Learnables: expected network object",
        ));
    };
    project_learnables(&network, "get.Learnables")
}

#[runtime_builtin(
    name = "set.Learnables",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::model"
)]
pub(super) async fn set_learnables_builtin(
    network: Value,
    learnables: Value,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(deep_learning_error(
            "set.Learnables",
            "set.Learnables: at most one output is supported",
        ));
    }
    let Value::Object(network) = network else {
        return Err(deep_learning_error(
            "set.Learnables",
            "set.Learnables: expected network object",
        ));
    };
    apply_learnables(network, learnables, "set.Learnables")
}

#[runtime_builtin(
    name = "get.State",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::model"
)]
pub(super) async fn get_state_builtin(network: Value) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(deep_learning_error(
            "get.State",
            "get.State: at most one output is supported",
        ));
    }
    let Value::Object(network) = network else {
        return Err(deep_learning_error(
            "get.State",
            "get.State: expected network object",
        ));
    };
    project_state(&network, "get.State")
}

#[runtime_builtin(
    name = "set.State",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::model"
)]
pub(super) async fn set_state_builtin(network: Value, state: Value) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(deep_learning_error(
            "set.State",
            "set.State: at most one output is supported",
        ));
    }
    let Value::Object(network) = network else {
        return Err(deep_learning_error(
            "set.State",
            "set.State: expected network object",
        ));
    };
    apply_state(network, state, "set.State")
}
pub(in crate::builtins) use properties::{
    feature_input_width, fully_connected_output_width, positive_property_usize, tensor_property,
    tensor_property_value,
};
pub(in crate::builtins) use topology::{
    validate_forward_layers, validate_sequential_connections, validate_sequential_topology,
};

pub(crate) const DLNETWORK_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::DL_NETWORK;
pub(crate) const SERIES_NETWORK_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::SERIES_NETWORK;
pub(crate) const DAG_NETWORK_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::DAG_NETWORK;

/// Install network property protocols in the active session, including when
/// a value was restored rather than constructed in this invocation.
pub(crate) fn ensure_network_classes_registered() {
    for identity in [DLNETWORK_CLASS, SERIES_NETWORK_CLASS, DAG_NETWORK_CLASS] {
        super::class_registration::ensure_network_class_registered(identity);
    }
}

#[runtime_builtin(
    name = "dlnetwork",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::deep_learning::model"
)]
async fn dlnetwork_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    dlnetwork_in_runtime(runtime.as_ref(), requested_outputs, args).await
}

#[cfg(test)]
pub(super) async fn dlnetwork_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dlnetwork_in_runtime(None, crate::output_count::current_output_count(), args).await
}

async fn dlnetwork_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(deep_learning_error(
            "dlnetwork",
            "dlnetwork: at most one output is supported",
        ));
    }
    let Some((layers_value, rest)) = args.split_first() else {
        return Err(deep_learning_error(
            "dlnetwork",
            "dlnetwork: expected a layer collection or layerGraph object",
        ));
    };
    let options = parse_network_options(rest)?;
    let layers = layers_from_network_ref(layers_value, "dlnetwork")?;
    validate_forward_layers(layers, "dlnetwork")?;
    options.preflight_layers(layers, "dlnetwork")?;
    let mut layers = layers_from_network_value(
        args.into_iter()
            .next()
            .expect("validated first layer argument"),
        "dlnetwork",
    )?;
    if options.initialize {
        initialise_layers_in_runtime(runtime, &mut layers, "dlnetwork").await?;
    }
    network_object(
        DLNETWORK_CLASS,
        layers,
        options.into_properties(),
        "dlnetwork",
    )
}

#[runtime_builtin(
    name = "forward",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::deep_learning::model"
)]
async fn forward_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    network: Value,
    input: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    forward_in_runtime(runtime.as_ref(), requested_outputs, network, input, rest).await
}

#[cfg(test)]
pub(super) async fn forward_builtin(
    network: Value,
    input: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    forward_in_runtime(
        None,
        crate::output_count::current_output_count(),
        network,
        input,
        rest,
    )
    .await
}

async fn forward_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    network: Value,
    input: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if requested_outputs.is_some_and(|count| count > 2) {
        return Err(deep_learning_error(
            "forward",
            "forward: at most two outputs are supported",
        ));
    }
    if !rest.is_empty() {
        return Err(deep_learning_error(
            "forward",
            "forward: name-value options are not supported for RunMat network forward execution",
        ));
    }
    let network_object = require_network_object(&network, "forward")?;
    let requested_state = requested_outputs == Some(2);
    let outcome = execution::execute_network(execution::NetworkExecutionRequest {
        runtime,
        network: network_object,
        input: &input,
        caller: execution::NetworkCaller::Forward,
        output_layout: execution::NetworkOutputLayout::Preserve,
    })
    .await?;
    let mut outputs = output::present(
        network_object,
        &input,
        outcome,
        output::OutputPresentation::Forward,
        requested_state,
    )?;
    if requested_state {
        Ok(Value::OutputList(outputs))
    } else {
        Ok(outputs.remove(0))
    }
}
