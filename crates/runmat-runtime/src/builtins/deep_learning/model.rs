//! Sequential network construction and forward execution.

use super::layer_identity::ForwardLayerKind;
use super::{
    autodiff, deep_learning_error, layer_names, logical_scalar, numeric_vector, object,
    object_with_identity, parse_name_values, positive_usize, scalar_text, string_array,
    tensor_value, unsupported_error, DLARRAY_CLASS, ELU_LAYER_CLASS, FULLY_CONNECTED_LAYER_CLASS,
    LAYER_GRAPH_CLASS, RELU_LAYER_CLASS, SOFTMAX_LAYER_CLASS,
};
use crate::{builtins::common::tensor, BuiltinResult};
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ProviderExecutionError, ProviderPrecision,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, NumericDType, NumericStorage, ObjectInstance, StructValue, Tensor, Value,
};

mod admission;
mod construction;
mod gpu;
mod host;
mod parameters;
mod properties;
mod topology;

use admission::{
    forward_data, gpu_backed_dlarray_handle, network_consumed_integer, preflight_forward_network,
    prepare_input_for_host, prepare_network_for_host, require_network_object,
};
use construction::{layer_slice, parse_network_options};
use gpu::evaluate_network_gpu;
use host::{
    dlarray_format, elu_alpha, evaluate_network, floating_tensor_from_f64, input_tensor,
    transpose_2d,
};
use properties::normalize_numeric_tensor;

pub(in crate::builtins) use construction::{
    initialise_fully_connected_layers, is_deep_learning_network_object, layers_from_network_ref,
    layers_from_network_value, network_object,
};
pub(in crate::builtins) use parameters::{
    apply as apply_learnables, project as project_learnables, validate_matching_networks,
};

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
pub(in crate::builtins) use host::predict_deep_learning_object;
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
    builtin_path = "crate::builtins::deep_learning::model"
)]
pub(super) async fn dlnetwork_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
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
    let mut layers = layers_from_network_value(
        args.into_iter()
            .next()
            .expect("validated first layer argument"),
        "dlnetwork",
    )?;
    if options.initialize {
        initialise_fully_connected_layers(&mut layers, "dlnetwork")?;
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
    builtin_path = "crate::builtins::deep_learning::model"
)]
pub(super) async fn forward_builtin(
    network: Value,
    input: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(deep_learning_error(
            "forward",
            "forward: at most one output is supported",
        ));
    }
    if !rest.is_empty() {
        return Err(deep_learning_error(
            "forward",
            "forward: name-value options are not supported for RunMat network forward execution",
        ));
    }
    let network_object = require_network_object(&network, "forward")?;
    if autodiff::tape_is_active() && gpu_backed_dlarray_handle(&input).is_some() {
        return Err(unsupported_error(
            "forward",
            "forward: provider-resident network autodiff is not supported",
        ));
    }
    preflight_forward_network(network_object, &input, "forward")?;
    if crate::builtins::common::validation::value_has_native_integer_class(forward_data(&input))
        || network_consumed_integer(network_object)?
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::FORWARD_INTEGER_EXTENSION,
            "forward",
        )?;
    }
    if let Some(handle) = gpu_backed_dlarray_handle(&input) {
        let format = dlarray_format(&input);
        let output = evaluate_network_gpu(network_object, handle, "forward").await?;
        let wrapped = object(
            "dlarray",
            vec![
                (
                    "Data",
                    crate::builtins::common::gpu_helpers::resident_gpu_value(output)?,
                ),
                ("Format", Value::String(format.clone())),
                ("Labels", Value::String(format)),
            ],
        );
        return autodiff::annotate_dlarray_value(wrapped);
    }
    let prepared_network = prepare_network_for_host(network_object, "forward").await?;
    let input = prepare_input_for_host(input).await?;
    let wrap_dlarray =
        matches!(&input, Value::Object(object) if object.class_name.is(super::DLARRAY_CLASS));
    let format = dlarray_format(&input);
    let traced_input = (wrap_dlarray && autodiff::tape_is_active()).then(|| input.clone());
    let tensor = input_tensor(input, "forward")?;
    if let Some(traced_input) = traced_input {
        if let Some(value) =
            autodiff::record_network_forward(&prepared_network, &traced_input, "forward")?
        {
            return Ok(value);
        }
    }
    let output = evaluate_network(&prepared_network, tensor, "forward")?;
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
