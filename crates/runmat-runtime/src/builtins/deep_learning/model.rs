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
pub(in crate::builtins::deep_learning) mod execution;
mod gpu;
mod host;
mod initialization;
pub(in crate::builtins::deep_learning) mod parameters;
pub(in crate::builtins::deep_learning) mod plan;
pub(in crate::builtins::deep_learning) mod precision;
mod properties;
pub(in crate::builtins::deep_learning) mod state;
mod table_rows;
mod topology;

use admission::{forward_data, gpu_backed_dlarray_handle, require_network_object};
use construction::{layer_slice, parse_network_options};
use gpu::evaluate_network_gpu;
use host::{dlarray_format, elu_alpha, floating_tensor_from_f64, transpose_2d};
use plan::host::HostNetworkPlan;
use properties::normalize_numeric_tensor;

pub(in crate::builtins) use construction::{
    is_deep_learning_network_object, layers_from_network_ref, layers_from_network_value,
    network_object,
};
pub(in crate::builtins) use initialization::initialise_layers;
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
    options.preflight_layers(layers, "dlnetwork")?;
    let mut layers = layers_from_network_value(
        args.into_iter()
            .next()
            .expect("validated first layer argument"),
        "dlnetwork",
    )?;
    if options.initialize {
        initialise_layers(&mut layers, "dlnetwork").await?;
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
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
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
    if autodiff::tape_is_active() && gpu_backed_dlarray_handle(&input).is_some() {
        return Err(unsupported_error(
            "forward",
            "forward: provider-resident network autodiff is not supported",
        ));
    }
    let plan = HostNetworkPlan::admit(network_object, &input, "forward")?;
    if crate::builtins::common::validation::value_has_native_integer_class(forward_data(&input))
        || plan.has_integer_parameter()
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::FORWARD_INTEGER_EXTENSION,
            "forward",
        )?;
    }
    if let Some(handle) = gpu_backed_dlarray_handle(&input) {
        let layers = network_object
            .properties
            .get("Layers")
            .map(|value| layer_slice(value, "forward"))
            .transpose()?
            .unwrap_or(&[]);
        let gpu_layout = gpu::GpuForwardLayout::from_axes(plan.input_axes()).filter(|layout| {
            gpu::validate_gpu_forward_layers(layers, handle, *layout, "forward").is_ok()
        });
        if let Some(layout) = gpu_layout {
            let format = dlarray_format(&input);
            if let Some(output) =
                evaluate_network_gpu(network_object, handle, layout, "forward").await?
            {
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
                let prediction = autodiff::annotate_dlarray_value(wrapped)?;
                return if crate::output_count::current_output_count() == Some(2) {
                    Ok(Value::OutputList(vec![
                        prediction,
                        state::publish_updates(network_object, vec![], "forward")?,
                    ]))
                } else {
                    Ok(prediction)
                };
            }
        }
        if runmat_accelerate_api::handle_is_explicit(handle) {
            return Err(unsupported_error("forward", "forward: this network has no resident execution route for an explicit gpuArray input"));
        }
    }
    let wrap_dlarray =
        matches!(&input, Value::Object(object) if object.class_name.is(super::DLARRAY_CLASS));
    let original_format = dlarray_format(&input);
    let prepared = plan.prepare().await?;
    let host_input = prepared.prepare_input(&input).await?;
    let mut result = prepared.execute(host_input, autodiff::tape_is_active())?;
    let requested_state = crate::output_count::current_output_count() == Some(2);
    let nodes =
        autodiff::record_network_execution(network_object, &input, &mut result, requested_state)?;
    let format = if original_format.is_empty() {
        original_format
    } else {
        prepared.output_axes().layout().labels()
    };
    let data = match result.value {
        execution::host::network::HostValue::Real(tensor) => Value::Tensor(tensor),
        execution::host::network::HostValue::Complex(tensor) => Value::ComplexTensor(tensor),
    };
    let prediction = if wrap_dlarray {
        let wrapped = object(
            "dlarray",
            vec![
                ("Data", data),
                ("Format", Value::String(format.clone())),
                ("Labels", Value::String(format)),
            ],
        );
        wrapped
    } else {
        data
    };
    let state = if requested_state {
        Some(state::publish_updates(
            network_object,
            result.next_state,
            "forward",
        )?)
    } else {
        None
    };
    let (prediction, state) =
        autodiff::annotate_network_outputs(network_object, prediction, state, nodes)?;
    match state {
        Some(state) => Ok(Value::OutputList(vec![prediction, state])),
        None => Ok(prediction),
    }
}
