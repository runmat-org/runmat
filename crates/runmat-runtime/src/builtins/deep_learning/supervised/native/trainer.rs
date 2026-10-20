//! Sequential supervised updates over the same admitted kernels as `forward`.
//!
//! Learnables remain owned by network Layers and recurrent State by `_State`.
//! A minibatch reads the canonical State snapshot; next State is not carried
//! between shuffled observations. Only a complete typed parameter step is
//! published to a new network object.

use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use std::collections::BTreeMap;

use runmat_accelerate_api::handle_is_explicit;
use runmat_builtins::DeepLearningAxisLabel;
use runmat_value::{ObjectInstance, Tensor, Value};

use crate::builtins::deep_learning::{deep_learning_error, model, DLARRAY_CLASS};
use crate::BuiltinResult;

use super::super::{
    admit_training_cancellation, admit_training_observations, output_network, training_info,
    LossKind, ShuffleMode, TrainingConfig,
};
use super::{
    admission::{prediction_domain, TrainingCaller, TrainingTarget},
    dataset::BatchSelection,
    factors::FactorPlan,
    gradients::canonical_gradients,
    loss,
    optimizer::NativeOptimizer,
    progress, resources,
};
use model::{
    execution::host::network::{self, HostValue},
    plan::host::HostNetworkPlan,
};

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

fn explicit_resident(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => handle_is_explicit(handle),
        Value::Object(wrapper) if wrapper.class_name().is(DLARRAY_CLASS) => wrapper
            .properties()
            .get("Data")
            .is_some_and(explicit_resident),
        _ => false,
    }
}

fn admit_host_residency(
    source: &Value,
    input: &Value,
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<()> {
    if explicit_resident(input) {
        return Err(fail(
            function,
            "explicit gpuArray input cannot enter host training",
        ));
    }
    for row in model::parameters::view_layers(layers, function)? {
        if row.value.is_some_and(explicit_resident) {
            return Err(fail(
                function,
                "explicit gpuArray Learnables cannot enter host training",
            ));
        }
    }
    if let Value::Object(network) = source {
        if model::is_deep_learning_network_object(network)
            && model::state::view(network, function)?
                .iter()
                .any(|row| explicit_resident(row.value))
        {
            return Err(fail(
                function,
                "explicit gpuArray State cannot enter host training",
            ));
        }
    }
    Ok(())
}

fn batch_value(original: &Value, selected: Tensor) -> BuiltinResult<Value> {
    match original {
        Value::Object(wrapper) if wrapper.class_name().is(DLARRAY_CLASS) => {
            let mut wrapper = wrapper.clone();
            wrapper.replace_dl_property("Data", Value::Tensor(selected))?;
            Ok(Value::Object(wrapper))
        }
        _ => Ok(Value::Tensor(selected)),
    }
}

async fn initialized_network(
    runtime: Option<&crate::context::RuntimeContext>,
    source: &Value,
    caller: TrainingCaller,
) -> BuiltinResult<(ObjectInstance, crate::execution::RootedValueLease)> {
    let _source_roots = crate::execution::structure::replacement::lease(source, runtime)?;
    let function = caller.function();
    if let Value::Object(existing) = source {
        if model::is_deep_learning_network_object(existing) {
            let mut network = existing.clone();
            let Some(Value::Cell(layers)) = network.property("Layers") else {
                return Err(fail(function, "network is missing Layers"));
            };
            let prepared_layers = crate::execution::structure::replacement::cell(layers, runtime)?;
            let _layer_roots = prepared_layers.1;
            let mut layers = prepared_layers.0;
            let layer_values = layers
                .try_data_mut_unique()
                .expect("private Layers destination");
            let _initialized_roots =
                model::initialise_layers_in_runtime(runtime, layer_values, function).await?;
            network.replace_dl_property("Layers", Value::Cell(layers))?;
            model::state::initialize_missing(&mut network, function)?;
            let roots = crate::execution::structure::replacement::lease(
                &Value::Object(network.clone()),
                runtime,
            )?;
            return Ok((network, roots));
        }
    }
    let prepared_layers = model::layers_from_network_value(source.clone(), function)?;
    let _layer_roots = prepared_layers.1;
    let mut layers = prepared_layers.0;
    let _initialized_roots = model::initialise_layers_in_runtime(
        runtime,
        layers
            .try_data_mut_unique()
            .expect("private Layers destination"),
        function,
    )
    .await?;
    let class = match caller {
        TrainingCaller::Trainnet => model::DLNETWORK_CLASS,
        TrainingCaller::TrainNetwork => {
            if matches!(source, Value::Object(object) if object.class_name().is(crate::builtins::deep_learning::LAYER_GRAPH_CLASS))
            {
                model::DAG_NETWORK_CLASS
            } else {
                model::SERIES_NETWORK_CLASS
            }
        }
    };
    let Value::Object(network) =
        model::network_object_from_layer_cell(class, layers, BTreeMap::new(), function)?
    else {
        unreachable!("network constructor produces an object")
    };
    let roots = crate::execution::RootedValueLease::for_traced(&network)
        .map_err(|error| error.into_runtime_error())?;
    Ok((network, roots))
}

fn attach_training_result(
    network: &mut ObjectInstance,
    classes: Option<Value>,
    loss: LossKind,
    config: &TrainingConfig,
    losses: &[f64],
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    network.replace_dl_property(
        "TrainingInfo",
        training_info(config, losses, iteration, function)?,
    )?;
    network.replace_dl_property("TrainedWith", Value::String(config.solver.as_str().into()))?;
    network.replace_dl_property(
        "TrainingLoss",
        Value::String(
            match loss {
                LossKind::Regression => "mse",
                LossKind::Classification => "crossentropy",
            }
            .into(),
        ),
    )?;
    if let Some(classes) = classes {
        network.replace_dl_property("Classes", classes)?;
    }
    Ok(())
}

pub(in crate::builtins::deep_learning::supervised) async fn train(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    caller: TrainingCaller,
    input: &Value,
    target: &Value,
    source: &Value,
    declared_loss: Option<LossKind>,
    config: TrainingConfig,
) -> BuiltinResult<crate::execution::RootedValue> {
    let _argument_roots =
        crate::execution::RootedValueLease::for_values([input, target, source].into_iter())
            .map_err(|error| error.into_runtime_error())?;
    let function = caller.function();
    let layers = model::layers_from_network_ref(source, function)?;
    model::validate_forward_layers(layers, function)?;
    let geometry = model::plan::host::admit_training_geometry(source, input, function)?;
    resources::admit(source, layers, function)?;
    admit_host_residency(source, input, layers, function)?;
    let count = geometry
        .input_axes
        .axis_size(DeepLearningAxisLabel::Batch)
        .ok_or_else(|| fail(function, "training input has no observation axis"))?;
    admit_training_observations(Some(count), function)?;
    let admitted_target =
        TrainingTarget::admit(caller, declared_loss, layers, target, &geometry.output_axes)?;
    let loss_kind = admitted_target.loss();
    let classes = admitted_target.classes_value()?;
    let domain = prediction_domain(layers);
    let factors = FactorPlan::admit(layers, function)?;
    admit_training_cancellation(runtime, function)?;
    // Target preparation is deliberately ahead of initializer callbacks: a
    // malformed numerical target cannot consume random draws or user effects.
    let full_target = admitted_target.prepare(geometry.precision).await?;
    loss::admit_target(
        &full_target,
        &geometry.output_axes,
        loss_kind == LossKind::Classification,
        function,
    )?;
    let prepared_network = initialized_network(runtime, source, caller).await?;
    let mut network_roots = prepared_network.1;
    let mut network = prepared_network.0;
    let full_input = {
        let full_plan = HostNetworkPlan::admit(&network, input, function)?;
        let HostValue::Real(full_input) = full_plan.prepare_input(input).await? else {
            return Err(fail(function, "supervised training requires real input"));
        };
        full_input
    };
    let mut optimizer = NativeOptimizer::new(factors);
    let mut losses = Vec::new();
    let mut iteration = 0usize;
    let mut order: Vec<usize> = (0..count).collect();
    if matches!(config.shuffle, ShuffleMode::Once) {
        order.reverse();
    }
    for epoch in 0..config.max_epochs {
        admit_training_cancellation(runtime, function)?;
        if matches!(config.shuffle, ShuffleMode::EveryEpoch) {
            let amount = epoch % order.len();
            order.rotate_left(amount);
        }
        let mut epoch_loss = 0.0;
        let mut seen = 0usize;
        for indices in order.chunks(config.mini_batch_size) {
            admit_training_cancellation(runtime, function)?;
            iteration = iteration
                .checked_add(1)
                .ok_or_else(|| fail(function, "iteration count overflow"))?;
            let input_selection = BatchSelection::admit(&geometry.input_axes, indices, function)?;
            let input_batch = batch_value(input, input_selection.select(&full_input, function)?)?;
            let target_selection = BatchSelection::admit(&geometry.output_axes, indices, function)?;
            let target_batch = target_selection.select(&full_target, function)?;
            let (mut execution, original_parameters) = {
                let plan = HostNetworkPlan::admit(&network, &input_batch, function)?;
                let prepared = plan.prepare_for_training().await?;
                let prepared_input = prepared.prepare_input(&input_batch).await?;
                let execution = prepared.execute(prepared_input, true)?;
                let original_parameters = prepared.original_parameters();
                (execution, original_parameters)
            };
            let HostValue::Real(prediction) = &execution.value else {
                return Err(fail(function, "supervised loss requires real output"));
            };
            let loss = match loss_kind {
                LossKind::Regression => {
                    loss::mse(prediction, &target_batch, &execution.axes, function)?
                }
                LossKind::Classification => loss::crossentropy(
                    prediction,
                    &target_batch,
                    &execution.axes,
                    domain,
                    function,
                )?,
            };
            let trace = execution
                .trace
                .take()
                .ok_or_else(|| fail(function, "shared host execution omitted training trace"))?;
            let reverse = network::backward(&trace, HostValue::Real(loss.output_gradient), &[])
                .map_err(|error| fail(function, error))?;
            let gradients = canonical_gradients(&network, reverse, function)?;
            let updates = optimizer.propose(
                &network,
                &original_parameters,
                gradients,
                &config,
                iteration,
                function,
            )?;
            let Value::Object(updated) =
                model::parameters::apply_updates(network, updates, function)?
            else {
                unreachable!("parameter owner returns a network object")
            };
            let updated_roots = crate::execution::RootedValueLease::for_traced(&updated)
                .map_err(|error| error.into_runtime_error())?;
            network = updated;
            network_roots = updated_roots;
            epoch_loss += loss.value * indices.len() as f64;
            seen += indices.len();
        }
        let epoch_loss = epoch_loss / seen as f64;
        losses.push(epoch_loss);
        progress::epoch(epoch + 1, config.max_epochs, epoch_loss, config.verbose)?;
    }
    admit_training_cancellation(runtime, function)?;
    attach_training_result(
        &mut network,
        classes,
        loss_kind,
        &config,
        &losses,
        iteration,
        function,
    )?;
    let info = if requested_outputs == Some(2) {
        Some(training_info(&config, &losses, iteration, function)?)
    } else {
        None
    };
    let output = output_network(Value::Object(network), info, function, requested_outputs)?;
    let rooted =
        crate::execution::RootedValue::new(output).map_err(|error| error.into_runtime_error())?;
    drop(network_roots);
    Ok(rooted)
}
