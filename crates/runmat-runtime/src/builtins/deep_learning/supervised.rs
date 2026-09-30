use std::collections::BTreeMap;

use runmat_macros::runtime_builtin;
use runmat_value::{NumericScalar, ObjectInstance, StructValue, Tensor, Value};

use crate::{builtins::common::tensor, BuiltinResult};

use super::layer_identity::ForwardLayerKind;
use super::{
    deep_learning_error, model, numeric_values, positive_usize, scalar_text, string_array,
};

#[runtime_builtin(
    name = "trainNetwork",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::supervised"
)]
pub(super) async fn train_network_builtin(mut args: Vec<Value>) -> BuiltinResult<Value> {
    admit_training_output_count("trainNetwork")?;
    if args.len() != 4 {
        return Err(deep_learning_error(
            "trainNetwork",
            "trainNetwork: expected X, Y, layers, and trainingOptions",
        ));
    }
    let config = TrainingConfig::from_value(&args[3], "trainNetwork")?;
    let admitted_layers = model::layers_from_network_ref(&args[2], "trainNetwork")?;
    model::validate_forward_layers(admitted_layers, "trainNetwork")?;
    let host_x = host_matrix(&args[0], "trainNetwork", "X")?;
    let observations = host_x
        .as_ref()
        .map(|matrix| matrix.rows)
        .or(resident_rows(&args[0], "trainNetwork")?);
    admit_training_observations(observations, "trainNetwork")?;
    let host_task = if matches!(args[1], Value::GpuTensor(_)) {
        None
    } else {
        observations
            .map(|rows| infer_train_network_task(admitted_layers, &args[1], rows))
            .transpose()?
    };
    let network_class = if matches!(
        &args[2],
        Value::Object(object) if object.class_name.is(super::LAYER_GRAPH_CLASS)
    ) {
        model::DAG_NETWORK_CLASS
    } else {
        model::SERIES_NETWORK_CLASS
    };
    let mut layers = take_training_layers(args.remove(2), "trainNetwork")?;
    model::initialise_fully_connected_layers(&mut layers, "trainNetwork")?;
    preflight_training_layers(&layers, "trainNetwork")?;
    let x = materialize_matrix(&args[0], host_x, "trainNetwork", "X").await?;
    let task = match host_task {
        Some(task) => task,
        None => {
            let y = crate::gather_if_needed_async(&args[1]).await?;
            infer_train_network_task(&layers, &y, x.rows)?
        }
    };
    gather_training_learnables(&mut layers).await?;
    train_and_output(layers, x, task, config, network_class, "trainNetwork")
}

#[runtime_builtin(
    name = "trainnet",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::supervised"
)]
pub(super) async fn trainnet_builtin(mut args: Vec<Value>) -> BuiltinResult<Value> {
    admit_training_output_count("trainnet")?;
    if args.len() != 5 {
        return Err(deep_learning_error(
            "trainnet",
            "trainnet: expected X, Y, net, lossFcn, and trainingOptions",
        ));
    }
    let loss = parse_trainnet_loss(&args[3])?;
    let config = TrainingConfig::from_value(&args[4], "trainnet")?;
    let admitted_layers = model::layers_from_network_ref(&args[2], "trainnet")?;
    model::validate_forward_layers(admitted_layers, "trainnet")?;
    let host_x = host_matrix(&args[0], "trainnet", "X")?;
    let observations = host_x
        .as_ref()
        .map(|matrix| matrix.rows)
        .or(resident_rows(&args[0], "trainnet")?);
    admit_training_observations(observations, "trainnet")?;
    let host_task = if matches!(args[1], Value::GpuTensor(_)) {
        None
    } else {
        observations
            .map(|rows| task_from_loss(loss, &args[1], rows))
            .transpose()?
    };
    let mut layers = take_training_layers(args.remove(2), "trainnet")?;
    model::initialise_fully_connected_layers(&mut layers, "trainnet")?;
    preflight_training_layers(&layers, "trainnet")?;
    let x = materialize_matrix(&args[0], host_x, "trainnet", "X").await?;
    let task = match host_task {
        Some(task) => task,
        None => {
            let y = crate::gather_if_needed_async(&args[1]).await?;
            task_from_loss(loss, &y, x.rows)?
        }
    };
    gather_training_learnables(&mut layers).await?;
    train_and_output(layers, x, task, config, model::DLNETWORK_CLASS, "trainnet")
}

mod admission;
use admission::*;

fn train_and_output(
    mut layers: Vec<Value>,
    x: Matrix,
    task: Task,
    config: TrainingConfig,
    network_class: runmat_types::StaticClassIdentity,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut state = OptimizerState::new(config.solver);
    // Do not reserve an arbitrary user-supplied epoch count before the first
    // cancellable unit of work has run.
    let mut losses = Vec::new();
    let mut iteration = 0usize;
    let mut order: Vec<usize> = (0..x.rows).collect();
    if matches!(config.shuffle, ShuffleMode::Once) {
        order.reverse();
    }

    for epoch in 0..config.max_epochs {
        admit_training_cancellation(function)?;
        if matches!(config.shuffle, ShuffleMode::EveryEpoch) && !order.is_empty() {
            let amount = epoch % order.len();
            order.rotate_left(amount);
        }
        let mut epoch_loss = 0.0;
        let mut seen = 0usize;
        for chunk in order.chunks(config.mini_batch_size) {
            admit_training_cancellation(function)?;
            iteration = iteration.checked_add(1).ok_or_else(|| {
                deep_learning_error(function, format!("{function}: iteration count overflow"))
            })?;
            let batch_x = x.select_rows(chunk, function)?;
            let batch_targets = task.select_rows(chunk, function)?;
            let (loss, grad, caches) =
                loss_and_gradient(&layers, &batch_x, &batch_targets, function)?;
            let batch_len = chunk.len();
            epoch_loss += loss * batch_len as f64;
            seen += batch_len;
            backward_update(
                &mut layers,
                caches,
                grad,
                &config,
                &mut state,
                iteration,
                function,
            )?;
        }
        let averaged = if seen == 0 {
            0.0
        } else {
            epoch_loss / seen as f64
        };
        losses.push(averaged);
    }

    admit_training_cancellation(function)?;
    let mut network = model::network_object(network_class, layers, BTreeMap::new(), function)?;
    attach_training_metadata(&mut network, &task, &config, &losses, iteration, function)?;
    let second_output = if crate::output_count::current_output_count() == Some(2) {
        Some(training_info(&config, &losses, iteration, function)?)
    } else {
        None
    };
    output_network(network, second_output, function)
}

fn output_network(
    network: Value,
    info: Option<Value>,
    function: &'static str,
) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        None => Ok(network),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![network])),
        Some(2) => Ok(Value::OutputList(vec![
            network,
            info.expect("second result is constructed for two outputs"),
        ])),
        Some(requested) => Err(deep_learning_error(
            function,
            format!("{function}: requested {requested} outputs, but at most 2 are supported"),
        )),
    }
}

fn attach_training_metadata(
    network: &mut Value,
    task: &Task,
    config: &TrainingConfig,
    losses: &[f64],
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    let Value::Object(object) = network else {
        return Ok(());
    };
    object.properties.insert(
        "TrainingInfo".to_string(),
        training_info(config, losses, iteration, function)?,
    );
    object.properties.insert(
        "TrainedWith".to_string(),
        Value::String(config.solver.as_str().into()),
    );
    object.properties.insert(
        "TrainingLoss".to_string(),
        Value::String(task.loss_name().to_string()),
    );
    if let Task::Classification(classification) = task {
        object.properties.insert(
            "Classes".to_string(),
            classification.classes_value(function)?,
        );
    }
    Ok(())
}

fn training_info(
    config: &TrainingConfig,
    losses: &[f64],
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut info = StructValue::new();
    info.insert("SolverName", Value::String(config.solver.as_str().into()));
    info.insert("MaxEpochs", Value::Num(config.max_epochs as f64));
    info.insert("MiniBatchSize", Value::Num(config.mini_batch_size as f64));
    info.insert("InitialLearnRate", Value::Num(config.learn_rate));
    info.insert("Iteration", Value::Num(iteration as f64));
    info.insert("Epoch", Value::Num(config.max_epochs as f64));
    info.insert("FinalLoss", Value::Num(*losses.last().unwrap_or(&0.0)));
    info.insert(
        "TrainingLoss",
        Value::Tensor(
            Tensor::new(losses.to_vec(), vec![losses.len(), 1])
                .map_err(|err| deep_learning_error(function, err))?,
        ),
    );
    info.insert(
        "StopReason",
        Value::String("MaxEpochs completed".to_string()),
    );
    Ok(Value::Struct(info))
}

mod matrix;
use matrix::*;
mod targets;
use targets::*;
mod config;
use config::*;
mod solver;
use solver::*;
mod optimizer;
use optimizer::*;
