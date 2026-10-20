//! Supervised public entrypoints over the shared typed network executor.

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::BuiltinResult;

use super::deep_learning_error;

mod admission;
mod config;
mod native;
mod targets;

use admission::*;
use config::*;
use native::admission::TrainingCaller;
use targets::*;

#[runtime_builtin(
    name = "trainNetwork",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::deep_learning::supervised"
)]
async fn train_network_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    train_network_in_runtime(runtime.as_ref(), requested_outputs, args).await
}

#[cfg(test)]
pub(super) async fn train_network_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    train_network_in_runtime(None, crate::output_count::current_output_count(), args).await
}

async fn train_network_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    admit_training_output_count("trainNetwork", requested_outputs)?;
    if args.len() != 4 {
        return Err(deep_learning_error(
            "trainNetwork",
            "trainNetwork: expected X, Y, layers, and trainingOptions",
        ));
    }
    let config = TrainingConfig::from_value(&args[3], "trainNetwork")?;
    native::trainer::train(
        runtime,
        requested_outputs,
        TrainingCaller::TrainNetwork,
        &args[0],
        &args[1],
        &args[2],
        None,
        config,
    )
    .await
}

#[runtime_builtin(
    name = "trainnet",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::deep_learning::supervised"
)]
async fn trainnet_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    trainnet_in_runtime(runtime.as_ref(), requested_outputs, args).await
}

#[cfg(test)]
pub(super) async fn trainnet_builtin(
    args: Vec<Value>,
) -> BuiltinResult<crate::execution::RootedValue> {
    trainnet_in_runtime(
        None,
        crate::output_count::current_output_count(),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .await
}

async fn trainnet_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    admit_training_output_count("trainnet", requested_outputs)?;
    if args.len() != 5 {
        return Err(deep_learning_error(
            "trainnet",
            "trainnet: expected X, Y, net, lossFcn, and trainingOptions",
        ));
    }
    let loss = parse_trainnet_loss(&args[3])?;
    let config = TrainingConfig::from_value(&args[4], "trainnet")?;
    native::trainer::train(
        runtime,
        requested_outputs,
        TrainingCaller::Trainnet,
        &args[0],
        &args[1],
        &args[2],
        Some(loss),
        config,
    )
    .await
}

fn output_network(
    network: Value,
    info: Option<Value>,
    function: &'static str,
    requested_outputs: Option<usize>,
) -> BuiltinResult<Value> {
    match requested_outputs {
        None => Ok(network),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![network])),
        Some(2) => Ok(Value::output_list(vec![
            network,
            info.expect("second result is constructed for two outputs"),
        ])),
        Some(requested) => Err(deep_learning_error(
            function,
            format!("{function}: requested {requested} outputs, but at most 2 are supported"),
        )),
    }
}

fn training_info(
    config: &TrainingConfig,
    losses: &[f64],
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut info = runmat_value::UnboundedStructValueBuilder::new();
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
                .map_err(|error| deep_learning_error(function, error))?,
        ),
    );
    info.insert(
        "StopReason",
        Value::string("MaxEpochs completed".to_string()),
    );
    Ok(Value::Struct(info.finish()))
}
