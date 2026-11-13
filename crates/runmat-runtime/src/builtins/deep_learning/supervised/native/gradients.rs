//! Bind shared host adjoints to canonical typed Learnables identities.

use std::collections::HashMap;

use runmat_value::{ObjectInstance, Tensor};

use crate::builtins::deep_learning::{
    deep_learning_error,
    model::{
        execution::host::network::{backward::StepGradients, HostBackwardResult},
        parameters::{self, ParameterKey, ParameterRole},
    },
};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

fn push(
    keys: &HashMap<(usize, ParameterRole), ParameterKey>,
    result: &mut Vec<(ParameterKey, Tensor)>,
    layer_index: usize,
    role: ParameterRole,
    gradient: Tensor,
    function: &'static str,
) -> BuiltinResult<()> {
    let key = keys
        .get(&(layer_index, role))
        .ok_or_else(|| fail(function, "host adjoint has no canonical Learnables role"))?;
    result.push((key.clone(), gradient));
    Ok(())
}

pub(super) fn canonical_gradients(
    network: &ObjectInstance,
    reverse: HostBackwardResult,
    function: &'static str,
) -> BuiltinResult<Vec<(ParameterKey, Tensor)>> {
    use ParameterRole as R;
    let keys = parameters::view(network, function)?
        .into_iter()
        .map(|row| ((row.key.layer_index, row.key.role), row.key))
        .collect::<HashMap<_, _>>();
    let mut result = Vec::new();
    for (index, gradients) in reverse.parameter_gradients {
        match gradients {
            StepGradients::None => {}
            StepGradients::FullyConnected { weights, bias }
            | StepGradients::Convolution1d { weights, bias } => {
                push(&keys, &mut result, index, R::Weights, weights, function)?;
                push(&keys, &mut result, index, R::Bias, bias, function)?;
            }
            StepGradients::LayerNormalization { scale, offset } => {
                push(&keys, &mut result, index, R::Scale, scale, function)?;
                push(&keys, &mut result, index, R::Offset, offset, function)?;
            }
            StepGradients::Recurrent {
                input_weights,
                recurrent_weights,
                bias,
                initial_hidden: _,
                initial_cell: _,
            } => {
                push(
                    &keys,
                    &mut result,
                    index,
                    R::InputWeights,
                    input_weights,
                    function,
                )?;
                push(
                    &keys,
                    &mut result,
                    index,
                    R::RecurrentWeights,
                    recurrent_weights,
                    function,
                )?;
                push(&keys, &mut result, index, R::Bias, bias, function)?;
            }
        }
    }
    Ok(result)
}
