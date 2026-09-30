//! Typed adjoints from the same host kernels used by forward execution.
use super::*;
use model::execution::host::network::backward::StepGradients;
use model::state::StateUpdate;

pub(in crate::builtins::deep_learning::autodiff) fn reverse(
    execution: &NetworkTapeRecord,
    output: NetworkOutput,
    gradient: Tensor,
) -> BuiltinResult<Vec<(usize, Tensor)>> {
    let (prediction, states) = match output {
        NetworkOutput::Prediction => (gradient, Vec::new()),
        NetworkOutput::State { layer_index, role } => (
            tensor::zeros_with_dtype(&execution.output_shape, execution.output_dtype)
                .map_err(|error| deep_learning_error("dlgradient", error))?,
            vec![StateUpdate {
                layer_index,
                role,
                value: gradient,
            }],
        ),
    };
    let gradients =
        model::plan::host::reverse(&execution.trace, HostValue::Real(prediction), &states)?;
    let mut result = Vec::new();
    if let HostValue::Real(gradient) = gradients.input_gradient {
        append(&mut result, execution.input.as_ref(), gradient)?;
    } else {
        return Err(unsupported_error(
            "dlgradient",
            "dlgradient: complex network input adjoints require complex tape support",
        ));
    }
    for (layer, gradients) in gradients.parameter_gradients {
        let mut parameter =
            |role, value| append(&mut result, execution.parameters.get(&(layer, role)), value);
        match gradients {
            StepGradients::None => {}
            StepGradients::FullyConnected { weights, bias }
            | StepGradients::Convolution1d { weights, bias } => {
                parameter(ParameterRole::Weights, weights)?;
                parameter(ParameterRole::Bias, bias)?;
            }
            StepGradients::LayerNormalization { scale, offset } => {
                parameter(ParameterRole::Scale, scale)?;
                parameter(ParameterRole::Offset, offset)?;
            }
            StepGradients::Recurrent {
                input_weights,
                recurrent_weights,
                bias,
                initial_hidden,
                initial_cell,
            } => {
                parameter(ParameterRole::InputWeights, input_weights)?;
                parameter(ParameterRole::RecurrentWeights, recurrent_weights)?;
                parameter(ParameterRole::Bias, bias)?;
                append(
                    &mut result,
                    execution.states.get(&(layer, StateRole::HiddenState)),
                    initial_hidden,
                )?;
                append(
                    &mut result,
                    execution.states.get(&(layer, StateRole::CellState)),
                    initial_cell,
                )?;
            }
        }
    }
    Ok(result)
}

fn append(
    result: &mut Vec<(usize, Tensor)>,
    binding: Option<&ParentBinding>,
    gradient: Tensor,
) -> BuiltinResult<()> {
    let Some(binding) = binding else {
        return Ok(());
    };
    let shape = &binding.shape;
    let count = shape
        .iter()
        .try_fold(1usize, |n, &size| n.checked_mul(size));
    if count != Some(tensor::tensor_element_len(&gradient)) {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: kernel adjoint shape differs from its tape parent",
        ));
    }
    // Prepared execution precision can differ from the original learnable.
    // Convert only at this explicit derivative-to-parent boundary.
    let storage = gradient
        .into_numeric_storage()
        .map_err(|error| deep_learning_error("dlgradient", error))?;
    let storage = match (storage, binding.dtype) {
        (storage @ NumericStorage::F32(_), NumericDType::F32)
        | (storage @ NumericStorage::F64(_), NumericDType::F64) => storage,
        (NumericStorage::F32(values), NumericDType::F64) => {
            NumericStorage::F64(values.into_iter().map(f64::from).collect())
        }
        (NumericStorage::F64(values), NumericDType::F32) => {
            NumericStorage::F32(values.into_iter().map(|value| value as f32).collect())
        }
        _ => {
            return Err(deep_learning_error(
                "dlgradient",
                "dlgradient: network adjoints require floating tape parents",
            ))
        }
    };
    let value = Tensor::from_numeric_storage(storage, shape.clone())
        .map_err(|error| deep_learning_error("dlgradient", error))?;
    result.push((binding.node, value));
    Ok(())
}
