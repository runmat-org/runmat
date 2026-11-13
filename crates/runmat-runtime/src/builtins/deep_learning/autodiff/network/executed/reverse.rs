//! Typed adjoints from the same host kernels used by forward execution.
use super::*;
use model::execution::host::network::backward::StepGradients;
use model::state::StateUpdate;

pub(in crate::builtins::deep_learning::autodiff) fn reverse(
    execution: &Rc<NetworkTapeRecord>,
    nodes: &[Node],
    adjoints: &HashMap<usize, Tensor>,
) -> BuiltinResult<Vec<(usize, Tensor)>> {
    let mut prediction = None;
    let mut states = Vec::new();
    for id in execution.recorded_outputs.clone() {
        let Some(gradient) = adjoints.get(&id) else {
            continue;
        };
        let Some(Node {
            kind:
                NodeKind::Network {
                    execution: recorded,
                    output,
                },
            ..
        }) = nodes.get(id)
        else {
            return Err(deep_learning_error(
                "dlgradient",
                "dlgradient: network output is missing from its tape",
            ));
        };
        if !Rc::ptr_eq(execution, recorded) {
            return Err(deep_learning_error(
                "dlgradient",
                "dlgradient: network output belongs to a different trace",
            ));
        }
        match *output {
            NetworkOutput::Prediction => prediction = Some(gradient.clone()),
            NetworkOutput::State { layer_index, role } => states.push(StateUpdate {
                layer_index,
                role,
                value: gradient.clone(),
            }),
        }
    }
    let prediction = match prediction {
        Some(gradient) => {
            let HostValue::Real(gradient) = model::execution::present_host(
                HostValue::Real(gradient),
                &execution.output_axes,
                execution.trace.output_axes(),
                "dlgradient",
            )?
            else {
                unreachable!("a real axis permutation preserves its value kind")
            };
            gradient
        }
        None => tensor::zeros_with_dtype(
            execution.trace.output_axes().shape(),
            execution.output_dtype,
        )
        .map_err(|error| deep_learning_error("dlgradient", error))?,
    };
    #[cfg(test)]
    execution
        .reverse_calls
        .set(execution.reverse_calls.get() + 1);
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
    let gradient = model::precision::project_owned(gradient, binding.dtype, "dlgradient")?;
    let value = gradient
        .reshape(shape.clone())
        .map_err(|error| deep_learning_error("dlgradient", error))?;
    result.push((binding.node, value));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_precision_parent_reshape_preserves_the_native_allocation() {
        let gradient = Tensor::from_f32(vec![1.25, -2.5], vec![2, 1]).unwrap();
        let parent = ParentBinding {
            node: 7,
            shape: vec![1, 2],
            dtype: NumericDType::F32,
        };
        let mut result = Vec::new();
        append(&mut result, Some(&parent), gradient.clone()).unwrap();
        assert_eq!(result[0].0, 7);
        assert_eq!(result[0].1.shape(), &[1, 2]);
        assert_eq!(result[0].1.as_f32_slice().unwrap(), &[1.25, -2.5]);
        assert!(gradient
            .host_buffer()
            .shares_allocation_with(result[0].1.host_buffer()));
    }

    #[test]
    fn prepared_single_adjoint_restores_original_double_parent_class() {
        let parent = ParentBinding {
            node: 0,
            shape: vec![2, 1],
            dtype: NumericDType::F64,
        };
        let mut result = Vec::new();
        append(
            &mut result,
            Some(&parent),
            Tensor::from_f32(vec![1.25, -2.5], vec![1, 2]).unwrap(),
        )
        .unwrap();
        assert_eq!(result[0].1.as_f64_slice().unwrap(), &[1.25, -2.5]);
        assert_eq!(result[0].1.shape(), &[2, 1]);
    }

    #[test]
    fn invalid_parent_size_does_not_publish_an_adjoint() {
        let parent = ParentBinding {
            node: 0,
            shape: vec![3, 1],
            dtype: NumericDType::F64,
        };
        let mut result = Vec::new();
        assert!(append(
            &mut result,
            Some(&parent),
            Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()
        )
        .is_err());
        assert!(result.is_empty());
    }
}
