use super::super::*;
use std::collections::HashMap;

fn matmul_grad_lhs(grad: &Tensor, lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    let grad_values = floating_tensor_values(grad, "dlgradient")?;
    let rhs_values = floating_tensor_values(rhs, "dlgradient")?;
    let mut data = vec![0.0; tensor::tensor_element_len(lhs)];
    for row in 0..lhs.rows {
        for col in 0..lhs.cols {
            let mut acc = 0.0;
            for k in 0..rhs.cols {
                acc += grad_values[row + k * grad.rows] * rhs_values[col + k * rhs.rows];
            }
            data[row + col * lhs.rows] = acc;
        }
    }
    floating_tensor_from_f64(data, lhs.shape.clone(), lhs.numeric_dtype(), "dlgradient")
}

fn matmul_grad_rhs(grad: &Tensor, lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    let grad_values = floating_tensor_values(grad, "dlgradient")?;
    let lhs_values = floating_tensor_values(lhs, "dlgradient")?;
    let mut data = vec![0.0; tensor::tensor_element_len(rhs)];
    for row in 0..rhs.rows {
        for col in 0..rhs.cols {
            let mut acc = 0.0;
            for k in 0..lhs.rows {
                acc += lhs_values[k + row * lhs.rows] * grad_values[k + col * grad.rows];
            }
            data[row + col * rhs.rows] = acc;
        }
    }
    floating_tensor_from_f64(data, rhs.shape.clone(), rhs.numeric_dtype(), "dlgradient")
}

pub(super) fn backward(loss_node: usize) -> BuiltinResult<HashMap<usize, Tensor>> {
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape",
        )
    })?;
    // The reverse pass invokes only local tensor arithmetic. Borrowing the
    // graph for this synchronous traversal avoids copying every recorded
    // activation and operand payload before computing gradients.
    let tape_guard = tape.borrow();
    let nodes = &tape_guard.nodes;
    let mut grads: HashMap<usize, Tensor> = HashMap::new();
    let loss_dtype = nodes
        .get(loss_node)
        .map(|node| node.value.numeric_dtype())
        .unwrap_or(NumericDType::F64);
    grads.insert(
        loss_node,
        floating_tensor_from_f64(vec![1.0], vec![1, 1], loss_dtype, "dlgradient")?,
    );
    for id in (0..=loss_node).rev() {
        let Some(grad) = grads.get(&id).cloned() else {
            continue;
        };
        let Some(node) = nodes.get(id) else {
            continue;
        };
        let grad_values = floating_tensor_values(&grad, "dlgradient")?;
        match &node.kind {
            NodeKind::Leaf => {}
            NodeKind::Add {
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
            } => {
                accumulate_parent(
                    &mut grads,
                    *lhs,
                    reduce_gradient(
                        &grad_values,
                        lhs_shape,
                        &node.value.shape,
                        node_dtype(&nodes, *lhs, node.value.numeric_dtype()),
                    )?,
                );
                accumulate_parent(
                    &mut grads,
                    *rhs,
                    reduce_gradient(
                        &grad_values,
                        rhs_shape,
                        &node.value.shape,
                        node_dtype(&nodes, *rhs, node.value.numeric_dtype()),
                    )?,
                );
            }
            NodeKind::Sub {
                lhs,
                rhs,
                lhs_shape,
                rhs_shape,
            } => {
                accumulate_parent(
                    &mut grads,
                    *lhs,
                    reduce_gradient(
                        &grad_values,
                        lhs_shape,
                        &node.value.shape,
                        node_dtype(&nodes, *lhs, node.value.numeric_dtype()),
                    )?,
                );
                let rhs_dtype = node_dtype(&nodes, *rhs, node.value.numeric_dtype());
                let rhs_grad =
                    reduce_gradient(&grad_values, rhs_shape, &node.value.shape, rhs_dtype)?;
                let rhs_shape = rhs_grad.shape.clone();
                let rhs_grad = floating_tensor_from_f64(
                    floating_tensor_values(&rhs_grad, "dlgradient")?
                        .into_iter()
                        .map(|value| -value)
                        .collect(),
                    rhs_shape,
                    rhs_dtype,
                    "dlgradient",
                )?;
                accumulate_parent(&mut grads, *rhs, rhs_grad);
            }
            NodeKind::Mul {
                lhs,
                rhs,
                lhs_data: lhs_values,
                rhs_data: rhs_values,
                lhs_shape,
                rhs_shape,
            } => {
                if let Some(parent) = *lhs {
                    let data = grad_values
                        .iter()
                        .zip(rhs_values)
                        .map(|(g, r)| g * r)
                        .collect::<Vec<_>>();
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        reduce_gradient(
                            &data,
                            lhs_shape,
                            &node.value.shape,
                            node_dtype(&nodes, Some(parent), node.value.numeric_dtype()),
                        )?,
                    );
                }
                if let Some(parent) = *rhs {
                    let data = grad_values
                        .iter()
                        .zip(lhs_values)
                        .map(|(g, l)| g * l)
                        .collect::<Vec<_>>();
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        reduce_gradient(
                            &data,
                            rhs_shape,
                            &node.value.shape,
                            node_dtype(&nodes, Some(parent), node.value.numeric_dtype()),
                        )?,
                    );
                }
            }
            NodeKind::Div {
                lhs,
                rhs,
                lhs_data: lhs_values,
                rhs_data: rhs_values,
                lhs_shape,
                rhs_shape,
            } => {
                if let Some(parent) = *lhs {
                    let data = grad_values
                        .iter()
                        .zip(rhs_values)
                        .map(|(g, r)| g / r)
                        .collect::<Vec<_>>();
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        reduce_gradient(
                            &data,
                            lhs_shape,
                            &node.value.shape,
                            node_dtype(&nodes, Some(parent), node.value.numeric_dtype()),
                        )?,
                    );
                }
                if let Some(parent) = *rhs {
                    let data = grad_values
                        .iter()
                        .zip(lhs_values)
                        .zip(rhs_values)
                        .map(|((g, l), r)| -g * l / (r * r))
                        .collect::<Vec<_>>();
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        reduce_gradient(
                            &data,
                            rhs_shape,
                            &node.value.shape,
                            node_dtype(&nodes, Some(parent), node.value.numeric_dtype()),
                        )?,
                    );
                }
            }
            NodeKind::MatMul {
                lhs,
                rhs,
                lhs_value,
                rhs_value,
            } => {
                if let Some(parent) = *lhs {
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        matmul_grad_lhs(&grad, lhs_value, rhs_value)?,
                    );
                }
                if let Some(parent) = *rhs {
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        matmul_grad_rhs(&grad, lhs_value, rhs_value)?,
                    );
                }
            }
            NodeKind::FullyConnected {
                input,
                weights,
                bias,
                input_value,
                weights_value,
            } => {
                let input_values = floating_tensor_values(input_value, "dlgradient")?;
                let weights_values = floating_tensor_values(weights_value, "dlgradient")?;
                let mut input_grad = vec![0.0; input_values.len()];
                let mut weights_grad = vec![0.0; weights_values.len()];
                let mut bias_grad = vec![0.0; weights_value.rows];
                for row in 0..input_value.rows {
                    for out_col in 0..weights_value.rows {
                        let g = grad_values[row + out_col * grad.rows];
                        bias_grad[out_col] += g;
                        for feature in 0..input_value.cols {
                            input_grad[row + feature * input_value.rows] +=
                                g * weights_values[out_col + feature * weights_value.rows];
                            weights_grad[out_col + feature * weights_value.rows] +=
                                input_values[row + feature * input_value.rows] * g;
                        }
                    }
                }
                accumulate_parent(
                    &mut grads,
                    *input,
                    floating_tensor_from_f64(
                        input_grad,
                        input_value.shape.clone(),
                        input_value.numeric_dtype(),
                        "dlgradient",
                    )?,
                );
                accumulate_parent(
                    &mut grads,
                    *weights,
                    floating_tensor_from_f64(
                        weights_grad,
                        weights_value.shape.clone(),
                        weights_value.numeric_dtype(),
                        "dlgradient",
                    )?,
                );
                accumulate_parent(
                    &mut grads,
                    *bias,
                    floating_tensor_from_f64(
                        bias_grad,
                        vec![weights_value.rows, 1],
                        node_dtype(&nodes, *bias, weights_value.numeric_dtype()),
                        "dlgradient",
                    )?,
                );
            }
            NodeKind::SumAll { input, input_shape } => {
                let seed = grad_values.iter().copied().sum::<f64>();
                let count = input_shape.iter().product();
                accumulate_parent(
                    &mut grads,
                    Some(*input),
                    floating_tensor_from_f64(
                        vec![seed; count],
                        input_shape.clone(),
                        node_dtype(&nodes, Some(*input), node.value.numeric_dtype()),
                        "dlgradient",
                    )?,
                );
            }
            NodeKind::Relu { input, input_value } => {
                let input_values = floating_tensor_values(input_value, "dlgradient")?;
                let data = grad_values
                    .iter()
                    .zip(&input_values)
                    .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                    .collect::<Vec<_>>();
                accumulate_parent(
                    &mut grads,
                    Some(*input),
                    floating_tensor_from_f64(
                        data,
                        input_value.shape.clone(),
                        input_value.numeric_dtype(),
                        "dlgradient",
                    )?,
                );
            }
            NodeKind::Elu {
                input,
                input_value,
                alpha,
            } => {
                let input_values = floating_tensor_values(input_value, "dlgradient")?;
                let data = grad_values
                    .iter()
                    .zip(&input_values)
                    .map(|(g, x)| if *x > 0.0 { *g } else { *g * alpha * x.exp() })
                    .collect::<Vec<_>>();
                accumulate_parent(
                    &mut grads,
                    Some(*input),
                    floating_tensor_from_f64(
                        data,
                        input_value.shape.clone(),
                        input_value.numeric_dtype(),
                        "dlgradient",
                    )?,
                );
            }
            NodeKind::Softmax {
                input,
                output_value,
            } => {
                let output_values = floating_tensor_values(output_value, "dlgradient")?;
                let mut data = vec![0.0; output_values.len()];
                for row in 0..output_value.rows {
                    let dot = (0..output_value.cols)
                        .map(|col| {
                            let idx = row + col * output_value.rows;
                            grad_values[idx] * output_values[idx]
                        })
                        .sum::<f64>();
                    for col in 0..output_value.cols {
                        let idx = row + col * output_value.rows;
                        data[idx] = output_values[idx] * (grad_values[idx] - dot);
                    }
                }
                accumulate_parent(
                    &mut grads,
                    Some(*input),
                    floating_tensor_from_f64(
                        data,
                        output_value.shape.clone(),
                        output_value.numeric_dtype(),
                        "dlgradient",
                    )?,
                );
            }
        }
    }
    Ok(grads)
}

fn reduce_gradient(
    data: &[f64],
    parent_shape: &[usize],
    output_shape: &[usize],
    dtype: NumericDType,
) -> BuiltinResult<Tensor> {
    let parent_len = parent_shape.iter().product::<usize>();
    if parent_len == data.len() && parent_shape == output_shape {
        return floating_tensor_from_f64(data.to_vec(), parent_shape.to_vec(), dtype, "dlgradient");
    }
    if parent_len == 1 {
        return floating_tensor_from_f64(
            vec![data.iter().sum()],
            parent_shape.to_vec(),
            dtype,
            "dlgradient",
        );
    }
    Err(deep_learning_error(
        "dlgradient",
        "dlgradient: implicit-expansion gradient reduction is only supported for scalar broadcasts",
    ))
}

fn accumulate_parent(grads: &mut HashMap<usize, Tensor>, id: Option<usize>, grad: Tensor) {
    let Some(id) = id else {
        return;
    };
    grads
        .entry(id)
        .and_modify(|existing| {
            for index in 0..tensor::tensor_element_len(existing) {
                let current = floating_scalar_f64(
                    existing
                        .numeric_value_at(index)
                        .expect("gradient index remains valid"),
                );
                let increment = floating_scalar_f64(
                    grad.numeric_value_at(index)
                        .expect("gradient index remains valid"),
                );
                existing
                    .set_numeric_assignment_at(index, NumericScalar::F64(current + increment))
                    .expect("same-shape floating gradient accumulation remains valid");
            }
        })
        .or_insert(grad);
}
