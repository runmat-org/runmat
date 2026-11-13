use super::super::*;
use std::collections::{HashMap, HashSet};

fn gradient_matmul_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn matmul_grad_lhs(grad: &Tensor, lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    let grad_values = floating_tensor_values(grad, "dlgradient")?;
    let rhs_values = floating_tensor_values(rhs, "dlgradient")?;
    let (lhs_rows, lhs_cols) = gradient_matmul_dimensions(lhs.shape());
    let (rhs_rows, rhs_cols) = gradient_matmul_dimensions(rhs.shape());
    let (grad_rows, _) = gradient_matmul_dimensions(grad.shape());
    let mut data = vec![0.0; tensor::tensor_element_len(lhs)];
    for row in 0..lhs_rows {
        for col in 0..lhs_cols {
            let mut acc = 0.0;
            for k in 0..rhs_cols {
                acc += grad_values[row + k * grad_rows] * rhs_values[col + k * rhs_rows];
            }
            data[row + col * lhs_rows] = acc;
        }
    }
    floating_tensor_from_f64(
        data,
        lhs.shape().to_vec(),
        lhs.numeric_dtype(),
        "dlgradient",
    )
}

fn matmul_grad_rhs(grad: &Tensor, lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    let grad_values = floating_tensor_values(grad, "dlgradient")?;
    let lhs_values = floating_tensor_values(lhs, "dlgradient")?;
    let (lhs_rows, _) = gradient_matmul_dimensions(lhs.shape());
    let (rhs_rows, rhs_cols) = gradient_matmul_dimensions(rhs.shape());
    let (grad_rows, _) = gradient_matmul_dimensions(grad.shape());
    let mut data = vec![0.0; tensor::tensor_element_len(rhs)];
    for row in 0..rhs_rows {
        for col in 0..rhs_cols {
            let mut acc = 0.0;
            for k in 0..lhs_rows {
                acc += lhs_values[k + row * lhs_rows] * grad_values[k + col * grad_rows];
            }
            data[row + col * rhs_rows] = acc;
        }
    }
    floating_tensor_from_f64(
        data,
        rhs.shape().to_vec(),
        rhs.numeric_dtype(),
        "dlgradient",
    )
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
    let mut reversed_networks = HashSet::new();
    let loss_dtype = nodes
        .get(loss_node)
        .and_then(|node| node.value.numeric_dtype())
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
        if let NodeKind::Network { execution, .. } = &node.kind {
            // Outputs of one immutable network trace are contiguous and all
            // precede their downstream operations. At the first reached
            // output, every prediction/State seed has therefore accumulated.
            if reversed_networks.insert(Rc::as_ptr(execution)) {
                for (parent, gradient) in
                    super::super::network::reverse_execution(execution, nodes, &grads)?
                {
                    accumulate_parent(&mut grads, Some(parent), gradient);
                }
            }
            continue;
        }
        let value = node.value.host().ok_or_else(|| {
            deep_learning_error(
                "dlgradient",
                "dlgradient: resident node requires resident reverse",
            )
        })?;
        let grad_values = floating_tensor_values(&grad, "dlgradient")?;
        match &node.kind {
            NodeKind::Leaf => {}
            NodeKind::ResidentNetwork { .. } => {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: resident network requires resident reverse",
                ));
            }
            NodeKind::Network { .. } => unreachable!("network uses its native typed reverse path"),
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
                        value.shape(),
                        node_dtype(nodes, *lhs, value.numeric_dtype()),
                    )?,
                );
                accumulate_parent(
                    &mut grads,
                    *rhs,
                    reduce_gradient(
                        &grad_values,
                        rhs_shape,
                        value.shape(),
                        node_dtype(nodes, *rhs, value.numeric_dtype()),
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
                        value.shape(),
                        node_dtype(nodes, *lhs, value.numeric_dtype()),
                    )?,
                );
                let rhs_dtype = node_dtype(nodes, *rhs, value.numeric_dtype());
                let rhs_grad = reduce_gradient(&grad_values, rhs_shape, value.shape(), rhs_dtype)?;
                let rhs_shape = rhs_grad.shape().to_vec();
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
                            value.shape(),
                            node_dtype(nodes, Some(parent), value.numeric_dtype()),
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
                            value.shape(),
                            node_dtype(nodes, Some(parent), value.numeric_dtype()),
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
                            value.shape(),
                            node_dtype(nodes, Some(parent), value.numeric_dtype()),
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
                            value.shape(),
                            node_dtype(nodes, Some(parent), value.numeric_dtype()),
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
            NodeKind::SumAll { input, input_shape } => {
                let seed = grad_values.iter().copied().sum::<f64>();
                let count = input_shape.iter().product();
                accumulate_parent(
                    &mut grads,
                    Some(*input),
                    floating_tensor_from_f64(
                        vec![seed; count],
                        input_shape.clone(),
                        node_dtype(nodes, Some(*input), value.numeric_dtype()),
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
                let current = existing
                    .numeric_value_at(index)
                    .expect("gradient index remains valid");
                let increment = grad
                    .numeric_value_at(index)
                    .expect("gradient index remains valid");
                let sum = match (current, increment) {
                    (NumericScalar::F32(current), NumericScalar::F32(increment)) => {
                        NumericScalar::F32(current + increment)
                    }
                    (NumericScalar::F32(current), NumericScalar::F64(increment)) => {
                        NumericScalar::F32(current + increment as f32)
                    }
                    (NumericScalar::F64(current), NumericScalar::F32(increment)) => {
                        NumericScalar::F64(current + f64::from(increment))
                    }
                    (NumericScalar::F64(current), NumericScalar::F64(increment)) => {
                        NumericScalar::F64(current + increment)
                    }
                    _ => unreachable!("the host autodiff tape records floating gradients only"),
                };
                existing
                    .set_numeric_assignment_at(index, sum)
                    .expect("same-shape floating gradient accumulation remains valid");
            }
        })
        .or_insert(grad);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn matmul_backward_keeps_rank_zero_and_rank_one_cached_geometry() {
        assert_eq!(gradient_matmul_dimensions(&[]), (0, 0));
        assert_eq!(gradient_matmul_dimensions(&[2]), (1, 2));

        let seed = Tensor::new(vec![3.0], vec![1, 1]).expect("gradient seed");
        let rank_zero = Tensor::new(vec![7.0], vec![]).expect("rank-zero operand");
        let rhs = Tensor::new(vec![10.0, 20.0], vec![2, 1]).expect("rhs");
        let rank_zero_gradient =
            matmul_grad_lhs(&seed, &rank_zero, &rhs).expect("rank-zero local backward kernel");
        assert!(rank_zero_gradient.shape().is_empty());
        assert_eq!(rank_zero_gradient.materialize_f64(), vec![0.0]);

        let rank_one = Tensor::new(vec![1.0, 2.0], vec![2]).expect("rank-one operand");
        let rank_one_gradient =
            matmul_grad_lhs(&seed, &rank_one, &rhs).expect("rank-one local backward kernel");
        assert_eq!(rank_one_gradient.shape(), [2]);
        assert_eq!(rank_one_gradient.materialize_f64(), vec![30.0, 60.0]);
    }
}
