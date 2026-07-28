use std::cell::RefCell;
use std::collections::HashMap;

use runmat_builtins::{CellArray, NumericDType, ObjectInstance, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::{any_type, deep_learning_error, gather_args, model, unsupported_error};

pub(super) const DLARRAY_CLASS: &str = "dlarray";
const AD_NODE_PROPERTY: &str = "__runmat_ad_node";

thread_local! {
    static ACTIVE_TAPE: RefCell<Option<Tape>> = const { RefCell::new(None) };
}

#[derive(Clone)]
struct Tape {
    nodes: Vec<Node>,
}

#[derive(Clone)]
struct Node {
    value: Tensor,
    kind: NodeKind,
}

#[derive(Clone)]
enum NodeKind {
    Leaf,
    Add {
        lhs: Option<usize>,
        rhs: Option<usize>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },
    Sub {
        lhs: Option<usize>,
        rhs: Option<usize>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },
    Mul {
        lhs: Option<usize>,
        rhs: Option<usize>,
        lhs_data: Vec<f64>,
        rhs_data: Vec<f64>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },
    Div {
        lhs: Option<usize>,
        rhs: Option<usize>,
        lhs_data: Vec<f64>,
        rhs_data: Vec<f64>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },
    MatMul {
        lhs: Option<usize>,
        rhs: Option<usize>,
        lhs_value: Tensor,
        rhs_value: Tensor,
    },
    FullyConnected {
        input: Option<usize>,
        weights: Option<usize>,
        bias: Option<usize>,
        input_value: Tensor,
        weights_value: Tensor,
    },
    SumAll {
        input: usize,
        input_shape: Vec<usize>,
    },
    Relu {
        input: usize,
        input_value: Tensor,
    },
    Elu {
        input: usize,
        input_value: Tensor,
        alpha: f64,
    },
    Softmax {
        input: usize,
        output_value: Tensor,
    },
}

pub(super) fn tape_is_active() -> bool {
    ACTIVE_TAPE.with(|slot| slot.borrow().is_some())
}

pub(super) async fn with_tape_context(
    function: Value,
    args: &[Value],
    requested_outputs: usize,
) -> BuiltinResult<Value> {
    ACTIVE_TAPE.with(|slot| {
        *slot.borrow_mut() = Some(Tape { nodes: Vec::new() });
    });
    let prepared = match prepare_tape_inputs(args.to_vec()) {
        Ok(args) => args,
        Err(err) => {
            ACTIVE_TAPE.with(|slot| *slot.borrow_mut() = None);
            return Err(err);
        }
    };
    let result = crate::call_feval_async_with_outputs(function, &prepared, requested_outputs).await;
    ACTIVE_TAPE.with(|slot| *slot.borrow_mut() = None);
    result
}

pub(super) fn prepare_tape_inputs(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    values.into_iter().map(annotate_tree).collect()
}

pub(super) fn annotate_tree(value: Value) -> BuiltinResult<Value> {
    if !tape_is_active() {
        return Ok(value);
    }
    match value {
        Value::Object(mut object) if object.class_name == DLARRAY_CLASS => {
            annotate_dlarray_object(&mut object)?;
            Ok(Value::Object(object))
        }
        Value::Object(mut object) if model::is_deep_learning_network_object(&object) => {
            annotate_network_object(&mut object)?;
            Ok(Value::Object(object))
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            let variables = crate::builtins::table::table_variables(&object)
                .map_err(|err| deep_learning_error("dlfeval", err.to_string()))?;
            let mut updated = StructValue::new();
            for (name, value) in variables.fields {
                updated.insert(name, annotate_tree(value)?);
            }
            crate::builtins::table::table_replace_variables_like(&object, updated)
                .map_err(|err| deep_learning_error("dlfeval", err.to_string()))
        }
        Value::Object(object) => Ok(Value::Object(object)),
        Value::Struct(mut st) => {
            for value in st.fields.values_mut() {
                *value = annotate_tree(value.clone())?;
            }
            Ok(Value::Struct(st))
        }
        Value::Cell(cell) => {
            let data = cell
                .data
                .into_iter()
                .map(annotate_tree)
                .collect::<BuiltinResult<Vec<_>>>()?;
            CellArray::new_with_shape(data, cell.shape)
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("dlfeval", err))
        }
        other => Ok(other),
    }
}

pub(super) fn annotate_dlarray_value(value: Value) -> BuiltinResult<Value> {
    if !tape_is_active() {
        return Ok(value);
    }
    match value {
        Value::Object(mut object) if object.class_name == DLARRAY_CLASS => {
            annotate_dlarray_object(&mut object)?;
            Ok(Value::Object(object))
        }
        other => Ok(other),
    }
}

fn annotate_dlarray_object(object: &mut ObjectInstance) -> BuiltinResult<()> {
    if object.properties.contains_key(AD_NODE_PROPERTY) {
        return Ok(());
    }
    let data =
        object.properties.get("Data").cloned().ok_or_else(|| {
            deep_learning_error("dlarray", "dlarray: traced object is missing Data")
        })?;
    let tensor = host_tensor_from_value(&data, "dlarray", "Data")?;
    let node = push_node(tensor, NodeKind::Leaf)?;
    object
        .properties
        .insert(AD_NODE_PROPERTY.to_string(), Value::Num(node as f64));
    Ok(())
}

fn annotate_network_object(object: &mut ObjectInstance) -> BuiltinResult<()> {
    let layers_value = object.properties.get("Layers").cloned();
    let Some(Value::Cell(mut layers)) = layers_value else {
        return Ok(());
    };
    for layer in &mut layers.data {
        let Value::Object(layer_object) = layer else {
            continue;
        };
        if layer_object.class_name != "nnet.cnn.layer.FullyConnectedLayer" {
            continue;
        }
        for parameter in ["Weights", "Bias"] {
            if let Some(value) = layer_object.properties.get(parameter).cloned() {
                let wrapped = match value {
                    Value::Object(object) if object.class_name == DLARRAY_CLASS => {
                        annotate_dlarray_value(Value::Object(object))?
                    }
                    Value::Tensor(_) | Value::Num(_) | Value::Int(_) => {
                        let mut wrapper = ObjectInstance::new(DLARRAY_CLASS.to_string());
                        wrapper.properties.insert("Data".to_string(), value);
                        wrapper
                            .properties
                            .insert("Format".to_string(), Value::String(String::new()));
                        wrapper
                            .properties
                            .insert("Labels".to_string(), Value::String(String::new()));
                        annotate_dlarray_object(&mut wrapper)?;
                        Value::Object(wrapper)
                    }
                    other => other,
                };
                layer_object
                    .properties
                    .insert(parameter.to_string(), wrapped);
            }
        }
    }
    object
        .properties
        .insert("Layers".to_string(), Value::Cell(layers.clone()));
    object.properties.insert(
        "Learnables".to_string(),
        model::learnables_struct(&layers.data, "dlfeval")?,
    );
    Ok(())
}

pub(super) fn dlarray_node_id(value: &Value) -> Option<usize> {
    let Value::Object(object) = value else {
        return None;
    };
    if object.class_name != DLARRAY_CLASS {
        return None;
    }
    object
        .properties
        .get(AD_NODE_PROPERTY)
        .and_then(|value| super::nonnegative_usize(value, "dlarray", AD_NODE_PROPERTY))
}

pub(super) fn dlarray_format_and_labels(value: &Value) -> (Value, Value) {
    let Value::Object(object) = value else {
        return (Value::String(String::new()), Value::String(String::new()));
    };
    (
        object
            .properties
            .get("Format")
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
        object
            .properties
            .get("Labels")
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    )
}

pub(super) fn dlarray_data(value: &Value, function: &'static str) -> BuiltinResult<Value> {
    match value {
        Value::Object(object) if object.class_name == DLARRAY_CLASS => {
            object.properties.get("Data").cloned().ok_or_else(|| {
                deep_learning_error(function, format!("{function}: dlarray is missing Data"))
            })
        }
        other => Ok(other.clone()),
    }
}

pub(super) fn host_tensor_from_value(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(object) if object.class_name == DLARRAY_CLASS => {
            let data = object.properties.get("Data").ok_or_else(|| {
                deep_learning_error(function, format!("{function}: dlarray is missing Data"))
            })?;
            host_tensor_from_value(data, function, label)
        }
        Value::GpuTensor(_) => Err(unsupported_error(
            function,
            format!("{function}: automatic differentiation for GPU-backed dlarray values requires provider-resident tape kernels"),
        )),
        other => tensor::value_into_tensor_for(function, other.clone())
            .map_err(|err| deep_learning_error(function, format!("{function}: {label} {err}"))),
    }
}

fn push_node(value: Tensor, kind: NodeKind) -> BuiltinResult<usize> {
    ACTIVE_TAPE.with(|slot| {
        let mut tape = slot.borrow_mut();
        let Some(tape) = tape.as_mut() else {
            return Err(deep_learning_error(
                "dlgradient",
                "dlgradient: no active automatic differentiation tape; call it inside dlfeval",
            ));
        };
        let id = tape.nodes.len();
        tape.nodes.push(Node { value, kind });
        Ok(id)
    })
}

fn value_for_node(id: usize) -> BuiltinResult<Tensor> {
    ACTIVE_TAPE.with(|slot| {
        let tape = slot.borrow();
        let Some(tape) = tape.as_ref() else {
            return Err(deep_learning_error(
                "dlgradient",
                "dlgradient: no active automatic differentiation tape",
            ));
        };
        tape.nodes
            .get(id)
            .map(|node| node.value.clone())
            .ok_or_else(|| deep_learning_error("dlgradient", "dlgradient: invalid tape node"))
    })
}

fn push_binary_node(
    value: Tensor,
    kind: impl FnOnce(Option<usize>, Option<usize>, Vec<usize>, Vec<usize>) -> NodeKind,
    lhs: &Payload,
    rhs: &Payload,
) -> BuiltinResult<Option<usize>> {
    if !tape_is_active() || (lhs.node.is_none() && rhs.node.is_none()) {
        return Ok(None);
    }
    Ok(Some(push_node(
        value,
        kind(
            lhs.node,
            rhs.node,
            lhs.tensor.shape.clone(),
            rhs.tensor.shape.clone(),
        ),
    )?))
}

#[runtime_builtin(
    name = "dlgradient",
    category = "deep_learning",
    summary = "Differentiate a traced scalar dlarray loss with respect to dlarray variables.",
    keywords = "dlgradient,deep learning,automatic differentiation,gradient",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::DLGRADIENT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::autodiff"
)]
pub(super) async fn dlgradient_builtin(loss: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let targets = gather_args(rest).await?;
    if targets.is_empty() {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: expected at least one dlarray, learnables, or dlnetwork target",
        ));
    }
    let loss = annotate_dlarray_value(loss)?;
    let Some(loss_node) = dlarray_node_id(&loss) else {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: loss must be a traced dlarray value produced inside dlfeval",
        ));
    };
    let loss_tensor = value_for_node(loss_node)?;
    if loss_tensor.data.len() != 1 {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: loss must be scalar",
        ));
    }
    let gradients = backward(loss_node)?;
    let mut outputs = Vec::with_capacity(targets.len());
    for target in targets {
        outputs.push(gradient_tree_for_target(&target, &gradients)?);
    }
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(n) if n > 1 => Ok(Value::OutputList(
            (0..n)
                .map(|idx| outputs.get(idx).cloned().unwrap_or(Value::Num(f64::NAN)))
                .collect(),
        )),
        _ if outputs.len() == 1 => Ok(outputs.remove(0)),
        _ => Ok(Value::OutputList(outputs)),
    }
}

fn backward(loss_node: usize) -> BuiltinResult<HashMap<usize, Tensor>> {
    let nodes = ACTIVE_TAPE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|tape| tape.nodes.clone())
            .ok_or_else(|| {
                deep_learning_error(
                    "dlgradient",
                    "dlgradient: no active automatic differentiation tape",
                )
            })
    })?;
    let mut grads: HashMap<usize, Tensor> = HashMap::new();
    grads.insert(
        loss_node,
        Tensor::new(vec![1.0], vec![1, 1]).map_err(|err| deep_learning_error("dlgradient", err))?,
    );
    for id in (0..=loss_node).rev() {
        let Some(grad) = grads.get(&id).cloned() else {
            continue;
        };
        let Some(node) = nodes.get(id) else {
            continue;
        };
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
                    reduce_gradient(&grad.data, lhs_shape, &node.value.shape)?,
                );
                accumulate_parent(
                    &mut grads,
                    *rhs,
                    reduce_gradient(&grad.data, rhs_shape, &node.value.shape)?,
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
                    reduce_gradient(&grad.data, lhs_shape, &node.value.shape)?,
                );
                let mut rhs_grad = reduce_gradient(&grad.data, rhs_shape, &node.value.shape)?;
                for value in &mut rhs_grad.data {
                    *value = -*value;
                }
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
                    let data = grad
                        .data
                        .iter()
                        .zip(rhs_values)
                        .map(|(g, r)| g * r)
                        .collect::<Vec<_>>();
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        reduce_gradient(&data, lhs_shape, &node.value.shape)?,
                    );
                }
                if let Some(parent) = *rhs {
                    let data = grad
                        .data
                        .iter()
                        .zip(lhs_values)
                        .map(|(g, l)| g * l)
                        .collect::<Vec<_>>();
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        reduce_gradient(&data, rhs_shape, &node.value.shape)?,
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
                    let data = grad
                        .data
                        .iter()
                        .zip(rhs_values)
                        .map(|(g, r)| g / r)
                        .collect::<Vec<_>>();
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        reduce_gradient(&data, lhs_shape, &node.value.shape)?,
                    );
                }
                if let Some(parent) = *rhs {
                    let data = grad
                        .data
                        .iter()
                        .zip(lhs_values)
                        .zip(rhs_values)
                        .map(|((g, l), r)| -g * l / (r * r))
                        .collect::<Vec<_>>();
                    accumulate_parent(
                        &mut grads,
                        Some(parent),
                        reduce_gradient(&data, rhs_shape, &node.value.shape)?,
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
                let mut input_grad = vec![0.0; input_value.data.len()];
                let mut weights_grad = vec![0.0; weights_value.data.len()];
                let mut bias_grad = vec![0.0; weights_value.rows];
                for row in 0..input_value.rows {
                    for out_col in 0..weights_value.rows {
                        let g = grad.data[row + out_col * grad.rows];
                        bias_grad[out_col] += g;
                        for feature in 0..input_value.cols {
                            input_grad[row + feature * input_value.rows] +=
                                g * weights_value.data[out_col + feature * weights_value.rows];
                            weights_grad[out_col + feature * weights_value.rows] +=
                                input_value.data[row + feature * input_value.rows] * g;
                        }
                    }
                }
                accumulate_parent(
                    &mut grads,
                    *input,
                    Tensor::new(input_grad, input_value.shape.clone())
                        .map_err(|err| deep_learning_error("dlgradient", err))?,
                );
                accumulate_parent(
                    &mut grads,
                    *weights,
                    Tensor::new(weights_grad, weights_value.shape.clone())
                        .map_err(|err| deep_learning_error("dlgradient", err))?,
                );
                accumulate_parent(
                    &mut grads,
                    *bias,
                    Tensor::new(bias_grad, vec![weights_value.rows, 1])
                        .map_err(|err| deep_learning_error("dlgradient", err))?,
                );
            }
            NodeKind::SumAll { input, input_shape } => {
                let seed = grad.data.iter().copied().sum::<f64>();
                let count = input_shape.iter().product();
                let tensor = Tensor::new(vec![seed; count], input_shape.clone())
                    .map_err(|err| deep_learning_error("dlgradient", err))?;
                accumulate_parent(&mut grads, Some(*input), tensor);
            }
            NodeKind::Relu { input, input_value } => {
                let data = grad
                    .data
                    .iter()
                    .zip(&input_value.data)
                    .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                    .collect::<Vec<_>>();
                accumulate_parent(
                    &mut grads,
                    Some(*input),
                    Tensor::new(data, input_value.shape.clone())
                        .map_err(|err| deep_learning_error("dlgradient", err))?,
                );
            }
            NodeKind::Elu {
                input,
                input_value,
                alpha,
            } => {
                let data = grad
                    .data
                    .iter()
                    .zip(&input_value.data)
                    .map(|(g, x)| if *x > 0.0 { *g } else { *g * alpha * x.exp() })
                    .collect::<Vec<_>>();
                accumulate_parent(
                    &mut grads,
                    Some(*input),
                    Tensor::new(data, input_value.shape.clone())
                        .map_err(|err| deep_learning_error("dlgradient", err))?,
                );
            }
            NodeKind::Softmax {
                input,
                output_value,
            } => {
                let mut data = vec![0.0; output_value.data.len()];
                for row in 0..output_value.rows {
                    let dot = (0..output_value.cols)
                        .map(|col| {
                            let idx = row + col * output_value.rows;
                            grad.data[idx] * output_value.data[idx]
                        })
                        .sum::<f64>();
                    for col in 0..output_value.cols {
                        let idx = row + col * output_value.rows;
                        data[idx] = output_value.data[idx] * (grad.data[idx] - dot);
                    }
                }
                accumulate_parent(
                    &mut grads,
                    Some(*input),
                    Tensor::new(data, output_value.shape.clone())
                        .map_err(|err| deep_learning_error("dlgradient", err))?,
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
) -> BuiltinResult<Tensor> {
    let parent_len = parent_shape.iter().product::<usize>();
    if parent_len == data.len() && parent_shape == output_shape {
        return Tensor::new(data.to_vec(), parent_shape.to_vec())
            .map_err(|err| deep_learning_error("dlgradient", err));
    }
    if parent_len == 1 {
        return Tensor::new(vec![data.iter().sum()], parent_shape.to_vec())
            .map_err(|err| deep_learning_error("dlgradient", err));
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
            for (dst, src) in existing.data.iter_mut().zip(&grad.data) {
                *dst += src;
            }
        })
        .or_insert(grad);
}

fn gradient_tree_for_target(
    target: &Value,
    gradients: &HashMap<usize, Tensor>,
) -> BuiltinResult<Value> {
    match target {
        Value::Object(object) if object.class_name == DLARRAY_CLASS => {
            let Some(id) = dlarray_node_id(target) else {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: target dlarray was not traced inside the active dlfeval",
                ));
            };
            let tensor = if let Some(tensor) = gradients.get(&id).cloned() {
                tensor
            } else {
                let target_tensor = host_tensor_from_value(target, "dlgradient", "target")?;
                Tensor::new(vec![0.0; target_tensor.data.len()], target_tensor.shape.clone())
                    .map_err(|err| deep_learning_error("dlgradient", err))?
            };
            let (format, labels) = dlarray_format_and_labels(target);
            Ok(super::object(
                DLARRAY_CLASS,
                vec![
                    ("Data", Value::Tensor(tensor)),
                    ("Format", format),
                    ("Labels", labels),
                ],
            ))
        }
        Value::Object(object) if model::is_deep_learning_network_object(object) => {
            if let Some(learnables) = object.properties.get("Learnables") {
                gradient_tree_for_target(learnables, gradients)
            } else {
                Ok(Value::Struct(StructValue::new()))
            }
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
            let variables = crate::builtins::table::table_variables(object)
                .map_err(|err| deep_learning_error("dlgradient", err.to_string()))?;
            let mut updated = StructValue::new();
            for (name, value) in variables.fields {
                updated.insert(name, gradient_tree_for_target(&value, gradients)?);
            }
            crate::builtins::table::table_replace_variables_like(object, updated)
                .map_err(|err| deep_learning_error("dlgradient", err.to_string()))
        }
        Value::Struct(st) => {
            if is_learnables_struct(st) {
                let mut out = st.clone();
                let value = st.fields.get("Value").ok_or_else(|| {
                    deep_learning_error("dlgradient", "dlgradient: learnables tree is missing Value")
                })?;
                out.insert("Value", gradient_tree_for_target(value, gradients)?);
                return Ok(Value::Struct(out));
            }
            let mut out = StructValue::new();
            for (name, value) in &st.fields {
                out.insert(name.clone(), gradient_tree_for_target(value, gradients)?);
            }
            Ok(Value::Struct(out))
        }
        Value::Cell(cell) => {
            let data = cell
                .data
                .iter()
                .map(|value| gradient_tree_for_target(value, gradients))
                .collect::<BuiltinResult<Vec<_>>>()?;
            CellArray::new_with_shape(data, cell.shape.clone())
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("dlgradient", err))
        }
        other => Err(deep_learning_error(
            "dlgradient",
            format!("dlgradient: target must be a traced dlarray, learnables tree, or dlnetwork, got {other:?}"),
        )),
    }
}

fn is_learnables_struct(value: &StructValue) -> bool {
    ["Layer", "Parameter", "Value"]
        .iter()
        .all(|field| value.fields.contains_key(*field))
}

#[derive(Clone)]
struct Payload {
    tensor: Tensor,
    node: Option<usize>,
    format: Value,
    labels: Value,
    dtype: NumericDType,
}

impl Payload {
    fn parse(value: Value, function: &'static str) -> BuiltinResult<Self> {
        let node = dlarray_node_id(&value);
        let (format, labels) = dlarray_format_and_labels(&value);
        let data = dlarray_data(&value, function)?;
        let tensor = host_tensor_from_value(&data, function, "operand")?;
        let dtype = tensor.dtype;
        Ok(Self {
            tensor,
            node,
            format,
            labels,
            dtype,
        })
    }
}

fn wrap_result(payload: &Payload, tensor: Tensor, node: Option<usize>) -> Value {
    let mut out = ObjectInstance::new(DLARRAY_CLASS.to_string());
    out.properties
        .insert("Data".to_string(), Value::Tensor(tensor));
    out.properties
        .insert("Format".to_string(), payload.format.clone());
    out.properties
        .insert("Labels".to_string(), payload.labels.clone());
    if let Some(id) = node {
        out.properties
            .insert(AD_NODE_PROPERTY.to_string(), Value::Num(id as f64));
    }
    Value::Object(out)
}

#[runtime_builtin(
    name = "dlarray.plus",
    category = "deep_learning",
    summary = "Add dlarray values and record tape gradients when active.",
    keywords = "dlarray,plus,deep learning,autodiff",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::autodiff"
)]
pub(super) fn dlarray_plus_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(
        lhs,
        rhs,
        "plus",
        |a, b| a + b,
        |lhs, rhs, out| {
            push_binary_node(
                out,
                |l, r, ls, rs| NodeKind::Add {
                    lhs: l,
                    rhs: r,
                    lhs_shape: ls,
                    rhs_shape: rs,
                },
                lhs,
                rhs,
            )
        },
    )
}

#[runtime_builtin(
    name = "dlarray.minus",
    category = "deep_learning",
    summary = "Subtract dlarray values and record tape gradients when active.",
    keywords = "dlarray,minus,deep learning,autodiff",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::autodiff"
)]
pub(super) fn dlarray_minus_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(
        lhs,
        rhs,
        "minus",
        |a, b| a - b,
        |lhs, rhs, out| {
            push_binary_node(
                out,
                |l, r, ls, rs| NodeKind::Sub {
                    lhs: l,
                    rhs: r,
                    lhs_shape: ls,
                    rhs_shape: rs,
                },
                lhs,
                rhs,
            )
        },
    )
}

#[runtime_builtin(
    name = "dlarray.times",
    category = "deep_learning",
    summary = "Multiply dlarray values elementwise and record tape gradients when active.",
    keywords = "dlarray,times,deep learning,autodiff",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::autodiff"
)]
pub(super) fn dlarray_times_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(
        lhs,
        rhs,
        "times",
        |a, b| a * b,
        |lhs, rhs, out| {
            let (lhs_values, rhs_values, _) =
                tensor::binary_numeric_tensors(&lhs.tensor, &rhs.tensor, "dlarray.times", "times")?;
            if !tape_is_active() || (lhs.node.is_none() && rhs.node.is_none()) {
                return Ok(None);
            }
            Ok(Some(push_node(
                out,
                NodeKind::Mul {
                    lhs: lhs.node,
                    rhs: rhs.node,
                    lhs_data: lhs_values,
                    rhs_data: rhs_values,
                    lhs_shape: lhs.tensor.shape.clone(),
                    rhs_shape: rhs.tensor.shape.clone(),
                },
            )?))
        },
    )
}

#[runtime_builtin(
    name = "dlarray.rdivide",
    category = "deep_learning",
    summary = "Divide dlarray values elementwise and record tape gradients when active.",
    keywords = "dlarray,rdivide,deep learning,autodiff",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::autodiff"
)]
pub(super) fn dlarray_rdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    dlarray_binary(
        lhs,
        rhs,
        "rdivide",
        |a, b| a / b,
        |lhs, rhs, out| {
            let (lhs_values, rhs_values, _) = tensor::binary_numeric_tensors(
                &lhs.tensor,
                &rhs.tensor,
                "dlarray.rdivide",
                "rdivide",
            )?;
            if !tape_is_active() || (lhs.node.is_none() && rhs.node.is_none()) {
                return Ok(None);
            }
            Ok(Some(push_node(
                out,
                NodeKind::Div {
                    lhs: lhs.node,
                    rhs: rhs.node,
                    lhs_data: lhs_values,
                    rhs_data: rhs_values,
                    lhs_shape: lhs.tensor.shape.clone(),
                    rhs_shape: rhs.tensor.shape.clone(),
                },
            )?))
        },
    )
}

fn dlarray_binary(
    lhs: Value,
    rhs: Value,
    function: &'static str,
    op: impl Fn(f64, f64) -> f64,
    record: impl FnOnce(&Payload, &Payload, Tensor) -> BuiltinResult<Option<usize>>,
) -> BuiltinResult<Value> {
    let lhs = Payload::parse(lhs, function)?;
    let rhs = Payload::parse(rhs, function)?;
    let (lhs_values, rhs_values, shape) =
        tensor::binary_numeric_tensors(&lhs.tensor, &rhs.tensor, function, function)?;
    let data = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(a, b)| op(*a, *b))
        .collect::<Vec<_>>();
    let out = Tensor::new_with_dtype(data, shape, lhs.dtype)
        .map_err(|err| deep_learning_error(function, err))?;
    let node = record(&lhs, &rhs, out.clone())?;
    Ok(wrap_result(&lhs, out, node))
}

#[runtime_builtin(
    name = "dlarray.mtimes",
    category = "deep_learning",
    summary = "Matrix-multiply dlarray values and record tape gradients when active.",
    keywords = "dlarray,mtimes,deep learning,autodiff",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::autodiff"
)]
pub(super) fn dlarray_mtimes_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    let lhs = Payload::parse(lhs, "mtimes")?;
    let rhs = Payload::parse(rhs, "mtimes")?;
    let out = crate::builtins::common::linalg::matmul_real(&lhs.tensor, &rhs.tensor)
        .map_err(|err| deep_learning_error("mtimes", err))?;
    let node = if tape_is_active() && (lhs.node.is_some() || rhs.node.is_some()) {
        Some(push_node(
            out.clone(),
            NodeKind::MatMul {
                lhs: lhs.node,
                rhs: rhs.node,
                lhs_value: lhs.tensor.clone(),
                rhs_value: rhs.tensor.clone(),
            },
        )?)
    } else {
        None
    };
    Ok(wrap_result(&lhs, out, node))
}

#[runtime_builtin(
    name = "dlarray.sum",
    category = "deep_learning",
    summary = "Reduce dlarray values to a scalar sum and record tape gradients when active.",
    keywords = "dlarray,sum,deep learning,autodiff",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::autodiff"
)]
pub(super) fn dlarray_sum_builtin(input: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest
        .iter()
        .any(|value| !matches!(value, Value::String(text) if text.eq_ignore_ascii_case("all")))
    {
        return Err(deep_learning_error(
            "sum",
            "sum: dlarray autodiff supports scalar reductions over all elements",
        ));
    }
    let payload = Payload::parse(input, "sum")?;
    let total = payload.tensor.data.iter().sum::<f64>();
    let out =
        Tensor::new(vec![total], vec![1, 1]).map_err(|err| deep_learning_error("sum", err))?;
    let node = if tape_is_active() {
        payload
            .node
            .map(|id| {
                push_node(
                    out.clone(),
                    NodeKind::SumAll {
                        input: id,
                        input_shape: payload.tensor.shape.clone(),
                    },
                )
            })
            .transpose()?
    } else {
        None
    };
    Ok(wrap_result(&payload, out, node))
}

pub(super) fn record_activation(
    input: &Value,
    output: Tensor,
    activation: ActivationKind,
) -> BuiltinResult<Value> {
    let payload = Payload::parse(input.clone(), "forward")?;
    let node = if tape_is_active() {
        payload
            .node
            .map(|id| {
                let kind = match activation {
                    ActivationKind::Relu => NodeKind::Relu {
                        input: id,
                        input_value: payload.tensor.clone(),
                    },
                    ActivationKind::Elu { alpha } => NodeKind::Elu {
                        input: id,
                        input_value: payload.tensor.clone(),
                        alpha,
                    },
                    ActivationKind::Softmax => NodeKind::Softmax {
                        input: id,
                        output_value: output.clone(),
                    },
                };
                push_node(output.clone(), kind)
            })
            .transpose()?
    } else {
        None
    };
    Ok(wrap_result(&payload, output, node))
}

pub(super) enum ActivationKind {
    Relu,
    Elu { alpha: f64 },
    Softmax,
}

pub(super) fn record_network_forward(
    network: &ObjectInstance,
    input: &Value,
    function: &'static str,
) -> BuiltinResult<Option<Value>> {
    if !tape_is_active() {
        return Ok(None);
    }
    let layers = network
        .properties
        .get("Layers")
        .cloned()
        .map(|value| super::layers_from_value(value, function))
        .transpose()?
        .unwrap_or_default();
    model::validate_forward_layers(&layers, function)?;
    let mut current = input.clone();
    let mut saw_input = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        current = match layer.class_name.as_str() {
            "nnet.cnn.layer.FeatureInputLayer" => {
                let tensor = host_tensor_from_value(&current, function, "input")?;
                let expected = model::feature_input_width(&layer, function)?;
                if tensor.cols != expected {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: input has {} features, but featureInputLayer expects {expected}",
                            tensor.cols
                        ),
                    ));
                }
                saw_input = true;
                current
            }
            "nnet.cnn.layer.FullyConnectedLayer" => {
                record_fully_connected(&current, &layer, function)?
            }
            "nnet.cnn.layer.ReLULayer" => {
                let input_tensor = host_tensor_from_value(&current, function, "input")?;
                let output = map_tensor(input_tensor, |value| value.max(0.0), function)?;
                record_activation(&current, output, ActivationKind::Relu)?
            }
            "nnet.cnn.layer.ELULayer" => {
                let input_tensor = host_tensor_from_value(&current, function, "input")?;
                let alpha = layer
                    .properties
                    .get("Alpha")
                    .map(|value| super::numeric_values(value, function, "Alpha"))
                    .transpose()?
                    .and_then(|values| values.first().copied())
                    .unwrap_or(1.0);
                let output = map_tensor(
                    input_tensor,
                    |value| {
                        if value > 0.0 {
                            value
                        } else {
                            alpha * (value.exp() - 1.0)
                        }
                    },
                    function,
                )?;
                record_activation(&current, output, ActivationKind::Elu { alpha })?
            }
            "nnet.cnn.layer.SoftmaxLayer" => {
                let input_tensor = host_tensor_from_value(&current, function, "input")?;
                let output = softmax_rows(input_tensor, function)?;
                record_activation(&current, output, ActivationKind::Softmax)?
            }
            "nnet.cnn.layer.ClassificationOutputLayer" | "nnet.cnn.layer.RegressionOutputLayer" => {
                current
            }
            other => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: unsupported layer type '{other}'"),
                ));
            }
        };
    }
    if !saw_input {
        return Err(deep_learning_error(
            function,
            format!("{function}: network must start with a supported input layer"),
        ));
    }
    Ok(Some(current))
}

fn record_fully_connected(
    input: &Value,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Value> {
    let input_payload = Payload::parse(input.clone(), function)?;
    let weights_value = layer.properties.get("Weights").cloned().ok_or_else(|| {
        deep_learning_error(function, format!("{function}: layer is missing Weights"))
    })?;
    let bias_value = layer.properties.get("Bias").cloned().ok_or_else(|| {
        deep_learning_error(function, format!("{function}: layer is missing Bias"))
    })?;
    let weights = Payload::parse(weights_value, function)?;
    let bias = Payload::parse(bias_value, function)?;
    if weights.tensor.shape.len() > 2 || weights.tensor.cols != input_payload.tensor.cols {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Weights must be outputSize-by-{}",
                input_payload.tensor.cols
            ),
        ));
    }
    if bias.tensor.data.len() != weights.tensor.rows {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Bias must have {} elements",
                weights.tensor.rows
            ),
        ));
    }
    let mut out = vec![0.0; input_payload.tensor.rows * weights.tensor.rows];
    for row in 0..input_payload.tensor.rows {
        for out_col in 0..weights.tensor.rows {
            let mut acc = bias.tensor.data[out_col];
            for feature in 0..input_payload.tensor.cols {
                acc += input_payload.tensor.data[row + feature * input_payload.tensor.rows]
                    * weights.tensor.data[out_col + feature * weights.tensor.rows];
            }
            out[row + out_col * input_payload.tensor.rows] = acc;
        }
    }
    let output = Tensor::new(out, vec![input_payload.tensor.rows, weights.tensor.rows])
        .map_err(|err| deep_learning_error(function, err))?;
    let node = if input_payload.node.is_some() || weights.node.is_some() || bias.node.is_some() {
        Some(push_node(
            output.clone(),
            NodeKind::FullyConnected {
                input: input_payload.node,
                weights: weights.node,
                bias: bias.node,
                input_value: input_payload.tensor.clone(),
                weights_value: weights.tensor.clone(),
            },
        )?)
    } else {
        None
    };
    Ok(wrap_result(&input_payload, output, node))
}

fn map_tensor(
    input: Tensor,
    f: impl Fn(f64) -> f64,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    Tensor::new_with_dtype(
        input.data.into_iter().map(f).collect(),
        input.shape,
        input.dtype,
    )
    .map_err(|err| deep_learning_error(function, err))
}

fn softmax_rows(input: Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    let mut out = vec![0.0; input.data.len()];
    for row in 0..input.rows {
        let max_value = (0..input.cols)
            .map(|col| input.data[row + col * input.rows])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0;
        for col in 0..input.cols {
            let value = (input.data[row + col * input.rows] - max_value).exp();
            out[row + col * input.rows] = value;
            denom += value;
        }
        if !denom.is_finite() || denom <= 0.0 {
            return Err(deep_learning_error(
                function,
                format!("{function}: softmax produced invalid normalization"),
            ));
        }
        for col in 0..input.cols {
            out[row + col * input.rows] /= denom;
        }
    }
    Tensor::new(out, input.shape).map_err(|err| deep_learning_error(function, err))
}

fn matmul_grad_lhs(grad: &Tensor, lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    let mut data = vec![0.0; lhs.data.len()];
    for row in 0..lhs.rows {
        for col in 0..lhs.cols {
            let mut acc = 0.0;
            for k in 0..rhs.cols {
                acc += grad.data[row + k * grad.rows] * rhs.data[col + k * rhs.rows];
            }
            data[row + col * lhs.rows] = acc;
        }
    }
    Tensor::new(data, lhs.shape.clone()).map_err(|err| deep_learning_error("dlgradient", err))
}

fn matmul_grad_rhs(grad: &Tensor, lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    let mut data = vec![0.0; rhs.data.len()];
    for row in 0..rhs.rows {
        for col in 0..rhs.cols {
            let mut acc = 0.0;
            for k in 0..lhs.rows {
                acc += lhs.data[k + row * lhs.rows] * grad.data[k + col * grad.rows];
            }
            data[row + col * rhs.rows] = acc;
        }
    }
    Tensor::new(data, rhs.shape.clone()).map_err(|err| deep_learning_error("dlgradient", err))
}
