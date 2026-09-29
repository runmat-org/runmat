use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, IntValue, NumericDType, NumericScalar, NumericStorage, ObjectInstance, StructValue,
    Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::layer_identity::ForwardLayerKind;
use super::{any_type, deep_learning_error, model, unsupported_error};

pub(super) const DLARRAY_CLASS: runmat_types::StaticClassIdentity = super::DLARRAY_CLASS;
const AD_NODE_PROPERTY: &str = "__runmat_ad_node";
const AD_TAPE_PROPERTY: &str = "__runmat_ad_tape";
static NEXT_TAPE_ID: AtomicU64 = AtomicU64::new(1);

runmat_thread_local::runmat_thread_local! {
    // Only installed for a callback future's poll or destruction. A suspended
    // callback never leaves its tape active for another execution on the thread.
    static ACTIVE_TAPES: RefCell<Vec<Rc<RefCell<Tape>>>> = const { RefCell::new(Vec::new()) };
}

fn active_tape() -> Option<Rc<RefCell<Tape>>> {
    ACTIVE_TAPES.with(|slot| slot.borrow().last().cloned())
}

struct TapeActivation(Rc<RefCell<Tape>>);

impl TapeActivation {
    fn enter(tape: Rc<RefCell<Tape>>) -> Self {
        ACTIVE_TAPES.with(|slot| slot.borrow_mut().push(tape.clone()));
        Self(tape)
    }
}

impl Drop for TapeActivation {
    fn drop(&mut self) {
        ACTIVE_TAPES.with(|slot| {
            let popped = slot.borrow_mut().pop();
            debug_assert!(popped.is_some_and(|tape| Rc::ptr_eq(&tape, &self.0)));
        });
    }
}

struct TapeFuture<F> {
    tape: Rc<RefCell<Tape>>,
    future: Option<Pin<Box<F>>>,
}

impl<F: Future> Future for TapeFuture<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let _activation = TapeActivation::enter(this.tape.clone());
        this.future
            .as_mut()
            .expect("tape future is live while polled")
            .as_mut()
            .poll(context)
    }
}

impl<F> Drop for TapeFuture<F> {
    fn drop(&mut self) {
        let _activation = TapeActivation::enter(self.tape.clone());
        self.future.take();
    }
}

#[derive(Clone)]
struct Tape {
    id: u64,
    nodes: Vec<Node>,
}

impl Tape {
    fn new() -> Self {
        let id = NEXT_TAPE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .expect("automatic differentiation tape identity space exhausted");
        Self {
            id,
            nodes: Vec::new(),
        }
    }
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
    active_tape().is_some()
}

pub(super) async fn with_tape_context(
    function: Value,
    args: &[Value],
    requested_outputs: usize,
) -> BuiltinResult<Value> {
    let tape = Rc::new(RefCell::new(Tape::new()));
    TapeFuture {
        tape,
        future: Some(Box::pin(async move {
            let prepared = prepare_tape_inputs(args.to_vec())?;
            crate::call_feval_async_with_outputs(function, &prepared, requested_outputs).await
        })),
    }
    .await
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
        Value::StructArray(array) => array.try_map_values(annotate_tree).map(Value::StructArray),
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
    if dlarray_node_id(&Value::Object(object.clone())).is_some() {
        return Ok(());
    }
    let data =
        object.properties.get("Data").cloned().ok_or_else(|| {
            deep_learning_error("dlarray", "dlarray: traced object is missing Data")
        })?;
    let tensor = host_tensor_from_value(&data, "dlarray", "Data")?;
    let node = push_node(tensor, NodeKind::Leaf)?;
    let tape_id = active_tape()
        .expect("node insertion requires active tape")
        .borrow()
        .id;
    object.properties.insert(
        AD_NODE_PROPERTY.to_string(),
        Value::Int(IntValue::U64(
            u64::try_from(node).expect("node index fits u64"),
        )),
    );
    object.properties.insert(
        AD_TAPE_PROPERTY.to_string(),
        Value::Int(IntValue::U64(tape_id)),
    );
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
        if !layer_object
            .class_name
            .is(super::FULLY_CONNECTED_LAYER_CLASS)
        {
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
    let tape = active_tape()?;
    let tape = tape.borrow();
    let tape_id = tape.id;
    let Value::Object(object) = value else {
        return None;
    };
    if object.class_name != DLARRAY_CLASS {
        return None;
    }
    if object.properties.get(AD_TAPE_PROPERTY) != Some(&Value::Int(IntValue::U64(tape_id))) {
        return None;
    }
    let id = object
        .properties
        .get(AD_NODE_PROPERTY)
        .and_then(|value| match value {
            Value::Int(IntValue::U64(id)) => usize::try_from(*id).ok(),
            _ => None,
        })?;
    (id < tape.nodes.len()).then_some(id)
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
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape; call it inside dlfeval",
        )
    })?;
    let mut tape = tape.borrow_mut();
    let id = tape.nodes.len();
    tape.nodes.push(Node { value, kind });
    Ok(id)
}

fn value_for_node(id: usize) -> BuiltinResult<Tensor> {
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape",
        )
    })?;
    let value = tape.borrow().nodes.get(id).map(|node| node.value.clone());
    value.ok_or_else(|| deep_learning_error("dlgradient", "dlgradient: invalid tape node"))
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
    if rest.is_empty() {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: expected at least one dlarray, learnables, or dlnetwork target",
        ));
    }
    let requested = crate::output_count::current_output_count().unwrap_or(1);
    if requested > rest.len() {
        return Err(deep_learning_error(
            "dlgradient",
            format!(
                "dlgradient: requested {requested} outputs for {} gradient targets",
                rest.len()
            ),
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
    if tensor::tensor_element_len(&loss_tensor) != 1 {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: loss must be scalar",
        ));
    }
    let gradients = backward(loss_node)?;
    let mut outputs = Vec::with_capacity(requested);
    for target in rest.iter().take(requested) {
        outputs.push(gradient_tree_for_target(target, &gradients)?);
    }
    if requested == 1 {
        Ok(outputs.remove(0))
    } else {
        Ok(Value::OutputList(outputs))
    }
}

fn backward(loss_node: usize) -> BuiltinResult<HashMap<usize, Tensor>> {
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape",
        )
    })?;
    let nodes = tape.borrow().nodes.clone();
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
                floating_tensor_from_f64(
                    vec![0.0; tensor::tensor_element_len(&target_tensor)],
                    target_tensor.shape.clone(),
                    target_tensor.numeric_dtype(),
                    "dlgradient",
                )?
            };
            let (format, labels) = dlarray_format_and_labels(target);
            Ok(super::object_with_identity(
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
        Value::StructArray(array) => array
            .clone()
            .try_map_values(|value| gradient_tree_for_target(&value, gradients))
            .map(Value::StructArray),
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
        let data = dlarray_data(&value, function)?;
        if crate::builtins::common::validation::value_has_native_integer_class(&data) {
            return Err(deep_learning_error(
                function,
                format!("{function}: native integer dlarray operands require explicit double or single conversion"),
            ));
        }
        let node = dlarray_node_id(&value);
        let (format, labels) = dlarray_format_and_labels(&value);
        let tensor = host_tensor_from_value(&data, function, "operand")?;
        let dtype = tensor.numeric_dtype();
        if !matches!(dtype, NumericDType::F64 | NumericDType::F32) {
            return Err(deep_learning_error(
                function,
                format!("{function}: dlarray arithmetic requires double or single data"),
            ));
        }
        Ok(Self {
            tensor,
            node,
            format,
            labels,
            dtype,
        })
    }
}

fn floating_tensor_values(tensor: &Tensor, function: &'static str) -> BuiltinResult<Vec<f64>> {
    match tensor
        .clone()
        .into_numeric_storage()
        .map_err(|err| deep_learning_error(function, err))?
    {
        NumericStorage::F64(values) => Ok(values),
        NumericStorage::F32(values) => Ok(values.into_iter().map(f64::from).collect::<Vec<_>>()),
        storage => Err(deep_learning_error(
            function,
            format!(
                "{function}: automatic differentiation requires double or single tensors, got {}",
                storage.class_name()
            ),
        )),
    }
}

fn floating_tensor_from_f64(
    data: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    match dtype {
        NumericDType::F64 | NumericDType::F32 => Tensor::new_with_dtype(data, shape, dtype)
            .map_err(|err| deep_learning_error(function, err)),
        dtype => Err(deep_learning_error(
            function,
            format!(
                "{function}: automatic differentiation cannot construct {} gradients",
                dtype.class_name()
            ),
        )),
    }
}

fn floating_scalar_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        value => panic!(
            "automatic differentiation gradient unexpectedly used {} storage",
            value.class_name()
        ),
    }
}

fn node_dtype(nodes: &[Node], id: Option<usize>, fallback: NumericDType) -> NumericDType {
    id.and_then(|id| nodes.get(id))
        .map(|node| node.value.numeric_dtype())
        .unwrap_or(fallback)
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
        out.properties.insert(
            AD_NODE_PROPERTY.to_string(),
            Value::Int(IntValue::U64(
                u64::try_from(id).expect("node index fits u64"),
            )),
        );
        let tape_id = active_tape()
            .expect("recorded output requires active tape")
            .borrow()
            .id;
        out.properties.insert(
            AD_TAPE_PROPERTY.to_string(),
            Value::Int(IntValue::U64(tape_id)),
        );
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
    let computed = crate::builtins::common::linalg::matmul_real(&lhs.tensor, &rhs.tensor)
        .map_err(|err| deep_learning_error("mtimes", err))?;
    let out = if lhs.dtype == NumericDType::F64 {
        computed
    } else {
        let shape = computed.shape.clone();
        let NumericStorage::F64(values) = computed
            .into_numeric_storage()
            .map_err(|err| deep_learning_error("mtimes", err))?
        else {
            unreachable!("real matrix multiplication returns double storage")
        };
        floating_tensor_from_f64(values, shape, lhs.dtype, "mtimes")?
    };
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
    let total = floating_tensor_values(&payload.tensor, "sum")?
        .iter()
        .sum::<f64>();
    let out = floating_tensor_from_f64(vec![total], vec![1, 1], payload.dtype, "sum")?;
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
        current = match ForwardLayerKind::from_identity(&layer.class_name) {
            Some(ForwardLayerKind::FeatureInput) => {
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
            Some(ForwardLayerKind::FullyConnected) => {
                record_fully_connected(&current, &layer, function)?
            }
            Some(ForwardLayerKind::Relu) => {
                let input_tensor = host_tensor_from_value(&current, function, "input")?;
                let output = map_tensor(input_tensor, |value| value.max(0.0), function)?;
                record_activation(&current, output, ActivationKind::Relu)?
            }
            Some(ForwardLayerKind::Elu) => {
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
            Some(ForwardLayerKind::Softmax) => {
                let input_tensor = host_tensor_from_value(&current, function, "input")?;
                let output = softmax_rows(input_tensor, function)?;
                record_activation(&current, output, ActivationKind::Softmax)?
            }
            Some(kind) if kind.is_output() => current,
            _ => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: unsupported layer type '{}'", layer.class_name),
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
    if tensor::tensor_element_len(&bias.tensor) != weights.tensor.rows {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Bias must have {} elements",
                weights.tensor.rows
            ),
        ));
    }
    let input_values = floating_tensor_values(&input_payload.tensor, function)?;
    let weight_values = floating_tensor_values(&weights.tensor, function)?;
    let bias_values = floating_tensor_values(&bias.tensor, function)?;
    let mut out = vec![0.0; input_payload.tensor.rows * weights.tensor.rows];
    for row in 0..input_payload.tensor.rows {
        for out_col in 0..weights.tensor.rows {
            let mut acc = bias_values[out_col];
            for feature in 0..input_payload.tensor.cols {
                acc += input_values[row + feature * input_payload.tensor.rows]
                    * weight_values[out_col + feature * weights.tensor.rows];
            }
            out[row + out_col * input_payload.tensor.rows] = acc;
        }
    }
    let output = floating_tensor_from_f64(
        out,
        vec![input_payload.tensor.rows, weights.tensor.rows],
        input_payload.dtype,
        function,
    )?;
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
    let dtype = input.numeric_dtype();
    let shape = input.shape.clone();
    let data = floating_tensor_values(&input, function)?
        .into_iter()
        .map(f)
        .collect();
    floating_tensor_from_f64(data, shape, dtype, function)
}

fn softmax_rows(input: Tensor, function: &'static str) -> BuiltinResult<Tensor> {
    let dtype = input.numeric_dtype();
    let input_values = floating_tensor_values(&input, function)?;
    let mut out = vec![0.0; input_values.len()];
    for row in 0..input.rows {
        let max_value = (0..input.cols)
            .map(|col| input_values[row + col * input.rows])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0;
        for col in 0..input.cols {
            let value = (input_values[row + col * input.rows] - max_value).exp();
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
    floating_tensor_from_f64(out, input.shape, dtype, function)
}

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

#[cfg(test)]
mod struct_array_tests {
    use super::*;
    use runmat_value::{IntValue, IntegerStorage, StructArray};

    #[test]
    fn suspended_and_dropped_callback_futures_do_not_publish_their_tapes() {
        let first_tape = Rc::new(RefCell::new(Tape::new()));
        let second_tape = Rc::new(RefCell::new(Tape::new()));
        let make_future = |tape: Rc<RefCell<Tape>>| TapeFuture {
            tape,
            future: Some(Box::pin(async {
                assert!(tape_is_active());
                push_node(Tensor::new(vec![1.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
                futures::future::pending::<()>().await;
            })),
        };
        let mut first = Box::pin(make_future(first_tape.clone()));
        let mut second = Box::pin(make_future(second_tape.clone()));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(first.as_mut().poll(&mut context).is_pending());
        assert!(!tape_is_active());
        assert!(second.as_mut().poll(&mut context).is_pending());
        assert!(!tape_is_active());
        assert_eq!(first_tape.borrow().nodes.len(), 1);
        assert_eq!(second_tape.borrow().nodes.len(), 1);
        drop(first);
        drop(second);
        assert!(!tape_is_active());
    }

    #[test]
    fn nested_tape_activation_restores_the_outer_graph() {
        let outer = Rc::new(RefCell::new(Tape::new()));
        let inner = Rc::new(RefCell::new(Tape::new()));
        let _outer = TapeActivation::enter(outer.clone());
        push_node(Tensor::new(vec![1.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
        {
            let _inner = TapeActivation::enter(inner.clone());
            push_node(Tensor::new(vec![2.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
        }
        push_node(Tensor::new(vec![3.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
        assert_eq!(outer.borrow().nodes.len(), 2);
        assert_eq!(inner.borrow().nodes.len(), 1);
    }

    #[test]
    fn traced_node_ids_are_exact_and_belong_only_to_the_active_tape() {
        let outer = Rc::new(RefCell::new(Tape::new()));
        let inner = Rc::new(RefCell::new(Tape::new()));
        let _outer = TapeActivation::enter(outer.clone());
        let traced = annotate_dlarray_value(dlarray(4.0)).expect("outer traced value");
        assert_eq!(dlarray_node_id(&traced), Some(0));
        let Value::Object(object) = &traced else {
            panic!("dlarray")
        };
        assert_eq!(
            object.properties.get(AD_NODE_PROPERTY),
            Some(&Value::Int(IntValue::U64(0)))
        );
        {
            let _inner = TapeActivation::enter(inner.clone());
            assert_eq!(dlarray_node_id(&traced), None);
            let retraced = annotate_dlarray_value(traced.clone()).expect("inner leaf");
            assert_eq!(dlarray_node_id(&retraced), Some(0));
            assert_eq!(inner.borrow().nodes.len(), 1);
        }
        assert_eq!(dlarray_node_id(&traced), Some(0));
        assert_eq!(outer.borrow().nodes.len(), 1);

        let mut forged = traced;
        let Value::Object(object) = &mut forged else {
            panic!("dlarray")
        };
        object
            .properties
            .insert(AD_NODE_PROPERTY.into(), Value::Num(0.0));
        assert_eq!(dlarray_node_id(&forged), None);
        object
            .properties
            .insert(AD_NODE_PROPERTY.into(), Value::Int(IntValue::U64(u64::MAX)));
        assert_eq!(dlarray_node_id(&forged), None);
    }

    fn wrapped(data: Value) -> Value {
        let mut object = ObjectInstance::new(DLARRAY_CLASS.to_string());
        object.properties.insert("Data".into(), data);
        Value::Object(object)
    }

    fn dlarray(value: f64) -> Value {
        wrapped(Value::Num(value))
    }

    fn element(value: f64) -> StructValue {
        let mut structure = StructValue::new();
        structure.insert("payload", dlarray(value));
        structure
    }

    #[test]
    fn annotation_and_gradient_preserve_typed_struct_array_shape_and_schema() {
        let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
        let input = Value::StructArray(
            StructArray::new(vec![element(1.0), element(2.0)], vec![2, 1]).unwrap(),
        );
        let annotated = annotate_tree(input).expect("annotate structure array");
        let gradient = gradient_tree_for_target(&annotated, &HashMap::new())
            .expect("gradient structure array");

        let Value::StructArray(gradient) = gradient else {
            panic!("expected typed structure array");
        };
        assert_eq!(gradient.shape(), &[2, 1]);
        assert_eq!(
            gradient
                .field_names()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["payload"]
        );
        for value in gradient.field_values("payload").unwrap() {
            let Value::Object(object) = value else {
                panic!("expected gradient dlarray");
            };
            let Value::Tensor(data) = object.properties.get("Data").unwrap() else {
                panic!("expected gradient tensor");
            };
            assert_eq!(data.materialize_f64(), [0.0]);
        }
    }

    #[test]
    fn dlarray_constructor_records_active_tape_and_rejects_resident_tape_input() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .expect("resident data");
            let tape = Rc::new(RefCell::new(Tape::new()));
            let _activation = TapeActivation::enter(tape.clone());
            let host = futures::executor::block_on(super::super::training::dlarray_builtin(
                Value::Num(2.0),
                vec![],
            ))
            .expect("host dlarray is tape-tracked");
            assert!(matches!(host, Value::Object(_)));
            assert_eq!(tape.borrow().nodes.len(), 1);

            provider.reset_telemetry();
            let error = futures::executor::block_on(super::super::training::dlarray_builtin(
                Value::GpuTensor(resident),
                vec![],
            ))
            .expect_err("resident autodiff needs a tape kernel");
            assert_eq!(error.identifier(), Some("RunMat:deepLearning:Unsupported"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn single_matrix_product_retains_owning_input_precision() {
        let left = wrapped(Value::Tensor(
            Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
        ));
        let right = wrapped(Value::Tensor(
            Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap(),
        ));
        let result = dlarray_mtimes_builtin(left, right).expect("single matrix product");
        let Value::Object(result) = result else {
            panic!("dlarray result")
        };
        let Value::Tensor(data) = result.properties.get("Data").unwrap() else {
            panic!("tensor result")
        };
        assert_eq!(data.shape, vec![2, 2]);
        assert_eq!(
            data.clone().into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![23.0, 34.0, 31.0, 46.0])
        );
    }

    #[test]
    fn arithmetic_rejects_mutated_native_integer_data_without_rounding() {
        let large = u64::MAX;
        let integer_data = wrapped(Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).unwrap(),
        ));
        for result in [
            dlarray_plus_builtin(integer_data.clone(), dlarray(1.0)),
            dlarray_mtimes_builtin(integer_data.clone(), dlarray(1.0)),
            dlarray_sum_builtin(integer_data, vec![Value::String("all".into())]),
            dlarray_times_builtin(wrapped(Value::Int(IntValue::U64(large))), dlarray(1.0)),
        ] {
            let error = result.expect_err("integer payload must not widen to double");
            assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        }
    }

    #[test]
    fn gradient_output_arity_rejects_before_nested_provider_access() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .expect("resident target");
            let nested =
                Value::Cell(CellArray::new(vec![Value::GpuTensor(resident)], 1, 1).unwrap());
            provider.reset_telemetry();
            let _outputs = crate::output_count::push_output_count(Some(2));
            let error = futures::executor::block_on(dlgradient_builtin(dlarray(1.0), vec![nested]))
                .expect_err("one target cannot provide two outputs");
            assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn gradient_nested_resident_target_rejects_without_gather() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .expect("resident target");
            let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
            let loss = annotate_dlarray_value(dlarray(1.0)).expect("traced loss");
            let nested =
                Value::Cell(CellArray::new(vec![Value::GpuTensor(resident)], 1, 1).unwrap());
            provider.reset_telemetry();
            let error = futures::executor::block_on(dlgradient_builtin(loss, vec![nested]))
                .expect_err("untraced resident target");
            assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }
}
