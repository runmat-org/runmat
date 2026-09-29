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

mod gradient;
pub(super) use gradient::dlgradient_builtin;

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

mod array_methods;
pub(super) use array_methods::{
    dlarray_minus_builtin, dlarray_mtimes_builtin, dlarray_plus_builtin, dlarray_rdivide_builtin,
    dlarray_sum_builtin, dlarray_times_builtin,
};

mod network;
pub(super) use network::record_network_forward;

#[cfg(test)]
mod tests;
