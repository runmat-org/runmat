use super::*;
mod input;
mod provenance;
pub(in crate::builtins::deep_learning) use input::annotate_dlarray_value;
#[cfg(test)]
pub(super) use input::annotate_tree;
use input::prepare_tape_inputs;
use provenance::DataProvenance;

pub(super) const AD_NODE_PROPERTY: &str = "__runmat_ad_node";
pub(super) const AD_TAPE_PROPERTY: &str = "__runmat_ad_tape";

pub(super) fn attach_node_to_object(object: &mut ObjectInstance, node: usize) {
    let tape_id = active_tape()
        .expect("recorded node requires active tape")
        .borrow()
        .id;
    object.properties.insert(
        AD_NODE_PROPERTY.into(),
        Value::Int(IntValue::U64(
            u64::try_from(node).expect("node index fits u64"),
        )),
    );
    object
        .properties
        .insert(AD_TAPE_PROPERTY.into(), Value::Int(IntValue::U64(tape_id)));
}
static NEXT_TAPE_ID: AtomicU64 = AtomicU64::new(1);

runmat_thread_local::runmat_thread_local! {
    // Only installed for a callback future's poll or destruction. A suspended
    // callback never leaves its tape active for another execution on the thread.
    static ACTIVE_TAPES: RefCell<Vec<Rc<RefCell<Tape>>>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn active_tape() -> Option<Rc<RefCell<Tape>>> {
    ACTIVE_TAPES.with(|slot| slot.borrow().last().cloned())
}

pub(super) struct TapeActivation(Rc<RefCell<Tape>>);

impl TapeActivation {
    pub(super) fn enter(tape: Rc<RefCell<Tape>>) -> Self {
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

pub(super) struct TapeFuture<F> {
    pub(super) tape: Rc<RefCell<Tape>>,
    pub(super) future: Option<Pin<Box<F>>>,
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

pub(super) struct Tape {
    pub(super) id: u64,
    pub(super) nodes: Vec<Node>,
}

impl Tape {
    pub(super) fn new() -> Self {
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

pub(super) struct Node {
    pub(super) value: NodeValue,
    pub(super) kind: NodeKind,
    provenance: DataProvenance,
}

/// One physical numerical payload. A resident node never owns a fictitious
/// host tensor or a mirrored copy of its device contents.
#[derive(Clone, Debug)]
pub(super) enum NodeValue {
    Host(Tensor),
    Resident(runmat_accelerate_api::GpuTensorHandle),
}

impl NodeValue {
    pub(super) fn shape(&self) -> &[usize] {
        match self {
            Self::Host(value) => &value.shape,
            Self::Resident(value) => value.shape(),
        }
    }

    pub(super) fn numeric_dtype(&self) -> Option<NumericDType> {
        match self {
            Self::Host(value) => Some(value.numeric_dtype()),
            Self::Resident(value) => match runmat_accelerate_api::handle_precision(value) {
                Some(runmat_accelerate_api::ProviderPrecision::F32) => Some(NumericDType::F32),
                Some(runmat_accelerate_api::ProviderPrecision::F64) => Some(NumericDType::F64),
                None => None,
            },
        }
    }

    pub(super) fn host(&self) -> Option<&Tensor> {
        match self {
            Self::Host(value) => Some(value),
            Self::Resident(_) => None,
        }
    }
}

pub(super) enum NodeKind {
    Leaf,
    ResidentNetwork {
        execution: Rc<super::network::ResidentTapeRecord>,
    },
    Network {
        execution: Rc<super::network::NetworkTapeRecord>,
        output: super::network::NetworkOutput,
    },
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
    SumAll {
        input: usize,
        input_shape: Vec<usize>,
    },
}

pub(in crate::builtins::deep_learning) fn tape_is_active() -> bool {
    active_tape().is_some()
}

pub(in crate::builtins::deep_learning) async fn with_tape_context(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    args: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<Value> {
    let tape = Rc::new(RefCell::new(Tape::new()));
    TapeFuture {
        tape,
        future: Some(Box::pin(async move {
            let prepared = prepare_tape_inputs(args)?;
            match runtime {
                Some(runtime) => {
                    crate::call_feval_async_with_outputs_in_context(
                        runtime,
                        function,
                        &prepared,
                        requested_outputs,
                    )
                    .await
                }
                None => {
                    crate::call_feval_async_with_outputs(function, &prepared, requested_outputs)
                        .await
                }
            }
        })),
    }
    .await
}

pub(in crate::builtins::deep_learning) fn dlarray_node_id(value: &Value) -> Option<usize> {
    let Value::Object(object) = value else {
        return None;
    };
    dlarray_object_node_id(object)
}

pub(super) fn dlarray_object_node_id(object: &ObjectInstance) -> Option<usize> {
    let tape = active_tape()?;
    let tape = tape.borrow();
    let tape_id = tape.id;
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
    let node = tape.nodes.get(id)?;
    let data = object.properties.get("Data")?;
    node.provenance.matches(data, &node.value).then_some(id)
}

pub(super) fn push_node(value: Tensor, kind: NodeKind) -> BuiltinResult<usize> {
    push_node_with_provenance(NodeValue::Host(value), kind, DataProvenance::Dense)
}

pub(super) fn push_resident_node(
    value: runmat_accelerate_api::GpuTensorHandle,
    kind: NodeKind,
) -> BuiltinResult<usize> {
    let provenance = DataProvenance::Resident(value.view_identity());
    push_node_with_provenance(NodeValue::Resident(value), kind, provenance)
}

pub(super) fn push_source_node(value: Tensor, data: &Value) -> BuiltinResult<usize> {
    let provenance = DataProvenance::from_data(data).ok_or_else(|| {
        deep_learning_error(
            "dlarray",
            "dlarray: tape source has unsupported Data storage",
        )
    })?;
    push_node_with_provenance(NodeValue::Host(value), NodeKind::Leaf, provenance)
}

pub(super) fn push_resident_source_node(
    value: runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<usize> {
    push_resident_node(value, NodeKind::Leaf)
}

fn push_node_with_provenance(
    value: NodeValue,
    kind: NodeKind,
    provenance: DataProvenance,
) -> BuiltinResult<usize> {
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape; call it inside dlfeval",
        )
    })?;
    let mut tape = tape.borrow_mut();
    let id = tape.nodes.len();
    tape.nodes.push(Node {
        value,
        kind,
        provenance,
    });
    Ok(id)
}

pub(super) fn value_for_node(id: usize) -> BuiltinResult<Tensor> {
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape",
        )
    })?;
    let value = tape.borrow().nodes.get(id).map(|node| node.value.clone());
    match value {
        Some(NodeValue::Host(value)) => Ok(value),
        Some(NodeValue::Resident(_)) => Err(deep_learning_error(
            "dlgradient",
            "dlgradient: expected host tape node storage",
        )),
        None => Err(deep_learning_error(
            "dlgradient",
            "dlgradient: invalid tape node",
        )),
    }
}

pub(super) fn node_value(id: usize) -> BuiltinResult<NodeValue> {
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape",
        )
    })?;
    let value = tape
        .borrow()
        .nodes
        .get(id)
        .map(|node| node.value.clone())
        .ok_or_else(|| deep_learning_error("dlgradient", "dlgradient: invalid tape node"));
    value
}

pub(super) fn push_binary_node(
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
