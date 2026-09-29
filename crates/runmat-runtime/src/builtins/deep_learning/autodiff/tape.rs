use super::*;
mod input;
pub(super) use input::{annotate_dlarray_value, annotate_tree, prepare_tape_inputs};

pub(super) const AD_NODE_PROPERTY: &str = "__runmat_ad_node";
pub(super) const AD_TAPE_PROPERTY: &str = "__runmat_ad_tape";
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

#[derive(Clone)]
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

#[derive(Clone)]
pub(super) struct Node {
    pub(super) value: Tensor,
    pub(super) kind: NodeKind,
}

#[derive(Clone)]
pub(super) enum NodeKind {
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

pub(super) fn push_node(value: Tensor, kind: NodeKind) -> BuiltinResult<usize> {
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

pub(super) fn value_for_node(id: usize) -> BuiltinResult<Tensor> {
    let tape = active_tape().ok_or_else(|| {
        deep_learning_error(
            "dlgradient",
            "dlgradient: no active automatic differentiation tape",
        )
    })?;
    let value = tape.borrow().nodes.get(id).map(|node| node.value.clone());
    value.ok_or_else(|| deep_learning_error("dlgradient", "dlgradient: invalid tape node"))
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
