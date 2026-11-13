use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use std::rc::Weak;
mod input;
mod provenance;
pub(in crate::builtins::deep_learning) use input::annotate_dlarray_value;
#[cfg(test)]
pub(super) use input::annotate_tree;
use input::prepare_tape_inputs;
use provenance::DataProvenance;

pub(super) const AD_NODE_PROPERTY: &str = "__runmat_ad_node";
pub(super) const AD_TAPE_PROPERTY: &str = "__runmat_ad_tape";

pub(super) fn attach_node_to_object(object: &mut ObjectInstance, node: usize) -> BuiltinResult<()> {
    let tape_id = active_tape()
        .expect("recorded node requires active tape")
        .borrow()
        .id;
    object.replace_dl_property(
        AD_NODE_PROPERTY,
        Value::Int(IntValue::U64(
            u64::try_from(node).expect("node index fits u64"),
        )),
    )?;
    object.replace_dl_property(AD_TAPE_PROPERTY, Value::Int(IntValue::U64(tape_id)))?;
    Ok(())
}
static NEXT_TAPE_ID: AtomicU64 = AtomicU64::new(1);

runmat_thread_local::runmat_thread_local! {
    // Only installed for a callback future's poll or destruction. A suspended
    // callback never leaves its tape active for another execution on the thread.
    // A context without a tape masks an enclosing callback while it is active.
    static ACTIVE_TAPES: RefCell<Vec<Option<Rc<RefCell<Tape>>>>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn active_tape() -> Option<Rc<RefCell<Tape>>> {
    ACTIVE_TAPES.with(|slot| slot.borrow().last().cloned().flatten())
}

/// Opaque, invocation-local authority to poll one callback against its tape.
/// Retained callback contexts cannot keep a completed graph alive.
#[derive(Clone)]
pub(crate) struct TapePollAuthority(Weak<RefCell<Tape>>);

impl TapePollAuthority {
    pub(super) fn new(tape: &Rc<RefCell<Tape>>) -> Self {
        Self(Rc::downgrade(tape))
    }
}

pub(super) struct TapeActivation(Option<Rc<RefCell<Tape>>>);

/// Selecting no tape is a no-op in the ordinary case. It masks a live outer
/// tape only when another invocation is actually nested beneath that poll.
pub(crate) struct TapePollGuard {
    _activation: Option<TapeActivation>,
}

impl TapePollGuard {
    pub(crate) fn select(authority: Option<&TapePollAuthority>) -> Self {
        let selected = authority.and_then(|authority| authority.0.upgrade());
        if selected.is_none() && active_tape().is_none() {
            Self { _activation: None }
        } else {
            Self {
                _activation: Some(TapeActivation::select_tape(selected)),
            }
        }
    }
}

impl TapeActivation {
    pub(super) fn enter(tape: Rc<RefCell<Tape>>) -> Self {
        Self::select_tape(Some(tape))
    }

    fn select_tape(tape: Option<Rc<RefCell<Tape>>>) -> Self {
        ACTIVE_TAPES.with(|slot| slot.borrow_mut().push(tape.clone()));
        Self(tape)
    }
}

impl Drop for TapeActivation {
    fn drop(&mut self) {
        ACTIVE_TAPES.with(|slot| {
            let popped = slot.borrow_mut().pop();
            debug_assert!(popped.is_some_and(|tape| match (tape, &self.0) {
                (Some(actual), Some(expected)) => Rc::ptr_eq(&actual, expected),
                (None, None) => true,
                _ => false,
            }));
        });
    }
}

#[cfg(test)]
pub(in crate::builtins::deep_learning) fn with_test_tape<R>(run: impl FnOnce() -> R) -> (R, usize) {
    let tape = Rc::new(RefCell::new(Tape::new()));
    let activation = TapeActivation::enter(tape.clone());
    let result = run();
    drop(activation);
    let recorded = tape.borrow().nodes.len();
    (result, recorded)
}

#[cfg(test)]
pub(in crate::builtins::deep_learning) fn test_tape_node_count() -> usize {
    active_tape()
        .expect("test tape must be active")
        .borrow()
        .nodes
        .len()
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
            Self::Host(value) => value.shape(),
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
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
) -> BuiltinResult<crate::execution::RootedValue> {
    let _input_roots = crate::execution::RootedValueLease::for_values(
        std::iter::once(&function).chain(args.iter()),
    )
    .map_err(|error| error.into_runtime_error())?;
    let tape = Rc::new(RefCell::new(Tape::new()));
    let selected_runtime = runtime
        .cloned()
        .map(|runtime| runtime.with_tape_poll_authority(TapePollAuthority::new(&tape)));
    TapeFuture {
        tape,
        future: Some(Box::pin(async move {
            let prepared = prepare_tape_inputs(&args, selected_runtime.as_ref()).await?;
            crate::call::feval::invoke_retained(
                selected_runtime.as_ref(),
                function,
                prepared,
                requested_outputs,
                None,
            )
            .await
            .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
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
    if !object.class_name().is(DLARRAY_CLASS) {
        return None;
    }
    if object.properties().get(AD_TAPE_PROPERTY) != Some(&Value::Int(IntValue::U64(tape_id))) {
        return None;
    }
    let id = object
        .properties()
        .get(AD_NODE_PROPERTY)
        .and_then(|value| match value {
            Value::Int(IntValue::U64(id)) => usize::try_from(*id).ok(),
            _ => None,
        })?;
    let node = tape.nodes.get(id)?;
    let data = object.properties().get("Data")?;
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
            lhs.tensor.shape().to_vec(),
            rhs.tensor.shape().to_vec(),
        ),
    )?))
}

#[cfg(test)]
mod unselected_scope_tests {
    use super::*;
    use crate::context::{PreparedUnselectedScope, RuntimeContext};
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::profiling::{current_command_profiling, CommandProfilingRoot};
    use runmat_value::{get_display_format, set_display_format, FormatMode};
    use std::{
        cell::Cell,
        future::Future,
        pin::Pin,
        task::{Context, Poll},
    };

    struct Probe(Rc<Cell<bool>>);
    impl Future for Probe {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            assert!(crate::context::legacy::active().is_none());
            assert!(active_tape().is_none());
            assert!(current_command_profiling().is_none());
            set_display_format(FormatMode::ShortG);
            Poll::Pending
        }
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            assert!(crate::context::legacy::active().is_none());
            assert!(active_tape().is_none());
            assert!(current_command_profiling().is_none());
            assert_eq!(get_display_format(), FormatMode::ShortG);
            self.0.set(true);
        }
    }
    #[test]
    fn explicit_original_none_masks_and_restores_every_foreign_poll_projection() {
        let prior = get_display_format();
        let unselected = PreparedUnselectedScope::prepare(None).unwrap();
        let tape = Rc::new(RefCell::new(Tape::new()));
        let profiling = CommandProfilingRoot::new();
        let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_tape_poll_authority(TapePollAuthority::new(&tape))
            .with_command_profiling(profiling.receipt());
        foreign.display_format_owner().set(FormatMode::Long);
        let entered = foreign.enter();
        assert!(PreparedUnselectedScope::prepare(Some(&foreign)).is_err());
        assert!(Rc::ptr_eq(&active_tape().unwrap(), &tape));
        assert!(current_command_profiling()
            .unwrap()
            .same_command(&profiling.receipt()));
        let dropped = Rc::new(Cell::new(false));
        let mut work = Box::pin(unselected.scope(Probe(dropped.clone())));
        assert!(work
            .as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
            .is_pending());
        assert_eq!(
            crate::context::legacy::active().unwrap().local_identity(),
            foreign.local_identity()
        );
        assert_eq!(get_display_format(), FormatMode::Long);
        assert!(Rc::ptr_eq(&active_tape().unwrap(), &tape));
        assert!(current_command_profiling()
            .unwrap()
            .same_command(&profiling.receipt()));
        drop(work);
        assert!(dropped.get());
        assert_eq!(get_display_format(), FormatMode::Long);
        assert!(Rc::ptr_eq(&active_tape().unwrap(), &tape));
        assert!(current_command_profiling()
            .unwrap()
            .same_command(&profiling.receipt()));
        drop(entered);
        assert_eq!(get_display_format(), FormatMode::ShortG);
        set_display_format(prior);
    }
}
