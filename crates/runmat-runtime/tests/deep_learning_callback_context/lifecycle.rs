//! Scoped callback futures must own their session while suspended, without
//! leaving an active callback scope or retaining the session after disposal.
use super::*;
use futures::task::noop_waker;
use runmat_runtime::RuntimeError;
use std::{
    cell::Cell,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

#[path = "lifecycle/training.rs"]
mod training;

#[derive(Clone, Copy)]
enum Outcome {
    Identity,
    Initializer,
    Error,
    Hold,
}

thread_local! {
    static OUTCOME: Cell<Outcome> = const { Cell::new(Outcome::Identity) };
    static STARTED: RefCell<Vec<(runmat_runtime::context::RuntimeContextLocalId, Option<usize>)>> =
        const { RefCell::new(Vec::new()) };
    static POLLS: Cell<usize> = const { Cell::new(0) };
}

struct SuspendedCallback {
    runtime: RuntimeContext,
    origin: runmat_runtime::context::RuntimeContextLocalId,
    requested_outputs: Option<usize>,
    input: Option<runmat_runtime::execution::RootedValue>,
    outcome: Outcome,
    first_poll: bool,
}

impl Future for SuspendedCallback {
    type Output = Result<runmat_runtime::execution::RootedValueSequence, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        assert_eq!(self.runtime.local_identity(), self.origin);
        assert_eq!(self.requested_outputs, Some(1));
        POLLS.with(|polls| polls.set(polls.get() + 1));
        if self.first_poll {
            self.first_poll = false;
            return Poll::Pending;
        }
        match self.outcome {
            Outcome::Hold => Poll::Pending,
            Outcome::Error => Poll::Ready(Err(RuntimeError::new("selected callback error"))),
            Outcome::Identity => {
                let input = self.input.take().expect("callback input");
                Poll::Ready(input.with_owned(|input| {
                    let sequence = ValueSequence::single(input)
                        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
                    runmat_runtime::execution::RootedValueSequence::new(sequence)
                        .map_err(|error| error.into_runtime_error())
                }))
            }
            Outcome::Initializer => {
                let input = self.input.take().expect("shape argument");
                let Value::Tensor(shape) = input.value() else {
                    panic!("initializer receives a shape tensor");
                };
                let dimensions: Vec<usize> = shape
                    .host_buffer()
                    .materialize_f64()
                    .into_iter()
                    .map(|size| size as usize)
                    .collect();
                let count = dimensions.iter().product();
                let sequence = ValueSequence::single(Value::Tensor(
                    Tensor::new(vec![0.25; count], dimensions).expect("fixture shape"),
                ))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
                .and_then(|sequence| {
                    runmat_runtime::execution::RootedValueSequence::new(sequence)
                        .map_err(|error| error.into_runtime_error())
                });
                Poll::Ready(sequence)
            }
        }
    }
}

fn suspended_callback(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let runtime = invocation
        .context
        .expect("selected callback has its runtime");
    let requested_outputs = invocation.requested_outputs;
    STARTED.with(|calls| {
        calls
            .borrow_mut()
            .push((runtime.local_identity(), requested_outputs))
    });
    let input = invocation
        .args
        .get(0)
        .cloned()
        .map(runmat_runtime::execution::RootedValue::new)
        .transpose();
    let input = match input {
        Ok(input) => input,
        Err(error) => {
            return runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
                Err(error.into_runtime_error())
            })
        }
    };
    runmat_runtime::builtin::admitted_builtin_future(
        None,
        "fixture",
        SuspendedCallback {
            origin: runtime.local_identity(),
            runtime,
            requested_outputs,
            input,
            outcome: OUTCOME.with(Cell::get),
            first_poll: true,
        },
    )
}

struct PendingService;

impl RuntimeBuiltinService for PendingService {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        let mut bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        if name.eq_ignore_ascii_case("deal") || name.eq_ignore_ascii_case("plus") {
            for binding in &mut bindings {
                binding.implementation = suspended_callback;
            }
        }
        bindings
    }
}

fn scoped_runtime() -> (RuntimeContext, Rc<PendingService>) {
    let service = Rc::new(PendingService);
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(service.clone()));
    (runtime, service)
}

fn reset(outcome: Outcome) {
    OUTCOME.with(|mode| mode.set(outcome));
    STARTED.with(|calls| calls.borrow_mut().clear());
    POLLS.with(|polls| polls.set(0));
}

fn invoke(name: &str, runtime: RuntimeContext, args: &[Value]) -> RuntimeBuiltinFuture {
    let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
    let [binding] = bindings.as_slice() else {
        panic!("{name} must have one registered binding");
    };
    (binding.implementation)(RuntimeBuiltinInvocation::new(args, Some(runtime), Some(1)))
}

fn poll_once(
    future: &mut RuntimeBuiltinFuture,
) -> Poll<Result<runmat_runtime::execution::RootedValueSequence, RuntimeError>> {
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    future.as_mut().poll(&mut context)
}

fn calls() -> Vec<(
    runmat_runtime::context::RuntimeContextLocalId,
    Option<usize>,
)> {
    STARTED.with(|calls| calls.borrow().clone())
}

#[test]
fn pending_dlfeval_drop_releases_runtime_without_replaying_callback() {
    reset(Outcome::Hold);
    let (runtime, service) = scoped_runtime();
    let weak = Rc::downgrade(&service);
    let id = runtime.local_identity();
    let args = [Value::FunctionHandle("deal".into()), Value::Num(7.0)];
    let mut future = invoke("dlfeval", runtime.clone(), &args);
    assert!(poll_once(&mut future).is_pending());
    assert_eq!(calls(), vec![(id, Some(1))]);
    assert_eq!(POLLS.with(Cell::get), 1);
    drop(runtime);
    drop(service);
    assert!(weak.upgrade().is_some(), "suspended callback owns session");
    drop(future);
    assert!(
        weak.upgrade().is_none(),
        "drop releases session and service"
    );
    assert_eq!(calls(), vec![(id, Some(1))]);
}

#[test]
fn cancelling_pending_dlfeval_then_dropping_releases_runtime() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};

    reset(Outcome::Hold);
    let source = CancellationSource::new();
    let (runtime, service) = scoped_runtime();
    let runtime = runtime.with_cancellation_token(source.token());
    let weak = Rc::downgrade(&service);
    let id = runtime.local_identity();
    let args = [Value::FunctionHandle("deal".into()), Value::Num(7.0)];
    let mut future = invoke("dlfeval", runtime.clone(), &args);
    assert!(poll_once(&mut future).is_pending());
    source.cancel(CancellationReason::User);
    assert!(runtime.is_cancelled());
    drop(runtime);
    drop(service);
    assert!(weak.upgrade().is_some());
    drop(future);
    assert!(weak.upgrade().is_none());
    assert_eq!(calls(), vec![(id, Some(1))]);
    assert_eq!(POLLS.with(Cell::get), 1);
}

#[test]
fn pending_dlfeval_resume_publishes_once_and_error_retains_cause() {
    for outcome in [Outcome::Identity, Outcome::Error] {
        reset(outcome);
        let (runtime, service) = scoped_runtime();
        let weak = Rc::downgrade(&service);
        let id = runtime.local_identity();
        let args = [Value::FunctionHandle("deal".into()), Value::Num(7.0)];
        let mut future = invoke("dlfeval", runtime.clone(), &args);
        assert!(poll_once(&mut future).is_pending());
        let finished = poll_once(&mut future);
        match outcome {
            Outcome::Identity => assert!(matches!(
                finished,
                Poll::Ready(Ok(sequence)) if sequence.as_slice() == [Value::Num(7.0)]
                    && sequence.kind() == runmat_value::ValueSequenceKind::Single
            )),
            Outcome::Error => assert!(
                matches!(finished, Poll::Ready(Err(error)) if error.to_string().contains("selected callback error"))
            ),
            _ => unreachable!(),
        }
        assert_eq!(calls(), vec![(id, Some(1))]);
        assert_eq!(POLLS.with(Cell::get), 2);
        drop(runtime);
        drop(service);
        drop(future);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn pending_dlfeval_retains_borrowed_resident_until_result_is_released() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };

    runmat_gc::gc_test_context(|| {
        reset(Outcome::Identity);
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .expect("owned resident input");
        let resident = allocation.handle.clone();
        let weak_resident = resident.downgrade();
        let identity = resident.allocation_identity();
        let weak_allocation = resident.allocation_lease().downgrade();
        let retirement = allocation.probe.clone();
        drop(allocation);
        let (runtime, service) = scoped_runtime();
        let weak_service = Rc::downgrade(&service);
        let id = runtime.local_identity();
        let args = [
            Value::FunctionHandle("deal".into()),
            Value::GpuTensor(resident.clone()),
        ];
        let mut future = invoke("dlfeval", runtime.clone(), &args);
        assert!(poll_once(&mut future).is_pending());
        drop(args);
        drop(resident);
        runmat_gc::gc_collect_major().expect("collect while callback is suspended");
        assert!(
            weak_resident.acquire().is_ok(),
            "suspended callback owns input"
        );
        weak_allocation
            .acquire()
            .expect("suspended callback retains its allocation lease")
            .validate_live()
            .unwrap();
        assert_eq!(retirement.retirements(), 0);
        let Poll::Ready(Ok(sequence)) = poll_once(&mut future) else {
            panic!("callback must publish the resident alias");
        };
        assert_eq!(sequence.kind(), runmat_value::ValueSequenceKind::Single);
        sequence.with_owned(|sequence| {
            let Value::GpuTensor(output) = sequence
                .into_output_storage()
                .try_into_first()
                .expect("unique callback output")
                .expect("one callback output")
            else {
                panic!("callback must publish one resident alias");
            };
            assert_eq!(calls(), vec![(id, Some(1))]);
            // Publication may create a provenance view. The physical lease, not
            // the original view's weak reference, owns the shared allocation.
            assert_eq!(output.allocation_identity(), identity);
            output.allocation_lease().validate_live().unwrap();
            assert!(output.allocation_lease().shares_allocation_with(
                &weak_allocation
                    .acquire()
                    .expect("published output retains the original allocation lease")
            ));
            assert_eq!(retirement.retirements(), 0);
            drop(output);
        });
        drop(runtime);
        drop(service);
        drop(future);
        runmat_gc::gc_collect_major().expect("collect after callback completion");
        assert!(weak_allocation.acquire().is_none(), "final lease retired");
        assert_eq!(retirement.retirements(), 1);
        assert!(weak_service.upgrade().is_none(), "session service retired");
    });
}

#[test]
fn pending_initializer_resumes_once_and_network_publishes_only_after_success() {
    reset(Outcome::Initializer);
    let (runtime, service) = scoped_runtime();
    let weak = Rc::downgrade(&service);
    let id = runtime.local_identity();
    let layers = training_layers(false);
    let args = [layers.value().clone()];
    let mut future = invoke("dlnetwork", runtime.clone(), &args);
    assert!(poll_once(&mut future).is_pending());
    assert_eq!(calls(), vec![(id, Some(1))]);
    let Poll::Ready(Ok(sequence)) = poll_once(&mut future) else {
        panic!("network should publish after one callback completion");
    };
    assert_eq!(sequence.kind(), runmat_value::ValueSequenceKind::Single);
    assert!(matches!(sequence.as_slice(), [Value::Object(_)]));
    assert_eq!(calls(), vec![(id, Some(1))]);
    drop(runtime);
    drop(service);
    drop(future);
    assert!(weak.upgrade().is_none());
}

#[test]
fn dropping_pending_initializer_does_not_publish_partial_network() {
    reset(Outcome::Hold);
    let (runtime, service) = scoped_runtime();
    let weak = Rc::downgrade(&service);
    let id = runtime.local_identity();
    let layers = training_layers(false);
    let args = [layers.value().clone()];
    let original = args[0].clone();
    let mut future = invoke("dlnetwork", runtime.clone(), &args);
    assert!(poll_once(&mut future).is_pending());
    assert_eq!(calls(), vec![(id, Some(1))]);
    drop(future);
    assert_eq!(args[0], original, "caller layers remain unchanged");
    drop(runtime);
    drop(service);
    assert!(weak.upgrade().is_none());
}

#[test]
fn failed_pending_initializer_preserves_cause_and_caller_layers() {
    use std::error::Error;

    reset(Outcome::Error);
    let (runtime, service) = scoped_runtime();
    let weak = Rc::downgrade(&service);
    let id = runtime.local_identity();
    let layers = training_layers(false);
    let args = [layers.value().clone()];
    let original = args[0].clone();
    let mut future = invoke("dlnetwork", runtime.clone(), &args);
    assert!(poll_once(&mut future).is_pending());
    let Poll::Ready(Err(error)) = poll_once(&mut future) else {
        panic!("initializer error must end this execution");
    };
    assert!(error.to_string().contains("initializer callback failed"));
    assert!(error
        .source()
        .is_some_and(|source| source.to_string().contains("selected callback error")));
    assert_eq!(args[0], original, "failed initialization is not published");
    assert_eq!(calls(), vec![(id, Some(1))]);
    drop(runtime);
    drop(service);
    drop(future);
    assert!(weak.upgrade().is_none());
}

#[test]
fn pending_dlupdate_network_traverses_canonical_learnables_without_replay() {
    reset(Outcome::Identity);
    let layers =
        block_on(call_builtin_async("featureInputLayer", &[Value::Num(1.0)])).expect("input layer");
    let dense = block_on(call_builtin_async(
        "fullyConnectedLayer",
        &[Value::Num(1.0)],
    ))
    .expect("dense layer");
    let mut layer_batch = runmat_runtime::execution::RootedValuesBuilder::with_capacity(2);
    layer_batch.push_owned(layers);
    layer_batch.push_owned(dense);
    let layer_batch = layer_batch.finish().expect("root layer batch");
    let source = layer_batch
        .with_unbounded_values(|layers| {
            runmat_runtime::execution::RootedValue::new(Value::Cell(
                CellArray::new(layers, 2, 1).unwrap(),
            ))
            .expect("root source layers")
        })
        .expect("fixture layer batch is unbounded");
    let network = block_on(
        source
            .with_owned_async(|source| async { call_builtin_async("dlnetwork", &[source]).await }),
    )
    .expect("network");
    let (runtime, service) = scoped_runtime();
    let weak = Rc::downgrade(&service);
    let id = runtime.local_identity();
    let args = [
        Value::FunctionHandle("plus".into()),
        network.value().clone(),
    ];
    let mut future = invoke("dlupdate", runtime.clone(), &args);
    assert!(poll_once(&mut future).is_pending());
    let result = loop {
        match poll_once(&mut future) {
            Poll::Pending => assert!(POLLS.with(Cell::get) <= 8, "bounded fixture"),
            Poll::Ready(result) => break result.expect("network update"),
        }
    };
    assert_eq!(result.kind(), runmat_value::ValueSequenceKind::Single);
    assert!(matches!(result.as_slice(), [Value::Object(_)]));
    assert_eq!(
        calls(),
        vec![(id, Some(1)); 2],
        "weights and bias once each"
    );
    assert_eq!(POLLS.with(Cell::get), 4);
    drop(runtime);
    drop(service);
    drop(future);
    assert!(weak.upgrade().is_none());
}
