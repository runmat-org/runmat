use super::{call_builtin_async_in_context, call_builtin_async_with_outputs_in_context};
use crate::builtin::{RuntimeBuiltinBinding, RuntimeBuiltinInvocation};
use crate::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use crate::execution::RuntimeExecutionService;
use futures::channel::oneshot;
use futures::task::noop_waker_ref;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};
use runmat_value::{Value, ValueSequence};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

const NAME: &str = "__output_projection_pending_fixture";

type OutputObservation = (
    Option<usize>,
    crate::output_count::InvocationOutputScope,
    Option<usize>,
);
type ControlledCall<F> = (
    Pin<Box<F>>,
    oneshot::Receiver<OutputObservation>,
    oneshot::Sender<()>,
);

struct CallControl {
    reached: oneshot::Sender<OutputObservation>,
    release: oneshot::Receiver<()>,
    completed: Rc<Cell<usize>>,
    presentation_outputs: usize,
}

thread_local! {
    static CONTROLS: RefCell<VecDeque<CallControl>> = const { RefCell::new(VecDeque::new()) };
}

fn pending_output_fixture(
    invocation: RuntimeBuiltinInvocation<'_>,
) -> crate::builtin::RuntimeBuiltinFuture {
    let requested_outputs = invocation.requested_outputs;
    let control = CONTROLS.with(|controls| {
        controls
            .borrow_mut()
            .pop_front()
            .expect("test call control")
    });
    crate::builtin::admitted_builtin_future(None, "fixture", async move {
        let entry = (
            requested_outputs,
            crate::output_count::current_output_scope(),
            crate::output_context::requested_output_count(),
        );
        let _ = control.reached.send(entry);
        control.release.await.expect("test release");
        let expected_scope = requested_outputs.map_or(
            crate::output_count::InvocationOutputScope::Implicit,
            crate::output_count::InvocationOutputScope::Requested,
        );
        assert_eq!(crate::output_count::current_output_scope(), expected_scope);
        assert_eq!(
            crate::output_context::requested_output_count(),
            Some(control.presentation_outputs)
        );
        control.completed.set(control.completed.get() + 1);
        let sequence = ValueSequence::single(Value::Num(requested_outputs.unwrap_or(0) as f64))
            .map_err(crate::sequence::sequence_error_to_runtime)?;
        crate::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

struct PendingOutputService;

impl RuntimeBuiltinService for PendingOutputService {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        if name != NAME {
            return Vec::new();
        }
        vec![RuntimeBuiltinBinding::new(
            BuiltinBindingIdentity {
                builtin: BuiltinCatalogIdentity { name: NAME },
                variant: "default",
            },
            pending_output_fixture,
        )]
    }
}

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_builtin(Rc::new(PendingOutputService)),
    )
}

fn controlled_call(
    runtime: &RuntimeContext,
    requested_outputs: Option<usize>,
    presentation_outputs: usize,
    completed: Rc<Cell<usize>>,
) -> ControlledCall<impl Future<Output = Result<crate::execution::RootedValue, crate::RuntimeError>>>
{
    let (reached_tx, reached_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    CONTROLS.with(|controls| {
        controls.borrow_mut().push_back(CallControl {
            reached: reached_tx,
            release: release_rx,
            completed,
            presentation_outputs,
        });
    });
    let selected = runtime.clone();
    let call_context = runtime.clone();
    let future = selected.scope(async move {
        let _presentation = crate::output_context::push_output_count(presentation_outputs)
            .expect("output-count scope");
        match requested_outputs {
            Some(count) => {
                call_builtin_async_with_outputs_in_context(&call_context, NAME, &[], count).await
            }
            None => call_builtin_async_in_context(&call_context, NAME, &[]).await,
        }
    });
    (Box::pin(future), reached_rx, release_tx)
}

fn poll_once<T>(future: Pin<&mut impl Future<Output = T>>) -> Poll<T> {
    let mut context = Context::from_waker(noop_waker_ref());
    future.poll(&mut context)
}

#[test]
fn pending_sibling_builtin_calls_keep_exact_output_projections() {
    let runtime = runtime();
    let completed = Rc::new(Cell::new(0));
    let (mut first, mut first_reached, first_release) =
        controlled_call(&runtime, Some(1), 1, Rc::clone(&completed));
    let (mut second, mut second_reached, second_release) =
        controlled_call(&runtime, Some(2), 2, Rc::clone(&completed));

    assert!(poll_once(first.as_mut()).is_pending());
    assert_eq!(
        first_reached.try_recv().unwrap(),
        Some((
            Some(1),
            crate::output_count::InvocationOutputScope::Requested(1),
            Some(1)
        ))
    );
    assert!(poll_once(second.as_mut()).is_pending());
    assert_eq!(
        second_reached.try_recv().unwrap(),
        Some((
            Some(2),
            crate::output_count::InvocationOutputScope::Requested(2),
            Some(2)
        ))
    );

    drop(first);
    drop(first_release);
    assert_eq!(
        completed.get(),
        0,
        "dropping pending work must not replay it"
    );
    second_release.send(()).unwrap();
    assert!(matches!(
        poll_once(second.as_mut()),
        Poll::Ready(Ok(value)) if matches!(&*value, Value::Num(2.0))
    ));
    assert_eq!(completed.get(), 1);
    assert_eq!(crate::output_count::current_output_count(), None);
    assert_eq!(crate::output_context::requested_output_count(), None);

    let (mut next, mut next_reached, next_release) =
        controlled_call(&runtime, Some(1), 3, Rc::clone(&completed));
    assert!(poll_once(next.as_mut()).is_pending());
    assert_eq!(
        next_reached.try_recv().unwrap(),
        Some((
            Some(1),
            crate::output_count::InvocationOutputScope::Requested(1),
            Some(3)
        ))
    );
    next_release.send(()).unwrap();
    assert!(matches!(
        poll_once(next.as_mut()),
        Poll::Ready(Ok(value)) if matches!(&*value, Value::Num(1.0))
    ));
    assert_eq!(completed.get(), 2);
    assert_eq!(crate::output_count::current_output_count(), None);
    assert_eq!(crate::output_context::requested_output_count(), None);

    CONTROLS.with(|controls| assert!(controls.borrow().is_empty()));
}

#[test]
fn pending_implicit_output_scope_stays_distinct_from_sibling_and_unscoped_calls() {
    let runtime = runtime();
    let completed = Rc::new(Cell::new(0));
    let (mut implicit, mut implicit_reached, implicit_release) =
        controlled_call(&runtime, None, 4, Rc::clone(&completed));
    let (mut requested, mut requested_reached, requested_release) =
        controlled_call(&runtime, Some(2), 2, Rc::clone(&completed));

    assert!(poll_once(implicit.as_mut()).is_pending());
    assert_eq!(
        implicit_reached.try_recv().unwrap(),
        Some((
            None,
            crate::output_count::InvocationOutputScope::Implicit,
            Some(4)
        ))
    );
    assert!(poll_once(requested.as_mut()).is_pending());
    assert_eq!(
        requested_reached.try_recv().unwrap(),
        Some((
            Some(2),
            crate::output_count::InvocationOutputScope::Requested(2),
            Some(2)
        ))
    );

    drop(requested);
    drop(requested_release);
    implicit_release.send(()).unwrap();
    assert!(matches!(
        poll_once(implicit.as_mut()),
        Poll::Ready(Ok(value)) if matches!(&*value, Value::Num(0.0))
    ));
    assert_eq!(completed.get(), 1);
    assert_eq!(
        crate::output_count::current_output_scope(),
        crate::output_count::InvocationOutputScope::Unscoped
    );
    assert_eq!(crate::output_context::requested_output_count(), None);
    CONTROLS.with(|controls| assert!(controls.borrow().is_empty()));
}
