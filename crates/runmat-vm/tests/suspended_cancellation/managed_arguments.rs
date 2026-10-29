//! Real VM dispatch, not a helper-only root test. This module has no accel gate.
use super::support;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    GpuTensorStorage, NumericElementType,
};
use runmat_runtime::builtin::{
    RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation,
};
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_runtime::execution::{RootedValue, RootedValueSequence};
use runmat_value::{HandleRef, Value, ValueSequence};
use std::{
    cell::RefCell,
    future::Future,
    rc::Rc,
    task::{Context, Poll},
};

struct State {
    target: Option<runmat_gc::GcHandle>,
    probe: Option<RetirementProbe>,
    receiver: Option<futures::channel::oneshot::Receiver<bool>>,
    cell_output: bool,
}
thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}
struct InstalledState;
impl Drop for InstalledState {
    fn drop(&mut self) {
        STATE.with(|state| state.borrow_mut().take());
    }
}
fn state() -> Rc<RefCell<State>> {
    STATE.with(|state| state.borrow().as_ref().unwrap().clone())
}

fn produce(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let mut object = runmat_value::ObjectInstance::new("ManagedInput");
        object.replace_property_unbounded("data".into(), Value::GpuTensor(allocation.handle));
        let root = runmat_gc::gc_allocate_rooted(Value::Object(object)).unwrap();
        let target = root.handle();
        let value = Value::HandleObject(HandleRef {
            class_name: "ManagedInput".into(),
            target,
            valid: true,
        });
        let state = state();
        let mut state = state.borrow_mut();
        state.target = Some(target);
        state.probe = Some(allocation.probe);
        let value = if state.cell_output {
            Value::Cell(runmat_value::CellArray::new(vec![value], 1, 1).unwrap())
        } else {
            value
        };
        let output = RootedValueSequence::new(
            ValueSequence::single(value)
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
        )
        .map_err(|error| error.into_runtime_error())?;
        drop(state);
        root.unroot().unwrap();
        Ok(output)
    })
}

fn consume(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let argument = RootedValue::new(invocation.args.get(0).expect("managed argument").clone())
        .map_err(|error| error.into_runtime_error());
    let receiver = state()
        .borrow_mut()
        .receiver
        .take()
        .expect("single invocation");
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
        let argument = argument?;
        if receiver.await.expect("test sender") {
            argument.try_map_owned(
                |argument| {
                    ValueSequence::single(argument)
                        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
                },
                |error| error.into_runtime_error(),
            )
        } else {
            Err(runmat_runtime::RuntimeError::new(
                "expected managed argument failure",
            ))
        }
    })
}
struct Builtins;
impl RuntimeBuiltinService for Builtins {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        let implementation: runmat_runtime::builtin::RuntimeBuiltinImplementation = match name {
            "sin" => produce,
            "cos" => consume,
            _ => return vec![],
        };
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
        vec![RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            implementation,
        )]
    }
}

#[derive(Clone, Copy, Debug)]
enum Terminal {
    Drop,
    Cancel,
    Error,
    Success,
}

#[test]
fn actual_vm_managed_arguments_survive_suspension_and_release_at_terminal_boundary() {
    for (source, cell_output) in [
        ("x = cos(sin(0));", false),
        ("x = feval(@cos, sin(0));", false),
        ("x = cos(sin(0){:});", true),
        (
            "function y = passthrough(v); y = cos(v); end; x = passthrough(sin(0));",
            false,
        ),
        (
            "function y = checked(v); arguments; v {mustBeA('ManagedInput')}; end; y = cos(v); end; x = checked(sin(0));",
            false,
        ),
    ] {
        for terminal in [
            Terminal::Drop,
            Terminal::Cancel,
            Terminal::Error,
            Terminal::Success,
        ] {
            runmat_gc::gc_test_context(|| run_case(source, cell_output, terminal));
        }
    }
}

fn run_case(source_text: &str, cell_output: bool, terminal: Terminal) {
    runmat_vm::reset_thread_state_for_tests();
    let (sender, receiver) = futures::channel::oneshot::channel();
    let state = Rc::new(RefCell::new(State {
        target: None,
        probe: None,
        receiver: Some(receiver),
        cell_output,
    }));
    STATE.with(|slot| *slot.borrow_mut() = Some(state.clone()));
    let _installed = InstalledState;
    let bytecode = support::compile_source(source_text).expect(source_text);
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Builtins)));
    let vars = vec![];
    let mut execution = Box::pin(runmat_vm::interpret_workspace(
        &bytecode,
        runmat_vm::InterpreterVars::from_owned(vars),
        None,
        runtime,
        None,
        None,
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(
        execution
            .as_mut()
            .poll(&mut context)
            .map(|execution| execution.outcome)
            .is_pending(),
        "{source_text}"
    );
    let target = state.borrow().target.expect("producer ran");
    let probe = state.borrow().probe.as_ref().unwrap().clone();
    runmat_gc::gc_collect_major().unwrap();
    assert!(
        runmat_gc::gc_with_value(&target, |_| ()).is_ok(),
        "{source_text}"
    );
    assert_eq!(probe.retirements(), 0);
    let mut output_root = None;
    match terminal {
        Terminal::Drop => {}
        Terminal::Cancel => {
            cancellation.cancel(runmat_execution::CancellationReason::User);
            assert!(matches!(
                execution
                    .as_mut()
                    .poll(&mut context)
                    .map(|execution| execution.outcome),
                Poll::Ready(Err(_))
            ));
        }
        Terminal::Error | Terminal::Success => {
            sender.send(matches!(terminal, Terminal::Success)).unwrap();
            let Poll::Ready(result) = execution
                .as_mut()
                .poll(&mut context)
                .map(|execution| execution.outcome)
            else {
                panic!("consumer must finish")
            };
            if matches!(terminal, Terminal::Success) {
                let runmat_vm::InterpreterOutcome::Completed(completion) = result.unwrap();
                let slot = bytecode
                    .var_names
                    .iter()
                    .find_map(|(slot, name)| (name == "x").then_some(*slot))
                    .unwrap();
                let published_target_matches = {
                    let value = completion
                        .values
                        .get(slot)
                        .expect("completed handle remains in its compiled slot");
                    matches!(&*value, Value::HandleObject(handle) if handle.target == target)
                };
                assert!(published_target_matches);
                output_root = Some(runmat_gc::ExplicitRoot::new(target).unwrap());
            } else {
                assert!(result.is_err());
            }
        }
    }
    drop(execution);

    runmat_vm::reset_thread_state_for_tests();
    runmat_gc::gc_collect_major().unwrap();
    assert_eq!(
        probe.retirements(),
        usize::from(output_root.is_none()),
        "{source_text}: {terminal:?} must release unpublished storage"
    );
    drop(output_root);
    runmat_gc::gc_collect_major().unwrap();
    assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
    assert_eq!(probe.retirements(), 1);
}
