//! Real VM dispatch, not a helper-only root test. This module has no accel gate.
use super::support;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    GpuTensorStorage, NumericElementType,
};
use runmat_runtime::builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture};
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_value::{HandleRef, Value};
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

fn produce(_: &[Value]) -> RuntimeBuiltinFuture {
    Box::pin(async {
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let root = runmat_gc::gc_allocate_rooted(Value::GpuTensor(allocation.handle)).unwrap();
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
        root.unroot().unwrap();
        Ok(if state.cell_output {
            Value::Cell(runmat_value::CellArray::new(vec![value], 1, 1).unwrap())
        } else {
            value
        })
    })
}

fn consume(args: &[Value]) -> RuntimeBuiltinFuture {
    let argument = args.first().expect("managed argument").clone();
    let receiver = state()
        .borrow_mut()
        .receiver
        .take()
        .expect("single invocation");
    Box::pin(async move {
        if receiver.await.expect("test sender") {
            Ok(argument)
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

#[derive(Clone, Copy)]
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
            "function y = checked(v); arguments; v {cos}; end; y = v; end; x = checked(sin(0));",
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
    let mut vars = vec![];
    let mut execution = Box::pin(runmat_vm::interpret_with_vars_in_context(
        &bytecode, &mut vars, None, runtime,
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(
        execution.as_mut().poll(&mut context).is_pending(),
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
                execution.as_mut().poll(&mut context),
                Poll::Ready(Err(_))
            ));
        }
        Terminal::Error | Terminal::Success => {
            sender.send(matches!(terminal, Terminal::Success)).unwrap();
            let Poll::Ready(result) = execution.as_mut().poll(&mut context) else {
                panic!("consumer must finish")
            };
            if matches!(terminal, Terminal::Success) {
                let runmat_vm::InterpreterOutcome::Completed(completion) = result.unwrap();
                let slot = bytecode
                    .var_names
                    .iter()
                    .find_map(|(slot, name)| (name == "x").then_some(*slot))
                    .unwrap();
                assert!(
                    matches!(&completion.values[slot], Value::HandleObject(handle) if handle.target == target)
                );
                output_root = Some(runmat_gc::ExplicitRoot::new(target).unwrap());
            } else {
                assert!(result.is_err());
            }
        }
    }
    drop(execution);
    drop(vars);
    runmat_vm::reset_thread_state_for_tests();
    runmat_gc::gc_collect_major().unwrap();
    assert_eq!(probe.retirements(), usize::from(output_root.is_none()));
    drop(output_root);
    runmat_gc::gc_collect_major().unwrap();
    assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
    assert_eq!(probe.retirements(), 1);
}
