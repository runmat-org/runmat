use super::{interpret_workspace, InterpreterOutcome};
use crate::bytecode::program::{Bytecode, FunctionBytecode, FunctionRegistry};
use crate::bytecode::Instr;
use futures::executor::block_on;
use futures::task::noop_waker;
use runmat_hir::FunctionId;
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_value::{Value, ValueSequence};
use std::future::Future;
use std::rc::Rc;
use std::sync::{mpsc, Mutex};
use std::task::{Context, Poll};

struct ProcessDispatchControl {
    reached: mpsc::Sender<()>,
    release: futures::channel::oneshot::Receiver<()>,
}

static PROCESS_DISPATCH_CONTROL: Mutex<Option<ProcessDispatchControl>> = Mutex::new(None);
static PROCESS_DISPATCH_TEST_LOCK: Mutex<()> = Mutex::new(());
const NESTED_PROCESS_DEPTH: usize = 64;

struct ProcessLeafRetirement;

impl Drop for ProcessLeafRetirement {
    fn drop(&mut self) {
        eprintln!("vm-process-dispatch phase: leaf future retired");
    }
}

fn pending_process_builtin(
    _invocation: runmat_runtime::builtin::RuntimeBuiltinInvocation<'_>,
) -> runmat_runtime::builtin::RuntimeBuiltinFuture {
    let control = PROCESS_DISPATCH_CONTROL
        .lock()
        .expect("process dispatch control lock")
        .take()
        .expect("process dispatch control");
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
        let _retirement = ProcessLeafRetirement;
        eprintln!("vm-process-dispatch phase: leaf entered before reach notification");
        control.reached.send(()).expect("test still receives entry");
        control.release.await.expect("release process builtin");
        eprintln!("vm-process-dispatch phase: leaf resumed after release");
        runmat_runtime::execution::RootedValueSequence::new(
            ValueSequence::single(Value::Num(17.0))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
        )
        .map_err(|error| error.into_runtime_error())
    })
}

struct ProcessBuiltinService;

impl RuntimeBuiltinService for ProcessBuiltinService {
    fn bindings_by_name(&self, name: &str) -> Vec<runmat_runtime::builtin::RuntimeBuiltinBinding> {
        if name != "javaObject" {
            return Vec::new();
        }
        vec![runmat_runtime::builtin::RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::BuiltinCatalogIdentity { name: "javaObject" },
                variant: "default",
            },
            pending_process_builtin,
        )]
    }
}

fn process_leaf_instructions() -> Vec<Instr> {
    vec![
        Instr::LoadString("java.lang.String".into()),
        Instr::CallBuiltinMulti("javaObject".into(), 1, 1),
        Instr::StoreVar(0),
        Instr::Return,
    ]
}

fn nested_process_bytecode(depth: usize) -> Bytecode {
    if depth == 0 {
        return Bytecode::with_instructions(
            vec![
                Instr::LoadString("java.lang.String".into()),
                Instr::CallBuiltinMulti("javaObject".into(), 1, 1),
                Instr::StoreVar(0),
                Instr::Return,
            ],
            1,
        );
    }

    let functions = (1..=depth)
        .map(|index| {
            let instructions = if index == depth {
                process_leaf_instructions()
            } else {
                vec![
                    Instr::CallSemanticFunctionMulti(FunctionId(index + 1), 0, 1),
                    Instr::StoreVar(0),
                    Instr::Return,
                ]
            };
            (
                FunctionId(index),
                FunctionBytecode {
                    function: FunctionId(index),
                    display_name: format!("process_chain_{index}"),
                    instructions,
                    var_count: 1,
                    output_slots: vec![0],
                    ..FunctionBytecode::default()
                },
            )
        })
        .collect();
    let registry = FunctionRegistry::new(functions);
    assert_eq!(
        registry.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Process,
        "the selected static chain must stay on the native process stack"
    );
    let mut bytecode = Bytecode::with_instructions(
        vec![
            Instr::CallSemanticFunctionMulti(FunctionId(1), 0, 1),
            Instr::StoreVar(0),
            Instr::Return,
        ],
        1,
    );
    bytecode.function_registry = registry;
    bytecode
}

fn run_process_fixture_on_bounded_stack(depth: usize) {
    let _test_guard = PROCESS_DISPATCH_TEST_LOCK
        .lock()
        .expect("serialize process dispatch fixtures");
    let (reached_tx, reached_rx) = mpsc::channel();
    let (release_tx, release_rx) = futures::channel::oneshot::channel();
    *PROCESS_DISPATCH_CONTROL
        .lock()
        .expect("process dispatch control lock") = Some(ProcessDispatchControl {
        reached: reached_tx,
        release: release_rx,
    });

    let execution = std::thread::Builder::new()
        .name("vm-process-dispatch".into())
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                RuntimeServicePorts::default().with_builtin(Rc::new(ProcessBuiltinService)),
            );
            let bytecode = nested_process_bytecode(depth);
            let vars = Vec::new();
            eprintln!("vm-process-dispatch depth={depth} phase: interpreter poll begins");
            let outcome = block_on(interpret_workspace(
                &bytecode,
                crate::InterpreterVars::from_owned(vars),
                Some("process_dispatch_fixture"),
                runtime,
                None,
                None,
            ))
            .outcome
            .expect("Process builtin execution");
            eprintln!("vm-process-dispatch depth={depth} phase: interpreter completed");
            let InterpreterOutcome::Completed(completion) = outcome;
            assert_eq!(completion.values, vec![Value::Num(17.0)]);
        })
        .expect("spawn bounded-stack VM thread");

    eprintln!("vm-process-dispatch depth={depth} phase: awaiting leaf entry");
    reached_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("Process builtin reached a genuine Pending boundary");
    eprintln!("vm-process-dispatch depth={depth} phase: leaf entry observed; releasing");
    release_tx.send(()).expect("release Process builtin");
    eprintln!("vm-process-dispatch depth={depth} phase: awaiting thread retirement");
    execution
        .join()
        .expect("bounded-stack VM thread must not overflow");
    eprintln!("vm-process-dispatch depth={depth} phase: thread retired");
}

#[test]
fn process_stack_builtin_can_suspend_on_a_bounded_native_thread_stack() {
    run_process_fixture_on_bounded_stack(0);
}

#[test]
fn nested_process_stack_calls_reach_a_pending_leaf_on_a_bounded_native_thread_stack() {
    run_process_fixture_on_bounded_stack(NESTED_PROCESS_DEPTH);
}

#[test]
fn nested_process_stack_calls_retire_a_pending_leaf_on_drop() {
    let _test_guard = PROCESS_DISPATCH_TEST_LOCK
        .lock()
        .expect("serialize process dispatch fixtures");
    let (reached_tx, reached_rx) = mpsc::channel();
    let (release_tx, release_rx) = futures::channel::oneshot::channel();
    *PROCESS_DISPATCH_CONTROL
        .lock()
        .expect("process dispatch control lock") = Some(ProcessDispatchControl {
        reached: reached_tx,
        release: release_rx,
    });

    let execution = std::thread::Builder::new()
        .name("vm-process-dispatch-drop".into())
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                RuntimeServicePorts::default().with_builtin(Rc::new(ProcessBuiltinService)),
            );
            let bytecode = nested_process_bytecode(NESTED_PROCESS_DEPTH);
            let vars = Vec::new();
            let mut pending = Box::pin(interpret_workspace(
                &bytecode,
                crate::InterpreterVars::from_owned(vars),
                Some("process_dispatch_fixture"),
                runtime,
                None,
                None,
            ));
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);
            eprintln!("vm-process-dispatch-drop phase: first poll begins");
            assert!(matches!(pending.as_mut().poll(&mut cx), Poll::Pending));
            eprintln!("vm-process-dispatch-drop phase: first poll returned Pending");
            eprintln!("vm-process-dispatch-drop phase: pending chain drop begins");
            drop(pending);
            eprintln!("vm-process-dispatch-drop phase: pending chain dropped");
        })
        .expect("spawn bounded-stack VM thread");

    eprintln!("vm-process-dispatch-drop phase: awaiting leaf entry");
    reached_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("Process builtin reached a genuine Pending boundary");
    eprintln!("vm-process-dispatch-drop phase: leaf entry observed; awaiting thread retirement");
    execution
        .join()
        .expect("dropping a suspended Process chain must not overflow");
    eprintln!("vm-process-dispatch-drop phase: thread retired");
    assert!(
        release_tx.send(()).is_err(),
        "leaf must have retired without replay"
    );
}
