use super::{interpret_with_vars_in_context, InterpreterOutcome};
use crate::bytecode::program::Bytecode;
use crate::bytecode::Instr;
use futures::executor::block_on;
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_value::{Value, ValueSequence};
use std::rc::Rc;
use std::sync::{mpsc, Mutex};

struct ProcessDispatchControl {
    reached: mpsc::Sender<()>,
    release: futures::channel::oneshot::Receiver<()>,
}

static PROCESS_DISPATCH_CONTROL: Mutex<Option<ProcessDispatchControl>> = Mutex::new(None);

fn pending_process_builtin(
    _invocation: runmat_runtime::builtin::RuntimeBuiltinInvocation<'_>,
) -> runmat_runtime::builtin::RuntimeBuiltinFuture {
    let control = PROCESS_DISPATCH_CONTROL
        .lock()
        .expect("process dispatch control lock")
        .take()
        .expect("process dispatch control");
    Box::pin(async move {
        control.reached.send(()).expect("test still receives entry");
        control.release.await.expect("release process builtin");
        ValueSequence::single(Value::Num(17.0))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
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

#[test]
fn process_stack_builtin_can_suspend_on_a_bounded_native_thread_stack() {
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
        .spawn(|| {
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                RuntimeServicePorts::default().with_builtin(Rc::new(ProcessBuiltinService)),
            );
            let bytecode = Bytecode::with_instructions(
                vec![
                    Instr::LoadString("java.lang.String".into()),
                    Instr::CallBuiltinMulti("javaObject".into(), 1, 1),
                    Instr::ReturnValue,
                ],
                0,
            );
            let mut vars = Vec::new();
            let outcome = block_on(interpret_with_vars_in_context(
                &bytecode,
                &mut vars,
                Some("process_dispatch_fixture"),
                runtime,
            ))
            .expect("Process builtin execution");
            let InterpreterOutcome::Completed(completion) = outcome;
            assert_eq!(completion.values, vec![Value::Num(17.0)]);
        })
        .expect("spawn bounded-stack VM thread");

    reached_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("Process builtin reached a genuine Pending boundary");
    release_tx.send(()).expect("release Process builtin");
    execution
        .join()
        .expect("bounded-stack VM thread must not overflow");
}
