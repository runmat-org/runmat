#[path = "support/mod.rs"]
mod support;

#[path = "suspended_cancellation/managed_arguments.rs"]
mod managed_arguments;

#[path = "suspended_cancellation/resident_validator.rs"]
mod resident_validator;

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_runtime::builtin::RuntimeBuiltinBinding;
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use std::future::Future;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll, Wake, Waker};

static CALLS: AtomicUsize = AtomicUsize::new(0);
static DROPS: AtomicUsize = AtomicUsize::new(0);
struct PendingDrop;
impl Drop for PendingDrop {
    fn drop(&mut self) {
        assert!(
            runmat_runtime::context::legacy::active().is_some(),
            "origin context on destruction"
        );
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}
struct PendingBuiltin;

#[test]
fn execution_scope_preserves_normal_caught_language_errors() {
    runmat_gc::gc_test_context(execution_scope_preserves_normal_caught_language_errors_inner);
}

fn execution_scope_preserves_normal_caught_language_errors_inner() {
    let bytecode = support::compile_source("try; error('expected'); catch; x = 42; end").unwrap();
    let source = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let vars = vec![];
    let admission = runmat_vm::workspace_admission(
        bytecode
            .var_names
            .iter()
            .map(|(slot, name)| (name.clone(), *slot))
            .collect(),
        Default::default(),
    );
    assert!(runmat_runtime::context::legacy::active().is_none());
    let execution = futures::executor::block_on(runmat_vm::interpret_workspace(
        &bytecode,
        runmat_vm::InterpreterVars::from_owned(vars),
        None,
        runtime,
        Some(admission),
        None,
    ));
    assert!(runmat_runtime::context::legacy::active().is_none());
    let outcome = execution.outcome.expect("language catch remains available");
    let runmat_vm::InterpreterOutcome::Completed(completion) = outcome;
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "x").then_some(*slot))
        .unwrap();
    let value = completion
        .values
        .get(slot)
        .expect("completed value remains in its compiled slot");
    assert_eq!(*value, runmat_value::Value::Num(42.0));
    let workspace = execution
        .workspace
        .expect("rich public caller receives exact closed frame without ambient context");
    assert_eq!(
        *workspace
            .state
            .vars
            .get(slot)
            .expect("closed workspace slot"),
        runmat_value::Value::Num(42.0)
    );
    assert!(!source.token().is_cancelled());
}

impl RuntimeBuiltinService for PendingBuiltin {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        if name != "sin" {
            return vec![];
        }
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
        vec![RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            |_| {
                runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
                    CALLS.fetch_add(1, Ordering::SeqCst);
                    let _drop = PendingDrop;
                    futures::future::pending().await
                })
            },
        )]
    }
}
#[derive(Default)]
struct Wakes(AtomicUsize);
impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn production_vm_cancellation_wakes_pending_dispatch_and_bypasses_catch() {
    runmat_gc::gc_test_context(
        production_vm_cancellation_wakes_pending_dispatch_and_bypasses_catch_inner,
    );
}

fn production_vm_cancellation_wakes_pending_dispatch_and_bypasses_catch_inner() {
    runmat_vm::reset_thread_state_for_tests();
    let bytecode = support::compile_source("try; x = sin(1); catch; x = sin(2); end").unwrap();
    let source = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        source.token(),
    )
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(PendingBuiltin)));
    let vars = vec![];
    let mut execution = Box::pin(runmat_vm::interpret_workspace(
        &bytecode,
        runmat_vm::InterpreterVars::from_owned(vars),
        None,
        runtime,
        None,
        None,
    ));
    let wakes = Arc::new(Wakes::default());
    let waker = Waker::from(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(execution
        .as_mut()
        .poll(&mut cx)
        .map(|execution| execution.outcome)
        .is_pending());
    assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(DROPS.load(Ordering::SeqCst), 0);
    source.cancel(CancellationReason::User);
    assert!(wakes.0.load(Ordering::SeqCst) > 0);
    let Poll::Ready(Err(error)) = execution
        .as_mut()
        .poll(&mut cx)
        .map(|execution| execution.outcome)
    else {
        panic!("cancellation must terminalize suspended dispatch");
    };
    assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
    assert_eq!(CALLS.load(Ordering::SeqCst), 1, "catch body must not run");
    assert_eq!(
        DROPS.load(Ordering::SeqCst),
        1,
        "retained completed future released work"
    );
    drop(execution);
    assert_eq!(DROPS.load(Ordering::SeqCst), 1);
    assert!(runmat_runtime::context::legacy::active().is_none());
    runmat_vm::reset_thread_state_for_tests();
    for cancel in [false, true] {
        abandoned_await_releases_only_its_running_service_future(cancel);
    }
}

fn abandoned_await_releases_only_its_running_service_future(cancel: bool) {
    use runmat_runtime::execution::{
        AwaitAction, DeferredCall, DeferredInvocation, DeferredProgram, ExecutionServiceError,
        RuntimeExecutionService, RuntimeExecutionServices,
    };
    use runmat_value::Value;

    runmat_vm::reset_thread_state_for_tests();
    let variables = std::collections::HashMap::from([("work".to_string(), 0)]);
    let ast = runmat_parser::parse("try; x = await(work); catch; x = sin(2); end").unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::new(&variables)).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let bytecode = runmat_vm::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
    let service = Rc::new(RuntimeExecutionService::new());
    let deferred = || DeferredCall {
        spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
        invocation: DeferredInvocation::callable(
            runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                runmat_types::CallableIdentity::Builtin(runmat_types::BuiltinId("sin".into())),
                vec![Value::Num(1.0)],
                1,
                runmat_types::CallableFallbackPolicy::None,
                runmat_runtime::call::descriptor::CallableCallKind::Direct,
            ),
        ),
        retry: runmat_execution::RetryPolicy::Never,
        program_revision: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        console: None,
        capabilities: Default::default(),
        program: DeferredProgram::Local,
    };
    let owned = service.create_future(deferred()).unwrap();
    let sibling = service.create_future(deferred()).unwrap();
    let source = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(service.clone(), source.token())
        .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(PendingBuiltin)));
    let vars = vec![Value::Future(owned.clone().into())];
    let calls_before = CALLS.load(Ordering::SeqCst);
    let drops_before = DROPS.load(Ordering::SeqCst);
    let mut execution = Box::pin(runmat_vm::interpret_workspace(
        &bytecode,
        runmat_vm::InterpreterVars::from_owned(vars),
        None,
        runtime,
        None,
        None,
    ));
    let wakes = Arc::new(Wakes::default());
    let waker = Waker::from(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(execution
        .as_mut()
        .poll(&mut cx)
        .map(|execution| execution.outcome)
        .is_pending());
    assert_eq!(CALLS.load(Ordering::SeqCst), calls_before + 1);
    if cancel {
        source.cancel(CancellationReason::User);
        assert!(wakes.0.load(Ordering::SeqCst) > 0);
        assert!(matches!(
            execution
                .as_mut()
                .poll(&mut cx)
                .map(|execution| execution.outcome),
            Poll::Ready(Err(_))
        ));
        assert!(matches!(
            service.begin_await(Value::Future(owned.clone().into())),
            Err(ExecutionServiceError::Cancelled)
        ));
    }
    drop(execution);
    assert_eq!(DROPS.load(Ordering::SeqCst), drops_before + 1);
    assert_eq!(
        CALLS.load(Ordering::SeqCst),
        calls_before + 1,
        "catch must not replay work"
    );
    assert!(matches!(
        service.begin_await(Value::Future(owned.into())),
        Err(ExecutionServiceError::Cancelled)
    ));
    assert!(matches!(
        service.begin_await(Value::Future(sibling.clone().into())),
        Ok(AwaitAction::ExecuteFuture { .. })
    ));
    service
        .complete_future(
            (&sibling).into(),
            Ok(runmat_runtime::execution::RootedValueSequence::new(
                runmat_value::ValueSequence::empty(),
            )
            .unwrap()),
        )
        .unwrap();
    assert!(runmat_runtime::context::legacy::active().is_none());
    runmat_vm::reset_thread_state_for_tests();
}
