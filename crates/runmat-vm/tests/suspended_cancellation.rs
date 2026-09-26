#[path = "support/mod.rs"]
mod support;

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
                Box::pin(async {
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
    runmat_vm::reset_thread_state_for_tests();
    let bytecode = support::compile_source("try; x = sin(1); catch; x = sin(2); end").unwrap();
    let source = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        source.token(),
    )
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(PendingBuiltin)));
    let mut vars = vec![];
    let mut execution = Box::pin(runmat_vm::interpret_with_vars_in_context(
        &bytecode, &mut vars, None, runtime,
    ));
    let wakes = Arc::new(Wakes::default());
    let waker = Waker::from(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(execution.as_mut().poll(&mut cx).is_pending());
    assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(DROPS.load(Ordering::SeqCst), 0);
    source.cancel(CancellationReason::User);
    assert!(wakes.0.load(Ordering::SeqCst) > 0);
    let Poll::Ready(Err(error)) = execution.as_mut().poll(&mut cx) else {
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
}
