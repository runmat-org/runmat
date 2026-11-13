//! Stateful test control selected by the invocation's exact builtin service.
use futures::channel::oneshot;
use runmat_runtime::builtin::{
    RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation,
};
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_value::{Value, ValueSequence};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

struct GatedSin {
    gate: RefCell<Option<oneshot::Receiver<()>>>,
    reached: RefCell<Option<oneshot::Sender<()>>>,
}

thread_local! {
    // Bindings are function pointers, so the fixture indexes its concrete
    // controls weakly. The invocation's explicit service selects the owner;
    // neither ambient runtime nor lookup order chooses a gate.
    static CONTROLS: RefCell<Vec<Weak<GatedSin>>> = const { RefCell::new(Vec::new()) };
}

fn invoke(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let runtime = invocation.context.expect("selected invocation runtime");
    let selected = runtime.service_ports().builtin().expect("selected service");
    let control = CONTROLS.with(|controls| {
        controls
            .borrow()
            .iter()
            .filter_map(Weak::upgrade)
            .find(|control| {
                let service: &dyn RuntimeBuiltinService = control.as_ref();
                std::ptr::addr_eq(service, selected.as_ref())
            })
            .expect("exact selected fixture service")
    });
    let pending = control
        .gate
        .borrow_mut()
        .take()
        .expect("no gate invocation replay");
    let reached = control
        .reached
        .borrow_mut()
        .take()
        .expect("no handler replay");
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
        reached.send(()).expect("reached receiver remains live");
        pending
            .await
            .map_err(|_| runmat_runtime::RuntimeError::new("sin gate abandoned"))?;
        runmat_runtime::execution::RootedValueSequence::new(
            ValueSequence::single(Value::Num(0.0))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
        )
        .map_err(|error| error.into_runtime_error())
    })
}

impl RuntimeBuiltinService for GatedSin {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        if name != "sin" {
            return Vec::new();
        }
        vec![RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::SIN_CATALOG_ENTRY.identity,
                variant: "default",
            },
            invoke,
        )]
    }
}

pub(super) fn runtime_with_gate(
    base: RuntimeContext,
) -> (RuntimeContext, oneshot::Sender<()>, oneshot::Receiver<()>) {
    let (sender, receiver) = oneshot::channel();
    let (reached_sender, reached_receiver) = oneshot::channel();
    let control = Rc::new(GatedSin {
        gate: RefCell::new(Some(receiver)),
        reached: RefCell::new(Some(reached_sender)),
    });
    CONTROLS.with(|controls| {
        let mut controls = controls.borrow_mut();
        controls.retain(|control| control.strong_count() != 0);
        controls.push(Rc::downgrade(&control));
    });
    let runtime = base.with_service_ports(RuntimeServicePorts::default().with_builtin(control));
    (runtime, sender, reached_receiver)
}
