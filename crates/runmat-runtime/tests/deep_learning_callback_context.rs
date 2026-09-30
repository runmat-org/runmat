use futures::executor::block_on;
use runmat_runtime::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation},
    call_builtin_async_with_outputs_in_context,
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use runmat_value::Value;
use std::{cell::RefCell, rc::Rc};

thread_local! {
    static CALLBACKS: RefCell<Vec<(runmat_runtime::context::RuntimeContextLocalId, Option<usize>)>> =
        const { RefCell::new(Vec::new()) };
}

fn selected_callback(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let runtime = invocation
        .context
        .expect("callback retains the invoking runtime");
    let outputs = invocation.requested_outputs;
    Box::pin(async move {
        CALLBACKS.with(|calls| calls.borrow_mut().push((runtime.local_identity(), outputs)));
        Ok(match outputs {
            Some(2) => Value::OutputList(vec![Value::Num(23.0), Value::Num(29.0)]),
            _ => Value::Num(23.0),
        })
    })
}

struct SelectedCallback;

impl RuntimeBuiltinService for SelectedCallback {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        let mut bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        if name.eq_ignore_ascii_case("deal") || name.eq_ignore_ascii_case("plus") {
            for binding in &mut bindings {
                binding.implementation = selected_callback;
            }
        }
        bindings
    }
}

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(SelectedCallback)))
}

#[test]
fn dlfeval_callback_uses_invoking_runtime_and_structural_output_count() {
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    let runtime = runtime();
    let output = block_on(call_builtin_async_with_outputs_in_context(
        &runtime,
        "dlfeval",
        &[Value::FunctionHandle("deal".into()), Value::Num(1.0)],
        2,
    ))
    .expect("selected callback");
    assert_eq!(
        output,
        Value::OutputList(vec![Value::Num(23.0), Value::Num(29.0)])
    );
    CALLBACKS.with(|calls| {
        assert_eq!(
            calls.borrow().as_slice(),
            &[(runtime.local_identity(), Some(2))]
        );
    });
}

#[test]
fn dlfeval_registered_binding_uses_invocation_count_not_ambient_count() {
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    let runtime = runtime();
    let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name("dlfeval");
    let [binding] = bindings.as_slice() else {
        panic!("dlfeval must have one canonical binding");
    };
    let args = [Value::FunctionHandle("deal".into()), Value::Num(1.0)];
    let _ambient = runmat_runtime::output_count::push_output_count(Some(1));
    let output = block_on((binding.implementation)(RuntimeBuiltinInvocation::new(
        &args,
        Some(runtime.clone()),
        Some(2),
    )))
    .expect("registered callback");
    assert_eq!(
        output,
        Value::OutputList(vec![Value::Num(23.0), Value::Num(29.0)])
    );
    CALLBACKS.with(|calls| {
        assert_eq!(
            calls.borrow().as_slice(),
            &[(runtime.local_identity(), Some(2))]
        );
    });
}

#[test]
fn dlupdate_leaf_callback_uses_invoking_runtime_once() {
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    let runtime = runtime();
    let output = block_on(call_builtin_async_with_outputs_in_context(
        &runtime,
        "dlupdate",
        &[Value::FunctionHandle("plus".into()), Value::Num(1.0)],
        1,
    ))
    .expect("selected leaf callback");
    assert_eq!(output, Value::Num(23.0));
    CALLBACKS.with(|calls| {
        assert_eq!(
            calls.borrow().as_slice(),
            &[(runtime.local_identity(), Some(1))]
        );
    });
}

#[test]
fn unpolled_dlfeval_releases_its_runtime_and_callback_service() {
    let service = Rc::new(SelectedCallback);
    let weak_service = Rc::downgrade(&service);
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(service.clone()));
    let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name("dlfeval");
    let [binding] = bindings.as_slice() else {
        panic!("dlfeval must have one canonical binding");
    };
    let args = [Value::FunctionHandle("deal".into()), Value::Num(1.0)];
    let future = (binding.implementation)(RuntimeBuiltinInvocation::new(
        &args,
        Some(runtime.clone()),
        Some(1),
    ));
    drop(runtime);
    drop(service);
    assert!(
        weak_service.upgrade().is_some(),
        "the future owns its runtime"
    );
    drop(future);
    assert!(
        weak_service.upgrade().is_none(),
        "dropping the future releases its runtime"
    );
}
