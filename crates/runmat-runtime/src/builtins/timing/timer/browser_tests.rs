//! Browser event-loop witness for session-owned timer execution.

use super::*;
use crate::context::{BrowserLocalTaskSupervisor, RuntimeServicePorts};
use crate::execution::RuntimeExecutionService;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test(async)]
async fn browser_start_returns_before_callback_and_wait_joins_it() {
    let supervisor = BrowserLocalTaskSupervisor::new();
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let owner = runtime.local_identity();
    let runtime = runtime.with_service_ports(
        RuntimeServicePorts::default().with_local_tasks(supervisor.submission_port(owner)),
    );
    let calls = Arc::new(Mutex::new(0usize));
    let recorded = calls.clone();
    let scope = runtime.enter();
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, _args, requested_outputs| {
            assert_eq!(requested_outputs, 0);
            let recorded = recorded.clone();
            Box::pin(async move {
                *recorded.lock().unwrap() += 1;
                crate::execution::RootedValueSequence::new(crate::sequence::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())
            })
        },
    )))
    .expect("call-port scope");
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 1);
    let callback = crate::user_functions::test_support::exact_local_value("tick", binding.clone());
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding);
    drop(scope);
    let timer_args = [Value::String("TimerFcn".into()), callback];
    let timer_args = crate::call::BuiltinArguments::from_borrowed(&timer_args, Some(&runtime))
        .expect("root timer arguments");
    let timer = timer_builtin(Some(runtime.clone()), timer_args)
        .await
        .expect("timer");
    timer_start_builtin(Some(runtime.clone()), Some(0), timer.clone())
        .await
        .expect("start");
    assert_eq!(*calls.lock().unwrap(), 0);
    timer_wait_builtin(Some(runtime.clone()), Some(0), timer.clone())
        .await
        .expect("wait");
    assert_eq!(*calls.lock().unwrap(), 1);
    let Value::HandleObject(handle) = timer else {
        panic!("timer handle");
    };
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
    supervisor.shutdown();
}
