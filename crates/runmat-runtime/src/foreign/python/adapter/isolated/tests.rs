use std::rc::Rc;

use super::super::{invoke_isolated, PythonAdapter};
use super::IsolatedCallGuard;
use crate::context::{ForeignCall, RuntimeContext};
use crate::execution::{RootedValues, RuntimeExecutionService};
use crate::foreign::python::isolation::IsolatedPythonClient;
use crate::foreign::{ForeignAdapter, ForeignHandleRegistry};

fn call() -> ForeignCall {
    ForeignCall {
        adapter: runmat_python::PYTHON_ADAPTER_ID.into(),
        symbol: "must-not-spawn".into(),
        arguments: crate::call::RootedFunctionArguments::ordinary(
            RootedValues::new(Vec::new()).unwrap(),
        ),
        requested_outputs: 1,
    }
}

#[tokio::test]
async fn both_ordinary_entry_routes_fence_unretired_prior_process() {
    let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
    let mut configuration = adapter.configuration.borrow().clone();
    configuration.execution_mode = runmat_python::PythonExecutionMode::OutOfProcess;
    adapter.configure(configuration).unwrap();
    let state = adapter.isolated_state();
    // Obtain real process evidence through the canonical spawn owner, never
    // fabricate a retirement result or mutate the evidence's private flag.
    let old =
        IsolatedPythonClient::pending_test_client_with_evidence(state.spawn_unverified.clone())
            .await;
    assert!(state.spawn_unverified.is_unverified());
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let error = invoke_isolated(state.clone(), context.clone(), call())
        .await
        .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Python:RetirementUnverified")
    );
    assert!(!state.call_active.get());
    assert!(state.client.borrow().is_none());
    let error = adapter
        .invoke(context, call())
        .await
        .into_invoked_result()
        .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Python:RetirementUnverified")
    );
    assert!(
        state.spawn_unverified.is_unverified(),
        "rejection cannot clear prior evidence"
    );

    // Canonical bounded process retirement alone makes a replacement eligible.
    drop(old);
    assert!(!state.spawn_unverified.is_unverified());
    let guard = IsolatedCallGuard::begin(state);
    guard.admit_replacement_spawn().unwrap();
    guard.finish_running();
}

#[test]
fn terminal_restart_failure_keeps_original_error_and_abandons_guard() {
    let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
    let mut state = adapter.isolated_state();
    // A missing registry entry exercises the actual fallible restart path.
    state.host_identity = "not-registered".into();
    state.call_active.set(true);
    let guard = IsolatedCallGuard::begin(state.clone());
    let primary = crate::build_runtime_error("original cancelled invocation")
        .with_identifier("RunMat:Python:Cancelled")
        .build();
    let error = guard.finish_terminal::<()>(Err(primary)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Python:Cancelled"));
    let cleanup = error
        .source
        .as_deref()
        .expect("restart cause retained")
        .source()
        .and_then(|cause| cause.downcast_ref::<crate::RuntimeError>())
        .expect("typed registry cleanup cause");
    assert_eq!(cleanup.identifier(), Some("RunMat:Foreign:HostUnavailable"));
    assert!(!state.call_active.get());
    assert!(state.terminated.get());
    assert!(state.client.borrow().is_none());
}
