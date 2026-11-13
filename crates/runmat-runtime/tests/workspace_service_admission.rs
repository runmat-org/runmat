use std::cell::RefCell;
use std::rc::Rc;

use futures::executor::block_on;
#[cfg(not(target_arch = "wasm32"))]
use runmat_mex::MexHostServices;
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::workspace::{BaseWorkspaceOwner, BaseWorkspaceService, WorkspaceDelta};
use runmat_value::Value;

fn runtime_with_workspace() -> (RuntimeContext, BaseWorkspaceOwner) {
    let owner = BaseWorkspaceOwner::new();
    let service = Rc::new(BaseWorkspaceService::new(owner.clone()));
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default()
            .with_workspace(service.clone())
            .with_base_workspace(service),
    );
    (runtime, owner)
}

async fn call(
    runtime: &RuntimeContext,
    name: &str,
    arguments: &[Value],
) -> Result<runmat_runtime::execution::RootedValue, runmat_runtime::RuntimeError> {
    call_outputs(runtime, name, arguments, 1).await
}

async fn call_outputs(
    runtime: &RuntimeContext,
    name: &str,
    arguments: &[Value],
    outputs: usize,
) -> Result<runmat_runtime::execution::RootedValue, runmat_runtime::RuntimeError> {
    runmat_runtime::call_builtin_async_with_outputs_in_context(runtime, name, arguments, outputs)
        .await
}

#[test]
fn introspection_builtins_preserve_closed_workspace_diagnostics() {
    let (runtime, owner) = runtime_with_workspace();
    owner.close().unwrap();

    for (name, arguments) in [
        ("who", Vec::new()),
        ("whos", Vec::new()),
        ("which", vec![Value::from("answer")]),
        ("open", vec![Value::from("answer")]),
        ("exist", vec![Value::from("answer"), Value::from("var")]),
        ("syms", vec![Value::from("answer")]),
        ("clear", Vec::new()),
        ("clearvars", Vec::new()),
    ] {
        let error = block_on(call(&runtime, name, &arguments)).expect_err(name);
        assert_eq!(
            error.identifier(),
            Some("RunMat:Workspace:Closed"),
            "{name}: {error:?}"
        );
        assert_eq!(error.context.builtin.as_deref(), Some(name));
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn mex_base_callbacks_preserve_closed_workspace_diagnostics_and_global_isolation() {
    let (runtime, owner) = runtime_with_workspace();
    let services = runmat_runtime::foreign::RuntimeMexHostServices::new(runtime);
    owner.close().unwrap();

    let read = services.get_variable("base", "answer").unwrap_err();
    assert_eq!(read.identifier.as_deref(), Some("RunMat:Workspace:Closed"));
    let write = services
        .put_variable("caller", "answer", Value::Num(1.0))
        .unwrap_err();
    assert_eq!(write.identifier.as_deref(), Some("RunMat:Workspace:Closed"));

    services
        .put_variable("global", "answer", Value::Num(2.0))
        .unwrap();
    let output = services
        .get_variable("global", "answer")
        .unwrap()
        .expect("global value remains available");
    output
        .publish(&mut |value| {
            assert_eq!(value, Value::Num(2.0));
            Ok(())
        })
        .unwrap();
}

#[test]
fn mat_workspace_admission_precedes_file_effects_and_load_publication() {
    let directory = tempfile::tempdir().unwrap();
    let rejected_path = directory.path().join("rejected.mat");
    let (closed_runtime, closed_owner) = runtime_with_workspace();
    closed_owner.close().unwrap();

    let error = block_on(call_outputs(
        &closed_runtime,
        "save",
        &[Value::from(rejected_path.to_string_lossy().as_ref())],
        0,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    assert!(!rejected_path.exists());

    let source_path = directory.path().join("source.mat");
    let (source_runtime, source_owner) = runtime_with_workspace();
    let mut delta = WorkspaceDelta::new();
    delta.assign("answer", Value::Num(42.0)).unwrap();
    source_owner.apply(delta).unwrap();
    block_on(call_outputs(
        &source_runtime,
        "save",
        &[Value::from(source_path.to_string_lossy().as_ref())],
        0,
    ))
    .unwrap();

    let error = block_on(call_outputs(
        &closed_runtime,
        "load",
        &[Value::from(source_path.to_string_lossy().as_ref())],
        0,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
}

#[test]
fn failed_clear_preflight_does_not_clear_dynamic_functions() {
    let (runtime, owner) = runtime_with_workspace();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let observed = requests.clone();
    runtime.set_dynamic_function_clearer(Some(Rc::new(move |runtime, request| {
        observed.borrow_mut().push(request);
        runmat_runtime::execution::admit_domain_continuation_with(&runtime, || async { Ok(()) })
    })));
    owner.close().unwrap();

    let error = block_on(call(&runtime, "clear", &[Value::from("all")])).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    assert!(requests.borrow().is_empty());
}

#[test]
fn workspace_introspection_is_session_scoped_and_excludes_persistents() {
    let (first, first_owner) = runtime_with_workspace();
    let (second, second_owner) = runtime_with_workspace();
    let mut first_delta = WorkspaceDelta::new();
    first_delta.assign("first", Value::Num(1.0)).unwrap();
    first_owner.apply(first_delta).unwrap();
    let mut second_delta = WorkspaceDelta::new();
    second_delta.assign("second", Value::Num(2.0)).unwrap();
    second_owner.apply(second_delta).unwrap();

    {
        let _scope = first.enter();
        runmat_runtime::workspace::session::store_global_named("first_global", Value::Num(3.0))
            .expect("scoped global write should succeed");
        runmat_runtime::workspace::session::store_persistent_named(
            "worker",
            "hidden_persistent",
            Value::Num(4.0),
        );
    }

    let first_names = names(block_on(call(&first, "who", &[])).unwrap());
    let second_names = names(block_on(call(&second, "who", &[])).unwrap());
    assert_eq!(first_names, vec!["first"]);
    assert_eq!(second_names, vec!["second"]);
    {
        let _scope = second.enter();
        assert_eq!(
            runmat_runtime::workspace::session::global_value("first_global"),
            None
        );
        assert_eq!(
            runmat_runtime::workspace::session::persistent_named_value(
                "worker",
                "hidden_persistent"
            ),
            None
        );
    }
}

fn names(value: runmat_runtime::execution::RootedValue) -> Vec<String> {
    let Value::Cell(values) = value.value() else {
        panic!("expected who cell result, got {:?}", value.value());
    };
    values
        .data()
        .iter()
        .map(|value| String::try_from(value).expect("name text"))
        .collect()
}
