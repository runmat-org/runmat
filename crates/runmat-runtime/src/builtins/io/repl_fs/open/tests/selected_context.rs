use super::*;
use crate::context::document_test_support::{
    controlled_context, poll_pending_until_host_starts, poll_ready,
};
use crate::context::{HostDocumentRequest, RuntimeContext};

async fn dispatch(
    runtime: &RuntimeContext,
    args: &[Value],
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::call_builtin_async_with_outputs_in_context(runtime, BUILTIN_NAME, args, 0).await
}

#[test]
fn selected_context_without_host_never_falls_back_for_variable_or_file() {
    let _guard = workspace_guard();
    ensure_test_resolver();
    set_workspace(&[("answer", Value::Num(42.0))]);
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("selected-no-host.m");
    std::fs::write(&path, "value = 1;\n").expect("fixture");

    let (ambient, ambient_host, _ambient_release) = controlled_context();
    let selected = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let _ambient = ambient.enter();
    for arguments in [
        vec![Value::from("answer")],
        vec![Value::String(path.to_string_lossy().into_owned().into())],
    ] {
        let error = block_on(dispatch(&selected, &arguments)).expect_err("selected host required");
        assert_eq!(
            error.identifier(),
            Some("RunMat:host:DocumentInteractionUnavailable")
        );
    }
    assert_eq!(ambient_host.started.get(), 0);
    assert!(ambient_host.request.borrow().is_none());
}

#[test]
fn variable_and_file_routes_keep_exact_pending_selected_hosts_and_drop_lifetimes() {
    let _guard = workspace_guard();
    ensure_test_resolver();
    set_workspace(&[("answer", Value::Num(42.0))]);
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("selected-file.m");
    std::fs::write(&path, "value = 1;\n").expect("fixture");
    let canonical_path = std::fs::canonicalize(&path).expect("canonical fixture");

    let (ambient, ambient_host, _ambient_release) = controlled_context();
    let (variable_context, variable_host, variable_release) = controlled_context();
    let (file_context, file_host, file_release) = controlled_context();
    let _ambient = ambient.enter();
    let variable_args = [Value::from("answer")];
    let file_args = [Value::String(path.to_string_lossy().into_owned().into())];
    let mut variable = Box::pin(dispatch(&variable_context, &variable_args));
    let mut file = Box::pin(dispatch(&file_context, &file_args));

    poll_pending_until_host_starts(&mut variable, &variable_host);
    poll_pending_until_host_starts(&mut file, &file_host);
    assert!(matches!(
        variable_host.request.borrow().as_ref(),
        Some(HostDocumentRequest::OpenVariable { name }) if name == "answer"
    ));
    assert!(matches!(
        file_host.request.borrow().as_ref(),
        Some(HostDocumentRequest::OpenFile { path }) if path == &canonical_path
    ));
    assert_eq!(ambient_host.started.get(), 0);

    file_release.send(Ok(())).expect("release file host");
    assert!(poll_ready(&mut file).is_ok());
    assert_eq!(file_host.completed.get(), 1);
    assert_eq!(variable_host.completed.get(), 0);
    drop(variable);
    assert_eq!(variable_host.dropped.get(), 1);
    assert!(
        variable_release.send(Ok(())).is_err(),
        "dropped variable request cannot replay"
    );
    assert_eq!(file_host.dropped.get(), 0);
}
