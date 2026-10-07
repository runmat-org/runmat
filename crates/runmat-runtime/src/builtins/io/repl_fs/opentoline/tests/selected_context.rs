use super::*;
use crate::context::{HostDocumentRequest, RuntimeContext};
use fixture::{args, controlled_context, poll_pending_until_host_starts, poll_ready};
use futures::FutureExt;
use std::error::Error;
use std::rc::Rc;

mod fixture;

#[test]
fn selected_context_without_host_never_falls_back_to_conflicting_ambient_host() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("no-selected-host.m");
    std::fs::write(&path, "value = 1;\n").expect("fixture");

    let (ambient_context, ambient_host, _ambient_release) = controlled_context();
    let selected = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let _ambient = ambient_context.enter();
    let selected_args = args(&path, 1.0);
    let error = futures::executor::block_on(crate::call_builtin_async_with_outputs_in_context(
        &selected,
        BUILTIN_NAME,
        &selected_args,
        0,
    ))
    .expect_err("selected context has no document host");

    assert_eq!(
        error.identifier(),
        Some("RunMat:host:DocumentInteractionUnavailable")
    );
    assert_eq!(ambient_host.started.get(), 0);
    assert!(ambient_host.request.borrow().is_none());
}

#[test]
fn sibling_pending_navigation_keeps_exact_selected_hosts_and_drop_lifetimes() {
    let directory = tempdir().expect("tempdir");
    let first_path = directory.path().join("first.m");
    let second_path = directory.path().join("second.m");
    std::fs::write(&first_path, "first = 1;\n").expect("first fixture");
    std::fs::write(&second_path, "second = 2;\n").expect("second fixture");

    let (first_context, first_host, first_release) = controlled_context();
    let (second_context, second_host, second_release) = controlled_context();
    let _ambient = first_context.enter();
    let first_args = args(&first_path, 1.0);
    let second_args = args(&second_path, 2.0);
    let mut first = Box::pin(crate::call_builtin_async_with_outputs_in_context(
        &first_context,
        BUILTIN_NAME,
        &first_args,
        0,
    ));
    let mut second = Box::pin(crate::call_builtin_async_with_outputs_in_context(
        &second_context,
        BUILTIN_NAME,
        &second_args,
        0,
    ));

    poll_pending_until_host_starts(&mut first, &first_host);
    poll_pending_until_host_starts(&mut second, &second_host);
    assert!(matches!(
        first_host.request.borrow().as_ref(),
        Some(HostDocumentRequest::Navigate { path, location })
            if path == &std::fs::canonicalize(&first_path).unwrap() && location.line.get() == 1
    ));
    assert!(matches!(
        second_host.request.borrow().as_ref(),
        Some(HostDocumentRequest::Navigate { path, location })
            if path == &std::fs::canonicalize(&second_path).unwrap() && location.line.get() == 2
    ));

    second_release.send(Ok(())).expect("release second host");
    poll_ready(&mut second).expect("second navigation");
    assert_eq!(second_host.completed.get(), 1);
    assert_eq!(first_host.completed.get(), 0);
    assert!(first.as_mut().now_or_never().is_none());
    drop(first);
    assert_eq!(first_host.dropped.get(), 1);
    assert!(
        first_release.send(Ok(())).is_err(),
        "dropped selected host request cannot replay"
    );
    assert_eq!(second_host.dropped.get(), 0);
}

#[test]
fn selected_host_retirement_keeps_typed_cause_and_never_uses_ambient_host() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("retired.m");
    std::fs::write(&path, "value = 1;\n").expect("fixture");

    let (ambient_context, ambient_host, _ambient_release) = controlled_context();
    let (selected_context, selected_host, selected_release) = controlled_context();
    let _ambient = ambient_context.enter();
    let selected_args = args(&path, 1.0);
    let mut call = Box::pin(crate::call_builtin_async_with_outputs_in_context(
        &selected_context,
        BUILTIN_NAME,
        &selected_args,
        0,
    ));
    poll_pending_until_host_starts(&mut call, &selected_host);
    drop(selected_release);
    let error = poll_ready(&mut call).expect_err("selected retirement");
    assert_eq!(
        error.identifier(),
        Some("RunMat:host:SelectedDocumentRetired")
    );
    let source = Error::source(&error)
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("opentoline retains selected host cause");
    assert_eq!(
        source.identifier(),
        Some("RunMat:host:SelectedDocumentRetired")
    );
    assert_eq!(selected_host.completed.get(), 1);
    assert_eq!(ambient_host.started.get(), 0);
    assert!(ambient_host.request.borrow().is_none());
}
