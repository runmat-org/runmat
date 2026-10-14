use crate::context::RuntimeContext;
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_value::Value;
use std::error::Error;
use std::rc::Rc;

mod fixture;
use fixture::{controlled_context, poll_pending, poll_ready, DialogKind, DialogRequest};

struct Case {
    name: &'static str,
    kind: DialogKind,
    args: Vec<Value>,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "uigetfile",
            kind: DialogKind::Open,
            args: Vec::new(),
        },
        Case {
            name: "uiputfile",
            kind: DialogKind::Save,
            args: Vec::new(),
        },
        Case {
            name: "uigetdir",
            kind: DialogKind::Directory,
            args: Vec::new(),
        },
    ]
}

#[test]
fn selected_context_without_host_returns_cancellation_without_ambient_fallback() {
    for case in cases() {
        let (ambient, ambient_host, _release) = controlled_context(case.kind);
        let selected =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
        let _ambient = ambient.enter();
        let requested_outputs = 1;
        let result = block_on(crate::call_builtin_async_with_outputs_in_context(
            &selected,
            case.name,
            &case.args,
            requested_outputs,
        ))
        .expect("headless dialog cancellation");
        let Value::OutputList(outputs) = result.value() else {
            panic!("{} must return its requested output list", case.name)
        };
        assert_eq!(outputs.len(), requested_outputs, "{} arity", case.name);
        assert_eq!(outputs[0], Value::Num(0.0), "{} cancellation", case.name);
        assert_eq!(ambient_host.started.get(), 0, "{} ambient host", case.name);
        assert!(ambient_host.request.borrow().is_none());
    }
}

#[test]
fn selected_siblings_reach_pending_and_drop_without_cross_host_replay() {
    for case in cases() {
        let (ambient, ambient_host, _ambient_release) = controlled_context(case.kind);
        let (first_context, first_host, first_release) = controlled_context(case.kind);
        let (second_context, second_host, second_release) = controlled_context(case.kind);
        let _ambient = ambient.enter();
        let mut first = Box::pin(crate::call_builtin_async_with_outputs_in_context(
            &first_context,
            case.name,
            &case.args,
            1,
        ));
        let mut second = Box::pin(crate::call_builtin_async_with_outputs_in_context(
            &second_context,
            case.name,
            &case.args,
            1,
        ));

        poll_pending(&mut first, &first_host);
        poll_pending(&mut second, &second_host);
        assert_request(case.kind, first_host.request.borrow().as_ref());
        assert_request(case.kind, second_host.request.borrow().as_ref());
        assert_eq!(ambient_host.started.get(), 0);

        second_release.send(Ok(())).expect("release second host");
        poll_ready(&mut second).expect("second dialog");
        assert_eq!(second_host.completed.get(), 1);
        drop(first);
        assert_eq!(first_host.dropped.get(), 1);
        assert!(first_release.send(Ok(())).is_err());
        assert_eq!(second_host.dropped.get(), 0);
    }
}

fn assert_request(kind: DialogKind, request: Option<&DialogRequest>) {
    match (kind, request) {
        (DialogKind::Open, Some(DialogRequest::Open(request))) => {
            assert!(!request.multiselect);
            assert!(!request.filters.is_empty());
        }
        (DialogKind::Save, Some(DialogRequest::Save(request))) => {
            assert!(!request.filters.is_empty());
        }
        (DialogKind::Directory, Some(DialogRequest::Directory(request))) => {
            assert!(request.default_path.is_none());
        }
        other => panic!("wrong selected dialog request: {other:?}"),
    }
}

#[test]
fn selected_host_failure_keeps_builtin_identifier_and_typed_source() {
    for case in cases() {
        let (context, host, release) = controlled_context(case.kind);
        let mut call = Box::pin(crate::call_builtin_async_with_outputs_in_context(
            &context, case.name, &case.args, 1,
        ));
        poll_pending(&mut call, &host);
        release
            .send(Err(crate::build_runtime_error("selected dialog retired")
                .with_identifier("RunMat:host:SelectedDialogRetired")
                .build()))
            .expect("release failure");
        let error = poll_ready(&mut call).expect_err("selected failure");
        assert_eq!(
            error.identifier(),
            Some(match case.kind {
                DialogKind::Open => "RunMat:uigetfile:HostError",
                DialogKind::Save => "RunMat:uiputfile:HostError",
                DialogKind::Directory => "RunMat:uigetdir:HostError",
            })
        );
        let source = Error::source(&error)
            .and_then(|source| source.downcast_ref::<RuntimeError>())
            .expect("typed selected host cause");
        assert_eq!(
            source.identifier(),
            Some("RunMat:host:SelectedDialogRetired")
        );
    }
}
