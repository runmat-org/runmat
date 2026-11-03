use crate::{builtins::common::env, context::RuntimeContext, execution::RuntimeExecutionService};
use std::rc::Rc;

#[test]
fn host_capture_omits_native_bookkeeping_without_relaxing_snapshot_validation() {
    let entries = vec![
        ("=C:".into(), "C:\\workspace".into()),
        ("VISIBLE".into(), "kept".into()),
    ];
    assert!(runmat_execution::EnvironmentSnapshot::new(entries.clone()).is_err());
    let captured = super::EnvironmentState::from_host_entries(entries);
    assert_eq!(captured.vars(), [("VISIBLE".into(), "kept".into())]);
    assert!(runmat_execution::EnvironmentSnapshot::new(captured.vars()).is_ok());
}

fn context(value: &str) -> RuntimeContext {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    context
        .install_environment_snapshot(vec![("RUNMAT_CONTEXT_ENV".into(), value.into())])
        .unwrap();
    context
}

#[test]
fn selected_constructor_does_not_inherit_unselected_environment_or_clock() {
    let runtime = RuntimeContext::selected_native_base(
        Rc::new(RuntimeExecutionService::new()),
        None,
        false,
        false,
    )
    .unwrap();
    assert!(runtime.environment_snapshot().is_empty());
    assert!(runtime.service_ports().timing().is_err());
    assert!(runtime.search_path().unwrap().current_string().is_empty());
    let frozen =
        runmat_execution::EnvironmentSnapshot::new(vec![("SELECTED_ONLY".into(), "frozen".into())])
            .unwrap();
    let deterministic = RuntimeContext::selected_native_base(
        Rc::new(RuntimeExecutionService::new()),
        Some(&frozen),
        false,
        false,
    )
    .unwrap();
    assert_eq!(
        deterministic.environment_snapshot(),
        [("SELECTED_ONLY".into(), "frozen".into())]
    );
    assert!(deterministic.service_ports().timing().is_err());
    {
        let _scope = deterministic.enter();
        crate::builtins::common::path_state::set_path_string("invocation-only")
            .expect("session-owned path does not require an environment write");
    }
    assert_eq!(
        deterministic.search_path().unwrap().current_string(),
        "invocation-only"
    );
    assert!(runtime.search_path().unwrap().current_string().is_empty());
}

#[test]
fn nested_scopes_mutations_and_unwind_preserve_environment_ownership() {
    let host = std::env::var_os("RUNMAT_CONTEXT_ENV");
    let first = context("first");
    let second = context("second");
    {
        let _first = first.enter();
        assert_eq!(env::var("RUNMAT_CONTEXT_ENV").unwrap(), "first");
        assert!(env::var("HOME").is_err());
        assert_eq!(env::vars(), [("RUNMAT_CONTEXT_ENV".into(), "first".into())]);
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _second = second.enter();
            env::set_var("RUNMAT_CONTEXT_ENV", "changed");
            panic!("intentional scoped unwind");
        }));
        assert!(failure.is_err());
        assert_eq!(env::var("RUNMAT_CONTEXT_ENV").unwrap(), "first");
        env::remove_var("RUNMAT_CONTEXT_ENV");
        assert!(env::var("RUNMAT_CONTEXT_ENV").is_err());
    }
    assert_eq!(
        second.environment_snapshot(),
        [("RUNMAT_CONTEXT_ENV".into(), "changed".into())]
    );
    assert_eq!(std::env::var_os("RUNMAT_CONTEXT_ENV"), host);
}

#[test]
fn malformed_snapshot_replacement_is_atomic() {
    let context = context("original");
    for entries in [
        vec![("a".into(), "1".into()), ("a".into(), "2".into())],
        vec![("z".into(), "1".into()), ("a".into(), "2".into())],
        vec![("bad=name".into(), "1".into())],
        vec![("name".into(), "bad\0value".into())],
    ] {
        assert!(context.install_environment_snapshot(entries).is_err());
        assert_eq!(
            context.environment_snapshot(),
            [("RUNMAT_CONTEXT_ENV".into(), "original".into())]
        );
    }
}

#[test]
fn alternating_async_polls_use_their_original_session_environment() {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll},
    };
    let first = context("first");
    let second = context("second");
    let mut first_future = Box::pin(first.scope(futures::future::poll_fn(|_| {
        assert_eq!(env::var("RUNMAT_CONTEXT_ENV").unwrap(), "first");
        Poll::<()>::Pending
    })));
    let mut second_future = Box::pin(second.scope(async {
        assert_eq!(env::var("RUNMAT_CONTEXT_ENV").unwrap(), "second");
    }));
    let waker = futures::task::noop_waker();
    let mut task = Context::from_waker(&waker);
    assert!(Pin::new(&mut first_future).poll(&mut task).is_pending());
    assert!(Pin::new(&mut second_future).poll(&mut task).is_ready());
    assert!(Pin::new(&mut first_future).poll(&mut task).is_pending());
}
