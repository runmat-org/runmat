use super::*;

thread_local! {
    static OBSERVED_TARGET: std::cell::Cell<Option<runmat_gc::GcHandle>> = const {
        std::cell::Cell::new(None)
    };
}

struct WitnessReset;

impl Drop for WitnessReset {
    fn drop(&mut self) {
        OBSERVED_TARGET.with(|target| target.set(None));
    }
}

const HANDLE_CLASS: &str = r#"
classdef PendingHandle < handle
    properties
        x
    end
    methods
        function obj = PendingHandle()
            obj.x = 7;
        end
    end
end
"#;

const SOURCE: &str = r#"
function output = first(value)
    global order_counter;
    global pending_handle_witness;
    order_counter = order_counter + 1;
    pending_handle_witness = value;
    output = struct('second', @(next) value.x + next);
end
global order_counter;
order_counter = 0;
s = struct('first', @first);
result = s.first(PendingHandle()).second(input('next: '));
assert(result == 10);
"#;

fn take_temporary_target() -> runmat_gc::GcHandle {
    let value = runmat_runtime::workspace::session::global_value("pending_handle_witness")
        .expect("the first invocation exposes its temporary to the test observer");
    let Value::HandleObject(handle) = value else {
        panic!("the temporary witness must be a managed handle")
    };
    let expected_class = runmat_types::ClassIdentity::new("PendingHandle").unwrap();
    assert_eq!(handle.class_name, expected_class);
    let Value::Object(object) =
        runmat_gc::gc_clone_value(&handle.target).expect("the observed temporary is live")
    else {
        panic!("the managed target must contain the authored object")
    };
    assert_eq!(object.class_name(), &expected_class);
    assert_eq!(object.properties().get("x"), Some(&Value::Num(7.0)));
    runmat_runtime::workspace::session::store_global_named(
        "pending_handle_witness",
        Value::Num(0.0),
    )
    .expect("retire the observer's managed-handle alias");
    assert_eq!(
        runmat_runtime::workspace::session::global_value("pending_handle_witness"),
        Some(Value::Num(0.0)),
        "the observer must not retain a managed alias"
    );
    // The observer retains only an unrooted identity. The anonymous closure
    // produced by the invocation must keep its own capture alive while pending.
    handle.target
}

fn path_request(session: &RunMatSession, path: &std::path::Path) -> crate::abi::ExecutionRequest {
    crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Path(path.to_string_lossy().into_owned()),
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    )
}

#[test]
fn temporary_managed_receiver_survives_native_pending_argument_and_major_collection() {
    runmat_gc::gc_test_context(|| {
        let _witness = WitnessReset;
        let _cwd_lock = super::super::cwd_lock();
        let directory = tempfile::TempDir::new().expect("class fixture directory");
        std::fs::write(directory.path().join("PendingHandle.m"), HANDLE_CLASS)
            .expect("write handle class");
        let source = directory.path().join("main.m");
        std::fs::write(&source, SOURCE).expect("write invocation script");
        let _cwd = super::super::push_cwd(directory.path());

        let mut session = hot_session();
        let ready = Arc::new(InputReadiness::new(true));
        let calls = Arc::new(AtomicUsize::new(0));
        let prefix_seen = Arc::new(AtomicBool::new(false));
        session.install_async_input_handler({
            let ready = Arc::clone(&ready);
            let calls = Arc::clone(&calls);
            let prefix_seen = Arc::clone(&prefix_seen);
            move |_| {
                let ready = Arc::clone(&ready);
                let calls = Arc::clone(&calls);
                let prefix_seen = Arc::clone(&prefix_seen);
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    OBSERVED_TARGET.with(|target| target.set(Some(take_temporary_target())));
                    if runmat_runtime::workspace::session::global_value("order_counter")
                        == Some(Value::Num(1.0))
                    {
                        prefix_seen.store(true, Ordering::SeqCst);
                    }
                    ready.wait().await;
                    Ok(crate::execution::InputResponse::Line("3".into()))
                }
            }
        });

        let mut native = 0;
        for _ in 0..4 {
            let request = path_request(&session, &source);
            let response = futures::executor::block_on(session.execute_request(request));
            let outcome = response.result.expect("warm managed receiver invocation");
            assert!(
                outcome.diagnostics.iter().all(|diagnostic| {
                    diagnostic.severity != crate::abi::DiagnosticSeverity::Error
                }),
                "{:?}",
                outcome.diagnostics
            );
            native += usize::from(outcome.used_jit);
        }
        assert!(
            native >= 2,
            "temporary managed receiver must execute native"
        );
        assert!(prefix_seen.load(Ordering::SeqCst));

        ready.set(false);
        prefix_seen.store(false, Ordering::SeqCst);
        let before = calls.load(Ordering::SeqCst);
        let request = path_request(&session, &source);
        let mut execution = Box::pin(session.execute_request(request));
        poll_until_input(execution.as_mut(), &calls, before + 1);
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
        assert!(prefix_seen.load(Ordering::SeqCst));
        let pending_target = OBSERVED_TARGET
            .with(|target| target.get())
            .expect("pending target identity");
        runmat_gc::gc_collect_major().expect("collect while managed receiver is pending");
        assert!(runmat_gc::gc_clone_value(&pending_target).is_ok());
        ready.set(true);
        let response = futures::executor::block_on(execution);
        let outcome = response.result.expect("resume managed receiver invocation");
        assert!(
            outcome.used_jit,
            "resumed managed receiver must execute native"
        );
        assert!(
            outcome
                .diagnostics
                .iter()
                .all(|diagnostic| { diagnostic.severity != crate::abi::DiagnosticSeverity::Error }),
            "{:?}",
            outcome.diagnostics
        );
        assert_eq!(calls.load(Ordering::SeqCst), before + 1, "no replay");
        assert_eq!(
            session.get_variables().unwrap().get("result"),
            Some(&Value::Num(10.0))
        );

        ready.set(false);
        let before_drop = calls.load(Ordering::SeqCst);
        let request = path_request(&session, &source);
        let mut abandoned = Box::pin(session.execute_request(request));
        poll_until_input(abandoned.as_mut(), &calls, before_drop + 1);
        assert_eq!(calls.load(Ordering::SeqCst), before_drop + 1);
        let abandoned_target = OBSERVED_TARGET
            .with(|target| target.get())
            .expect("abandoned target identity");
        runmat_gc::gc_collect_major().expect("collect while abandoned invocation is still live");
        assert!(runmat_gc::gc_clone_value(&abandoned_target).is_ok());
        drop(abandoned);
        runmat_gc::gc_collect_major().expect("collect after dropping managed pending invocation");
        assert!(
            runmat_gc::gc_clone_value(&abandoned_target).is_err(),
            "dropping the whole invocation must release its temporary capture"
        );
        assert_eq!(calls.load(Ordering::SeqCst), before_drop + 1);
        ready.set(true);
        let request = path_request(&session, &source);
        let response = futures::executor::block_on(session.execute_request(request));
        let recovered = response
            .result
            .expect("recover after dropped managed invocation");
        assert!(recovered.used_jit, "recovery must execute native");
        assert!(
            recovered
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.severity != crate::abi::DiagnosticSeverity::Error),
            "{:?}",
            recovered.diagnostics
        );
        assert_eq!(calls.load(Ordering::SeqCst), before_drop + 2);
    });
}
