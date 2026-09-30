use super::*;

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
    order_counter = order_counter + 1;
    output = struct('second', @(next) value.x + next);
end
global order_counter;
order_counter = 0;
s = struct('first', @first);
result = s.first(PendingHandle()).second(input('next: '));
assert(result == 10);
"#;

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
        let _cwd_lock = super::super::cwd_lock();
        let directory = tempfile::TempDir::new().expect("class fixture directory");
        std::fs::write(directory.path().join("PendingHandle.m"), HANDLE_CLASS)
            .expect("write handle class");
        let source = directory.path().join("main.m");
        std::fs::write(&source, SOURCE).expect("write invocation script");
        let _cwd = super::super::push_cwd(directory.path());

        let mut session = hot_session();
        let ready = Arc::new(AtomicBool::new(true));
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
                    if runmat_runtime::workspace::session::global_value("order_counter")
                        == Some(Value::Num(1.0))
                    {
                        prefix_seen.store(true, Ordering::SeqCst);
                    }
                    std::future::poll_fn(|_| {
                        if ready.load(Ordering::SeqCst) {
                            Poll::Ready(())
                        } else {
                            Poll::Pending
                        }
                    })
                    .await;
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

        ready.store(false, Ordering::SeqCst);
        prefix_seen.store(false, Ordering::SeqCst);
        let before = calls.load(Ordering::SeqCst);
        let request = path_request(&session, &source);
        let mut execution = Box::pin(session.execute_request(request));
        let waker = futures::task::noop_waker();
        assert!(Future::poll(execution.as_mut(), &mut Context::from_waker(&waker)).is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
        assert!(prefix_seen.load(Ordering::SeqCst));
        runmat_gc::gc_collect_major().expect("collect while managed receiver is pending");
        ready.store(true, Ordering::SeqCst);
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
            session.get_variables().get("result"),
            Some(&Value::Num(10.0))
        );

        ready.store(false, Ordering::SeqCst);
        let before_drop = calls.load(Ordering::SeqCst);
        let request = path_request(&session, &source);
        let mut abandoned = Box::pin(session.execute_request(request));
        assert!(Future::poll(abandoned.as_mut(), &mut Context::from_waker(&waker)).is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), before_drop + 1);
        drop(abandoned);
        runmat_gc::gc_collect_major().expect("collect after dropping managed pending invocation");
        assert_eq!(calls.load(Ordering::SeqCst), before_drop + 1);
        ready.store(true, Ordering::SeqCst);
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
