use super::*;
use crate::context::{RuntimeContext, RuntimeServicePorts};
use runmat_execution::console::{ConsoleCompletion, ConsoleRejection};
use std::rc::Rc;

fn context(host: Rc<PortableConsoleHost>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_host(host))
}

#[test]
fn retaining_host_has_no_duplicate_buffer_and_evalc_remains_hidden() {
    let host = Rc::new(
        PortableConsoleHost::new(ConsoleLimits {
            max_records: 3,
            max_text_bytes: 100,
        })
        .unwrap(),
    );
    let runtime = context(host.clone());
    let _scope = runtime.enter();
    crate::console::record_console_output(ConsoleStream::Stdout, "first");
    let guard = crate::console::begin_capture().expect("capture admitted");
    crate::console::record_console_output(ConsoleStream::Stderr, "hidden");
    assert_eq!(guard.finish().expect("capture finished"), "hidden");
    crate::console::record_clear_screen();
    crate::console::record_console_output(ConsoleStream::Stderr, "last");
    assert!(crate::console::take_thread_buffer().is_empty());
    let transcript = host.finish(true).unwrap();
    transcript.validate().unwrap();
    assert_eq!(
        transcript
            .records()
            .iter()
            .map(|r| r.text.as_ref())
            .collect::<Vec<_>>(),
        vec!["first", "", "last"]
    );
    assert_eq!(transcript.completion(), ConsoleCompletion::Complete);
    assert!(host.finish(true).is_err());
    assert_eq!(
        host.check_execution().unwrap_err().identifier(),
        Some("RunMat:console:Sealed")
    );
}

#[test]
fn portable_host_rejects_queued_line_and_keypress_without_consuming_them() {
    let host = Rc::new(
        PortableConsoleHost::new(ConsoleLimits {
            max_records: 2,
            max_text_bytes: 100,
        })
        .unwrap(),
    );
    let runtime = context(host);
    let _scope = runtime.enter();
    crate::interaction::push_queued_response(Ok(crate::interaction::InteractionResponse::Line(
        "ambient".into(),
    )));
    let line = futures::executor::block_on(crate::interaction::request_line_async("", false));
    assert_eq!(
        line.unwrap_err().identifier(),
        Some("RunMat:interaction:NoninteractiveHost")
    );
    {
        let interactive = runtime
            .clone()
            .with_service_ports(RuntimeServicePorts::default());
        let _interactive_scope = interactive.enter();
        assert_eq!(
            futures::executor::block_on(crate::interaction::request_line_async("", false)).unwrap(),
            "ambient"
        );
    }
    crate::interaction::push_queued_response(Ok(crate::interaction::InteractionResponse::KeyPress));
    let key = futures::executor::block_on(crate::interaction::wait_for_key_async(""));
    assert_eq!(
        key.unwrap_err().identifier(),
        Some("RunMat:interaction:NoninteractiveHost")
    );
}

#[test]
fn builtin_capture_failure_is_sticky_without_marking_user_cancellation() {
    let host = Rc::new(
        PortableConsoleHost::new(ConsoleLimits {
            max_records: 1,
            max_text_bytes: 1,
        })
        .unwrap(),
    );
    let runtime = context(host.clone());
    let _scope = runtime.enter();
    let result = futures::executor::block_on(crate::dispatcher::call_builtin_async(
        "disp",
        &[runmat_value::Value::Num(123.0)],
    ));
    assert_eq!(
        result.unwrap_err().identifier(),
        Some("RunMat:console:CaptureRejected")
    );
    assert!(!runtime.cancellation().is_cancelled());
    assert!(
        futures::executor::block_on(crate::dispatcher::call_builtin_async(
            "cos",
            &[runmat_value::Value::Num(0.0)]
        ))
        .is_err()
    );
    let transcript = host.finish(true).unwrap();
    assert!(transcript.records().is_empty());
    assert_eq!(
        transcript.completion(),
        ConsoleCompletion::Rejected {
            reason: ConsoleRejection::TextByteLimit,
        }
    );
}

#[test]
fn interleaved_invocations_retain_only_their_original_console_writes() {
    let first = Rc::new(
        PortableConsoleHost::new(ConsoleLimits {
            max_records: 2,
            max_text_bytes: 100,
        })
        .unwrap(),
    );
    let second = Rc::new(
        PortableConsoleHost::new(ConsoleLimits {
            max_records: 2,
            max_text_bytes: 100,
        })
        .unwrap(),
    );
    let first_context = context(first.clone());
    let second_context = context(second.clone());
    async fn writes(label: &str) {
        crate::console::record_console_output(ConsoleStream::Stdout, label);
        let mut yielded = false;
        futures::future::poll_fn(|context| {
            if yielded {
                std::task::Poll::Ready(())
            } else {
                yielded = true;
                context.waker().wake_by_ref();
                std::task::Poll::Pending
            }
        })
        .await;
        crate::console::record_console_output(ConsoleStream::Stderr, label);
    }
    futures::executor::block_on(futures::future::join(
        first_context.scope(writes("first")),
        second_context.scope(writes("second")),
    ));
    for (host, label) in [(first, "first"), (second, "second")] {
        let transcript = host.finish(true).unwrap();
        assert_eq!(
            transcript
                .records()
                .iter()
                .map(|record| record.text.as_ref())
                .collect::<Vec<_>>(),
            vec![label, label]
        );
    }
}

#[test]
fn source_sink_failure_is_sticky_and_cannot_claim_complete_delivery() {
    struct FailedSink;
    impl PortableConsoleSink for FailedSink {
        fn emit(&self, _record: &ConsoleRecord) -> Result<(), String> {
            Err("original transport failure".into())
        }
    }
    let host = PortableConsoleHost::with_sink(
        ConsoleLimits {
            max_records: 4,
            max_text_bytes: 100,
        },
        Some(Rc::new(FailedSink)),
    )
    .unwrap();
    host.console(ConsoleStream::Stdout, "original".into());
    host.console(
        ConsoleStream::Stderr,
        "must not append after failure".into(),
    );
    assert!(host.check_execution().is_err());
    let retained = host.finish(true).unwrap();
    assert_eq!(retained.records().len(), 1);
    assert_eq!(retained.records()[0].text.as_ref(), "original");
    assert_eq!(retained.completion(), ConsoleCompletion::Interrupted);
}

#[test]
fn explicitly_unrequested_console_neither_retains_an_unbounded_buffer_nor_claims_capture() {
    let host = Rc::new(NonCapturingPortableHost);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_host(host.clone()));
    let _scope = runtime.enter();
    crate::console::reset_thread_buffer();
    for _ in 0..1000 {
        crate::console::record_console_output(ConsoleStream::Stdout, "unrequested");
    }
    assert!(crate::console::take_thread_buffer().is_empty());
    assert_eq!(host.console_limits(), None);
    host.check_execution().unwrap();
}
