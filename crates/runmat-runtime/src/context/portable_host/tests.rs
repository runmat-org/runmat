use super::*;
use crate::context::{RuntimeContext, RuntimeServicePorts};
use runmat_execution::console::{ConsoleCompletion, ConsoleRejection};
use std::rc::Rc;

fn context(host: Rc<PortableConsoleHost>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_host(host))
}

#[test]
fn shared_admission_keeps_original_child_prefix_scoped_after_parent_finish() {
    use crate::context::{
        RuntimeConsoleAdmission, RuntimeConsoleScopeIdentity, RuntimeConsoleScopeRef,
        RuntimeRetainedConsoleSource,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct TwoRecords(AtomicUsize);
    impl RuntimeConsoleAdmission for TwoRecords {
        fn max_scopes(&self) -> u32 {
            4
        }
        fn reserve(
            &self,
            _: RuntimeConsoleScopeIdentity,
            limits: ConsoleLimits,
        ) -> Result<(), String> {
            limits.validate().map_err(|error| error.to_string())
        }
        fn admit_record(
            &self,
            _: RuntimeConsoleScopeIdentity,
            _: ConsoleStream,
            _: &str,
        ) -> Result<(), String> {
            self.0
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |count| {
                    (count < 2).then_some(count + 1)
                })
                .map(|_| ())
                .map_err(|_| "shared record budget exhausted".into())
        }
        fn settle(
            &self,
            _: RuntimeConsoleScopeIdentity,
            transcript: &ConsoleTranscript,
        ) -> Result<(), String> {
            transcript.validate().map_err(|error| error.to_string())
        }
    }
    struct OriginalChild {
        identity: RuntimeConsoleScopeIdentity,
        transcript: Box<ConsoleTranscript>,
    }
    impl RuntimeRetainedConsoleSource for OriginalChild {
        fn source_identity(
            &self,
        ) -> Option<(runmat_execution::ExecutionScopeId, runmat_execution::PoolId)> {
            Some((self.identity.scope_id, self.identity.pool_id))
        }
        fn visit(
            &self,
            visitor: &mut dyn FnMut(RuntimeConsoleScopeRef<'_>) -> Result<(), String>,
        ) -> Result<(), String> {
            visitor(RuntimeConsoleScopeRef {
                identity: self.identity,
                transcript: &self.transcript,
            })
        }
    }
    let identity = |name: &[u8]| RuntimeConsoleScopeIdentity {
        scope_id: runmat_execution::ExecutionScopeId::derive(&[name]),
        pool_id: runmat_execution::PoolId::derive(&[name]),
        task_id: runmat_execution::TaskId::derive(&[name]),
        attempt_id: runmat_execution::identity::AttemptId::derive(&[name]),
    };
    let owner: Arc<dyn RuntimeConsoleAdmission> = Arc::new(TwoRecords(AtomicUsize::new(0)));
    let limits = ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    };
    let parent = PortableConsoleHost::new(limits).unwrap();
    parent
        .install_console_admission(owner.clone(), identity(b"parent"))
        .unwrap();
    parent.console(ConsoleStream::Stdout, "parent".into());
    let child = PortableConsoleHost::new(limits).unwrap();
    let child_identity = identity(b"child");
    child
        .install_console_admission(owner, child_identity)
        .unwrap();
    child.console(ConsoleStream::Stderr, "original child prefix".into());
    child.console(ConsoleStream::Stdout, "rejected".into());
    assert_eq!(
        child.check_execution().unwrap_err().identifier(),
        Some("RunMat:console:CaptureRejected")
    );
    let transcript = Box::new(child.finish(true).unwrap());
    let original_pointer = transcript.as_ref() as *const ConsoleTranscript;
    assert_eq!(transcript.completion(), ConsoleCompletion::Interrupted);
    let source = Arc::new(OriginalChild {
        identity: child_identity,
        transcript,
    });
    parent.retain_console_source(source.clone()).unwrap();
    parent.retain_console_source(source).unwrap(); // same driver is retained once
    let retained = parent.retained_console_source();
    let root = parent.finish(false).unwrap();
    drop(parent);
    assert_eq!(root.records()[0].sequence, 1);
    let mut seen = 0;
    retained
        .visit(&mut |scope| {
            seen += 1;
            assert_eq!(scope.identity, child_identity);
            assert_eq!(
                scope.transcript as *const ConsoleTranscript,
                original_pointer
            );
            assert_eq!(scope.transcript.records().len(), 1);
            assert_eq!(scope.transcript.records()[0].sequence, 1); // source-local, not globally renumbered
            assert_eq!(
                scope.transcript.records()[0].text.as_ref(),
                "original child prefix"
            );
            assert_eq!(
                scope.transcript.completion(),
                ConsoleCompletion::Interrupted
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, 1);
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
    crate::console::record_console_output(ConsoleStream::Stdout, "first")
        .expect("retain first console write");
    let guard = crate::console::begin_capture().expect("capture admitted");
    crate::console::record_console_output(ConsoleStream::Stderr, "hidden")
        .expect("capture hidden console write");
    assert_eq!(guard.finish().expect("capture finished"), "hidden");
    crate::console::record_clear_screen().expect("retain clear-screen record");
    crate::console::record_console_output(ConsoleStream::Stderr, "last")
        .expect("retain last console write");
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
        crate::console::record_console_output(ConsoleStream::Stdout, label)
            .expect("retain invocation stdout");
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
        crate::console::record_console_output(ConsoleStream::Stderr, label)
            .expect("retain invocation stderr");
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
        crate::console::record_console_output(ConsoleStream::Stdout, "unrequested")
            .expect("discard unrequested output without buffering");
    }
    assert!(crate::console::take_thread_buffer().is_empty());
    assert_eq!(host.console_limits(), None);
    host.check_execution().unwrap();
}
