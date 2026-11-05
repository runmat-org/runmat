use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView, ProviderDispatchStats,
    ProviderTelemetry, ThreadProviderGuard,
};
use runmat_execution::profiling::{current_command_profiling, CommandThreadCpuSampler};
use runmat_execution::profiling::{CommandProfilingRoot, ProviderObservationTerminal};
use runmat_runtime::context::{
    IsolatedMemoryPreferenceService, RuntimeContext, RuntimeServiceFuture, RuntimeUserPathService,
    UserPathMutation, UserPathServiceError, UserPathSnapshot,
};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::task::Poll;
use std::time::Duration;

struct DiagnosticProvider {
    resets: AtomicUsize,
    snapshots: AtomicUsize,
}

static PROVIDER: DiagnosticProvider = DiagnosticProvider {
    resets: AtomicUsize::new(0),
    snapshots: AtomicUsize::new(0),
};

impl AccelProvider for DiagnosticProvider {
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("profiling fixture does not upload"))
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow::anyhow!("profiling fixture does not download")) })
    }

    fn device_info(&self) -> String {
        "profiling diagnostic fixture".into()
    }

    fn device_id(&self) -> u32 {
        900_156
    }

    fn telemetry_snapshot(&self) -> ProviderTelemetry {
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        ProviderTelemetry {
            fused_elementwise: ProviderDispatchStats {
                count: 37,
                total_wall_time_ns: 900_000_000,
            },
            ..ProviderTelemetry::default()
        }
    }

    fn reset_telemetry(&self) {
        self.resets.fetch_add(1, Ordering::Relaxed);
    }
}

fn session() -> RunMatSession {
    let mut session = RunMatSession::with_options(false, false).expect("Core session");
    session
        .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
        .expect("isolated settings");
    session
}

struct StepCpuClock(AtomicUsize);

impl CommandThreadCpuSampler for StepCpuClock {
    fn sample_current_thread(&self) -> Option<Duration> {
        let read = self.0.fetch_add(1, Ordering::SeqCst).checked_add(1)?;
        Some(Duration::from_nanos(
            u64::try_from(read).ok()?.checked_mul(10)?,
        ))
    }
}

struct ObservingUserPath {
    inner: Rc<dyn RuntimeUserPathService>,
    selected_reads: Arc<AtomicUsize>,
}

impl RuntimeUserPathService for ObservingUserPath {
    fn read(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        let read = self.inner.read(runtime);
        let selected_reads = Arc::clone(&self.selected_reads);
        Box::pin(async move {
            assert!(
                current_command_profiling().is_some(),
                "hydration is inside command extent"
            );
            selected_reads.fetch_add(1, Ordering::SeqCst);
            read.await
        })
    }

    fn commit(
        &self,
        runtime: RuntimeContext,
        mutation: UserPathMutation,
    ) -> RuntimeServiceFuture<Result<UserPathSnapshot, UserPathServiceError>> {
        self.inner.commit(runtime, mutation)
    }
}

#[test]
fn source_hydration_and_cleanup_share_the_borrowed_cpu_command_extent() {
    let mut session = session();
    let reads = Arc::new(AtomicUsize::new(0));
    let old_user_path = session
        .runtime_context
        .service_ports()
        .user_path()
        .cloned()
        .expect("installed settings authority");
    let ports = session
        .runtime_context
        .service_ports()
        .clone()
        .with_user_path(Rc::new(ObservingUserPath {
            inner: old_user_path,
            selected_reads: Arc::clone(&reads),
        }));
    session.runtime_context = session.runtime_context.clone().with_service_ports(ports);
    let clock = Arc::new(StepCpuClock(AtomicUsize::new(0)));
    let root = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    session.inherited_command_profiling = Some(root.receipt());
    let cleanup_calls = Arc::new(AtomicUsize::new(0));
    session.install_async_input_handler({
        let cleanup_calls = Arc::clone(&cleanup_calls);
        move |_| {
            let cleanup_calls = Arc::clone(&cleanup_calls);
            async move {
                assert!(
                    current_command_profiling().is_some(),
                    "cleanup keeps original command"
                );
                cleanup_calls.fetch_add(1, Ordering::SeqCst);
                Ok(crate::execution::InputResponse::Line("0".into()))
            }
        }
    });
    let source = request(&session, "<cpu-command-extent>",
        "cleanup_scope();\nfunction cleanup_scope()\n c = onCleanup(@() input('cpu cleanup: '));\nend\n");
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let response = host.block_on(session.execute_request(source));
    let outcome = response.result.expect("actual source command");
    assert!(
        outcome.profiling.is_none(),
        "borrowed child cannot seal its caller"
    );
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
    assert!(clock.0.load(Ordering::SeqCst) >= 2);
    assert!(root.seal().cpu_thread_attributed.is_some());
}

fn request(session: &RunMatSession, name: &str, source: &str) -> crate::abi::ExecutionRequest {
    crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: name.into(),
            text: source.into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    )
}

fn assert_measured_total(response: crate::abi::ExecutionResponse) {
    let outcome = response.result.expect("actual Core execution outcome");
    assert!(
        outcome
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.severity != crate::abi::DiagnosticSeverity::Error),
        "unexpected diagnostics: {:?}",
        outcome.diagnostics
    );
    let profiling = outcome
        .profiling
        .as_ref()
        .expect("measured command summary");
    assert!(
        profiling.total_ms >= outcome.execution_time_ms,
        "outer command wall time includes inner execution and cleanup"
    );
    assert_eq!(profiling.cpu_ms, None);
    assert_eq!(profiling.gpu_ms, None);
    assert_eq!(profiling.kernel_count, None);
}

#[test]
fn interleaved_sessions_do_not_reset_or_borrow_shared_provider_diagnostics() {
    let mut first = session();
    let mut second = session();
    let first_request = request(&first, "<profiling-pending>", "pause(0.05); a = 11;");
    let second_request = request(&second, "<profiling-peer>", "b = 22;");
    let resets_before = PROVIDER.resets.load(Ordering::Relaxed);
    let snapshots_before = PROVIDER.snapshots.load(Ordering::Relaxed);
    let provider_guard = ThreadProviderGuard::set(Some(&PROVIDER));
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("single-lane test host");
    host.block_on(async {
        let mut pending = Box::pin(first.execute_request(first_request));
        futures::future::poll_fn(|context| match pending.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("pause command must genuinely suspend"),
        })
        .await;
        let peer = second.execute_request(second_request).await;
        assert_measured_total(peer);
        assert_measured_total(pending.await);
    });
    assert_eq!(PROVIDER.resets.load(Ordering::Relaxed), resets_before);
    assert_eq!(PROVIDER.snapshots.load(Ordering::Relaxed), snapshots_before);

    drop(provider_guard);
}

#[test]
fn real_test_child_borrows_the_callers_command_without_sealing_it() {
    let parent = session();
    let root = CommandProfilingRoot::new();
    let caller = parent
        .runtime_context()
        .clone()
        .with_command_profiling(root.receipt());
    let mut child = RunMatSession::with_options_on_parent_driver(
        false,
        false,
        &parent.local_task_driver().child_scope_factory(),
        &caller,
    )
    .expect("real child on parent driver");
    let snapshot = runmat_test::discovery::FrozenTestRunSnapshot::freeze(
        runmat_execution::Digest::sha256(b"profiling-child-graph").to_string(),
        "sha256:profiling-child-source",
        crate::program_environment(parent.compat_mode()),
        runmat_execution::Digest::sha256(b"profiling-child-config").to_string(),
        vec![runmat_test::discovery::SavedRunSource {
            owner_identity: "path:profiling-child".into(),
            relative_path: "childTest.m".into(),
            content: "%% passing section\nassert(2 + 2 == 4)\n".into(),
        }],
        Vec::new(),
    )
    .expect("frozen test source");
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("single-lane test host");
    let (run, child_response) = host.block_on(async {
        let run =
            parent
                .local_task_driver()
                .participate_foreground(child.run_test_snapshot(
                    &snapshot,
                    &runmat_test::descriptor::TestSelector::default(),
                ))
                .await
                .expect("parent drives child")
                .expect("child snapshot execution");
        let child_request = request(&child, "<profiling-child-source>", "child_value = 9;");
        let response = parent
            .local_task_driver()
            .participate_foreground(child.execute_request(child_request))
            .await
            .expect("parent drives child source request");
        (run, response)
    });
    assert_eq!(run.results.len(), 1);
    assert_eq!(
        run.results[0].state.disposition,
        runmat_test::result::TerminalDisposition::Passed
    );
    let child_outcome = child_response.result.expect("actual child source request");
    assert!(
        child_outcome.profiling.is_none(),
        "borrowed command cannot publish a premature parent snapshot"
    );
    child
        .runtime_context()
        .command_profiling()
        .expect("typed inherited origin")
        .begin_provider_operation()
        .expect("child did not seal its caller")
        .complete(ProviderObservationTerminal::Missing);
    let snapshot = root.seal();
    assert_eq!(snapshot.observations.expect("valid parent").missing, 1);
    assert!(child
        .runtime_context()
        .command_profiling()
        .expect("borrowed receipt retained")
        .begin_provider_operation()
        .is_none());
}

#[test]
fn standalone_executable_admits_and_seals_a_fresh_command_after_source_execution() {
    let mut session = session();
    let first = request(&session, "<profiling-before-unit>", "seed = 3;");
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("single-lane test host");
    host.block_on(async {
        assert_measured_total(session.execute_request(first).await);
        let previous = session
            .runtime_context()
            .command_profiling()
            .expect("source command origin")
            .clone();
        assert!(previous.begin_provider_operation().is_none());
        let unit = session
            .compile_executable_unit(
                crate::ExecutableSource::new(
                    "fixture",
                    "answer.m",
                    "function y = answer(); y = input('standalone: '); end\n",
                ),
                None,
            )
            .await
            .expect("real immutable unit");
        let admitted = Arc::new(AtomicUsize::new(0));
        session.install_async_input_handler({
            let admitted = Arc::clone(&admitted);
            move |request| {
                let admitted = Arc::clone(&admitted);
                async move {
                    assert_eq!(request.prompt, "standalone: ");
                    runmat_execution::profiling::current_command_profiling()
                        .expect("standalone invocation selected a command")
                        .begin_provider_operation()
                        .expect("standalone origin is open, not the prior sealed source origin")
                        .complete(ProviderObservationTerminal::Missing);
                    admitted.fetch_add(1, Ordering::SeqCst);
                    Ok(crate::execution::InputResponse::Line("7".into()))
                }
            }
        });
        let output = session
            .invoke_executable(
                &unit,
                crate::ProcedureInvocation {
                    target: crate::ProcedureTarget::Function("answer".into()),
                    arguments: crate::ProcedureArguments::Source(Vec::new()),
                    requested_outputs: 1,
                },
                &crate::InvocationControl::default(),
            )
            .await
            .expect("standalone invocation");
        assert_eq!(output.as_slice(), &[runmat_value::Value::Num(7.0)]);
        assert_eq!(admitted.load(Ordering::SeqCst), 1);
        let standalone = session
            .runtime_context()
            .command_profiling()
            .expect("standalone command origin");
        assert!(standalone.begin_provider_operation().is_none());
        assert!(previous.begin_provider_operation().is_none());
    });
}

#[test]
fn cleanup_callback_gate_prevents_command_publication_until_drained() {
    let mut session = session();
    let (entered_tx, entered_rx) = futures::channel::oneshot::channel::<()>();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    let (release_tx, release_rx) = futures::channel::oneshot::channel::<()>();
    let release_rx = Arc::new(Mutex::new(Some(release_rx)));
    session.install_async_input_handler(move |request| {
        assert_eq!(request.prompt, "cleanup gate: ");
        entered_tx
            .lock()
            .expect("entry signal")
            .take()
            .expect("cleanup callback must execute once")
            .send(())
            .expect("command observer remains live");
        let release = release_rx
            .lock()
            .expect("release gate")
            .take()
            .expect("cleanup input is not replayed");
        async move {
            release
                .await
                .map_err(|_| "cleanup input abandoned".to_string())?;
            Ok(crate::execution::InputResponse::Line("0".into()))
        }
    });
    let request = request(
        &session,
        "<profiling-cleanup-gate>",
        r#"
global cleanup_trace;
cleanup_trace = 0;
cleanup_scope();
assert(cleanup_trace == 3);
escaped = escaped_scope();
alias = escaped;
clear escaped;
assert(cleanup_trace == 3);
clear alias;
assert(cleanup_trace == 4);
overwritten = escaped_scope();
overwritten_alias = overwritten;
overwritten = [];
assert(cleanup_trace == 4);
overwritten_alias = [];
assert(cleanup_trace == 5);
caught_original = false;
try
    failing_scope();
catch original
    assert(strcmp(original.identifier, 'Cleanup:Original'));
    caught_original = true;
end
assert(caught_original);
function cleanup_scope()
    c = onCleanup(@cleanup_body);
end
function cleanup_body()
    input('cleanup gate: ');
    nested_scope();
    global cleanup_trace;
    assert(cleanup_trace == 2);
    cleanup_trace = 3;
end
function nested_scope()
    global cleanup_trace;
    cleanup_trace = 1;
    c = onCleanup(@nested_done);
end
function nested_done()
    global cleanup_trace;
    assert(cleanup_trace == 1);
    cleanup_trace = 2;
end
function c = escaped_scope()
    c = onCleanup(@escaped_done);
end
function escaped_done()
    global cleanup_trace;
    cleanup_trace = cleanup_trace + 1;
end
function failing_scope()
    c = onCleanup(@() error('Cleanup:Secondary', 'cleanup failure'));
    error('Cleanup:Original', 'original failure');
end
"#,
    );
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("single-lane test host");
    host.block_on(async {
        let mut command = Box::pin(session.execute_request(request));
        tokio::time::timeout(Duration::from_secs(10), async {
            match futures::future::select(command.as_mut(), entered_rx).await {
                futures::future::Either::Right((Ok(()), _pending_command)) => {}
                futures::future::Either::Right((Err(_), _pending_command)) => {
                    panic!("cleanup entry observer retired before callback")
                }
                futures::future::Either::Left((response, _pending_entry)) => {
                    panic!("command published before cleanup callback: {response:?}")
                }
            }
        })
        .await
        .expect("cleanup callback admission watchdog");
        release_tx.send(()).expect("cleanup callback still waiting");
        let response = command.await;
        assert_measured_total(response);
    });
}

#[test]
fn cancelled_source_command_seals_its_owner_and_later_command_admits_fresh_work() {
    for cancel in [false, true] {
        cancelled_source_command_preserves_workspace_and_seals_owner(cancel);
    }
}

fn cancelled_source_command_preserves_workspace_and_seals_owner(cancel: bool) {
    let mut session = session();
    let (entered_tx, entered_rx) = futures::channel::oneshot::channel::<()>();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    let (_unreleased_tx, unreleased_rx) = futures::channel::oneshot::channel::<()>();
    let unreleased_rx = Arc::new(Mutex::new(Some(unreleased_rx)));
    session.install_async_input_handler(move |_| {
        entered_tx
            .lock()
            .expect("entry signal")
            .take()
            .expect("one pending input")
            .send(())
            .expect("observer live");
        let pending = unreleased_rx
            .lock()
            .expect("pending input")
            .take()
            .expect("one pending input");
        async move {
            pending.await.map_err(|_| "abandoned input".to_string())?;
            Ok(crate::execution::InputResponse::Line("1".into()))
        }
    });
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("single-lane test host");
    host.block_on(async {
        let seed = request(&session, "<profiling-original>", "original = [11 12];");
        assert_measured_total(session.execute_request(seed).await);
        let original_storage = {
            let workspace = session
                .base_workspace_snapshot()
                .expect("original workspace");
            let Some(Value::Tensor(original)) = workspace.get("original") else {
                panic!("original tensor was not published")
            };
            original.as_f64_slice().expect("double original").as_ptr()
        };
        let first = request(
            &session,
            "<profiling-cancelled>",
            "prefix = 91; x = input('first: '); after_suspend = 99;",
        );
        let interrupt = session.interrupt_handle();
        let mut pending = Box::pin(session.execute_request(first));
        tokio::time::timeout(Duration::from_secs(10), async {
            match futures::future::select(pending.as_mut(), entered_rx).await {
                futures::future::Either::Right((Ok(()), _pending_command)) => {}
                futures::future::Either::Right((Err(_), _pending_command)) => {
                    panic!("pending input observer retired")
                }
                futures::future::Either::Left((response, _pending_entry)) => {
                    panic!("command did not suspend: {response:?}")
                }
            }
        })
        .await
        .expect("pending command watchdog");
        if cancel {
            interrupt.cancel(runmat_execution::CancellationReason::User);
            let response = pending.as_mut().await;
            match response.result {
                Ok(outcome) => assert!(outcome.diagnostics.iter().any(|diagnostic| {
                    diagnostic.code == "RunMat:ExecutionCancelled"
                        || diagnostic.code == "ExecutionCancelled"
                })),
                Err(crate::RunError::Runtime(error)) => assert!(matches!(
                    error.identifier(),
                    Some("RunMat:ExecutionCancelled" | "ExecutionCancelled")
                )),
                Err(error) => panic!("unexpected cancellation error: {error}"),
            }
        }
        drop(pending);
        {
            let workspace = session
                .base_workspace_snapshot()
                .expect("retained workspace");
            let Some(Value::Tensor(original)) = workspace.get("original") else {
                panic!("abandonment lost original workspace owner")
            };
            assert_eq!(original.as_f64_slice().unwrap().as_ptr(), original_storage);
            assert_eq!(original.shape(), &[1, 2]);
            assert_eq!(original.as_f64_slice(), Some([11.0, 12.0].as_slice()));
            assert_eq!(workspace.get("prefix"), Some(&Value::Num(91.0)));
            assert!(workspace.get("x").is_none());
            assert!(workspace.get("after_suspend").is_none());
        }
        // A completed error returns its moved owner through the ordinary rich
        // outcome; Drop must retire that same exact frame without leaving its
        // register file rooted in the legacy thread-local completion handoff.
        session
            .runtime_context()
            .clone()
            .scope(async {
                assert!(runmat_vm::take_updated_workspace_state().is_none());
                assert!(runmat_vm::take_updated_workspace_assigned_report().is_none());
            })
            .await;
        let old = session
            .runtime_context()
            .command_profiling()
            .expect("first command origin")
            .clone();
        assert!(
            old.begin_provider_operation().is_none(),
            "drop sealed original root"
        );
        session.install_async_input_handler(|_| async {
            runmat_execution::profiling::current_command_profiling()
                .expect("fresh command selected")
                .begin_provider_operation()
                .expect("fresh root admits work")
                .complete(ProviderObservationTerminal::Missing);
            Ok(crate::execution::InputResponse::Line("2".into()))
        });
        let later = request(&session, "<profiling-later>", "y = input('later: ');");
        assert_measured_total(session.execute_request(later).await);
        assert!(
            old.begin_provider_operation().is_none(),
            "old origin remains sealed"
        );
    });
}
