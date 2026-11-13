use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use std::future::Future;
use std::rc::Rc;
use std::task::{Context, Poll};

struct ManagedParallel(ManagedHostEffectLedger);

impl RuntimeParallelService for ManagedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn entry(text: &str, timestamp_ms: u64) -> ConsoleEntry {
    ConsoleEntry {
        stream: ConsoleStream::Stdout,
        text: text.into(),
        timestamp_ms,
    }
}

#[test]
fn append_thread_buffer_orders_entries_by_timestamp_stably() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let _scope = runtime.enter();
    reset_thread_buffer();
    append_thread_buffer(vec![entry("late", 20)]).expect("first append");
    append_thread_buffer(vec![entry("early", 10), entry("same-time", 20)]).expect("second append");

    let entries = take_thread_buffer();
    let texts = entries
        .into_iter()
        .map(|entry| entry.text)
        .collect::<Vec<_>>();
    assert_eq!(texts, vec!["early", "late", "same-time"]);
}

#[test]
fn statement_struct_output_uses_the_command_window_field_renderer() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let _scope = runtime.enter();
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(3.0));
    fields.insert(
        "algorithm",
        Value::CharArray(runmat_value::CharArray::new_row("quasi-newton")),
    );
    fields.insert(
        "identifier",
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
    );
    let value = Value::Struct(fields.finish());
    let body =
        "    iterations: 3\n    algorithm: 'quasi-newton'\n    identifier: 18446744073709551615";
    for (label, expected) in [
        (Some("output"), format!("output =\n{body}\n")),
        (None, format!("{body}\n")),
    ] {
        reset_thread_buffer();
        record_value_output(label, &value).unwrap();
        let entries = take_thread_buffer();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text.as_str(), expected);
        assert_eq!(take_last_value_output(), Some(value.clone()));
    }
}

#[test]
fn pending_sibling_captures_do_not_exchange_streams_or_drop_state() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let make = |label: &'static str, release: futures::channel::oneshot::Receiver<()>| {
        runtime.scope(async move {
            let outer = begin_capture().expect("outer capture");
            record_console_output(ConsoleStream::Stdout, format!("{label}:outer"))
                .expect("outer output");
            let nested = begin_capture().expect("nested capture");
            record_console_output(ConsoleStream::Stderr, format!("{label}:nested"))
                .expect("nested output");
            release.await.expect("release sibling");
            record_console_output(ConsoleStream::Stderr, format!("{label}:resumed"))
                .expect("resumed output");
            let nested = nested.finish_entries();
            record_console_output(ConsoleStream::Stdout, format!("{label}:after"))
                .expect("after output");
            let outer = outer.finish_entries();
            (outer, nested)
        })
    };
    let (first_tx, first_rx) = futures::channel::oneshot::channel();
    let (second_tx, second_rx) = futures::channel::oneshot::channel();
    let mut first = Box::pin(make("first", first_rx));
    let mut second = Box::pin(make("second", second_rx));
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert!(matches!(first.as_mut().poll(&mut cx), Poll::Pending));
    assert!(matches!(second.as_mut().poll(&mut cx), Poll::Pending));
    first_tx.send(()).expect("resume first");
    let Poll::Ready((outer, nested)) = first.as_mut().poll(&mut cx) else {
        panic!("first completes")
    };
    assert_eq!(
        outer
            .iter()
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>(),
        ["first:outer", "first:after"]
    );
    assert_eq!(
        nested
            .iter()
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>(),
        ["first:nested", "first:resumed"]
    );
    drop(second); // cancellation retires its nested and outer guards
    drop(second_tx);
    {
        let _scope = runtime.enter();
        record_console_output(ConsoleStream::Stdout, "post-cancel").expect("post-cancel output");
        assert!(take_thread_buffer()
            .iter()
            .any(|entry| entry.text == "post-cancel"));
    }
    let unrelated = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let _scope = unrelated.enter();
    record_console_output(ConsoleStream::Stdout, "unrelated").expect("unrelated output");
    assert!(take_thread_buffer()
        .iter()
        .any(|entry| entry.text == "unrelated"));
}

#[test]
fn capture_diverts_console_text_from_thread_buffer() {
    let _lock = runmat_filesystem::provider_override_lock();
    set_diary_enabled(false);
    reset_thread_buffer();
    let capture = begin_capture().expect("capture");
    record_console_line(ConsoleStream::Stdout, "inside").expect("inside output");
    let text = capture.finish().expect("ordinary capture text");
    record_console_line(ConsoleStream::Stdout, "outside").expect("outside output");

    assert_eq!(text, "inside\n");
    let entries = take_thread_buffer();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].text, "outside\n");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn managed_capture_retains_text_frame_and_entries_until_last_owner() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let entries = {
        let _scope = runtime.enter();
        let capture = begin_capture().expect("managed capture frame");
        record_console_output(ConsoleStream::Stdout, "retained").expect("managed captured text");
        let entries = capture.finish_entries();
        assert_eq!(entries[0].text, "retained");
        assert!(entries[0].text.has_managed_backing());
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        append_thread_buffer(entries).expect("admitted batch append");
        let mut external = String::with_capacity(48);
        external.push_str("external");
        append_thread_buffer(vec![ConsoleEntry {
            stream: ConsoleStream::Stdout,
            text: ConsoleText::from(external),
            timestamp_ms: now_ms(),
        }])
        .expect("adopt ordinary source text");
        let entries = take_thread_buffer();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|entry| entry.text.has_managed_backing()));
        assert!(entries.iter().any(|entry| entry.text == "retained"));
        assert!(entries.iter().any(|entry| entry.text == "external"));
        entries
    };
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(entries);
    // The context still owns its admitted, empty capture-stack projection.
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(runtime);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn forwarding_host_uses_the_existing_managed_console_owner_without_a_second_capture() {
    let _lock = runmat_filesystem::provider_override_lock();
    reset_thread_buffer();
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(ManagedParallel(ledger.clone())))
            .with_host(Rc::new(crate::context::PortableForwardingConsoleHost)),
    );
    let retained = {
        let _scope = runtime.enter();
        record_console_output(ConsoleStream::Stdout, "selected console")
            .expect("admitted stock-console text");
        let retained = take_thread_buffer();
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].text, "selected console");
        assert!(retained[0].text.has_managed_backing());
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        retained
    };
    drop(retained);
    drop(runtime);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn statement_emission_stops_before_download_and_wakes_pending_download_without_publication() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, AccelProvider, GpuTensorHandle, GpuTensorStorage,
        HostNumericDataOwned, HostNumericTensorOwned, HostTensorView,
    };
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    struct DownloadProbe {
        owner: TestAllocationOwner,
        calls: AtomicUsize,
        dropped: Arc<AtomicUsize>,
        gate: Mutex<
            Option<futures::channel::oneshot::Receiver<anyhow::Result<HostNumericTensorOwned>>>,
        >,
    }
    struct DownloadDrop(Arc<AtomicUsize>);
    impl Drop for DownloadDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl AccelProvider for DownloadProbe {
        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }
        fn device_info(&self) -> String {
            "controlled statement download; no physical GPU claim".into()
        }
        fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("not an upload fixture")
        }
        fn download<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("numeric download only") })
        }
        fn download_numeric<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let gate = self.gate.lock().unwrap().take();
            let drop = DownloadDrop(self.dropped.clone());
            Box::pin(async move {
                let _drop = drop;
                match gate {
                    Some(gate) => gate.await?,
                    None => Err(crate::build_runtime_error("original download failure")
                        .with_identifier("RunMat:StatementProbe:DownloadFailure")
                        .build()
                        .into()),
                }
            })
        }
    }
    struct WakeCount(AtomicUsize);
    impl futures::task::ArcWake for WakeCount {
        fn wake_by_ref(this: &Arc<Self>) {
            this.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let _provider_lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(DownloadProbe {
        owner: TestAllocationOwner::new(),
        calls: AtomicUsize::new(0),
        dropped: Arc::new(AtomicUsize::new(0)),
        gate: Mutex::new(None),
    }));
    // SAFETY: this fixture deliberately retains the provider for process life
    // and its test allocation owner minted a unique device identity.
    unsafe {
        runmat_accelerate_api::register_device_provider(provider);
    }
    let allocation = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![7.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let value = Value::GpuTensor(allocation.handle);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    for reason in [CancellationReason::User, CancellationReason::Deadline] {
        let source = CancellationSource::new();
        source.cancel(reason);
        let stopped = runtime.clone().with_cancellation_token(source.token());
        let error = futures::executor::block_on(stopped.scope(async {
            reset_thread_buffer();
            emit_statement_value(&stopped, Some("before"), &value).await
        }))
        .unwrap_err();
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
        assert_eq!(stopped.cancellation().reason(), Some(reason));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        let _scope = stopped.enter();
        assert!(take_thread_buffer().is_empty());
    }
    for (index, reason) in [CancellationReason::User, CancellationReason::Deadline]
        .into_iter()
        .enumerate()
    {
        let source = CancellationSource::new();
        let active = runtime.clone().with_cancellation_token(source.token());
        let (send, receive) = futures::channel::oneshot::channel();
        *provider.gate.lock().unwrap() = Some(receive);
        let wake = Arc::new(WakeCount(AtomicUsize::new(0)));
        let waker = futures::task::waker(wake.clone());
        let mut cx = Context::from_waker(&waker);
        let mut pending = Box::pin(active.scope(async {
            reset_thread_buffer();
            emit_statement_value(&active, Some("before"), &value).await
        }));
        assert!(pending.as_mut().poll(&mut cx).is_pending());
        source.cancel(reason);
        assert!(wake.0.load(Ordering::SeqCst) > 0);
        let Poll::Ready(Err(error)) = pending.as_mut().poll(&mut cx) else {
            panic!("cancel must not wait for physical download completion");
        };
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
        assert_eq!(provider.dropped.load(Ordering::SeqCst), index + 1);
        assert!(
            send.send(Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![7.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            }))
            .is_err(),
            "logical waiter retired without provider completion"
        );
        let _scope = active.enter();
        assert!(take_thread_buffer().is_empty());
        assert!(take_last_value_output().is_none());
    }
    let failure = futures::executor::block_on(runtime.scope(async {
        reset_thread_buffer();
        emit_statement_value(&runtime, Some("before"), &value).await
    }))
    .unwrap_err();
    assert_eq!(
        failure.identifier(),
        Some("RunMat:StatementProbe:DownloadFailure")
    );
    assert!(!crate::runtime_error::error_has_control_cancellation_origin(&failure));
    assert!(std::error::Error::source(&failure).is_some());
    let _scope = runtime.enter();
    assert!(take_thread_buffer().is_empty());
    assert!(take_last_value_output().is_none());
}
