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
