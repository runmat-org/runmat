use once_cell::sync::OnceCell;
use runmat_thread_local::runmat_thread_local;
use runmat_time::unix_timestamp_ms;
use runmat_value::Value;
use std::cell::RefCell;
use std::sync::{Arc, RwLock};

mod diary;
mod capture_projection;
pub(crate) use capture_projection::{CaptureLease, ConsoleCaptureProjection};
pub use diary::{
    diary_enabled, diary_filename, ensure_diary_writable, record_diary_command, set_diary_enabled,
    set_diary_enabled_checked, set_diary_filename, set_diary_filename_checked, take_diary_error,
    toggle_diary,
};
use diary::{write_diary_entry, DiaryState};

/// Identifies the console stream that received the text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleStream {
    Stdout,
    Stderr,
    ClearScreen,
}

/// Single console write (line or chunk) captured during execution.
#[derive(Clone, Debug)]
pub struct ConsoleEntry {
    pub stream: ConsoleStream,
    pub text: String,
    pub timestamp_ms: u64,
}

type StreamForwarder = dyn Fn(&ConsoleEntry) + Send + Sync + 'static;

runmat_thread_local! {
    static FALLBACK_STATE: RefCell<ConsoleState> = RefCell::new(ConsoleState::default());
}

static FORWARDER: OnceCell<RwLock<Option<Arc<StreamForwarder>>>> = OnceCell::new();

/// Guard for a dynamic command-window capture scope such as `evalc`.
pub struct ConsoleCaptureGuard {
    active: bool,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
    projection: Option<CaptureLease>,
}

#[derive(Debug, Default)]
pub(crate) struct ConsoleState {
    buffer: Vec<ConsoleEntry>,
    last_value_output: Option<Value>,
    captures: Vec<Vec<ConsoleEntry>>,
    diary: DiaryState,
}

fn now_ms() -> u64 {
    unix_timestamp_ms().min(u64::MAX as u128) as u64
}

/// Record console output for the current thread while also forwarding it to any
/// registered listener (used by wasm bindings for live streaming).
pub fn record_console_output(stream: ConsoleStream, text: impl Into<String>) {
    let entry = ConsoleEntry {
        stream,
        text: text.into(),
        timestamp_ms: now_ms(),
    };
    if let Some(runtime) = crate::context::legacy::active() {
        if runtime
            .console_capture_projection()
            .borrow()
            .record(entry.clone())
        {
            return;
        }
    }
    if with_console_state(|state| {
        if let Some(current) = state.captures.last_mut() {
            current.push(entry.clone());
            true
        } else {
            false
        }
    }) {
        return;
    }

    with_console_state(|state| state.buffer.push(entry.clone()));
    write_diary_entry(&entry);

    if let Some(context) = crate::context::legacy::active() {
        if let Some(host) = context.service_ports().host() {
            host.console(entry.stream, entry.text.clone());
            return;
        }
    }

    if let Some(forwarder) = FORWARDER
        .get()
        .and_then(|lock| lock.read().ok().map(|guard| guard.as_ref().cloned()))
        .flatten()
    {
        forwarder(&entry);
    }
}

/// Forward an entry that the Runtime host port received after the canonical
/// buffer and diary owners already recorded it. Host adapters use this to
/// preserve installed live-stream listeners without recursively recording.
pub fn forward_recorded_console_output(stream: ConsoleStream, text: String) {
    let entry = ConsoleEntry {
        stream,
        text,
        timestamp_ms: now_ms(),
    };
    if let Some(forwarder) = FORWARDER
        .get()
        .and_then(|lock| lock.read().ok().map(|guard| guard.as_ref().cloned()))
        .flatten()
    {
        forwarder(&entry);
    }
}

/// Record a control event that asks the host to clear the visible console.
pub fn record_clear_screen() {
    record_console_output(ConsoleStream::ClearScreen, String::new());
}

/// Record a line-oriented console entry, ensuring the stream text ends with a newline.
pub fn record_console_line(stream: ConsoleStream, text: impl Into<String>) {
    let mut text = text.into();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    record_console_output(stream, text);
}

/// Clears the per-thread console buffer. Call this before execution begins so
/// each run only returns fresh output.
pub fn reset_thread_buffer() {
    with_console_state(|state| {
        state.buffer.clear();
        state.last_value_output = None;
    });
}

/// Drain (and return) the buffered console entries for the current thread.
pub fn take_thread_buffer() -> Vec<ConsoleEntry> {
    with_console_state(|state| state.buffer.drain(..).collect())
}

/// Append console entries captured on another execution thread without
/// forwarding them again to live stream listeners.
pub fn append_thread_buffer(entries: impl IntoIterator<Item = ConsoleEntry>) {
    with_console_state(|state| {
        state.buffer.extend(entries);
        state.buffer.sort_by_key(|entry| entry.timestamp_ms);
    });
}

/// Install (or remove) a global forwarder for console output. Passing `None`
/// removes the current listener.
pub fn install_forwarder(forwarder: Option<Arc<StreamForwarder>>) {
    let lock = FORWARDER.get_or_init(|| RwLock::new(None));
    if let Ok(mut guard) = lock.write() {
        *guard = forwarder;
    }
}

/// Convenience helper to record formatted value output (matching MATLAB's `name = value` layout).
pub fn record_value_output(label: Option<&str>, value: &Value) {
    with_console_state(|state| state.last_value_output = Some(value.clone()));
    let value_text = match value {
        Value::Object(obj) if obj.is_class(runmat_types::standard::DATETIME) => {
            crate::builtins::datetime::datetime_display_text(value)
                .ok()
                .flatten()
                .unwrap_or_else(|| value.to_string())
        }
        Value::Object(obj) if obj.is_class(runmat_types::standard::DURATION) => {
            crate::builtins::duration::duration_display_text(value)
                .ok()
                .flatten()
                .unwrap_or_else(|| value.to_string())
        }
        _ => value.to_string(),
    };
    let text = if let Some(name) = label {
        if is_unlabeled_nd_page_display(&value_text) {
            inject_label_into_nd_page_headers(name, &value_text)
        } else if value_text.contains('\n') {
            format!("{name} =\n{value_text}")
        } else {
            format!("{name} = {value_text}")
        }
    } else {
        value_text
    };
    record_console_line(ConsoleStream::Stdout, text);
}

pub fn take_last_value_output() -> Option<Value> {
    with_console_state(|state| state.last_value_output.take())
}

/// Begin capturing command-window text for the current execution context.
///
/// While a capture is active, stdout/stderr entries are diverted into the
/// capture buffer instead of the normal execution stream, live forwarder, or
/// diary log. This matches MATLAB's `evalc` behavior, where `diary` is disabled
/// inside the captured evaluation.
pub fn begin_capture() -> ConsoleCaptureGuard {
    if let Some(runtime) = crate::context::legacy::active() {
        return ConsoleCaptureGuard {
            active: true,
            state: None,
            projection: Some(ConsoleCaptureProjection::begin(
                runtime.console_capture_projection(),
            )),
        };
    }
    let state = active_state();
    if let Some(state) = &state {
        state.console.borrow_mut().captures.push(Vec::new());
    } else {
        FALLBACK_STATE.with(|fallback| fallback.borrow_mut().captures.push(Vec::new()));
    }
    ConsoleCaptureGuard {
        active: true,
        state,
        projection: None,
    }
}

impl ConsoleCaptureGuard {
    pub fn finish(mut self) -> String {
        captured_text(self.take_entries())
    }

    /// Finish this capture and retain the stream identity of every entry.
    ///
    /// Foreign engine adapters use this form when the caller supplied
    /// distinct standard-output and standard-error stream buffers.
    pub fn finish_entries(mut self) -> Vec<ConsoleEntry> {
        self.take_entries()
    }

    fn take_entries(&mut self) -> Vec<ConsoleEntry> {
        self.active = false;
        if let Some(projection) = self.projection.take() {
            return projection.finish();
        }
        if let Some(state) = &self.state {
            state
                .console
                .borrow_mut()
                .captures
                .pop()
                .unwrap_or_default()
        } else {
            FALLBACK_STATE.with(|fallback| fallback.borrow_mut().captures.pop().unwrap_or_default())
        }
    }
}

impl Drop for ConsoleCaptureGuard {
    fn drop(&mut self) {
        if self.active {
            if self.projection.take().is_some() {
                return;
            }
            if let Some(state) = &self.state {
                state.console.borrow_mut().captures.pop();
            } else {
                FALLBACK_STATE.with(|fallback| {
                    fallback.borrow_mut().captures.pop();
                });
            }
        }
    }
}

fn captured_text(entries: Vec<ConsoleEntry>) -> String {
    let mut out = String::new();
    for entry in entries {
        if matches!(entry.stream, ConsoleStream::Stdout | ConsoleStream::Stderr) {
            out.push_str(&entry.text);
        }
    }
    out
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

fn with_console_state<R>(callback: impl FnOnce(&mut ConsoleState) -> R) -> R {
    if let Some(state) = active_state() {
        callback(&mut state.console.borrow_mut())
    } else {
        FALLBACK_STATE.with(|state| callback(&mut state.borrow_mut()))
    }
}

fn is_unlabeled_nd_page_display(text: &str) -> bool {
    text.lines()
        .any(|line| line.trim_start().starts_with("(:, :") && line.trim_end().ends_with('='))
}

fn inject_label_into_nd_page_headers(label: &str, text: &str) -> String {
    let mut out = String::new();
    for (idx, line) in text.lines().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("(:, :") && trimmed.trim_end().ends_with('=') {
            out.push_str(label);
            out.push_str(trimmed);
        } else {
            out.push_str(line);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    fn entry(text: &str, timestamp_ms: u64) -> ConsoleEntry {
        ConsoleEntry {
            stream: ConsoleStream::Stdout,
            text: text.to_string(),
            timestamp_ms,
        }
    }

    #[test]
    fn append_thread_buffer_orders_entries_by_timestamp_stably() {
        reset_thread_buffer();
        append_thread_buffer(vec![entry("late", 20)]);
        append_thread_buffer(vec![entry("early", 10), entry("same-time", 20)]);

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
                let outer = begin_capture();
                record_console_output(ConsoleStream::Stdout, format!("{label}:outer"));
                let nested = begin_capture();
                record_console_output(ConsoleStream::Stderr, format!("{label}:nested"));
                release.await.expect("release sibling");
                record_console_output(ConsoleStream::Stderr, format!("{label}:resumed"));
                let nested = nested.finish_entries();
                record_console_output(ConsoleStream::Stdout, format!("{label}:after"));
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
            record_console_output(ConsoleStream::Stdout, "post-cancel");
            assert!(take_thread_buffer()
                .iter()
                .any(|entry| entry.text == "post-cancel"));
        }
        let unrelated = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let _scope = unrelated.enter();
        record_console_output(ConsoleStream::Stdout, "unrelated");
        assert!(take_thread_buffer()
            .iter()
            .any(|entry| entry.text == "unrelated"));
    }

    #[test]
    fn capture_diverts_console_text_from_thread_buffer() {
        let _lock = runmat_filesystem::provider_override_lock();
        set_diary_enabled(false);
        reset_thread_buffer();
        let capture = begin_capture();
        record_console_line(ConsoleStream::Stdout, "inside");
        let text = capture.finish();
        record_console_line(ConsoleStream::Stdout, "outside");

        assert_eq!(text, "inside\n");
        let entries = take_thread_buffer();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "outside\n");
    }
}
