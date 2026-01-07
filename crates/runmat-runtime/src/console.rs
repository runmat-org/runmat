use once_cell::sync::OnceCell;
use runmat_builtins::Value;
use runmat_time::unix_timestamp_ms;
use std::cell::RefCell;
use std::sync::{Arc, RwLock};
use std::thread_local;

/// Identifies the console stream that received the text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleStream {
    Stdout,
    Stderr,
}

/// Single console write (line or chunk) captured during execution.
#[derive(Clone, Debug)]
pub struct ConsoleEntry {
    pub stream: ConsoleStream,
    pub text: String,
    pub timestamp_ms: u64,
}

type StreamForwarder = dyn Fn(&ConsoleEntry) + Send + Sync + 'static;

thread_local! {
    static THREAD_BUFFER: RefCell<Vec<ConsoleEntry>> = const { RefCell::new(Vec::new()) };
}

static FORWARDER: OnceCell<RwLock<Option<Arc<StreamForwarder>>>> = OnceCell::new();

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
    THREAD_BUFFER.with(|buf| buf.borrow_mut().push(entry.clone()));

    if let Some(forwarder) = FORWARDER
        .get()
        .and_then(|lock| lock.read().ok().map(|guard| guard.as_ref().cloned()))
        .flatten()
    {
        forwarder(&entry);
    }
}

/// Clears the per-thread console buffer. Call this before execution begins so
/// each run only returns fresh output.
pub fn reset_thread_buffer() {
    THREAD_BUFFER.with(|buf| buf.borrow_mut().clear());
}

/// Drain (and return) the buffered console entries for the current thread.
pub fn take_thread_buffer() -> Vec<ConsoleEntry> {
    THREAD_BUFFER.with(|buf| buf.borrow_mut().drain(..).collect())
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
    let value_text = value.to_string();
    let text = if let Some(name) = label {
        if value_text.contains('\n') {
            format!("{name} =\n{value_text}")
        } else {
            format!("{name} = {value_text}")
        }
    } else {
        value_text
    };
    record_console_output(ConsoleStream::Stdout, text);
}
