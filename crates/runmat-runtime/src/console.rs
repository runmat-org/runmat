use once_cell::sync::OnceCell;
use runmat_thread_local::runmat_thread_local;
use runmat_time::unix_timestamp_ms;
use runmat_value::Value;
use std::cell::RefCell;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

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
    static THREAD_BUFFER: RefCell<Vec<ConsoleEntry>> = const { RefCell::new(Vec::new()) };
    static LAST_VALUE_OUTPUT: RefCell<Option<Value>> = const { RefCell::new(None) };
    static CAPTURE_STACK: RefCell<Vec<Vec<ConsoleEntry>>> = const { RefCell::new(Vec::new()) };
    static DIARY_STATE: RefCell<DiaryState> = RefCell::new(DiaryState::default());
}

static FORWARDER: OnceCell<RwLock<Option<Arc<StreamForwarder>>>> = OnceCell::new();

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiaryStateSnapshot {
    pub enabled: bool,
    pub filename: PathBuf,
}

impl Default for DiaryStateSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            filename: PathBuf::from("diary"),
        }
    }
}

#[derive(Debug)]
struct DiaryState {
    enabled: bool,
    filename: PathBuf,
    last_error: Option<String>,
}

impl Default for DiaryState {
    fn default() -> Self {
        Self {
            enabled: false,
            filename: PathBuf::from("diary"),
            last_error: None,
        }
    }
}

impl From<DiaryStateSnapshot> for DiaryState {
    fn from(snapshot: DiaryStateSnapshot) -> Self {
        Self {
            enabled: snapshot.enabled,
            filename: snapshot.filename,
            last_error: None,
        }
    }
}

/// Guard for a dynamic command-window capture scope such as `evalc`.
pub struct ConsoleCaptureGuard {
    active: bool,
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
    if CAPTURE_STACK.with(|captures| {
        let mut captures = captures.borrow_mut();
        if let Some(current) = captures.last_mut() {
            current.push(entry.clone());
            true
        } else {
            false
        }
    }) {
        return;
    }

    THREAD_BUFFER.with(|buf| buf.borrow_mut().push(entry.clone()));
    write_diary_entry(&entry);

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
    THREAD_BUFFER.with(|buf| buf.borrow_mut().clear());
    LAST_VALUE_OUTPUT.with(|value| value.borrow_mut().take());
}

/// Drain (and return) the buffered console entries for the current thread.
pub fn take_thread_buffer() -> Vec<ConsoleEntry> {
    THREAD_BUFFER.with(|buf| buf.borrow_mut().drain(..).collect())
}

/// Append console entries captured on another execution thread without
/// forwarding them again to live stream listeners.
pub fn append_thread_buffer(entries: impl IntoIterator<Item = ConsoleEntry>) {
    THREAD_BUFFER.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.extend(entries);
        buf.sort_by_key(|entry| entry.timestamp_ms);
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
    LAST_VALUE_OUTPUT.with(|last| {
        *last.borrow_mut() = Some(value.clone());
    });
    let value_text = match value {
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::datetime_display_text(value)
                .ok()
                .flatten()
                .unwrap_or_else(|| value.to_string())
        }
        Value::Object(obj) if obj.is_class("duration") => {
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
    LAST_VALUE_OUTPUT.with(|value| value.borrow_mut().take())
}

/// Begin capturing command-window text for the current execution context.
///
/// While a capture is active, stdout/stderr entries are diverted into the
/// capture buffer instead of the normal execution stream, live forwarder, or
/// diary log. This matches MATLAB's `evalc` behavior, where `diary` is disabled
/// inside the captured evaluation.
pub fn begin_capture() -> ConsoleCaptureGuard {
    CAPTURE_STACK.with(|captures| captures.borrow_mut().push(Vec::new()));
    ConsoleCaptureGuard { active: true }
}

impl ConsoleCaptureGuard {
    pub fn finish(mut self) -> String {
        self.active = false;
        let entries =
            CAPTURE_STACK.with(|captures| captures.borrow_mut().pop().unwrap_or_default());
        captured_text(entries)
    }
}

impl Drop for ConsoleCaptureGuard {
    fn drop(&mut self) {
        if self.active {
            CAPTURE_STACK.with(|captures| {
                captures.borrow_mut().pop();
            });
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

pub fn diary_enabled() -> bool {
    DIARY_STATE.with(|state| state.borrow().enabled)
}

pub fn diary_filename() -> PathBuf {
    DIARY_STATE.with(|state| state.borrow().filename.clone())
}

pub fn diary_state_snapshot() -> DiaryStateSnapshot {
    DIARY_STATE.with(|state| {
        let state = state.borrow();
        DiaryStateSnapshot {
            enabled: state.enabled,
            filename: state.filename.clone(),
        }
    })
}

pub fn replace_diary_state(snapshot: DiaryStateSnapshot) -> DiaryStateSnapshot {
    DIARY_STATE.with(|state| {
        let previous = {
            let state = state.borrow();
            DiaryStateSnapshot {
                enabled: state.enabled,
                filename: state.filename.clone(),
            }
        };
        *state.borrow_mut() = DiaryState::from(snapshot);
        previous
    })
}

pub fn take_diary_error() -> Option<String> {
    DIARY_STATE.with(|state| state.borrow_mut().last_error.take())
}

pub fn set_diary_filename(filename: impl Into<PathBuf>) {
    DIARY_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.filename = filename.into();
        state.enabled = true;
        state.last_error = None;
    });
}

pub fn set_diary_filename_checked(filename: impl Into<PathBuf>) -> std::io::Result<()> {
    let filename = filename.into();
    open_diary_append(&filename).map(|_| ())?;
    set_diary_filename(filename);
    Ok(())
}

pub fn set_diary_enabled(enabled: bool) {
    DIARY_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.enabled = enabled;
        if enabled {
            state.last_error = None;
        }
    });
}

pub fn set_diary_enabled_checked(enabled: bool) -> std::io::Result<()> {
    if enabled {
        ensure_diary_writable()?;
    }
    set_diary_enabled(enabled);
    Ok(())
}

pub fn toggle_diary() -> std::io::Result<()> {
    if diary_enabled() {
        set_diary_enabled(false);
    } else {
        set_diary_enabled_checked(true)?;
    }
    Ok(())
}

pub fn ensure_diary_writable() -> std::io::Result<()> {
    let filename = diary_filename();
    open_diary_append(&filename).map(|_| ())
}

/// Append the top-level source text to the diary without echoing it to stdout.
pub fn record_diary_command(text: &str) {
    if !diary_enabled() {
        return;
    }
    let mut owned = text.to_string();
    if !owned.ends_with('\n') {
        owned.push('\n');
    }
    if let Err(err) = write_diary_text(&owned) {
        record_diary_failure(err);
    }
}

fn write_diary_entry(entry: &ConsoleEntry) {
    if !matches!(entry.stream, ConsoleStream::Stdout | ConsoleStream::Stderr) {
        return;
    }
    if diary_enabled() {
        if let Err(err) = write_diary_text(&entry.text) {
            record_diary_failure(err);
        }
    }
}

fn record_diary_failure(err: std::io::Error) {
    DIARY_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.enabled = false;
        state.last_error = Some(format!("diary: failed to write diary file ({err})"));
    });
}

fn write_diary_text(text: &str) -> std::io::Result<()> {
    let filename = diary_filename();
    let mut file = open_diary_append(&filename)?;
    file.write_all(text.as_bytes())?;
    file.flush()
}

fn open_diary_append(path: &PathBuf) -> std::io::Result<runmat_filesystem::File> {
    let mut options = runmat_filesystem::OpenOptions::new();
    options.create(true).append(true);
    options.open(path)
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
