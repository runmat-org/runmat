use once_cell::sync::OnceCell;
use runmat_thread_local::runmat_thread_local;
use runmat_time::unix_timestamp_ms;
use runmat_value::Value;
use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::{Arc, RwLock};

mod buffer;
mod capture_projection;
mod diary;
mod entries;
#[cfg(test)]
mod tests;
mod text;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    HostTextError,
};
use crate::context::overlay_storage::AdmittedOverlayVec;
use buffer::append_buffer_entry;
pub use buffer::{append_thread_buffer, reset_thread_buffer, take_thread_buffer};
pub(crate) use capture_projection::{CaptureLease, ConsoleCaptureProjection};
pub use diary::{
    diary_enabled, diary_filename, ensure_diary_writable, record_diary_command, set_diary_enabled,
    set_diary_enabled_checked, set_diary_filename, set_diary_filename_checked, take_diary_error,
    toggle_diary,
};
use diary::{write_diary_entry, DiaryState};
pub use entries::{CapturedConsoleEntries, CapturedConsoleEntriesIter};
pub use text::ConsoleText;

pub use runmat_execution::console::ConsoleStream;

/// Single console write (line or chunk) captured during execution.
#[derive(Clone, Debug)]
pub struct ConsoleEntry {
    pub stream: ConsoleStream,
    pub text: ConsoleText,
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

pub(crate) struct ConsoleState {
    buffer: AdmittedOverlayVec<ConsoleEntry>,
    last_value_output: Option<Value>,
    captures: Vec<Vec<ConsoleEntry>>,
    diary: DiaryState,
}

impl Default for ConsoleState {
    fn default() -> Self {
        Self {
            buffer: AdmittedOverlayVec::unbounded(),
            last_value_output: None,
            captures: Vec::new(),
            diary: DiaryState::default(),
        }
    }
}

impl std::fmt::Debug for ConsoleState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConsoleState")
            .field("buffer_entries", &self.buffer.len())
            .field("capture_depth", &self.captures.len())
            .field("has_last_value_output", &self.last_value_output.is_some())
            .field("diary", &self.diary)
            .finish()
    }
}

fn now_ms() -> u64 {
    unix_timestamp_ms().min(u64::MAX as u128) as u64
}

/// Record console output for the current thread while also forwarding it to any
/// registered listener (used by wasm bindings for live streaming).
pub fn record_console_output<'a>(
    stream: ConsoleStream,
    text: impl Into<Cow<'a, str>>,
) -> Result<(), crate::RuntimeError> {
    record_console_text(stream, text.into(), false)
}

fn record_console_text(
    stream: ConsoleStream,
    text: Cow<'_, str>,
    append_newline: bool,
) -> Result<(), crate::RuntimeError> {
    let needs_newline = append_newline && !text.ends_with('\n');
    let text_len = text
        .len()
        .checked_add(usize::from(needs_newline))
        .ok_or_else(|| console_admission(None, HostAllocationError::Overflow))?;
    let runtime = crate::context::legacy::active();
    let text = if let Some(runtime) = runtime.as_ref() {
        if HostAllocationReceipt::policy(Some(runtime))
            .map_err(|error| console_admission(Some(runtime), error))?
            == HostAllocationPolicy::Managed
        {
            let adopt_existing = matches!(&text, Cow::Owned(owned) if owned.capacity() >= text_len);
            let retained = ConsoleText::retained_bytes(match &text {
                Cow::Owned(owned) if adopt_existing => owned.capacity(),
                _ => text_len,
            })
            .map_err(|error| console_admission(Some(runtime), error))?;
            let receipt = HostAllocationReceipt::admit(
                Some(runtime),
                HostAllocationPlan::new(0, retained)
                    .map_err(|error| console_admission(Some(runtime), error))?,
            )
            .map_err(|error| console_admission(Some(runtime), error))?
            .ok_or_else(|| {
                console_admission(Some(runtime), HostAllocationError::MissingManagedOwner)
            })?;
            let permit = receipt
                .output_permit(runtime)
                .map_err(|error| console_admission(Some(runtime), error))?;
            let prepared = match text {
                Cow::Owned(owned) if adopt_existing => {
                    if needs_newline {
                        ConsoleText::try_adopt_owned_line(owned, &permit)
                    } else {
                        ConsoleText::try_adopt_owned(owned, &permit)
                    }
                }
                text => ConsoleText::try_initialize(text_len, &permit, |writer| {
                    std::fmt::Write::write_str(writer, text.as_ref())?;
                    if needs_newline {
                        std::fmt::Write::write_char(writer, '\n')?;
                    }
                    Ok(())
                }),
            };
            receipt
                .checkpoint(runtime)
                .map_err(|error| console_admission(Some(runtime), error))?;
            prepared.map_err(|error| console_text_error(runtime, error))?
        } else {
            let mut text = text.into_owned();
            if needs_newline {
                text.push('\n');
            }
            ConsoleText::from(text)
        }
    } else {
        let mut text = text.into_owned();
        if needs_newline {
            text.push('\n');
        }
        ConsoleText::from(text)
    };
    let entry = ConsoleEntry {
        stream,
        text,
        timestamp_ms: now_ms(),
    };
    let entry = if let Some(runtime) = runtime.as_ref() {
        match ConsoleCaptureProjection::record(runtime.console_capture_projection(), entry, runtime)
            .map_err(|error| console_admission(Some(runtime), error))?
        {
            capture_projection::CaptureRoute::Captured => return Ok(()),
            capture_projection::CaptureRoute::Uncaptured(entry) => entry,
        }
    } else {
        entry
    };
    let entry = match with_console_state(|state| {
        if let Some(current) = state.captures.last_mut() {
            current.push(entry);
            capture_projection::CaptureRoute::Captured
        } else {
            capture_projection::CaptureRoute::Uncaptured(entry)
        }
    }) {
        capture_projection::CaptureRoute::Captured => return Ok(()),
        capture_projection::CaptureRoute::Uncaptured(entry) => entry,
    };

    // A retaining host (remote execution) owns the console inventory itself.
    let host = crate::context::legacy::active()
        .and_then(|context| context.service_ports().host().cloned());
    if host
        .as_ref()
        .is_none_or(|host| !host.owns_console_retention())
    {
        append_buffer_entry(entry.clone(), runtime.as_ref())?;
    }
    write_diary_entry(&entry);

    if let Some(host) = host {
        host.console(entry.stream, entry.text);
        return Ok(());
    }

    forward_to_registered(&entry);
    Ok(())
}

fn forward_to_registered(entry: &ConsoleEntry) {
    if let Some(forwarder) = FORWARDER
        .get()
        .and_then(|lock| lock.read().ok().map(|guard| guard.as_ref().cloned()))
        .flatten()
    {
        forwarder(entry);
    }
}

/// The stock hosted-console port forwards the already-admitted text through
/// the same invocation listener as the ordinary no-host route. It does not
/// allocate a second capture or replace the managed thread-buffer owner.
pub(crate) fn forward_host_text(stream: ConsoleStream, text: ConsoleText) {
    let entry = ConsoleEntry {
        stream,
        text,
        timestamp_ms: now_ms(),
    };
    forward_to_registered(&entry);
}

pub(super) fn console_admission(
    runtime: Option<&crate::context::RuntimeContext>,
    error: HostAllocationError,
) -> crate::RuntimeError {
    let cancellation = runtime.map(crate::context::RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), "console", error)
}

pub(super) fn console_text_error(
    runtime: &crate::context::RuntimeContext,
    error: HostTextError,
) -> crate::RuntimeError {
    match error {
        HostTextError::Allocation(error) => {
            console_admission(Some(runtime), HostAllocationError::Numeric(error))
        }
        other => {
            crate::runtime_error::semantic_error("RunMat:console:TextOutput", other.to_string())
        }
    }
}

/// Forward an entry that the Runtime host port received after the canonical
/// buffer and diary owners already recorded it. Host adapters use this to
/// preserve installed live-stream listeners without recursively recording.
pub fn forward_recorded_console_output(stream: ConsoleStream, text: ConsoleText) {
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
pub fn record_clear_screen() -> Result<(), crate::RuntimeError> {
    record_console_output(ConsoleStream::ClearScreen, String::new())
}

/// Record a line-oriented console entry, ensuring the stream text ends with a newline.
pub fn record_console_line<'a>(
    stream: ConsoleStream,
    text: impl Into<Cow<'a, str>>,
) -> Result<(), crate::RuntimeError> {
    record_console_text(stream, text.into(), true)
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
pub fn record_value_output(label: Option<&str>, value: &Value) -> Result<(), crate::RuntimeError> {
    let retained = value.clone();
    let displaced = with_console_state(|state| state.last_value_output.replace(retained));
    drop(displaced);
    let value_text = match value {
        Value::Struct(_) => crate::builtins::io::disp::format_for_display(value).join("\n"),
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
    record_console_line(ConsoleStream::Stdout, text)
}

/// One statement-ordered value effect shared by bytecode and native execution.
/// The selected provider's gather completes before any console publication.
pub async fn emit_statement_value(
    runtime: &crate::context::RuntimeContext,
    label: Option<&str>,
    value: &Value,
) -> Result<(), crate::RuntimeError> {
    let mut cooperation = runtime.cpu_cooperation();
    let control = |cause: crate::execution::ExecutionServiceError| {
        cause.into_runtime_error_in_context(Some(runtime))
    };
    cooperation.checkpoint(0).await.map_err(control)?;
    // Poll cancellation before constructing/polling the gather. The selected
    // provider's physical callback keeps its own original download claims if
    // cancellation drops this logical waiter; no later statement can overtake
    // an output whose gather is still pending.
    let host_value = cooperation
        .await_cancellable(async {
            crate::dispatcher::gather_if_needed_in_context_async(value, Some(runtime)).await
        })
        .await
        .map_err(control)??;
    cooperation.checkpoint(0).await.map_err(control)?;
    record_value_output(label, &host_value)?;
    Ok(())
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
pub fn begin_capture() -> Result<ConsoleCaptureGuard, crate::RuntimeError> {
    if let Some(runtime) = crate::context::legacy::active() {
        return Ok(ConsoleCaptureGuard {
            active: true,
            state: None,
            projection: Some(
                ConsoleCaptureProjection::begin(runtime.console_capture_projection(), &runtime)
                    .map_err(|error| console_admission(Some(&runtime), error))?,
            ),
        });
    }
    let state = active_state();
    if let Some(state) = &state {
        state.console.borrow_mut().captures.push(Vec::new());
    } else {
        FALLBACK_STATE.with(|fallback| fallback.borrow_mut().captures.push(Vec::new()));
    }
    Ok(ConsoleCaptureGuard {
        active: true,
        state,
        projection: None,
    })
}

impl ConsoleCaptureGuard {
    pub fn finish(mut self) -> Result<String, crate::RuntimeError> {
        captured_text(self.take_entries())
    }

    /// Finish this capture and retain the stream identity of every entry.
    ///
    /// Foreign engine adapters use this form when the caller supplied
    /// distinct standard-output and standard-error stream buffers.
    pub fn finish_entries(mut self) -> CapturedConsoleEntries {
        self.take_entries()
    }

    fn take_entries(&mut self) -> CapturedConsoleEntries {
        self.active = false;
        if let Some(projection) = self.projection.take() {
            return projection.finish();
        }
        let entries = if let Some(state) = &self.state {
            state
                .console
                .borrow_mut()
                .captures
                .pop()
                .unwrap_or_default()
        } else {
            FALLBACK_STATE.with(|fallback| fallback.borrow_mut().captures.pop().unwrap_or_default())
        };
        entries.into()
    }
}

impl Drop for ConsoleCaptureGuard {
    fn drop(&mut self) {
        if self.active {
            if self.projection.take().is_some() {
                return;
            }
            let retired = if let Some(state) = &self.state {
                state.console.borrow_mut().captures.pop()
            } else {
                FALLBACK_STATE.with(|fallback| fallback.borrow_mut().captures.pop())
            };
            drop(retired);
        }
    }
}

fn captured_text(entries: CapturedConsoleEntries) -> Result<String, crate::RuntimeError> {
    if entries.iter().any(|entry| entry.text.has_managed_backing()) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:console:ManagedTextMaterialization",
            "captured managed console text requires an admitted language output owner",
        ));
    }
    let mut out = String::new();
    for entry in entries {
        if matches!(entry.stream, ConsoleStream::Stdout | ConsoleStream::Stderr) {
            out.push_str(&entry.text);
        }
    }
    Ok(out)
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

pub(super) fn with_console_state<R>(callback: impl FnOnce(&mut ConsoleState) -> R) -> R {
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
