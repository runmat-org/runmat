use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, OnceLock};

use runmat_logging::{LoggingGuard, RuntimeLogRecord, TraceEvent};
use runmat_thread_local::runmat_thread_local;
runmat_thread_local! {
    pub(crate) static FIGURE_EVENT_CALLBACK: RefCell<Option<js_sys::Function>> = RefCell::new(None);
}

pub(crate) static FIGURE_EVENT_OBSERVER: OnceLock<()> = OnceLock::new();

runmat_thread_local! {
    pub(crate) static JS_STDIN_HANDLER: RefCell<Option<js_sys::Function>> = RefCell::new(None);
}

runmat_thread_local! {
    pub(crate) static STDOUT_SUBSCRIBERS: RefCell<HashMap<u32, js_sys::Function>> =
        RefCell::new(HashMap::new());
}

pub(crate) type StdoutForwarder =
    Arc<dyn Fn(&runmat_runtime::console::ConsoleEntry) + Send + Sync + 'static>;
pub(crate) type RuntimeLogForwarder = Arc<dyn Fn(&RuntimeLogRecord) + Send + Sync + 'static>;
pub(crate) type TraceForwarder = Arc<dyn Fn(&[TraceEvent]) + Send + Sync + 'static>;

pub(crate) static STDOUT_FORWARDER: OnceLock<StdoutForwarder> = OnceLock::new();
pub(crate) static STDOUT_NEXT_ID: AtomicU32 = AtomicU32::new(1);

runmat_thread_local! {
    pub(crate) static RUNTIME_LOG_SUBSCRIBERS: RefCell<HashMap<u32, js_sys::Function>> =
        RefCell::new(HashMap::new());
}

pub(crate) static RUNTIME_LOG_FORWARDER: OnceLock<RuntimeLogForwarder> = OnceLock::new();
pub(crate) static RUNTIME_LOG_NEXT_ID: AtomicU32 = AtomicU32::new(1);

runmat_thread_local! {
    pub(crate) static TRACE_SUBSCRIBERS: RefCell<HashMap<u32, js_sys::Function>> =
        RefCell::new(HashMap::new());
}

pub(crate) static TRACE_FORWARDER: OnceLock<TraceForwarder> = OnceLock::new();
pub(crate) static TRACE_NEXT_ID: AtomicU32 = AtomicU32::new(1);
pub(crate) static LOGGING_GUARD: OnceLock<LoggingGuard> = OnceLock::new();
pub(crate) static LOG_FILTER_OVERRIDE: OnceLock<String> = OnceLock::new();

pub(crate) static PLOT_SURFACE_NEXT_ID: AtomicU32 = AtomicU32::new(1);

runmat_thread_local! {
    pub(crate) static LEGACY_PLOT_SURFACE_ID: RefCell<Option<u32>> = RefCell::new(None);
    pub(crate) static LEGACY_FIGURE_SURFACES: RefCell<HashMap<u32, u32>> = RefCell::new(HashMap::new());
}

pub(crate) fn clear_figure_event_callback() {
    FIGURE_EVENT_CALLBACK.with(|slot| {
        slot.replace(None);
    });
}

pub(crate) fn set_js_stdin_handler(handler: Option<js_sys::Function>) {
    JS_STDIN_HANDLER.with(|slot| *slot.borrow_mut() = handler);
}

pub(crate) fn js_stdin_handler() -> Option<js_sys::Function> {
    JS_STDIN_HANDLER.with(|slot| slot.borrow().clone())
}

pub(crate) fn figure_event_callback() -> Option<js_sys::Function> {
    FIGURE_EVENT_CALLBACK.with(|slot| slot.borrow().clone())
}

pub(crate) fn replace_figure_event_callback(callback: Option<js_sys::Function>) {
    FIGURE_EVENT_CALLBACK.with(|slot| {
        *slot.borrow_mut() = callback;
    });
}
