use std::backtrace::Backtrace;
use std::sync::Arc;
use std::sync::OnceLock;

use wasm_bindgen::prelude::JsValue;

use runmat_logging::{init_logging, set_runtime_log_hook, LoggingOptions};

use crate::runtime::state::{
    LOGGING_GUARD, LOG_FILTER_OVERRIDE, RUNTIME_LOG_FORWARDER, RUNTIME_LOG_SUBSCRIBERS,
    STDOUT_FORWARDER, STDOUT_SUBSCRIBERS, TRACE_FORWARDER, TRACE_SUBSCRIBERS,
};
use crate::wire::payloads::ConsoleStreamPayload;

pub(crate) fn ensure_runtime_log_forwarder_installed() {
    RUNTIME_LOG_FORWARDER.get_or_init(|| {
        let forwarder = Arc::new(|record: &runmat_logging::RuntimeLogRecord| {
            let js_value = serde_wasm_bindgen::to_value(record).unwrap_or(JsValue::NULL);
            RUNTIME_LOG_SUBSCRIBERS.with(|cell| {
                for cb in cell.borrow().values() {
                    let _ = cb.call1(&JsValue::NULL, &js_value);
                }
            });
        });
        let hook = forwarder.clone();
        set_runtime_log_hook(move |rec| {
            (hook)(rec);
        });
        forwarder
    });
}

pub(crate) fn ensure_trace_forwarder_installed() {
    use runmat_logging::{set_trace_hook, TraceEvent};

    TRACE_FORWARDER.get_or_init(|| {
        let forwarder = Arc::new(|events: &[TraceEvent]| {
            let js_value = serde_wasm_bindgen::to_value(events).unwrap_or(JsValue::NULL);
            TRACE_SUBSCRIBERS.with(|cell| {
                for cb in cell.borrow().values() {
                    let _ = cb.call1(&JsValue::NULL, &js_value);
                }
            });
        });
        let hook = forwarder.clone();
        set_trace_hook(move |events| {
            (hook)(events);
        });
        forwarder
    });
}

pub(crate) fn ensure_stdout_forwarder_installed() {
    let forwarder = STDOUT_FORWARDER.get_or_init(|| {
        Arc::new(dispatch_stdout_entry as fn(&runmat_runtime::console::ConsoleEntry))
            as Arc<dyn Fn(&runmat_runtime::console::ConsoleEntry) + Send + Sync + 'static>
    });
    runmat_runtime::console::install_forwarder(Some(forwarder.clone()));
}

fn dispatch_stdout_entry(entry: &runmat_runtime::console::ConsoleEntry) {
    let handlers: Vec<js_sys::Function> =
        STDOUT_SUBSCRIBERS.with(|cell| cell.borrow().values().cloned().collect());
    if handlers.is_empty() {
        return;
    }
    if let Ok(payload) =
        serde_wasm_bindgen::to_value(&ConsoleStreamPayload::from_console_entry(entry))
    {
        for handler in handlers {
            let _ = handler.call1(&JsValue::NULL, &payload);
        }
    }
}

pub(crate) fn init_logging_once() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(|info| {
                web_sys::console::error_1(&JsValue::from_str(
                    "RunMat panic hook invoked; forwarding to console_error_panic_hook",
                ));
                console_error_panic_hook::hook(info);
                let bt = Backtrace::force_capture();
                web_sys::console::error_1(&JsValue::from_str(&format!(
                    "RunMat panic backtrace:\n{bt:?}"
                )));
            }));
            let guard = init_logging(LoggingOptions {
                enable_otlp: false,
                enable_traces: true,
                pid: 1,
                default_filter: Some(
                    LOG_FILTER_OVERRIDE
                        .get()
                        .cloned()
                        .unwrap_or_else(|| "debug".to_string()),
                ),
            });
            let _ = LOGGING_GUARD.set(guard);
            ensure_runtime_log_forwarder_installed();
            ensure_trace_forwarder_installed();
        }
    });
}

pub(crate) fn set_log_filter_override(level: &str) {
    let normalized = level.trim();
    if normalized.is_empty() {
        return;
    }
    if LOGGING_GUARD.get().is_none() {
        let _ = LOG_FILTER_OVERRIDE.set(normalized.to_string());
    }
}
