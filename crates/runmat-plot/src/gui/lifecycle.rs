//! Figure lifecycle helpers for native GUI windows.
//!
//! This module lets embedding runtimes associate RunMat figure handles with
//! window instances so that lifecycle events (e.g. MATLAB's `close`) can
//! gracefully tear down the corresponding OS windows without polling.

use crate::plots::Figure;
use log::warn;
use once_cell::sync::{Lazy, OnceCell};
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub type CloseSignal = Arc<AtomicBool>;

pub struct WindowRegistration {
    handle: u32,
    signal: Option<CloseSignal>,
}

static WINDOW_SIGNALS: Lazy<Mutex<HashMap<u32, CloseSignal>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DesktopBackend {
    NativeWindow,
    GuiThread,
    SingleWindow,
}

static BACKEND_PREF: OnceCell<Vec<DesktopBackend>> = OnceCell::new();

/// Render a figure in an interactive window that is tied to a specific MATLAB
/// figure handle.
///
/// When the runtime later emits a `FigureEventKind::Closed` for the same handle,
/// calling [`request_close`] will trigger the associated window to shut down.
pub fn register_handle(handle: u32) -> Result<WindowRegistration, String> {
    if handle == 0 {
        return Ok(WindowRegistration {
            handle,
            signal: None,
        });
    }

    let signal = Arc::new(AtomicBool::new(false));
    {
        let mut map = WINDOW_SIGNALS
            .lock()
            .map_err(|_| "failed to track plot window".to_string())?;
        map.insert(handle, signal.clone());
    }

    Ok(WindowRegistration {
        handle,
        signal: Some(signal),
    })
}

impl WindowRegistration {
    pub fn signal(&self) -> Option<CloseSignal> {
        self.signal.as_ref().map(Arc::clone)
    }
}

impl Drop for WindowRegistration {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        if let Ok(mut map) = WINDOW_SIGNALS.lock() {
            map.remove(&self.handle);
        }
    }
}

pub fn render_figure(handle: u32, figure: Figure) -> Result<String, String> {
    if handle == 0 {
        return crate::show_interactive_platform_optimal(figure);
    }

    let registration = register_handle(handle)?;
    let signal = registration.signal();
    let mut last_err: Option<String> = None;

    for backend in backend_preference() {
        let fig_clone = figure.clone();
        let sig_clone = clone_signal(&signal);
        let attempt = match backend {
            DesktopBackend::NativeWindow => render_via_native(fig_clone, sig_clone),
            DesktopBackend::GuiThread => render_via_gui_thread(fig_clone, sig_clone),
            DesktopBackend::SingleWindow => render_via_single_window(fig_clone, sig_clone),
        };

        match attempt {
            Ok(msg) => return Ok(msg),
            Err(err) => {
                warn!("runmat-plot: backend {:?} failed: {}", backend, err);
                last_err = Some(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| "No interactive plotting backend succeeded".to_string()))
}

/// Request that the window associated with `handle` close.
pub fn request_close(handle: u32) {
    if let Ok(map) = WINDOW_SIGNALS.lock() {
        if let Some(signal) = map.get(&handle) {
            signal.store(true, Ordering::SeqCst);
        }
    }
}

fn backend_preference() -> &'static [DesktopBackend] {
    BACKEND_PREF
        .get_or_init(|| parse_backend_env().unwrap_or_else(default_backend_order))
        .as_slice()
}

fn parse_backend_env() -> Option<Vec<DesktopBackend>> {
    let raw = env::var("RUNMAT_PLOT_DESKTOP_BACKEND").ok()?;
    let mut list = Vec::new();
    for token in raw.split(',') {
        let trimmed = token.trim().to_ascii_lowercase();
        if trimmed.is_empty() {
            continue;
        }
        let backend = match trimmed.as_str() {
            "native" | "native_window" => DesktopBackend::NativeWindow,
            "gui" | "gui_thread" => DesktopBackend::GuiThread,
            "single" | "single_window" => DesktopBackend::SingleWindow,
            _ => continue,
        };
        list.push(backend);
    }
    if list.is_empty() {
        None
    } else {
        Some(list)
    }
}

fn default_backend_order() -> Vec<DesktopBackend> {
    vec![
        DesktopBackend::NativeWindow,
        DesktopBackend::GuiThread,
        DesktopBackend::SingleWindow,
    ]
}

fn clone_signal(signal: &Option<CloseSignal>) -> Option<CloseSignal> {
    signal.as_ref().map(Arc::clone)
}

fn render_via_native(figure: Figure, signal: Option<CloseSignal>) -> Result<String, String> {
    crate::gui::initialize_native_window()
        .map_err(|err| format!("native window init failed: {err}"))?;
    crate::gui::show_plot_native_window_with_signal(figure, signal)
}

fn render_via_gui_thread(figure: Figure, signal: Option<CloseSignal>) -> Result<String, String> {
    crate::gui::initialize_gui_manager()
        .map_err(|err| format!("GUI manager init failed: {err}"))?;
    match crate::gui::show_plot_global_with_signal(figure, signal) {
        Ok(result) => gui_result_to_string(result),
        Err(err) => gui_result_to_string(err),
    }
}

fn render_via_single_window(figure: Figure, signal: Option<CloseSignal>) -> Result<String, String> {
    crate::gui::single_window_manager::show_plot_sequential_with_signal(figure, signal)
}

fn gui_result_to_string(result: crate::gui::GuiOperationResult) -> Result<String, String> {
    match result {
        crate::gui::GuiOperationResult::Success(msg)
        | crate::gui::GuiOperationResult::Cancelled(msg) => Ok(msg),
        crate::gui::GuiOperationResult::Error { message, .. } => Err(message),
    }
}
