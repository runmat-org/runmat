//! Native cross-platform window management
//!
//! This module provides a robust, cross-platform native window implementation
//! that properly handles platform-specific requirements (especially macOS EventLoop)
//! while leveraging our world-class WGPU rendering engine.

use crate::plots::Figure;
use std::sync::{Arc, Mutex, OnceLock};

/// Result from native window operations
#[derive(Debug, Clone)]
pub enum NativeWindowResult {
    Success(String),
    Error(String),
    WindowClosed,
}

impl std::fmt::Display for NativeWindowResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeWindowResult::Success(msg) => write!(f, "Success: {msg}"),
            NativeWindowResult::Error(msg) => write!(f, "Error: {msg}"),
            NativeWindowResult::WindowClosed => write!(f, "Window closed by user"),
        }
    }
}

/// Native window manager that properly handles cross-platform requirements
pub struct NativeWindowManager {
    is_initialized: bool,
}

impl Default for NativeWindowManager {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeWindowManager {
    /// Create a new native window manager
    pub fn new() -> Self {
        Self {
            is_initialized: false,
        }
    }

    /// Initialize the native window system
    pub fn initialize(&mut self) -> Result<(), String> {
        if self.is_initialized {
            return Ok(());
        }

        // Verify main thread requirements on macOS
        #[cfg(target_os = "macos")]
        {
            if !crate::gui::thread_manager::is_main_thread() {
                return Err(
                    "Native window manager must be initialized on the main thread on macOS"
                        .to_string(),
                );
            }
        }

        self.is_initialized = true;
        Ok(())
    }

    /// Show a plot using native window with proper cross-platform handling
    pub fn show_plot_native(&self, figure: Figure) -> Result<NativeWindowResult, String> {
        if !self.is_initialized {
            return Err("Native window manager not initialized".to_string());
        }

        // On macOS, run directly on main thread
        #[cfg(target_os = "macos")]
        {
            self.show_plot_main_thread(figure)
        }

        // On other platforms, can use thread-based approach
        #[cfg(not(target_os = "macos"))]
        {
            self.show_plot_threaded(figure)
        }
    }

    /// Show plot directly on main thread (macOS)
    #[cfg(target_os = "macos")]
    fn show_plot_main_thread(&self, figure: Figure) -> Result<NativeWindowResult, String> {
        use pollster;

        // Create and run the plot window directly using our WGPU rendering engine
        let config = crate::gui::window::WindowConfig::default();
        // We don't need a tokio runtime for the direct approach on macOS
        // pollster handles the async execution for us

        // Use pollster to block on the async PlotWindow creation
        match pollster::block_on(crate::gui::PlotWindow::new(config)) {
            Ok(mut window) => {
                // Set the figure data
                window.set_figure(figure);

                // Run the window event loop (this will block until the window closes)
                match pollster::block_on(window.run()) {
                    Ok(_) => Ok(NativeWindowResult::Success(
                        "Plot window closed successfully".to_string(),
                    )),
                    Err(e) => Err(format!("Window runtime error: {e}")),
                }
            }
            Err(e) => Err(format!("Failed to create plot window: {e}")),
        }
    }

    /// Show plot using threaded approach (non-macOS)
    #[cfg(not(target_os = "macos"))]
    fn show_plot_threaded(&self, figure: Figure) -> Result<NativeWindowResult, String> {
        // For non-macOS platforms, we can use the existing thread-based approach
        // This would use the thread_manager system
        match crate::gui::show_plot_global(figure) {
            Ok(result) => match result {
                crate::gui::GuiOperationResult::Success(msg) => {
                    Ok(NativeWindowResult::Success(msg))
                }
                crate::gui::GuiOperationResult::Cancelled(_msg) => {
                    Ok(NativeWindowResult::WindowClosed)
                }
                crate::gui::GuiOperationResult::Error { message, .. } => {
                    Ok(NativeWindowResult::Error(message))
                }
            },
            Err(result) => match result {
                crate::gui::GuiOperationResult::Success(msg) => {
                    Ok(NativeWindowResult::Success(msg))
                }
                crate::gui::GuiOperationResult::Cancelled(_msg) => {
                    Ok(NativeWindowResult::WindowClosed)
                }
                crate::gui::GuiOperationResult::Error { message, .. } => Err(message),
            },
        }
    }
}

/// Global native window manager
static NATIVE_WINDOW_MANAGER: OnceLock<Arc<Mutex<NativeWindowManager>>> = OnceLock::new();

/// Initialize the native window system
pub fn initialize_native_window() -> Result<(), String> {
    let manager_mutex =
        NATIVE_WINDOW_MANAGER.get_or_init(|| Arc::new(Mutex::new(NativeWindowManager::new())));

    let mut manager = manager_mutex
        .lock()
        .map_err(|_| "Failed to acquire manager lock".to_string())?;

    manager.initialize()
}

/// Show a plot using native window
pub fn show_plot_native_window(figure: Figure) -> Result<String, String> {
    let manager_mutex = NATIVE_WINDOW_MANAGER
        .get()
        .ok_or_else(|| "Native window system not initialized".to_string())?;

    let manager = manager_mutex
        .lock()
        .map_err(|_| "Failed to acquire manager lock".to_string())?;

    match manager.show_plot_native(figure) {
        Ok(NativeWindowResult::Success(msg)) => Ok(msg),
        Ok(NativeWindowResult::WindowClosed) => Ok("Plot window closed by user".to_string()),
        Ok(NativeWindowResult::Error(msg)) => Err(msg),
        Err(msg) => Err(msg),
    }
}

/// Check if native window system is available
pub fn is_native_window_available() -> bool {
    NATIVE_WINDOW_MANAGER.get().is_some()
}
