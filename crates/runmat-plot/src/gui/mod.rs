//! GUI components for interactive plotting
//!
//! Provides interactive controls, widgets, and GUI layout
//! for desktop plotting applications.

#[cfg(feature = "gui")]
pub mod controls;

#[cfg(feature = "gui")]
pub mod lifecycle;
pub mod native_window;
#[cfg(feature = "gui")]
pub mod plot_overlay;
pub mod single_window_manager; // V8-caliber single window management
pub mod thread_manager; // Robust thread management
#[cfg(feature = "gui")]
pub mod widgets;
#[cfg(feature = "gui")]
pub mod window;
#[cfg(feature = "gui")]
pub mod window_impl; // Cross-platform native window management

#[cfg(feature = "gui")]
pub use controls::PlotControls;
#[cfg(feature = "gui")]
pub use plot_overlay::PlotOverlay;
#[cfg(feature = "gui")]
pub use window::*;

// Legacy IPC exports (deprecated)
// Legacy IPC system removed - using sequential window manager instead

// Thread manager exports
pub use thread_manager::{
    get_gui_manager, health_check_global, initialize_gui_manager, is_main_thread,
    register_main_thread, show_plot_global, show_plot_global_with_signal, GuiErrorCode,
    GuiOperationResult, GuiThreadManager,
};

// Native window exports
pub use native_window::{
    initialize_native_window, is_native_window_available, show_plot_native_window,
    show_plot_native_window_with_signal, NativeWindowManager, NativeWindowResult,
};

// Single window manager exports
pub use single_window_manager::{is_window_available, show_plot_sequential};
