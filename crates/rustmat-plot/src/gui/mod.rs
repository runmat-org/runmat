//! GUI components for interactive plotting
//! 
//! Provides interactive controls, widgets, and GUI layout
//! for desktop plotting applications.

#[cfg(feature = "gui")]
pub mod window;
#[cfg(feature = "gui")]
pub mod window_impl;
#[cfg(feature = "gui")]
pub mod plot_overlay;
#[cfg(feature = "gui")]
pub mod controls;
#[cfg(feature = "gui")]
pub mod widgets;
pub mod ipc; // Legacy IPC system (deprecated)
pub mod thread_manager; // Robust thread management
pub mod native_window; // Cross-platform native window management



#[cfg(feature = "gui")]
pub use window::*;
#[cfg(feature = "gui")]
pub use plot_overlay::PlotOverlay;
#[cfg(feature = "gui")]
pub use controls::PlotControls;

// Legacy IPC exports (deprecated)
pub use ipc::{GuiHandle, GuiManager, GuiMessage, init_gui_handle, get_gui_handle};

// Thread manager exports
pub use thread_manager::{
    GuiThreadManager, GuiOperationResult, GuiErrorCode,
    register_main_thread, is_main_thread, initialize_gui_manager,
    get_gui_manager, show_plot_global, health_check_global,
};

// Native window exports
pub use native_window::{
    NativeWindowManager, NativeWindowResult, initialize_native_window,
    show_plot_native_window, is_native_window_available,
};



