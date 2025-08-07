//! Single Window Manager - V8-caliber EventLoop management
//!
//! Ensures only one interactive plot window exists at a time, avoiding
//! winit's "EventLoop can't be recreated" error.

use crate::gui::window::WindowConfig;
use crate::plots::Figure;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Global flag to track if an EventLoop is currently active
static WINDOW_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Thread-safe window manager for sequential plot display
static WINDOW_MANAGER: Mutex<()> = Mutex::new(());

/// Show a plot using a single, managed window approach
pub fn show_plot_sequential(figure: Figure) -> Result<String, String> {
    // Acquire exclusive access to the window system
    let _guard = WINDOW_MANAGER
        .lock()
        .map_err(|_| "Window manager lock failed".to_string())?;

    // Check if a window is already active
    if WINDOW_ACTIVE.load(Ordering::Acquire) {
        return Err("Another plot window is already open. Please close it first.".to_string());
    }

    // Mark window as active
    WINDOW_ACTIVE.store(true, Ordering::Release);

    let result = show_plot_internal(figure);

    // Mark window as inactive when done
    WINDOW_ACTIVE.store(false, Ordering::Release);

    result
}

/// Internal function that actually creates and runs the window
fn show_plot_internal(figure: Figure) -> Result<String, String> {
    use crate::gui::PlotWindow;
    
    // Create window directly on the current thread (main thread)
    let config = WindowConfig::default();
    
    // Use existing runtime or create one if none exists
    let handle = tokio::runtime::Handle::try_current();
    
    match handle {
        Ok(handle) => {
            // Use existing runtime
            tokio::task::block_in_place(|| {
                handle.block_on(async {
                    let mut window = PlotWindow::new(config)
                        .await
                        .map_err(|e| format!("Failed to create plot window: {}", e))?;
                    
                    // Set the figure data
                    window.set_figure(figure);
                    
                    // Run the window (this will consume the EventLoop)
                    window.run().await.map_err(|e| format!("Window execution failed: {}", e))?;
                    
                    Ok("Plot window closed successfully".to_string())
                })
            })
        }
        Err(_) => {
            // No runtime available, create one
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("Failed to create async runtime: {}", e))?;
                
            rt.block_on(async {
                let mut window = PlotWindow::new(config)
                    .await
                    .map_err(|e| format!("Failed to create plot window: {}", e))?;
                
                // Set the figure data
                window.set_figure(figure);
                
                // Run the window (this will consume the EventLoop)
                window.run().await.map_err(|e| format!("Window execution failed: {}", e))?;
                
                Ok("Plot window closed successfully".to_string())
            })
        }
    }
}

/// Check if the window system is available
pub fn is_window_available() -> bool {
    !WINDOW_ACTIVE.load(Ordering::Acquire)
}