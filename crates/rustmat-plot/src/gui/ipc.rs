//! IPC system for communicating plot requests to the main GUI thread
//!
//! This module provides message passing capabilities to handle the macOS requirement
//! that EventLoop must be created on the main thread.

use crate::plots::Figure;
use std::sync::mpsc;

/// Messages sent from runtime threads to the main GUI thread
#[derive(Debug)]
pub enum GuiMessage {
    /// Request to show an interactive plot
    ShowPlot {
        figure: Figure,
        response: mpsc::Sender<Result<String, String>>,
    },
    /// Request to shut down the GUI thread
    Shutdown,
}

/// Handle for communicating with the GUI thread
#[derive(Clone)]
pub struct GuiHandle {
    sender: mpsc::Sender<GuiMessage>,
}

impl GuiHandle {
    /// Create a new GUI handle
    pub fn new(sender: mpsc::Sender<GuiMessage>) -> Self {
        Self { sender }
    }

    /// Request to show an interactive plot (blocking)
    pub fn show_plot(&self, figure: Figure) -> Result<String, String> {
        let (response_tx, response_rx) = mpsc::channel();
        
        let message = GuiMessage::ShowPlot {
            figure,
            response: response_tx,
        };

        // Send the message to the GUI thread
        self.sender.send(message)
            .map_err(|_| "GUI thread is not running".to_string())?;

        // Wait for the response
        response_rx.recv()
            .map_err(|_| "GUI thread response channel closed".to_string())?
    }

    /// Request GUI thread shutdown
    pub fn shutdown(&self) -> Result<(), String> {
        self.sender.send(GuiMessage::Shutdown)
            .map_err(|_| "GUI thread is not running".to_string())
    }
}

/// Manager for the GUI thread that processes messages
pub struct GuiManager {
    receiver: mpsc::Receiver<GuiMessage>,
}

impl GuiManager {
    /// Create a new GUI manager and return both the manager and a handle
    pub fn new() -> (Self, GuiHandle) {
        let (sender, receiver) = mpsc::channel();
        let manager = GuiManager { receiver };
        let handle = GuiHandle::new(sender);
        (manager, handle)
    }

    /// Run the GUI message loop (must be called on the main thread)
    #[cfg(feature = "gui")]
    pub async fn run(self) -> Result<(), String> {
        // Process messages
        loop {
            match self.receiver.recv() {
                Ok(GuiMessage::ShowPlot { figure, response }) => {
                    // Create a new plot window for each request
                    let result = match self.create_plot_window(figure).await {
                        Ok(result) => Ok(result),
                        Err(e) => Err(format!("Failed to show plot: {}", e)),
                    };
                    let _ = response.send(result);
                }
                Ok(GuiMessage::Shutdown) => {
                    break;
                }
                Err(_) => {
                    // Channel closed, exit
                    break;
                }
            }
        }

        Ok(())
    }

    #[cfg(not(feature = "gui"))]
    pub async fn run(self) -> Result<(), String> {
        // For non-GUI builds, just process shutdown messages
        loop {
            match self.receiver.recv() {
                Ok(GuiMessage::ShowPlot { response, .. }) => {
                    let _ = response.send(Err("GUI feature not enabled".to_string()));
                }
                Ok(GuiMessage::Shutdown) => {
                    break;
                }
                Err(_) => {
                    break;
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "gui")]
    async fn create_plot_window(&self, figure: Figure) -> Result<String, String> {
        use crate::gui::window::WindowConfig;
        use crate::gui::PlotWindow;

        let config = WindowConfig::default();
        let mut window = PlotWindow::new(config).await
            .map_err(|e| format!("Failed to create window: {}", e))?;

        // Set the figure data
        window.set_figure(figure);

        // Run the window event loop directly (blocking)
        // This must run on the main thread, especially on macOS
        if let Err(e) = window.run().await {
            return Err(format!("Window event loop error: {}", e));
        }

        Ok("Interactive plot window closed".to_string())
    }
}

use std::sync::OnceLock;

/// Global GUI handle (thread-safe)
static GUI_HANDLE: OnceLock<GuiHandle> = OnceLock::new();

/// Initialize the global GUI handle (called from main thread)
pub fn init_gui_handle(handle: GuiHandle) {
    let _ = GUI_HANDLE.set(handle);
}

/// Get the global GUI handle
pub fn get_gui_handle() -> Option<GuiHandle> {
    GUI_HANDLE.get().cloned()
}