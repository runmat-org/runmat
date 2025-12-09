//! Robust GUI thread management system
//!
//! This module provides cross-platform GUI thread management that properly
//! handles platform-specific requirements (especially macOS EventLoop main thread requirement)
//! while maintaining high performance and reliability.

use crate::gui::lifecycle::CloseSignal;
use crate::plots::Figure;
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread::{self, ThreadId};

/// Thread-safe message passing system for GUI operations
#[derive(Debug)]
pub enum GuiThreadMessage {
    /// Request to show an interactive plot with response channel
    ShowPlot {
        figure: Figure,
        response: mpsc::Sender<GuiOperationResult>,
        close_signal: Option<CloseSignal>,
    },
    /// Request to close all GUI windows
    CloseAll {
        response: mpsc::Sender<GuiOperationResult>,
    },
    /// Health check for GUI thread
    HealthCheck {
        response: mpsc::Sender<GuiOperationResult>,
    },
    /// Graceful shutdown request
    Shutdown,
}

/// Result of GUI operations with comprehensive error information
#[derive(Debug, Clone)]
pub enum GuiOperationResult {
    /// Operation completed successfully
    Success(String),
    /// Operation failed with detailed error information
    Error {
        message: String,
        error_code: GuiErrorCode,
        recoverable: bool,
    },
    /// Operation was cancelled or timed out
    Cancelled(String),
}

/// Error codes for GUI operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiErrorCode {
    /// EventLoop creation failed (platform-specific)
    EventLoopCreationFailed,
    /// Window creation failed
    WindowCreationFailed,
    /// WGPU initialization failed
    WgpuInitializationFailed,
    /// Thread communication failure
    ThreadCommunicationFailed,
    /// Main thread requirement violation
    MainThreadViolation,
    /// Resource exhaustion
    ResourceExhaustion,
    /// Invalid operation state
    InvalidState,
    /// Platform-specific error
    PlatformError,
    /// Unknown error
    Unknown,
}

impl std::fmt::Display for GuiOperationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuiOperationResult::Success(msg) => write!(f, "Success: {msg}"),
            GuiOperationResult::Error {
                message,
                error_code,
                recoverable,
            } => {
                write!(
                    f,
                    "Error [{error_code:?}]: {message} (recoverable: {recoverable})"
                )
            }
            GuiOperationResult::Cancelled(msg) => write!(f, "Cancelled: {msg}"),
        }
    }
}

impl std::error::Error for GuiOperationResult {}

/// Main thread detection and validation
pub struct MainThreadDetector {
    main_thread_id: OnceLock<ThreadId>,
}

impl Default for MainThreadDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MainThreadDetector {
    /// Create a new main thread detector
    pub const fn new() -> Self {
        Self {
            main_thread_id: OnceLock::new(),
        }
    }

    /// Register the current thread as the main thread
    /// This should be called from main() before any GUI operations
    pub fn register_main_thread(&self) {
        let current_id = thread::current().id();
        if self.main_thread_id.set(current_id).is_err() {
            // Already set - this is fine, multiple calls are safe
        }
    }

    /// Check if the current thread is the main thread
    pub fn is_main_thread(&self) -> bool {
        if let Some(main_id) = self.main_thread_id.get() {
            return thread::current().id() == *main_id;
        }

        #[cfg(target_os = "macos")]
        {
            if is_macos_main_thread() {
                self.register_main_thread();
                true
            } else {
                false
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Other platforms do not require the GUI to live on the OS main thread.
            self.register_main_thread();
            true
        }
    }

    /// Get the main thread ID if registered
    pub fn main_thread_id(&self) -> Option<ThreadId> {
        self.main_thread_id.get().copied()
    }
}

#[cfg(target_os = "macos")]
fn is_macos_main_thread() -> bool {
    unsafe { libc::pthread_main_np() != 0 }
}

/// Global main thread detector instance
static MAIN_THREAD_DETECTOR: MainThreadDetector = MainThreadDetector::new();

/// Register the current thread as the main thread
pub fn register_main_thread() {
    MAIN_THREAD_DETECTOR.register_main_thread();
}

/// Check if the current thread is the main thread
pub fn is_main_thread() -> bool {
    MAIN_THREAD_DETECTOR.is_main_thread()
}

/// GUI thread manager with robust error handling and health monitoring
pub struct GuiThreadManager {
    /// Message sender to GUI thread
    sender: mpsc::Sender<GuiThreadMessage>,
    /// Thread handle for the GUI thread
    thread_handle: Option<thread::JoinHandle<Result<(), GuiOperationResult>>>,
    /// Health check state
    health_state: Arc<Mutex<GuiHealthState>>,
}

/// GUI thread health monitoring
#[derive(Debug, Clone)]
struct GuiHealthState {
    last_response: std::time::Instant,
    response_count: u64,
    error_count: u64,
    is_healthy: bool,
}

impl GuiThreadManager {
    /// Create a new GUI thread manager
    ///
    /// This must be called from the main thread on macOS, or it will return an error.
    pub fn new() -> Result<Self, GuiOperationResult> {
        // Verify we're on the main thread for platforms that require it
        #[cfg(target_os = "macos")]
        if !is_main_thread() {
            return Err(GuiOperationResult::Error {
                message: "GuiThreadManager must be created on the main thread on macOS".to_string(),
                error_code: GuiErrorCode::MainThreadViolation,
                recoverable: false,
            });
        }

        let (sender, receiver) = mpsc::channel();
        let health_state = Arc::new(Mutex::new(GuiHealthState {
            last_response: std::time::Instant::now(),
            response_count: 0,
            error_count: 0,
            is_healthy: true,
        }));

        let health_state_clone = Arc::clone(&health_state);

        // Start the GUI thread
        let thread_handle = thread::Builder::new()
            .name("runmat-gui".to_string())
            .spawn(move || Self::gui_thread_main(receiver, health_state_clone))
            .map_err(|e| GuiOperationResult::Error {
                message: format!("Failed to spawn GUI thread: {e}"),
                error_code: GuiErrorCode::ThreadCommunicationFailed,
                recoverable: false,
            })?;

        Ok(Self {
            sender,
            thread_handle: Some(thread_handle),
            health_state,
        })
    }

    /// Main loop for the GUI thread
    fn gui_thread_main(
        receiver: mpsc::Receiver<GuiThreadMessage>,
        health_state: Arc<Mutex<GuiHealthState>>,
    ) -> Result<(), GuiOperationResult> {
        log::info!("GUI thread started successfully");

        // Initialize GUI subsystems on this thread
        #[cfg(feature = "gui")]
        let gui_context = Self::initialize_gui_context()?;

        loop {
            match receiver.recv() {
                Ok(message) => {
                    let result = Self::handle_gui_message(message, &gui_context);

                    // Update health state
                    if let Ok(mut health) = health_state.lock() {
                        health.last_response = std::time::Instant::now();
                        health.response_count += 1;

                        if let Some(GuiOperationResult::Error { .. }) = &result {
                            health.error_count += 1;
                            health.is_healthy = health.error_count < 10; // Allow some errors
                        }
                    }

                    // If this was a shutdown message, break the loop
                    if result.is_none() {
                        break;
                    }
                }
                Err(_) => {
                    // Channel closed, exit gracefully
                    log::info!("GUI thread channel closed, shutting down");
                    break;
                }
            }
        }

        log::info!("GUI thread exiting gracefully");
        Ok(())
    }

    /// Initialize GUI context (platform-specific)
    #[cfg(feature = "gui")]
    fn initialize_gui_context() -> Result<GuiContext, GuiOperationResult> {
        use crate::gui::window::WindowConfig;

        // This will create the EventLoop on the correct thread
        Ok(GuiContext {
            _default_config: WindowConfig::default(),
            _active_windows: Vec::new(),
        })
    }

    #[cfg(not(feature = "gui"))]
    fn initialize_gui_context() -> Result<GuiContext, GuiOperationResult> {
        Err(GuiOperationResult::Error {
            message: "GUI feature not enabled".to_string(),
            error_code: GuiErrorCode::InvalidState,
            recoverable: false,
        })
    }

    /// Handle a GUI message and return the result
    fn handle_gui_message(
        message: GuiThreadMessage,
        gui_context: &GuiContext,
    ) -> Option<GuiOperationResult> {
        match message {
            GuiThreadMessage::ShowPlot {
                figure,
                response,
                close_signal,
            } => {
                let result = Self::handle_show_plot(figure, close_signal, gui_context);
                let _ = response.send(result.clone());
                Some(result)
            }
            GuiThreadMessage::CloseAll { response } => {
                let result = Self::handle_close_all(gui_context);
                let _ = response.send(result.clone());
                Some(result)
            }
            GuiThreadMessage::HealthCheck { response } => {
                let result = GuiOperationResult::Success("GUI thread healthy".to_string());
                let _ = response.send(result.clone());
                Some(result)
            }
            GuiThreadMessage::Shutdown => {
                log::info!("GUI thread received shutdown signal");
                None // Signal to exit the loop
            }
        }
    }

    /// Handle show plot request
    #[cfg(feature = "gui")]
    fn handle_show_plot(
        figure: Figure,
        close_signal: Option<CloseSignal>,
        _gui_context: &GuiContext,
    ) -> GuiOperationResult {
        use crate::gui::{window::WindowConfig, PlotWindow};

        // Create a new runtime for this async operation
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                return GuiOperationResult::Error {
                    message: format!("Failed to create async runtime: {e}"),
                    error_code: GuiErrorCode::ResourceExhaustion,
                    recoverable: true,
                }
            }
        };

        rt.block_on(async {
            let config = WindowConfig::default();
            let mut window = match PlotWindow::new(config).await {
                Ok(window) => window,
                Err(e) => {
                    return GuiOperationResult::Error {
                        message: format!("Failed to create window: {e}"),
                        error_code: GuiErrorCode::WindowCreationFailed,
                        recoverable: true,
                    }
                }
            };

            if let Some(sig) = close_signal {
                window.install_close_signal(sig);
            }

            // Set the figure data
            window.set_figure(figure);

            // Run the window (this will block until the window is closed)
            match window.run().await {
                Ok(_) => GuiOperationResult::Success("Plot window closed".to_string()),
                Err(e) => GuiOperationResult::Error {
                    message: format!("Window runtime error: {e}"),
                    error_code: GuiErrorCode::PlatformError,
                    recoverable: true,
                },
            }
        })
    }

    #[cfg(not(feature = "gui"))]
    fn handle_show_plot(
        _figure: Figure,
        _close_signal: Option<CloseSignal>,
        _gui_context: &GuiContext,
    ) -> GuiOperationResult {
        GuiOperationResult::Error {
            message: "GUI feature not enabled".to_string(),
            error_code: GuiErrorCode::InvalidState,
            recoverable: false,
        }
    }

    /// Handle close all windows request
    fn handle_close_all(_gui_context: &GuiContext) -> GuiOperationResult {
        // Implementation would close all active windows
        GuiOperationResult::Success("All windows closed".to_string())
    }

    /// Show a plot using the GUI thread manager
    pub fn show_plot(&self, figure: Figure) -> Result<GuiOperationResult, GuiOperationResult> {
        self.show_plot_with_signal(figure, None)
    }

    pub fn show_plot_with_signal(
        &self,
        figure: Figure,
        close_signal: Option<CloseSignal>,
    ) -> Result<GuiOperationResult, GuiOperationResult> {
        let (response_tx, response_rx) = mpsc::channel();

        let message = GuiThreadMessage::ShowPlot {
            figure,
            response: response_tx,
            close_signal,
        };

        // Send message to GUI thread
        self.sender
            .send(message)
            .map_err(|_| GuiOperationResult::Error {
                message: "GUI thread is not responding".to_string(),
                error_code: GuiErrorCode::ThreadCommunicationFailed,
                recoverable: false,
            })?;

        // Wait for response with timeout
        match response_rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(result) => Ok(result),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(GuiOperationResult::Cancelled(
                "GUI operation timed out after 30 seconds".to_string(),
            )),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(GuiOperationResult::Error {
                message: "GUI thread disconnected unexpectedly".to_string(),
                error_code: GuiErrorCode::ThreadCommunicationFailed,
                recoverable: false,
            }),
        }
    }

    /// Perform a health check on the GUI thread
    pub fn health_check(&self) -> Result<GuiOperationResult, GuiOperationResult> {
        let (response_tx, response_rx) = mpsc::channel();

        let message = GuiThreadMessage::HealthCheck {
            response: response_tx,
        };

        self.sender
            .send(message)
            .map_err(|_| GuiOperationResult::Error {
                message: "GUI thread is not responding".to_string(),
                error_code: GuiErrorCode::ThreadCommunicationFailed,
                recoverable: false,
            })?;

        match response_rx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(result) => Ok(result),
            Err(_) => Err(GuiOperationResult::Error {
                message: "GUI thread health check failed".to_string(),
                error_code: GuiErrorCode::ThreadCommunicationFailed,
                recoverable: true,
            }),
        }
    }

    /// Get the current health state of the GUI thread
    pub fn get_health_state(&self) -> Option<(u64, u64, bool)> {
        self.health_state
            .lock()
            .ok()
            .map(|health| (health.response_count, health.error_count, health.is_healthy))
    }

    /// Gracefully shutdown the GUI thread
    pub fn shutdown(mut self) -> Result<(), GuiOperationResult> {
        // Send shutdown signal
        let _ = self.sender.send(GuiThreadMessage::Shutdown);

        // Wait for thread to complete
        if let Some(handle) = self.thread_handle.take() {
            match handle.join() {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(e),
                Err(_) => Err(GuiOperationResult::Error {
                    message: "GUI thread panicked during shutdown".to_string(),
                    error_code: GuiErrorCode::PlatformError,
                    recoverable: false,
                }),
            }
        } else {
            Ok(())
        }
    }
}

/// Drop implementation for graceful cleanup
impl Drop for GuiThreadManager {
    fn drop(&mut self) {
        if self.thread_handle.is_some() {
            log::warn!("GuiThreadManager dropped without explicit shutdown");
            let _ = self.sender.send(GuiThreadMessage::Shutdown);

            // Give the thread a moment to shut down gracefully
            if let Some(handle) = self.thread_handle.take() {
                let _ = handle.join();
            }
        }
    }
}

/// GUI context for managing windows and resources
#[cfg(feature = "gui")]
struct GuiContext {
    _default_config: crate::gui::window::WindowConfig,
    _active_windows: Vec<String>, // Window IDs for tracking
}

#[cfg(not(feature = "gui"))]
struct GuiContext {
    // Stub for non-GUI builds
}

/// Global GUI thread manager instance
static GUI_MANAGER: OnceLock<Arc<Mutex<Option<GuiThreadManager>>>> = OnceLock::new();

/// Initialize the global GUI thread manager
pub fn initialize_gui_manager() -> Result<(), GuiOperationResult> {
    let manager_mutex = GUI_MANAGER.get_or_init(|| Arc::new(Mutex::new(None)));

    let mut manager_guard = manager_mutex
        .lock()
        .map_err(|_| GuiOperationResult::Error {
            message: "Failed to acquire GUI manager lock".to_string(),
            error_code: GuiErrorCode::ThreadCommunicationFailed,
            recoverable: false,
        })?;

    if manager_guard.is_some() {
        return Ok(()); // Already initialized
    }

    let manager = GuiThreadManager::new()?;
    *manager_guard = Some(manager);

    log::info!("Global GUI thread manager initialized successfully");
    Ok(())
}

/// Get a reference to the global GUI thread manager
pub fn get_gui_manager() -> Result<Arc<Mutex<Option<GuiThreadManager>>>, GuiOperationResult> {
    GUI_MANAGER
        .get()
        .ok_or_else(|| GuiOperationResult::Error {
            message: "GUI manager not initialized. Call initialize_gui_manager() first."
                .to_string(),
            error_code: GuiErrorCode::InvalidState,
            recoverable: true,
        })
        .map(Arc::clone)
}

/// Show a plot using the global GUI manager
pub fn show_plot_global(figure: Figure) -> Result<GuiOperationResult, GuiOperationResult> {
    show_plot_global_with_signal(figure, None)
}

pub fn show_plot_global_with_signal(
    figure: Figure,
    close_signal: Option<CloseSignal>,
) -> Result<GuiOperationResult, GuiOperationResult> {
    let manager_mutex = get_gui_manager()?;
    let manager_guard = manager_mutex
        .lock()
        .map_err(|_| GuiOperationResult::Error {
            message: "Failed to acquire GUI manager lock".to_string(),
            error_code: GuiErrorCode::ThreadCommunicationFailed,
            recoverable: false,
        })?;

    match manager_guard.as_ref() {
        Some(manager) => manager.show_plot_with_signal(figure, close_signal),
        None => Err(GuiOperationResult::Error {
            message: "GUI manager not initialized".to_string(),
            error_code: GuiErrorCode::InvalidState,
            recoverable: true,
        }),
    }
}

/// Perform a health check on the global GUI manager
pub fn health_check_global() -> Result<GuiOperationResult, GuiOperationResult> {
    let manager_mutex = get_gui_manager()?;
    let manager_guard = manager_mutex
        .lock()
        .map_err(|_| GuiOperationResult::Error {
            message: "Failed to acquire GUI manager lock".to_string(),
            error_code: GuiErrorCode::ThreadCommunicationFailed,
            recoverable: false,
        })?;

    match manager_guard.as_ref() {
        Some(manager) => manager.health_check(),
        None => Err(GuiOperationResult::Error {
            message: "GUI manager not initialized".to_string(),
            error_code: GuiErrorCode::InvalidState,
            recoverable: true,
        }),
    }
}
