//! Main window management for interactive GUI plotting
//! 
//! Provides the main application window with integrated plot viewport
//! and control panels using winit and egui.



/// Configuration for the plot window
#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
    pub maximized: bool,
    pub vsync: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "RustMat Plot - Interactive Visualization".to_string(),
            width: 1200,
            height: 800,
            resizable: true,
            maximized: false,
            vsync: true,
        }
    }
}

/// Interactive plot window - simplified for now
#[cfg(feature = "gui")]
pub struct PlotWindow<'window> {
    config: WindowConfig,
    _lifetime: std::marker::PhantomData<&'window ()>,
}

#[cfg(feature = "gui")]
mod window_impl;

// Stub implementation for non-GUI builds
#[cfg(not(feature = "gui"))]
pub struct PlotWindow;

#[cfg(not(feature = "gui"))]
impl PlotWindow {
    pub async fn new(_config: WindowConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Err("GUI feature not enabled".into())
    }
    
    pub fn add_test_plot(&mut self) {
        // No-op for non-GUI builds
    }
    
    pub fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        Err("GUI feature not enabled".into())
    }
}