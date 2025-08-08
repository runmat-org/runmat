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
            title: "RunMat - Interactive Visualization | Powered by Dystr".to_string(),
            width: 1200,
            height: 800,
            resizable: true,
            maximized: false,
            vsync: true,
        }
    }
}

/// Interactive plot window with full WGPU rendering
#[cfg(feature = "gui")]
pub struct PlotWindow<'window> {
    pub window: std::sync::Arc<winit::window::Window>,
    pub event_loop: Option<winit::event_loop::EventLoop<()>>,
    pub plot_renderer: crate::core::PlotRenderer,
    pub plot_overlay: crate::gui::PlotOverlay,
    pub surface: wgpu::Surface<'window>,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,
    pub config: WindowConfig,
    pub mouse_position: glam::Vec2,
    pub is_mouse_over_plot: bool,
}

// The implementation is in window_impl.rs in the same directory

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
