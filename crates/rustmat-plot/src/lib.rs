//! RustMat Plot - World-Class Interactive Plotting Library
//! 
//! This library provides both static and interactive plotting capabilities
//! with GPU acceleration, comprehensive 2D/3D support, and Jupyter integration.

// Legacy static plotting support (always available)
use plotters::prelude::*;
use serde::Deserialize;
use std::env;
use std::fs;

pub mod simple_plots;

// Core architecture (always available for internal use)
pub mod core;

// High-level plot types
pub mod plots;

// Advanced modules
pub mod data;
pub mod export;
pub mod jupyter;
pub mod styling;

// Feature-gated modules
#[cfg(feature = "gui")]
pub mod gui;

// (Modules already declared above)

// Re-exports for convenience
pub use core::*;

// High-level API
#[cfg(feature = "gui")]
pub use gui::{PlotWindow, WindowConfig};

// Legacy IPC system (deprecated)
pub use gui::{GuiHandle, GuiManager, init_gui_handle, get_gui_handle};

// Robust GUI thread management
pub use gui::{
    GuiThreadManager, GuiOperationResult, GuiErrorCode,
    register_main_thread, is_main_thread, initialize_gui_manager,
    get_gui_manager, show_plot_global, health_check_global,
};

// Native window system
pub use gui::{
    NativeWindowManager, NativeWindowResult, initialize_native_window,
    show_plot_native_window, is_native_window_available,
};

// Styling and theming system
pub use styling::{
    PlotThemeConfig, ThemeVariant, ModernDarkTheme, Typography, Layout,
    validate_theme_config,
};





/// Environment variable specifying the path to the optional YAML config file.
pub const CONFIG_ENV: &str = "RUSTMAT_PLOT_CONFIG";

/// Plot options for customizing plot appearance
#[derive(Debug, Clone)]
pub struct PlotOptions {
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub z_label: Option<String>,
    pub x_scale: Option<String>,
    pub y_scale: Option<String>,
    pub line_style: LineStyle,
    pub marker_style: MarkerStyle,
    pub color_map: ColorMap,
    pub show_grid: bool,
    pub legend: Option<Vec<String>>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            title: None,
            x_label: None,
            y_label: None,
            z_label: None,
            x_scale: None,
            y_scale: None,
            line_style: LineStyle::Solid,
            marker_style: MarkerStyle::Circle,
            color_map: ColorMap::Default,
            show_grid: true,
            legend: None,
            width: None,
            height: None,
        }
    }
}

/// Line style options
#[derive(Debug, Clone, PartialEq)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
    None,
}

/// Marker style options
#[derive(Debug, Clone, PartialEq)]
pub enum MarkerStyle {
    Circle,
    Square,
    Triangle,
    Diamond,
    Plus,
    Cross,
    None,
}

/// Color map options
#[derive(Debug, Clone, PartialEq)]
pub enum ColorMap {
    Default,
    Jet,
    Hot,
    Cool,
    Gray,
    Spring,
    Summer,
    Autumn,
    Winter,
}

/// Style configuration loaded from YAML.
#[derive(Clone, Deserialize)]
pub struct PlotConfig {
    /// Width of the plot area.
    #[serde(default = "default_width")]
    pub width: u32,
    /// Height of the plot area.
    #[serde(default = "default_height")]
    pub height: u32,
    /// Color of line plots in hex form (`#rrggbb`).
    #[serde(default = "default_line_color")]
    pub line_color: String,
    /// Line width in pixels.
    #[serde(default = "default_line_width")]
    pub line_width: u32,
    /// Color of scatter plot points.
    #[serde(default = "default_scatter_color")]
    pub scatter_color: String,
    /// Radius of scatter plot points in pixels.
    #[serde(default = "default_marker_size")]
    pub marker_size: u32,
    /// Fill color for bar charts.
    #[serde(default = "default_bar_color")]
    pub bar_color: String,
    /// Fill color for histograms.
    #[serde(default = "default_hist_color")]
    pub hist_color: String,
    /// Background color of the drawing area.
    #[serde(default = "default_background")]
    pub background: String,
}

// Default values for configuration
fn default_line_color() -> String { "#0000ff".to_string() }
fn default_scatter_color() -> String { "#ff0000".to_string() }
fn default_bar_color() -> String { "#00ff00".to_string() }
fn default_hist_color() -> String { "#ffff00".to_string() }
fn default_background() -> String { "#ffffff".to_string() }
fn default_width() -> u32 { 800 }
fn default_height() -> u32 { 600 }
fn default_line_width() -> u32 { 2 }
fn default_marker_size() -> u32 { 4 }

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            line_color: default_line_color(),
            line_width: default_line_width(),
            scatter_color: default_scatter_color(),
            marker_size: default_marker_size(),
            bar_color: default_bar_color(),
            hist_color: default_hist_color(),
            background: default_background(),
        }
    }
}

/// Load configuration from environment variable or return default.
pub fn load_config() -> PlotConfig {
    if let Ok(config_path) = env::var(CONFIG_ENV) {
        match fs::read_to_string(&config_path) {
            Ok(yaml_str) => match serde_yaml::from_str(&yaml_str) {
                Ok(config) => return config,
                Err(e) => eprintln!("Warning: Failed to parse config {}: {}", config_path, e),
            },
            Err(e) => eprintln!("Warning: Failed to read config {}: {}", config_path, e),
        }
    }
    PlotConfig::default()
}

/// Parse a hex color string to RGBColor.
#[allow(dead_code)]
fn parse_color(hex: &str) -> Result<RGBColor, &'static str> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return Err("invalid color format");
    }
    let r = u8::from_str_radix(&hex[0..2], 16).map_err(|_| "invalid color")?;
    let g = u8::from_str_radix(&hex[2..4], 16).map_err(|_| "invalid color")?;
    let b = u8::from_str_radix(&hex[4..6], 16).map_err(|_| "invalid color")?;
    Ok(RGBColor(r, g, b))
}

// ===== WORLD-CLASS PLOTTING API =====

/// Create a line plot using the modern plotting system (backward compatibility)
pub fn plot_line(xs: &[f64], ys: &[f64], path: &str, _options: PlotOptions) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }
    
    // Use the new world-class plotting system
    let line_plot = plots::LinePlot::new(xs.to_vec(), ys.to_vec())
        .map_err(|e| format!("Failed to create line plot: {}", e))?
        .with_label("Data")
        .with_style(
            glam::Vec4::new(0.0, 0.4, 0.8, 1.0), // Blue
            2.0,
            plots::LineStyle::Solid
        );
    
    let mut figure = plots::Figure::new()
        .with_title("Line Plot")
        .with_labels("X", "Y")
        .with_grid(true);
    
    figure.add_line_plot(line_plot);
    
    // Export via simple_plots for now (fallback)
    if path.ends_with(".png") {
        simple_plots::line_plot_png(xs, ys, path, &_options)
    } else if path.ends_with(".svg") {
        simple_plots::line_plot_svg(xs, ys, path, &_options)
    } else {
        Err("Unsupported file format".to_string())
    }
}

/// Create a scatter plot using the modern plotting system (backward compatibility)
pub fn plot_scatter(xs: &[f64], ys: &[f64], path: &str, _options: PlotOptions) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }
    
    // Use the new world-class plotting system
    let scatter_plot = plots::ScatterPlot::new(xs.to_vec(), ys.to_vec())
        .map_err(|e| format!("Failed to create scatter plot: {}", e))?
        .with_label("Data")
        .with_style(
            glam::Vec4::new(0.8, 0.2, 0.2, 1.0), // Red
            5.0,
            plots::MarkerStyle::Circle
        );
    
    let mut figure = plots::Figure::new()
        .with_title("Scatter Plot")
        .with_labels("X", "Y")
        .with_grid(true);
    
    figure.add_scatter_plot(scatter_plot);
    
    // Export via simple_plots for now (fallback)
    if path.ends_with(".png") {
        simple_plots::scatter_plot_png(xs, ys, path, &_options)
    } else if path.ends_with(".svg") {
        simple_plots::scatter_plot_svg(xs, ys, path, &_options)
    } else {
        Err("Unsupported file format".to_string())
    }
}

/// Create a bar chart using the modern plotting system (backward compatibility)
pub fn plot_bar(labels: &[String], values: &[f64], path: &str, _options: PlotOptions) -> Result<(), String> {
    if labels.len() != values.len() {
        return Err("labels and values length mismatch".into());
    }
    
    // Use the new world-class plotting system
    let bar_chart = plots::BarChart::new(labels.to_vec(), values.to_vec())
        .map_err(|e| format!("Failed to create bar chart: {}", e))?
        .with_label("Values")
        .with_style(glam::Vec4::new(0.2, 0.6, 0.3, 1.0), 0.8); // Green bars with 80% width
    
    let mut figure = plots::Figure::new()
        .with_title("Bar Chart")
        .with_labels("Categories", "Values")
        .with_grid(true);
    
    figure.add_bar_chart(bar_chart);
    
    // Export via simple_plots for now (fallback)
    if path.ends_with(".png") {
        simple_plots::bar_chart_png(labels, values, path, &_options)
    } else {
        Err("SVG bar charts not yet implemented".to_string())
    }
}

/// Create a histogram using the modern plotting system (backward compatibility)
pub fn plot_histogram(data: &[f64], bins: usize, path: &str, _options: PlotOptions) -> Result<(), String> {
    // Use the new world-class plotting system
    let histogram = plots::Histogram::new(data.to_vec(), bins)
        .map_err(|e| format!("Failed to create histogram: {}", e))?
        .with_label("Frequency")
        .with_style(glam::Vec4::new(0.6, 0.3, 0.7, 1.0), false); // Purple, not normalized
    
    let mut figure = plots::Figure::new()
        .with_title("Histogram")
        .with_labels("Values", "Frequency")
        .with_grid(true);
    
    figure.add_histogram(histogram);
    
    // Export via simple_plots for now (fallback)
    if path.ends_with(".png") {
        simple_plots::histogram_png(data, bins, path, &_options)
    } else {
        Err("SVG histograms not yet implemented".to_string())
    }
}

// ===== INTERACTIVE GUI API =====

/// Launch an interactive plot window (requires GUI feature)
#[cfg(feature = "gui")]
pub async fn show_interactive() -> Result<(), Box<dyn std::error::Error>> {
    let config = WindowConfig::default();
    let mut window = PlotWindow::new(config).await?;
    
    // Add a test plot
    window.add_test_plot();
    
    // Run the event loop
    window.run().await?;
    
    Ok(())
}

/// High-level function to create an interactive plot (alias for show_interactive)
#[cfg(feature = "gui")]
pub async fn interactive_plot() -> Result<(), Box<dyn std::error::Error>> {
    show_interactive().await
}

/// Launch an interactive plot window with custom configuration (requires GUI feature)  
#[cfg(feature = "gui")]
pub async fn show_interactive_with_config(config: WindowConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut window = PlotWindow::new(config).await?;
    window.add_test_plot();
    window.run().await?;
    Ok(())
}

/// Launch an interactive plot window with a specific figure (requires GUI feature)
#[cfg(feature = "gui")]
pub async fn show_interactive_with_figure(figure: &plots::Figure) -> Result<(), Box<dyn std::error::Error>> {
    let config = WindowConfig::default();
    let mut window = PlotWindow::new(config).await?;
    
    // Add the actual figure data instead of test plot
    window.add_figure(figure);
    
    // Run the event loop
    window.run().await?;
    
    Ok(())
}

/// Placeholder for non-GUI builds
#[cfg(not(feature = "gui"))]
pub async fn show_interactive() -> Result<(), Box<dyn std::error::Error>> {
    Err("GUI feature not enabled. Build with --features gui to use interactive plotting.".into())
}

/// Placeholder for non-GUI builds
#[cfg(not(feature = "gui"))]
pub async fn interactive_plot() -> Result<(), Box<dyn std::error::Error>> {
    Err("GUI feature not enabled. Build with --features gui to use interactive plotting.".into())
}

/// Placeholder for non-GUI builds
#[cfg(not(feature = "gui"))]
pub async fn show_interactive_with_config(_config: ()) -> Result<(), Box<dyn std::error::Error>> {
    Err("GUI feature not enabled. Build with --features gui to use interactive plotting.".into())
}

/// Placeholder for non-GUI builds
#[cfg(not(feature = "gui"))]
pub async fn show_interactive_with_figure(_figure: &plots::Figure) -> Result<(), Box<dyn std::error::Error>> {
    Err("GUI feature not enabled. Build with --features gui to use interactive plotting.".into())
}

/// Show an interactive plot using the legacy IPC system (deprecated, works from any thread)
pub fn show_interactive_via_ipc(figure: plots::Figure) -> Result<String, String> {
    if let Some(gui_handle) = get_gui_handle() {
        gui_handle.show_plot(figure)
    } else {
        Err("GUI system not initialized. Make sure to call init_gui_system() from the main thread.".to_string())
    }
}

/// Show an interactive plot using the robust GUI thread manager (recommended)
pub fn show_interactive_robust(figure: plots::Figure) -> Result<String, String> {
    match show_plot_global(figure) {
        Ok(GuiOperationResult::Success(msg)) => Ok(msg),
        Ok(GuiOperationResult::Error { message, error_code: _, recoverable: _ }) => Err(message),
        Ok(GuiOperationResult::Cancelled(msg)) => Err(msg),
        Err(GuiOperationResult::Error { message, error_code: _, recoverable: _ }) => Err(message),
        Err(GuiOperationResult::Cancelled(msg)) => Err(msg),
        Err(GuiOperationResult::Success(msg)) => Ok(msg), // Shouldn't happen, but handle it
    }
}

/// Show an interactive plot with optimal platform compatibility
pub fn show_interactive_platform_optimal(figure: plots::Figure) -> Result<String, String> {
    // Try native window system first (handles macOS main thread requirements properly)
    if is_native_window_available() {
        match show_plot_native_window(figure.clone()) {
            Ok(result) => return Ok(result),
            Err(e) => {
                eprintln!("Native window failed: {}, trying thread-based approach", e);
            }
        }
    }

    // Fall back to thread-based GUI system for non-macOS or if native window failed
    match show_interactive_robust(figure.clone()) {
        Ok(result) => Ok(result),
        Err(_) => {
            // Final fallback to legacy IPC
            show_interactive_via_ipc(figure)
        }
    }
}