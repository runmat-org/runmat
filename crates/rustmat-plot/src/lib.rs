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

// Feature-gated modules
#[cfg(feature = "gui")]
pub mod gui;

#[cfg(feature = "jupyter")]
pub mod jupyter;

// Additional modules
pub mod plots;
pub mod data;
pub mod export;

// Re-exports for convenience
pub use core::*;

// High-level API
#[cfg(feature = "gui")]
pub use gui::{PlotWindow, WindowConfig};

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

// ===== PUBLIC PLOTTING FUNCTIONS =====

/// Plot a line series with full options support
pub fn plot_line(xs: &[f64], ys: &[f64], path: &str, options: PlotOptions) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }
    
    if path.ends_with(".png") {
        simple_plots::line_plot_png(xs, ys, path, &options)
    } else if path.ends_with(".svg") {
        simple_plots::line_plot_svg(xs, ys, path, &options)
    } else {
        Err("Unsupported file format".to_string())
    }
}

/// Create a scatter plot with full options support
pub fn plot_scatter(xs: &[f64], ys: &[f64], path: &str, options: PlotOptions) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }
    
    if path.ends_with(".png") {
        simple_plots::scatter_plot_png(xs, ys, path, &options)
    } else if path.ends_with(".svg") {
        simple_plots::scatter_plot_svg(xs, ys, path, &options)
    } else {
        Err("Unsupported file format".to_string())
    }
}

/// Create a bar chart with full options support
pub fn plot_bar(labels: &[String], values: &[f64], path: &str, options: PlotOptions) -> Result<(), String> {
    if labels.len() != values.len() {
        return Err("labels and values length mismatch".into());
    }
    
    if path.ends_with(".png") {
        simple_plots::bar_chart_png(labels, values, path, &options)
    } else {
        // For SVG, we'd need to implement bar_chart_svg
        Err("SVG bar charts not yet implemented".to_string())
    }
}

/// Create a histogram with full options support
pub fn plot_histogram(data: &[f64], bins: usize, path: &str, options: PlotOptions) -> Result<(), String> {
    if path.ends_with(".png") {
        simple_plots::histogram_png(data, bins, path, &options)
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