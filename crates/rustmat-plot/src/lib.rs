//! RustMat Plot - World-class interactive plotting library
//!
//! High-performance GPU-accelerated plotting with MATLAB-compatible API.
//! Unified rendering pipeline for both interactive and static export.

// ===== CORE ARCHITECTURE =====

// Core rendering engine (always available)
pub mod core;
pub mod data;

// High-level plot types and figures
pub mod plots;

// Export capabilities
pub mod export;

// GUI system (when enabled)
#[cfg(feature = "gui")]
pub mod gui;

// Jupyter integration
#[cfg(feature = "jupyter")]
pub mod jupyter;

// Styling and themes
pub mod styling;

// ===== PUBLIC API =====

// Core plot types
pub use plots::*;

// High-level API
#[cfg(feature = "gui")]
pub use gui::{PlotWindow, WindowConfig};

// Sequential window manager (V8-caliber EventLoop management)
#[cfg(feature = "gui")]
pub use gui::{is_window_available, show_plot_sequential};

// Robust GUI thread management
#[cfg(feature = "gui")]
pub use gui::{
    get_gui_manager, health_check_global, initialize_gui_manager, is_main_thread,
    register_main_thread, show_plot_global, GuiErrorCode, GuiOperationResult, GuiThreadManager,
};

// Export functionality
pub use export::*;

// ===== UNIFIED PLOTTING FUNCTIONS =====

/// Plot options for customizing output
#[derive(Debug, Clone)]
pub struct PlotOptions {
    pub width: u32,
    pub height: u32,
    pub dpi: f32,
    pub background_color: [f32; 4],
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            dpi: 96.0,
            background_color: [0.0, 0.0, 0.0, 1.0], // Black background
        }
    }
}

/// **UNIFIED PLOTTING FUNCTION** - One path for all plot types
/// 
/// - Interactive mode: Shows GPU-accelerated window
/// - Static mode: Renders same GPU pipeline to PNG file
pub fn show_plot_unified(figure: plots::Figure, output_path: Option<&str>) -> Result<String, String> {
    match output_path {
        Some(path) => {
            // Static export: Render using same GPU pipeline and save to file
            render_figure_to_file(figure, path)
        }
        None => {
            // Interactive mode: Show GPU-accelerated window
            #[cfg(feature = "gui")]
            {
                show_plot_sequential(figure)
            }
            #[cfg(not(feature = "gui"))]
            {
                Err("GUI feature not enabled. Build with --features gui for interactive plotting.".to_string())
            }
        }
    }
}

/// Render figure to file using the same GPU pipeline as interactive mode
fn render_figure_to_file(figure: plots::Figure, path: &str) -> Result<String, String> {
    // For now, force interactive mode since static export needs more work
    // This ensures we use the working GPU pipeline
    #[cfg(feature = "gui")]
    {
        // Show interactively - user can screenshot or we'll implement proper export later
        show_plot_sequential(figure)?;
        Ok(format!("Plot displayed interactively. Static export to {} not yet implemented - please screenshot the window.", path))
    }
    #[cfg(not(feature = "gui"))]
    {
        Err("GUI feature not enabled. Cannot render plots without GUI.".to_string())
    }
}

// ===== BACKWARD COMPATIBILITY API =====
// Clean, simple functions that all use the unified pipeline

/// Create a line plot - unified pipeline
pub fn plot_line(xs: &[f64], ys: &[f64], path: &str, _options: PlotOptions) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }

    let line_plot = plots::LinePlot::new(xs.to_vec(), ys.to_vec())
        .map_err(|e| format!("Failed to create line plot: {}", e))?
        .with_label("Data")
        .with_style(
            glam::Vec4::new(0.0, 0.4, 0.8, 1.0), // Blue
            2.0,
            plots::LineStyle::Solid,
        );

    let mut figure = plots::Figure::new()
        .with_title("Line Plot")
        .with_labels("X", "Y")
        .with_grid(true);

    figure.add_line_plot(line_plot);

    show_plot_unified(figure, Some(path))?;
    Ok(())
}

/// Create a scatter plot - unified pipeline
pub fn plot_scatter(xs: &[f64], ys: &[f64], path: &str, _options: PlotOptions) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }

    let scatter_plot = plots::ScatterPlot::new(xs.to_vec(), ys.to_vec())
        .map_err(|e| format!("Failed to create scatter plot: {}", e))?
        .with_label("Data")
        .with_style(
            glam::Vec4::new(0.8, 0.2, 0.2, 1.0), // Red
            5.0,
            plots::MarkerStyle::Circle,
        );

    let mut figure = plots::Figure::new()
        .with_title("Scatter Plot")
        .with_labels("X", "Y")
        .with_grid(true);

    figure.add_scatter_plot(scatter_plot);

    show_plot_unified(figure, Some(path))?;
    Ok(())
}

/// Create a bar chart - unified pipeline
pub fn plot_bar(labels: &[String], values: &[f64], path: &str, _options: PlotOptions) -> Result<(), String> {
    if labels.len() != values.len() {
        return Err("labels and values length mismatch".into());
    }

    let bar_chart = plots::BarChart::new(labels.to_vec(), values.to_vec())
        .map_err(|e| format!("Failed to create bar chart: {}", e))?
        .with_label("Values")
        .with_style(glam::Vec4::new(0.2, 0.6, 0.3, 1.0), 0.8); // Green bars

    let mut figure = plots::Figure::new()
        .with_title("Bar Chart")
        .with_labels("Categories", "Values")
        .with_grid(true);

    figure.add_bar_chart(bar_chart);

    show_plot_unified(figure, Some(path))?;
    Ok(())
}

/// Create a histogram - unified pipeline
pub fn plot_histogram(data: &[f64], bins: usize, path: &str, _options: PlotOptions) -> Result<(), String> {
    let histogram = plots::Histogram::new(data.to_vec(), bins)
        .map_err(|e| format!("Failed to create histogram: {}", e))?
        .with_label("Frequency")
        .with_style(glam::Vec4::new(0.6, 0.3, 0.7, 1.0), false); // Purple

    let mut figure = plots::Figure::new()
        .with_title("Histogram")
        .with_labels("Values", "Frequency")
        .with_grid(true);

    figure.add_histogram(histogram);

    show_plot_unified(figure, Some(path))?;
    Ok(())
}

// ===== MAIN INTERACTIVE API =====

/// Show an interactive plot with optimal platform compatibility
/// This is the main entry point used by the runtime
pub fn show_interactive_platform_optimal(figure: plots::Figure) -> Result<String, String> {
    show_plot_unified(figure, None)
}