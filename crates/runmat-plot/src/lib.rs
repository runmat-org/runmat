//! RunMat Plot - World-class interactive plotting library
//!
//! High-performance GPU-accelerated plotting.
//! Unified rendering pipeline for both interactive and static export.

// ===== CORE ARCHITECTURE =====

// Core rendering engine (always available)
pub mod context;
pub mod core;
pub mod data;
pub mod gpu;

// High-level plot types and figures
pub mod plots;

// Export capabilities
pub mod export;

pub use context::{install_shared_wgpu_context, shared_wgpu_context, SharedWgpuContext};

// GUI system (when enabled)
#[cfg(feature = "gui")]
pub mod gui;

// WASM/WebGPU bridge
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod web;

// Jupyter integration
#[cfg(feature = "jupyter")]
pub mod jupyter;

// Styling and themes
pub mod styling;

// ===== PUBLIC API =====

pub use core::scene::GpuVertexBuffer;

// Core plot types
// Avoid ambiguous re-exports: explicitly export plot types
pub use plots::{
    AreaPlot, ContourFillPlot, ContourPlot, Figure, ImagePlot, LinePlot, PieChart, QuiverPlot,
    Scatter3Plot, ScatterPlot, StairsPlot, StemPlot, SurfacePlot,
};

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
// Explicitly export image exporter to avoid collision with plots::image
pub use export::{image::*, vector::*};

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
pub fn show_plot_unified(
    figure: plots::Figure,
    output_path: Option<&str>,
) -> Result<String, String> {
    match output_path {
        Some(path) => {
            // Static export: Render using same GPU pipeline and save to file
            render_figure_to_file(figure, path)
        }
        None => {
            // Interactive mode: Show GPU-accelerated window
            #[cfg(feature = "gui")]
            {
                #[cfg(target_os = "macos")]
                {
                    if !is_main_thread() {
                        return Err("Interactive plotting is unavailable on macOS when called from a non-main thread. Launch RunMat from the main thread or set RUSTMAT_PLOT_MODE=headless for exports.".to_string());
                    }
                }
                show_plot_sequential(figure)
            }
            #[cfg(not(feature = "gui"))]
            {
                Err(
                    "GUI feature not enabled. Build with --features gui for interactive plotting."
                        .to_string(),
                )
            }
        }
    }
}

/// Render figure to file using the same GPU pipeline as interactive mode
#[cfg(not(target_arch = "wasm32"))]
fn render_figure_to_file(figure: plots::Figure, path: &str) -> Result<String, String> {
    use crate::export::ImageExporter;
    // Use the headless GPU exporter that shares the same render pipeline
    let rt =
        tokio::runtime::Runtime::new().map_err(|e| format!("Failed to create runtime: {e}"))?;
    rt.block_on(async move {
        let mut fig = figure.clone();
        let exporter = ImageExporter::new().await?;
        exporter.export_png(&mut fig, path).await?;
        Ok::<_, String>(format!("Saved plot to {path}"))
    })
}

#[cfg(target_arch = "wasm32")]
fn render_figure_to_file(_figure: plots::Figure, _path: &str) -> Result<String, String> {
    Err("Static image export is not available in wasm builds".to_string())
}

// ===== BACKWARD COMPATIBILITY API =====
// Clean, simple functions that all use the unified pipeline

/// Create a line plot - unified pipeline
pub fn plot_line(xs: &[f64], ys: &[f64], path: &str, _options: PlotOptions) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }

    let line_plot = plots::LinePlot::new(xs.to_vec(), ys.to_vec())
        .map_err(|e| format!("Failed to create line plot: {e}"))?
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
pub fn plot_scatter(
    xs: &[f64],
    ys: &[f64],
    path: &str,
    _options: PlotOptions,
) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }

    let scatter_plot = plots::ScatterPlot::new(xs.to_vec(), ys.to_vec())
        .map_err(|e| format!("Failed to create scatter plot: {e}"))?
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
pub fn plot_bar(
    labels: &[String],
    values: &[f64],
    path: &str,
    _options: PlotOptions,
) -> Result<(), String> {
    if labels.len() != values.len() {
        return Err("labels and values length mismatch".into());
    }

    let bar_chart = plots::BarChart::new(labels.to_vec(), values.to_vec())
        .map_err(|e| format!("Failed to create bar chart: {e}"))?
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
pub fn plot_histogram(
    data: &[f64],
    bins: usize,
    path: &str,
    _options: PlotOptions,
) -> Result<(), String> {
    if data.is_empty() {
        return Err("Cannot create histogram with empty data".to_string());
    }
    if bins == 0 {
        return Err("Number of bins must be greater than zero".to_string());
    }

    let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (min_val, max_val) = if (max_val - min_val).abs() < f64::EPSILON {
        (min_val - 0.5, max_val + 0.5)
    } else {
        (min_val, max_val)
    };
    let bin_width = (max_val - min_val) / bins as f64;
    let edges: Vec<f64> = (0..=bins).map(|i| min_val + i as f64 * bin_width).collect();
    // Count values
    let mut counts = vec![0u64; bins];
    for &v in data {
        let mut idx = bins;
        for i in 0..bins {
            if v >= edges[i] && v < edges[i + 1] {
                idx = i;
                break;
            }
        }
        if idx == bins && (v - edges[bins]).abs() < f64::EPSILON {
            idx = bins - 1;
        }
        if idx < bins {
            counts[idx] += 1;
        }
    }

    // Build bar labels and values
    let labels: Vec<String> = edges
        .windows(2)
        .map(|w| format!("[{:.3},{:.3})", w[0], w[1]))
        .collect();
    let values: Vec<f64> = counts.into_iter().map(|c| c as f64).collect();

    let bar_chart = plots::BarChart::new(labels, values)
        .map_err(|e| format!("Failed to create histogram bars: {e}"))?
        .with_label("Frequency")
        .with_style(glam::Vec4::new(0.6, 0.3, 0.7, 1.0), 0.9);

    let mut figure = plots::Figure::new()
        .with_title("Histogram")
        .with_labels("Values", "Frequency")
        .with_grid(true);
    figure.add_bar_chart(bar_chart);
    show_plot_unified(figure, Some(path))?;
    Ok(())
}

// ===== MAIN INTERACTIVE API =====

/// Show an interactive plot with optimal platform compatibility
/// This is the main entry point used by the runtime
pub fn show_interactive_platform_optimal(figure: plots::Figure) -> Result<String, String> {
    render_interactive_with_handle(0, figure)
}

/// Show an interactive plot that is tied to a specific MATLAB figure handle.
/// This allows embedding runtimes to request that a window close when the
/// corresponding figure lifecycle event fires.
pub fn render_interactive_with_handle(
    handle: u32,
    figure: plots::Figure,
) -> Result<String, String> {
    #[cfg(feature = "gui")]
    {
        if handle == 0 {
            show_plot_unified(figure, None)
        } else {
            gui::lifecycle::render_figure(handle, figure)
        }
    }
    #[cfg(not(feature = "gui"))]
    {
        let _ = handle;
        let _ = figure;
        Err(
            "GUI feature not enabled. Build with --features gui for interactive plotting."
                .to_string(),
        )
    }
}
