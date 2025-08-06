//! Modern plotting builtin functions using the world-class rustmat-plot system
//!
//! These functions integrate with the configuration system to provide
//! interactive GUI plotting, static exports, and Jupyter notebook support.

use rustmat_builtins::Matrix;
use rustmat_macros::runtime_builtin;
use rustmat_plot::jupyter::JupyterBackend;
use rustmat_plot::plots::figure::PlotElement;
use rustmat_plot::plots::*;
use std::env;

/// Determine the plotting mode from environment/config
fn get_plotting_mode() -> PlottingMode {
    // Check environment variables for plotting configuration
    if let Ok(mode) = env::var("RUSTMAT_PLOT_MODE") {
        match mode.to_lowercase().as_str() {
            "gui" => PlottingMode::Interactive,
            "headless" => PlottingMode::Static,
            "jupyter" => PlottingMode::Jupyter,
            _ => PlottingMode::Auto,
        }
    } else {
        PlottingMode::Auto
    }
}

/// Plotting mode configuration
#[derive(Debug, Clone, Copy)]
enum PlottingMode {
    /// Auto-detect based on environment
    Auto,
    /// Interactive GUI window
    Interactive,
    /// Static file export
    Static,
    /// Jupyter notebook
    Jupyter,
}

/// Execute a plot based on the current mode
fn execute_plot(mut figure: Figure) -> Result<String, String> {
    match get_plotting_mode() {
        PlottingMode::Interactive => interactive_export(&mut figure),
        PlottingMode::Static => static_export(&mut figure, "plot.png"),
        PlottingMode::Jupyter => jupyter_export(&mut figure),
        PlottingMode::Auto => {
            // Auto-detect: prefer Jupyter if available, otherwise interactive
            if env::var("JPY_PARENT_PID").is_ok() || env::var("JUPYTER_RUNTIME_DIR").is_ok() {
                jupyter_export(&mut figure)
            } else {
                interactive_export(&mut figure)
            }
        }
    }
}

/// Launch interactive GUI window using platform-optimal approach
fn interactive_export(figure: &mut Figure) -> Result<String, String> {
    // Use platform-optimal GUI system for best compatibility
    let figure_clone = figure.clone();

    // Try platform-optimal GUI system first
    match rustmat_plot::show_interactive_platform_optimal(figure_clone) {
        Ok(result) => Ok(result),
        Err(e) => {
            // GUI system failed, fall back to static export
            eprintln!("Interactive plotting failed, using static export: {}", e);
            static_export(figure, "plot.png")
        }
    }
}

/// Export plot as static file using the figure data
fn static_export(figure: &mut Figure, filename: &str) -> Result<String, String> {
    // Extract data from the first plot in the figure for fallback export
    // This is a simplified approach - ideally we'd render the entire figure
    if figure.is_empty() {
        return Err("No plots found in figure to export".to_string());
    }

    match figure.get_plot_mut(0) {
        Some(PlotElement::Line(line_plot)) => {
            let x_data: Vec<f64> = line_plot.x_data.iter().map(|&x| x as f64).collect();
            let y_data: Vec<f64> = line_plot.y_data.iter().map(|&y| y as f64).collect();

            rustmat_plot::plot_line(
                &x_data,
                &y_data,
                filename,
                rustmat_plot::PlotOptions::default(),
            )
            .map_err(|e| format!("Plot export failed: {}", e))?;

            Ok(format!("Plot saved to {}", filename))
        }
        Some(PlotElement::Scatter(scatter_plot)) => {
            let x_data: Vec<f64> = scatter_plot.x_data.iter().map(|&x| x as f64).collect();
            let y_data: Vec<f64> = scatter_plot.y_data.iter().map(|&y| y as f64).collect();

            rustmat_plot::plot_scatter(
                &x_data,
                &y_data,
                filename,
                rustmat_plot::PlotOptions::default(),
            )
            .map_err(|e| format!("Plot export failed: {}", e))?;

            Ok(format!("Scatter plot saved to {}", filename))
        }
        Some(PlotElement::Bar(bar_chart)) => {
            let values: Vec<f64> = bar_chart.values.iter().map(|&v| v as f64).collect();

            rustmat_plot::plot_bar(
                &bar_chart.labels,
                &values,
                filename,
                rustmat_plot::PlotOptions::default(),
            )
            .map_err(|e| format!("Plot export failed: {}", e))?;

            Ok(format!("Bar chart saved to {}", filename))
        }
        Some(PlotElement::Histogram(histogram)) => {
            let data: Vec<f64> = histogram.data.iter().map(|&d| d as f64).collect();

            rustmat_plot::plot_histogram(
                &data,
                histogram.bins,
                filename,
                rustmat_plot::PlotOptions::default(),
            )
            .map_err(|e| format!("Plot export failed: {}", e))?;

            Ok(format!("Histogram saved to {}", filename))
        }
        None => Err("No plots found in figure to export".to_string()),
    }
}

/// Export plot for Jupyter
fn jupyter_export(figure: &mut Figure) -> Result<String, String> {
    let mut backend = JupyterBackend::new();
    backend.display_figure(figure)
}

/// Extract numeric vector from a Matrix
fn extract_numeric_vector(matrix: &Matrix) -> Vec<f64> {
    matrix.data.clone()
}

#[runtime_builtin(name = "plot")]
fn plot_builtin(x: Matrix, y: Matrix) -> Result<String, String> {
    let x_data = extract_numeric_vector(&x);
    let y_data = extract_numeric_vector(&y);

    if x_data.len() != y_data.len() {
        return Err("X and Y data must have the same length".to_string());
    }

    // Create a line plot
    let line_plot = LinePlot::new(x_data.clone(), y_data.clone())
        .map_err(|e| format!("Failed to create line plot: {}", e))?
        .with_label("Data")
        .with_style(
            glam::Vec4::new(0.0, 0.4, 0.8, 1.0), // Blue
            3.0,
            LineStyle::Solid,
        );

    let mut figure = Figure::new()
        .with_title("Plot")
        .with_labels("X", "Y")
        .with_grid(true);

    figure.add_line_plot(line_plot);

    execute_plot(figure)
}

#[runtime_builtin(name = "scatter")]
fn scatter_builtin(x: Matrix, y: Matrix) -> Result<String, String> {
    let x_data = extract_numeric_vector(&x);
    let y_data = extract_numeric_vector(&y);

    if x_data.len() != y_data.len() {
        return Err("X and Y data must have the same length".to_string());
    }

    // Create modern scatter plot
    let scatter_plot = ScatterPlot::new(x_data, y_data)
        .map_err(|e| format!("Failed to create scatter plot: {}", e))?
        .with_label("Data")
        .with_style(
            glam::Vec4::new(0.8, 0.2, 0.2, 1.0), // Red
            5.0,
            MarkerStyle::Circle,
        );

    let mut figure = Figure::new()
        .with_title("Scatter Plot")
        .with_labels("X", "Y")
        .with_grid(true);

    figure.add_scatter_plot(scatter_plot);

    execute_plot(figure)
}

#[runtime_builtin(name = "bar")]
fn bar_builtin(values: Matrix) -> Result<String, String> {
    let data = extract_numeric_vector(&values);

    // Create simple string labels
    let labels: Vec<String> = (1..=data.len()).map(|i| format!("Item {}", i)).collect();

    // Create modern bar chart
    let bar_chart = BarChart::new(labels, data)
        .map_err(|e| format!("Failed to create bar chart: {}", e))?
        .with_label("Values")
        .with_style(glam::Vec4::new(0.2, 0.6, 0.3, 1.0), 0.8); // Green with 80% width

    let mut figure = Figure::new()
        .with_title("Bar Chart")
        .with_labels("Categories", "Values")
        .with_grid(true);

    figure.add_bar_chart(bar_chart);

    execute_plot(figure)
}

#[runtime_builtin(name = "hist")]
fn hist_builtin(values: Matrix) -> Result<String, String> {
    let data = extract_numeric_vector(&values);
    let bins = 10; // Default number of bins

    // Create modern histogram
    let histogram = Histogram::new(data, bins)
        .map_err(|e| format!("Failed to create histogram: {}", e))?
        .with_label("Frequency")
        .with_style(glam::Vec4::new(0.6, 0.3, 0.7, 1.0), false); // Purple, not normalized

    let mut figure = Figure::new()
        .with_title("Histogram")
        .with_labels("Values", "Frequency")
        .with_grid(true);

    figure.add_histogram(histogram);

    execute_plot(figure)
}

// 3D Plotting Functions

#[runtime_builtin(name = "surf")]
fn surf_builtin(x: Matrix, y: Matrix, z: Matrix) -> Result<String, String> {
    // Convert matrices to 3D surface format
    // For now, assume z is a flattened grid
    let x_data = extract_numeric_vector(&x);
    let y_data = extract_numeric_vector(&y);
    let z_data_flat = extract_numeric_vector(&z);

    // Reconstruct grid (assuming square for simplicity)
    let grid_size = (z_data_flat.len() as f64).sqrt() as usize;
    if grid_size * grid_size != z_data_flat.len() {
        return Err("Z data must form a square grid".to_string());
    }

    let mut z_grid = Vec::new();
    for i in 0..grid_size {
        let mut row = Vec::new();
        for j in 0..grid_size {
            row.push(z_data_flat[i * grid_size + j]);
        }
        z_grid.push(row);
    }

    // Create surface plot
    let surface = SurfacePlot::new(x_data, y_data, z_grid)
        .map_err(|e| format!("Failed to create surface plot: {}", e))?
        .with_colormap(ColorMap::Viridis)
        .with_label("Surface");

    // For now, return a placeholder until 3D integration is complete
    Ok(format!(
        "3D Surface plot created with {} points",
        surface.len()
    ))
}

#[runtime_builtin(name = "scatter3")]
fn scatter3_builtin(x: Matrix, y: Matrix, z: Matrix) -> Result<String, String> {
    let x_data = extract_numeric_vector(&x);
    let y_data = extract_numeric_vector(&y);
    let z_data = extract_numeric_vector(&z);

    if x_data.len() != y_data.len() || y_data.len() != z_data.len() {
        return Err("X, Y, and Z data must have the same length".to_string());
    }

    // Create 3D positions
    let positions: Vec<glam::Vec3> = x_data
        .iter()
        .zip(y_data.iter())
        .zip(z_data.iter())
        .map(|((&x, &y), &z)| glam::Vec3::new(x as f32, y as f32, z as f32))
        .collect();

    // Create point cloud
    let point_cloud = PointCloudPlot::new(positions)
        .with_default_color(glam::Vec4::new(0.0, 0.6, 0.8, 1.0))
        .with_label("3D Points");

    // For now, return a placeholder until 3D integration is complete
    Ok(format!(
        "3D Point cloud created with {} points",
        point_cloud.len()
    ))
}

#[runtime_builtin(name = "mesh")]
fn mesh_builtin(x: Matrix, y: Matrix, z: Matrix) -> Result<String, String> {
    // Similar to surf but wireframe mode
    let result = surf_builtin(x, y, z)?;
    Ok(result.replace("Surface", "Mesh (wireframe)"))
}
