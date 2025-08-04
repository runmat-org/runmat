//! Plotting builtin functions using the existing runtime_builtin system

use rustmat_builtins::Matrix;
use rustmat_macros::runtime_builtin;
use rustmat_plot::{plot_line, plot_scatter, plot_bar, plot_histogram, PlotOptions};

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
    
    let options = PlotOptions::default();
    let output_path = "plot.png".to_string();
    
    plot_line(&x_data, &y_data, &output_path, options)
        .map_err(|e| format!("Plot failed: {}", e))?;
    
    Ok(format!("Plot saved to {}", output_path))
}

#[runtime_builtin(name = "scatter")]  
fn scatter_builtin(x: Matrix, y: Matrix) -> Result<String, String> {
    let x_data = extract_numeric_vector(&x);
    let y_data = extract_numeric_vector(&y);
    
    if x_data.len() != y_data.len() {
        return Err("X and Y data must have the same length".to_string());
    }
    
    let options = PlotOptions::default();
    let output_path = "scatter.png".to_string();
    
    plot_scatter(&x_data, &y_data, &output_path, options)
        .map_err(|e| format!("Scatter plot failed: {}", e))?;
    
    Ok(format!("Scatter plot saved to {}", output_path))
}

#[runtime_builtin(name = "bar")]
fn bar_builtin(values: Matrix) -> Result<String, String> {
    let data = extract_numeric_vector(&values);
    
    // Create simple string labels
    let labels: Vec<String> = (1..=data.len()).map(|i| i.to_string()).collect();
    
    let options = PlotOptions::default();
    let output_path = "bar.png".to_string();
    
    plot_bar(&labels, &data, &output_path, options)
        .map_err(|e| format!("Bar chart failed: {}", e))?;
    
    Ok(format!("Bar chart saved to {}", output_path))
}

#[runtime_builtin(name = "hist")]
fn hist_builtin(values: Matrix) -> Result<String, String> {
    let data = extract_numeric_vector(&values);
    let bins = 10; // Default number of bins
    
    let options = PlotOptions::default();
    let output_path = "histogram.png".to_string();
    
    plot_histogram(&data, bins, &output_path, options)
        .map_err(|e| format!("Histogram failed: {}", e))?;
    
    Ok(format!("Histogram saved to {}", output_path))
}