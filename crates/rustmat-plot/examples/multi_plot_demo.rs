//! Multi-plot demonstration
//!
//! This example showcases the ability to overlay multiple plot types
//! in a single figure, demonstrating world-class plotting capabilities.

use glam::Vec4;
use rustmat_plot::plots::figure::matlab_compat;
use rustmat_plot::plots::{
    BarChart, Figure, Histogram, LinePlot, LineStyle, MarkerStyle, ScatterPlot,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 RustMat Multi-Plot Overlay Demo");
    println!("==================================");

    // Demo 1: Multiple Line Plots with Different Styles
    demo_multiple_lines()?;

    // Demo 2: Mixed Plot Types (Line + Scatter + Bar)
    demo_mixed_plot_types()?;

    // Demo 3: Data Analysis Visualization (Histogram + Fitted Line)
    demo_data_analysis()?;

    // Demo 4: MATLAB-Compatible Multi-Plot
    demo_matlab_compatibility()?;

    // Demo 5: Complex Overlay with All Plot Types
    demo_complex_overlay()?;

    println!("\n✅ All multi-plot demos completed successfully!");
    println!("🚀 RustMat plotting system supports seamless multi-plot overlays!");

    Ok(())
}

/// Demo 1: Multiple line plots with different styles overlaid
fn demo_multiple_lines() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Demo 1: Multiple Line Plots");
    println!("------------------------------");

    let mut figure = Figure::new()
        .with_title("Mathematical Functions Comparison")
        .with_labels("X", "Y")
        .with_grid(true);

    // Generate data for different mathematical functions
    let x: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();

    // Linear function: y = x
    let y_linear: Vec<f64> = x.to_vec();
    let linear_plot = LinePlot::new(x.clone(), y_linear)?
        .with_style(Vec4::new(0.0, 0.4, 0.7, 1.0), 2.0, LineStyle::Solid)
        .with_label("Linear (y = x)");

    // Quadratic function: y = x²/4
    let y_quadratic: Vec<f64> = x.iter().map(|&x| x * x / 4.0).collect();
    let quadratic_plot = LinePlot::new(x.clone(), y_quadratic)?
        .with_style(Vec4::new(0.8, 0.3, 0.1, 1.0), 2.0, LineStyle::Dashed)
        .with_label("Quadratic (y = x²/4)");

    // Sine function: y = 5*sin(x) + 5
    let y_sine: Vec<f64> = x.iter().map(|&x| 5.0 * (x * 0.5).sin() + 5.0).collect();
    let sine_plot = LinePlot::new(x.clone(), y_sine)?
        .with_style(Vec4::new(0.9, 0.7, 0.1, 1.0), 2.0, LineStyle::Dotted)
        .with_label("Sine (y = 5sin(x/2) + 5)");

    // Add all plots to the figure
    figure.add_line_plot(linear_plot);
    figure.add_line_plot(quadratic_plot);
    figure.add_line_plot(sine_plot);

    // Display statistics
    let stats = figure.statistics();
    println!(
        "  📊 Figure contains {} plots ({} visible)",
        stats.total_plots, stats.visible_plots
    );
    println!("  🏷️  Legend entries: {}", figure.legend_entries().len());

    // Generate render data (would be sent to GPU)
    let render_data = figure.render_data();
    println!("  🎨 Generated {} render batches", render_data.len());

    let bounds = figure.bounds();
    println!(
        "  📐 Combined bounds: ({:.1}, {:.1}) to ({:.1}, {:.1})",
        bounds.min.x, bounds.min.y, bounds.max.x, bounds.max.y
    );

    Ok(())
}

/// Demo 2: Mixed plot types (line + scatter + bar) in one figure
fn demo_mixed_plot_types() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Demo 2: Mixed Plot Types");
    println!("---------------------------");

    let mut figure = Figure::new()
        .with_title("Sales Data Analysis")
        .with_labels("Month", "Sales/Trend")
        .with_legend(true);

    // Monthly sales data (bar chart)
    let months = vec![
        "Jan".to_string(),
        "Feb".to_string(),
        "Mar".to_string(),
        "Apr".to_string(),
        "May".to_string(),
        "Jun".to_string(),
    ];
    let sales = vec![120.0, 135.0, 155.0, 180.0, 165.0, 190.0];

    let sales_bars = BarChart::new(months, sales.clone())?
        .with_style(Vec4::new(0.2, 0.6, 0.9, 0.8), 0.6)
        .with_outline(Vec4::new(0.0, 0.3, 0.6, 1.0), 1.5)
        .with_label("Monthly Sales");

    // Trend line (line plot)
    let x_months: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let trend_line = LinePlot::new(x_months.clone(), sales)?
        .with_style(Vec4::new(1.0, 0.3, 0.3, 1.0), 3.0, LineStyle::Solid)
        .with_label("Sales Trend");

    // Key performance indicators (scatter plot)
    let kpi_months: Vec<f64> = vec![1.0, 3.0, 5.0]; // Feb, Apr, Jun
    let kpi_values: Vec<f64> = vec![135.0, 180.0, 190.0];
    let kpi_points = ScatterPlot::new(kpi_months, kpi_values)?
        .with_style(Vec4::new(0.0, 0.8, 0.0, 1.0), 8.0, MarkerStyle::Star)
        .with_label("KPI Targets");

    // Add all plots
    figure.add_bar_chart(sales_bars);
    figure.add_line_plot(trend_line);
    figure.add_scatter_plot(kpi_points);

    // Analysis
    let stats = figure.statistics();
    println!(
        "  📊 Mixed figure: {} different plot types",
        stats.plot_type_counts.len()
    );
    println!(
        "  💾 Total memory usage: {} bytes",
        stats.total_memory_usage
    );

    // Test plot visibility toggling
    if let Some(_plot) = figure.get_plot_mut(1) {
        // Could hide/show plots dynamically
        println!("  👁️  Plot visibility can be toggled dynamically");
    }

    Ok(())
}

/// Demo 3: Data analysis with histogram and fitted curve
fn demo_data_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Demo 3: Data Analysis Visualization");
    println!("--------------------------------------");

    let mut figure = Figure::new()
        .with_title("Data Distribution Analysis")
        .with_labels("Value", "Frequency/Density")
        .with_background_color(Vec4::new(0.98, 0.98, 0.98, 1.0));

    // Generate sample data (normal distribution)
    let mut data = Vec::new();
    for i in 0..1000 {
        let x = (i as f64 - 500.0) / 100.0;
        let _y = (-0.5 * x * x).exp() + (rand::random::<f64>() - 0.5) * 0.2;
        data.push(x);
    }

    // Create histogram of the data
    let histogram = Histogram::new(data.clone(), 25)?
        .with_style(Vec4::new(0.7, 0.8, 1.0, 0.8), true) // Normalized
        .with_outline(Vec4::new(0.3, 0.4, 0.6, 1.0), 1.0)
        .with_label("Data Distribution");

    // Create theoretical curve (fitted line)
    let x_theory: Vec<f64> = (-50..=50).map(|i| i as f64 / 10.0).collect();
    let y_theory: Vec<f64> = x_theory
        .iter()
        .map(|&x| (-0.5 * x * x).exp() / (2.5066282746)) // Normalized Gaussian
        .collect();

    let theory_curve = LinePlot::new(x_theory, y_theory)?
        .with_style(Vec4::new(1.0, 0.2, 0.2, 1.0), 3.0, LineStyle::Solid)
        .with_label("Theoretical Fit");

    // Add sample points
    let sample_x: Vec<f64> = data.iter().take(50).step_by(2).cloned().collect();
    let sample_y: Vec<f64> = sample_x
        .iter()
        .map(|&x| (-0.5 * x * x).exp() / (2.5066282746) + (rand::random::<f64>() - 0.5) * 0.05)
        .collect();

    let sample_points = ScatterPlot::new(sample_x, sample_y)?
        .with_style(Vec4::new(0.2, 0.8, 0.2, 0.7), 4.0, MarkerStyle::Circle)
        .with_label("Sample Points");

    figure.add_histogram(histogram);
    figure.add_line_plot(theory_curve);
    figure.add_scatter_plot(sample_points);

    println!("  📈 Combined histogram + theoretical curve + sample points");
    println!("  🎯 Perfect for statistical analysis and model validation");

    Ok(())
}

/// Demo 4: MATLAB-compatible multi-plot creation
fn demo_matlab_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Demo 4: MATLAB Compatibility");
    println!("-------------------------------");

    // Create figure using MATLAB-style syntax
    let mut figure = matlab_compat::figure_with_title("MATLAB-Style Multi-Plot");

    // Multiple line plots with automatic color cycling
    let data_sets = vec![
        (
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 4.0, 9.0, 16.0],
            Some("x²".to_string()),
        ),
        (
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            Some("x".to_string()),
        ),
        (
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            Some("constant".to_string()),
        ),
    ];

    let line_indices = matlab_compat::plot_multiple_lines(&mut figure, data_sets)?;
    println!(
        "  📊 Created {} line plots with automatic color cycling",
        line_indices.len()
    );

    // Multiple scatter plots
    let scatter_data = vec![
        (
            vec![0.5, 1.5, 2.5, 3.5],
            vec![0.25, 2.25, 6.25, 12.25],
            Some("scatter 1".to_string()),
        ),
        (
            vec![0.2, 1.2, 2.2, 3.2],
            vec![0.04, 1.44, 4.84, 10.24],
            Some("scatter 2".to_string()),
        ),
    ];

    let scatter_indices = matlab_compat::scatter_multiple(&mut figure, scatter_data)?;
    println!(
        "  🎯 Added {} scatter plots with automatic styling",
        scatter_indices.len()
    );

    // Verify MATLAB-style automatic colors are different
    let legend = figure.legend_entries();
    println!(
        "  🎨 Automatic color cycling: {} unique colors generated",
        legend.len()
    );

    // Test that colors are indeed different (basic check)
    if legend.len() >= 2 {
        let color_different = legend[0].color != legend[1].color;
        println!("  ✅ Color cycling works: {color_different}");
    }

    Ok(())
}

/// Demo 5: Complex overlay with all plot types
fn demo_complex_overlay() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌟 Demo 5: Complex Multi-Plot Overlay");
    println!("-------------------------------------");

    let mut figure = Figure::new()
        .with_title("Production Data Dashboard")
        .with_labels("Time/Category", "Value")
        .with_limits((-1.0, 12.0), (-50.0, 300.0))
        .with_grid(true);

    // Production targets (bar chart)
    let targets = BarChart::new(
        vec![
            "Q1".to_string(),
            "Q2".to_string(),
            "Q3".to_string(),
            "Q4".to_string(),
        ],
        vec![200.0, 220.0, 250.0, 280.0],
    )?
    .with_style(Vec4::new(0.8, 0.9, 1.0, 0.6), 0.8)
    .with_outline(Vec4::new(0.4, 0.5, 0.7, 1.0), 1.0)
    .with_label("Quarterly Targets");

    // Actual production trend (line plot)
    let time: Vec<f64> = (0..=12).map(|i| i as f64).collect();
    let production: Vec<f64> = time
        .iter()
        .map(|&t| 180.0 + 8.0 * t + 2.0 * (t * 0.5).sin())
        .collect();

    let trend = LinePlot::new(time.clone(), production)?
        .with_style(Vec4::new(0.0, 0.6, 0.0, 1.0), 3.0, LineStyle::Solid)
        .with_label("Actual Production");

    // Quality control points (scatter plot)
    let qc_time: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let qc_values: Vec<f64> = vec![195.0, 210.0, 235.0, 248.0, 275.0];
    let quality_points = ScatterPlot::new(qc_time, qc_values)?
        .with_style(Vec4::new(1.0, 0.0, 0.0, 1.0), 6.0, MarkerStyle::Diamond)
        .with_label("QC Checkpoints");

    // Efficiency distribution (histogram - simulated monthly efficiency percentages)
    let efficiency_data: Vec<f64> = vec![
        82.0, 85.0, 88.0, 87.0, 90.0, 92.0, 89.0, 91.0, 94.0, 93.0, 95.0, 96.0, 84.0, 86.0, 89.0,
        88.0, 91.0, 93.0, 90.0, 92.0, 95.0, 94.0, 96.0, 97.0,
    ];

    let efficiency_hist = Histogram::new(efficiency_data, 8)?
        .with_style(Vec4::new(1.0, 0.8, 0.0, 0.7), false)
        .with_outline(Vec4::new(0.7, 0.5, 0.0, 1.0), 1.0)
        .with_label("Efficiency Distribution");

    // Add all plots to create a comprehensive dashboard
    figure.add_bar_chart(targets);
    figure.add_line_plot(trend);
    figure.add_scatter_plot(quality_points);
    figure.add_histogram(efficiency_hist);

    // Analyze the complex figure
    let stats = figure.statistics();
    println!("  📊 Complex figure statistics:");
    println!("     • Total plots: {}", stats.total_plots);
    println!("     • Visible plots: {}", stats.visible_plots);
    println!("     • Memory usage: {} bytes", stats.total_memory_usage);
    println!("     • Has legend: {}", stats.has_legend);

    // Show render data complexity
    let render_data = figure.render_data();
    let total_vertices: usize = render_data.iter().map(|rd| rd.vertices.len()).sum();
    println!("     • Total vertices: {total_vertices}");
    println!("     • Render batches: {}", render_data.len());

    // Test bounds computation across all plot types
    let bounds = figure.bounds();
    println!("  📐 Combined bounds across all plot types:");
    println!("     • X: {:.1} to {:.1}", bounds.min.x, bounds.max.x);
    println!("     • Y: {:.1} to {:.1}", bounds.min.y, bounds.max.y);

    println!("  🎯 Successfully overlaid 4 different plot types in one figure!");

    Ok(())
}

// Simple random number generation for the demo
mod rand {
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(1) };
    }

    pub fn random<T>() -> T
    where
        T: From<f64>,
    {
        SEED.with(|seed| {
            let s = seed.get();
            seed.set(s.wrapping_mul(1103515245).wrapping_add(12345));
            T::from((s >> 16) as f64 / 32768.0)
        })
    }
}
