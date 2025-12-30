//! Comprehensive integration tests for RunMat Plot
//!
//! Tests the complete plotting system including 2D plots, 3D plots,
//! Jupyter integration, and performance characteristics.

#[cfg(feature = "jupyter")]
use runmat_plot::jupyter::{JupyterBackend, OutputFormat};
use runmat_plot::plots::*;

#[test]
fn test_complete_2d_plotting_pipeline() {
    // Test data
    let x_data = (0..100).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
    let y_data = x_data.iter().map(|&x| x.sin()).collect::<Vec<_>>();

    // Create line plot
    let line_plot = LinePlot::new(x_data.clone(), y_data.clone())
        .unwrap()
        .with_label("sin(x)")
        .with_style(glam::Vec4::new(1.0, 0.0, 0.0, 1.0), 2.0, LineStyle::Solid);

    // Create scatter plot
    let scatter_data: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).cos()).collect();
    let scatter_x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();

    let scatter_plot = ScatterPlot::new(scatter_x, scatter_data)
        .unwrap()
        .with_label("cos(x)")
        .with_style(
            glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            5.0,
            MarkerStyle::Circle,
        );

    // Create figure and combine plots
    let mut figure = Figure::new()
        .with_title("2D Multi-Plot Example")
        .with_labels("X", "Y")
        .with_grid(true);

    figure.add_line_plot(line_plot);
    figure.add_scatter_plot(scatter_plot);

    // Verify figure properties
    assert_eq!(figure.len(), 2);
    assert!(figure.title.is_some());
    assert!(figure.grid_enabled);
    assert!(figure.legend_enabled);

    // Test statistics
    let stats = figure.statistics();
    assert_eq!(stats.total_plots, 2);
    assert_eq!(stats.visible_plots, 2);
    assert!(stats.total_memory_usage > 0);

    // Test legend entries
    let legend_entries = figure.legend_entries();
    assert_eq!(legend_entries.len(), 2);
    assert_eq!(legend_entries[0].label, "sin(x)");
    assert_eq!(legend_entries[1].label, "cos(x)");
}

#[test]
fn test_3d_surface_plotting() {
    // Create test surface data
    let x_range = (-2.0, 2.0);
    let y_range = (-2.0, 2.0);
    let resolution = (20, 20);

    // Create surface from function: z = x^2 - y^2 (saddle point)
    let surface = SurfacePlot::from_function(x_range, y_range, resolution, |x, y| x * x - y * y)
        .unwrap()
        .with_colormap(ColorMap::Viridis)
        .with_shading(ShadingMode::Smooth)
        .with_alpha(0.8)
        .with_label("Saddle Surface");

    // Verify surface properties
    assert_eq!(surface.x_data.len(), 20);
    assert_eq!(surface.y_data.len(), 20);
    assert_eq!(surface.z_data.as_ref().unwrap().len(), 20);
    assert_eq!(surface.colormap, ColorMap::Viridis);
    assert_eq!(surface.shading_mode, ShadingMode::Smooth);
    assert_eq!(surface.alpha, 0.8);
    assert_eq!(surface.label, Some("Saddle Surface".to_string()));

    // Test statistics
    let stats = surface.statistics();
    assert_eq!(stats.grid_points, 400); // 20 * 20
    assert_eq!(stats.triangle_count, 722); // (20-1) * (20-1) * 2 = 722
    assert_eq!(stats.x_resolution, 20);
    assert_eq!(stats.y_resolution, 20);
    assert!(stats.memory_usage > 0);

    // Test bounds calculation
    let mut surface_mut = surface;
    let bounds = surface_mut.bounds();
    assert_eq!(bounds.min.x, -2.0);
    assert_eq!(bounds.max.x, 2.0);
    assert_eq!(bounds.min.y, -2.0);
    assert_eq!(bounds.max.y, 2.0);
    // Z bounds should be around [-4, 4] for x^2 - y^2 on [-2,2]x[-2,2]
    assert!(bounds.min.z >= -5.0 && bounds.min.z <= -3.0);
    assert!(bounds.max.z >= 3.0 && bounds.max.z <= 5.0);
}

// Point cloud tests removed with point_cloud module deprecation

#[test]
fn test_colormap_functionality() {
    let colormaps = vec![
        ColorMap::Jet,
        ColorMap::Hot,
        ColorMap::Cool,
        ColorMap::Viridis,
        ColorMap::Plasma,
        ColorMap::Gray,
    ];

    for colormap in colormaps {
        // Test value mapping at different points
        let color_0 = colormap.map_value(0.0);
        let _color_half = colormap.map_value(0.5);
        let color_1 = colormap.map_value(1.0);

        // Verify colors are valid (0-1 range)
        assert!(color_0.x >= 0.0 && color_0.x <= 1.0);
        assert!(color_0.y >= 0.0 && color_0.y <= 1.0);
        assert!(color_0.z >= 0.0 && color_0.z <= 1.0);

        assert!(color_1.x >= 0.0 && color_1.x <= 1.0);
        assert!(color_1.y >= 0.0 && color_1.y <= 1.0);
        assert!(color_1.z >= 0.0 && color_1.z <= 1.0);

        // Verify different values give different colors (for most colormaps)
        if colormap != ColorMap::Gray {
            assert_ne!(color_0, color_1);
        }
    }

    // Test custom colormap
    let custom = ColorMap::Custom(
        glam::Vec4::new(1.0, 0.0, 0.0, 1.0), // Red
        glam::Vec4::new(0.0, 1.0, 0.0, 1.0), // Green
    );

    let custom_0 = custom.map_value(0.0);
    let custom_1 = custom.map_value(1.0);

    // Should interpolate from red to green
    assert!(custom_0.x > custom_0.y); // More red at 0
    assert!(custom_1.y > custom_1.x); // More green at 1
}

#[test]
fn test_matlab_compatibility_functions() {
    use runmat_plot::plots::surface::matlab_compat as surf_compat;

    // Test MATLAB-style surface plot
    let surf_x = vec![0.0, 1.0];
    let surf_y = vec![0.0, 1.0];
    let surf_z = vec![vec![0.0, 1.0], vec![1.0, 2.0]];

    let surface = surf_compat::surf(surf_x.clone(), surf_y.clone(), surf_z.clone()).unwrap();
    assert!(!surface.wireframe);

    let mesh = surf_compat::mesh(surf_x, surf_y, surf_z).unwrap();
    assert!(mesh.wireframe);

    // 3D scatter not available in plots module; runtime handles future scatter3
}

#[cfg(feature = "jupyter")]
#[test]
fn test_jupyter_integration() {
    // Test Jupyter backend creation
    let backend = JupyterBackend::new();
    assert_eq!(backend.output_format, OutputFormat::HTML);

    // Test different output formats
    let formats = vec![
        OutputFormat::PNG,
        OutputFormat::SVG,
        OutputFormat::HTML,
        OutputFormat::Base64,
        OutputFormat::PlotlyJSON,
    ];

    for format in formats {
        let backend_fmt = JupyterBackend::with_format(format);
        assert_eq!(backend_fmt.output_format, format);
    }

    // Test direct backend usage
    let line_plot = LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    let mut backend = JupyterBackend::new();

    // These should not panic and return valid output
    let display_result = backend.display_line_plot(&line_plot);
    assert!(display_result.is_ok());
}

#[test]
fn test_performance_characteristics() {
    use runmat_time::Instant;

    // Test large dataset handling
    let large_size = 10_000;
    let x_data: Vec<f64> = (0..large_size).map(|i| i as f64 * 0.001).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .map(|&x| x.sin() + 0.1 * (10.0 * x).sin())
        .collect();

    let start = Instant::now();
    let large_line_plot = LinePlot::new(x_data, y_data).unwrap();
    let creation_time = start.elapsed();

    // Creation should be reasonably fast
    assert!(creation_time.as_millis() < 100);

    // Memory usage should be reasonable
    let memory_usage = large_line_plot.estimated_memory_usage();
    let expected_min = large_size * 16; // 2 * f64 per point
    assert!(memory_usage >= expected_min);

    // Test statistics generation performance
    let start = Instant::now();
    let stats = large_line_plot.statistics();
    let stats_time = start.elapsed();

    assert!(stats_time.as_millis() < 10);
    assert_eq!(stats.point_count, large_size);

    // Test large surface plot
    let surface_start = Instant::now();
    let large_surface = SurfacePlot::from_function((-5.0, 5.0), (-5.0, 5.0), (100, 100), |x, y| {
        (x * x + y * y).sin()
    })
    .unwrap();
    let surface_creation_time = surface_start.elapsed();

    // Surface creation should complete in reasonable time
    assert!(surface_creation_time.as_millis() < 1000);

    let surface_stats = large_surface.statistics();
    assert_eq!(surface_stats.grid_points, 10_000); // 100 * 100
    assert_eq!(surface_stats.triangle_count, 19_602); // (100-1) * (100-1) * 2
}

#[test]
fn test_error_handling() {
    // Test invalid data dimensions
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0]; // Mismatched length

    assert!(LinePlot::new(x.clone(), y.clone()).is_err());
    assert!(ScatterPlot::new(x, y).is_err());

    // Test empty data
    let empty_x: Vec<f64> = vec![];
    let empty_y: Vec<f64> = vec![];

    assert!(LinePlot::new(empty_x.clone(), empty_y.clone()).is_err());
    assert!(ScatterPlot::new(empty_x, empty_y).is_err());

    // Test invalid surface data
    let surf_x = vec![0.0, 1.0];
    let surf_y = vec![0.0, 1.0, 2.0];
    let surf_z = vec![vec![0.0, 1.0], vec![1.0, 2.0]]; // Wrong dimensions

    assert!(SurfacePlot::new(surf_x, surf_y, surf_z).is_err());

    // Point cloud module removed; skipping invalid point cloud tests
}

#[test]
fn test_memory_efficiency() {
    // Test that plots don't duplicate data unnecessarily
    let data_size = 1000;
    let x_data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x * 2.0).collect();

    let line_plot = LinePlot::new(x_data.clone(), y_data.clone()).unwrap();
    let initial_memory = line_plot.estimated_memory_usage();

    // Create multiple plots from same data
    let scatter_plot = ScatterPlot::new(x_data.clone(), y_data.clone()).unwrap();

    // Each plot should have similar memory usage (not sharing data currently, but reasonable)
    let scatter_memory = scatter_plot.estimated_memory_usage();
    let memory_ratio = scatter_memory as f64 / initial_memory as f64;

    // Memory usage should be in similar range (within 50% of each other)
    assert!(memory_ratio > 0.5 && memory_ratio < 2.0);

    // Test that cached data is reused
    let mut line_plot_mut = line_plot;
    let _vertices1 = line_plot_mut.generate_vertices();
    let memory_after_cache = line_plot_mut.estimated_memory_usage();

    // Memory should increase due to cached vertices
    assert!(memory_after_cache > initial_memory);

    // Generating vertices again should not increase memory further
    let _vertices2 = line_plot_mut.generate_vertices();
    let memory_after_reuse = line_plot_mut.estimated_memory_usage();

    assert_eq!(memory_after_cache, memory_after_reuse);
}

#[test]
fn test_plot_styling_and_customization() {
    // Test comprehensive styling options
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 0.0];

    let styled_line = LinePlot::new(x.clone(), y.clone())
        .unwrap()
        .with_style(glam::Vec4::new(0.8, 0.2, 0.1, 0.9), 3.5, LineStyle::Dashed)
        .with_label("Styled Line");

    assert_eq!(styled_line.color, glam::Vec4::new(0.8, 0.2, 0.1, 0.9));
    assert_eq!(styled_line.line_width, 3.5);
    assert_eq!(styled_line.line_style, LineStyle::Dashed);
    assert_eq!(styled_line.label, Some("Styled Line".to_string()));

    // Test surface styling
    let surface = SurfacePlot::new(
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![vec![0.0, 1.0], vec![1.0, 2.0]],
    )
    .unwrap()
    .with_colormap(ColorMap::Hot)
    .with_shading(ShadingMode::Faceted)
    .with_wireframe(true)
    .with_alpha(0.7);

    assert_eq!(surface.colormap, ColorMap::Hot);
    assert_eq!(surface.shading_mode, ShadingMode::Faceted);
    assert!(surface.wireframe);
    assert_eq!(surface.alpha, 0.7);
}

#[test]
fn test_bounds_calculation_accuracy() {
    // Test with known data
    let x = vec![-3.0, -1.0, 0.0, 2.0, 4.0];
    let y = vec![-2.0, 0.0, 1.0, -1.0, 3.0];

    let mut line_plot = LinePlot::new(x, y).unwrap();
    let bounds = line_plot.bounds();

    assert_eq!(bounds.min.x, -3.0);
    assert_eq!(bounds.max.x, 4.0);
    assert_eq!(bounds.min.y, -2.0);
    assert_eq!(bounds.max.y, 3.0);
    assert_eq!(bounds.min.z, 0.0); // 2D plot
    assert_eq!(bounds.max.z, 0.0); // 2D plot

    // Test surface bounds with known function
    let mut surface = SurfacePlot::from_function(
        (-1.0, 1.0),
        (-1.0, 1.0),
        (3, 3),
        |x, y| x + y, // Simple linear function
    )
    .unwrap();

    let surface_bounds = surface.bounds();
    assert_eq!(surface_bounds.min.x, -1.0);
    assert_eq!(surface_bounds.max.x, 1.0);
    assert_eq!(surface_bounds.min.y, -1.0);
    assert_eq!(surface_bounds.max.y, 1.0);
    assert_eq!(surface_bounds.min.z, -2.0); // min(x+y) = -1 + -1 = -2
    assert_eq!(surface_bounds.max.z, 2.0); // max(x+y) = 1 + 1 = 2
}
