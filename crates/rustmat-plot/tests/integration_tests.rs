//! Integration tests demonstrating complete plotting workflows
//!
//! This module tests the plotting system working end-to-end:
//! - Plot creation and data management
//! - Scene graph integration
//! - Camera and bounds calculation
//! - Render data generation
//! - Performance characteristics

use rustmat_plot::core::{Scene, Camera, BoundingBox, SceneNode};
use rustmat_plot::plots::{LinePlot, ScatterPlot, LineStyle, MarkerStyle};
use glam::{Vec3, Vec4, Mat4};

#[test]
fn test_complete_plotting_workflow() {
    // Create test data
    let x_data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| x.sin()).collect();
    
    // Create a line plot
    let mut line_plot = LinePlot::new(x_data.clone(), y_data.clone()).unwrap()
        .with_style(Vec4::new(0.0, 0.5, 1.0, 1.0), 2.0, LineStyle::Solid)
        .with_label("Sine Wave");
    
    // Generate scatter data
    let scatter_x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let scatter_y: Vec<f64> = scatter_x.iter().map(|x| x.cos()).collect();
    
    let mut scatter_plot = ScatterPlot::new(scatter_x, scatter_y).unwrap()
        .with_style(Vec4::new(1.0, 0.0, 0.0, 1.0), 4.0, MarkerStyle::Circle)
        .with_label("Cosine Points");
    
    // Test render data generation
    let line_render_data = line_plot.render_data();
    let scatter_render_data = scatter_plot.render_data();
    
    // Verify render data is correctly generated
    assert!(!line_render_data.vertices.is_empty());
    assert!(!scatter_render_data.vertices.is_empty());
    assert_eq!(line_render_data.pipeline_type, rustmat_plot::core::PipelineType::Lines);
    assert_eq!(scatter_render_data.pipeline_type, rustmat_plot::core::PipelineType::Points);
    
    // Test bounds calculation
    let line_bounds = line_plot.bounds();
    let _scatter_bounds = scatter_plot.bounds();
    
    assert!(line_bounds.min.x >= -0.1); // Should start around 0
    assert!(line_bounds.max.x <= 10.1); // Should end around 10
    assert!(line_bounds.min.y >= -1.1); // Sine range
    assert!(line_bounds.max.y <= 1.1);
    
    // Test statistics
    let line_stats = line_plot.statistics();
    let scatter_stats = scatter_plot.statistics();
    
    assert_eq!(line_stats.point_count, 100);
    assert_eq!(scatter_stats.point_count, 20);
    assert!(line_stats.memory_usage > 0);
    assert!(scatter_stats.memory_usage > 0);
}

#[test]
fn test_scene_graph_integration() {
    // Create a scene
    let mut scene = Scene::new();
    
    // Create plots
    let x_data = vec![0.0, 1.0, 2.0, 3.0];
    let y_data = vec![0.0, 1.0, 0.0, 1.0];
    
    let mut line_plot = LinePlot::new(x_data.clone(), y_data.clone()).unwrap()
        .with_style(Vec4::new(0.0, 1.0, 0.0, 1.0), 1.0, LineStyle::Solid);
    
    let mut scatter_plot = ScatterPlot::new(x_data, y_data).unwrap()
        .with_style(Vec4::new(1.0, 0.0, 0.0, 1.0), 3.0, MarkerStyle::Square);
    
    // Create scene nodes
    let line_node = SceneNode {
        id: 0, // Will be set by scene
        name: "Line Plot".to_string(),
        transform: Mat4::IDENTITY,
        visible: true,
        cast_shadows: false,
        receive_shadows: false,
        parent: None,
        children: Vec::new(),
        render_data: Some(line_plot.render_data()),
        bounds: BoundingBox::from_points(&vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
        ]),
        lod_levels: Vec::new(),
        current_lod: 0,
    };
    
    let scatter_node = SceneNode {
        id: 0, // Will be set by scene
        name: "Scatter Plot".to_string(),
        transform: Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0)), // Offset up
        visible: true,
        cast_shadows: false,
        receive_shadows: false,
        parent: None,
        children: Vec::new(),
        render_data: Some(scatter_plot.render_data()),
        bounds: BoundingBox::from_points(&vec![
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(3.0, 3.0, 0.0),
        ]),
        lod_levels: Vec::new(),
        current_lod: 0,
    };
    
    // Add to scene
    let line_id = scene.add_node(line_node);
    let scatter_id = scene.add_node(scatter_node);
    
    // Verify scene contains our nodes
    assert!(scene.get_node(line_id).is_some());
    assert!(scene.get_node(scatter_id).is_some());
    
    // Test scene bounds calculation
    let world_bounds = scene.world_bounds();
    assert!(world_bounds.min.x <= 0.0);
    assert!(world_bounds.max.x >= 3.0);
    assert!(world_bounds.min.y <= 0.0);
    assert!(world_bounds.max.y >= 3.0);
    
    // Test visible nodes
    let visible_nodes = scene.get_visible_nodes();
    assert_eq!(visible_nodes.len(), 2);
    
    // Test scene statistics
    let stats = scene.statistics();
    assert_eq!(stats.total_nodes, 2);
    assert_eq!(stats.visible_nodes, 2);
    assert!(stats.total_vertices > 0);
}

#[test]
fn test_camera_integration() {
    // Create test data
    let x_data = vec![-5.0, 0.0, 5.0];
    let y_data = vec![-3.0, 0.0, 3.0];
    
    let mut plot = LinePlot::new(x_data, y_data).unwrap();
    let bounds = plot.bounds();
    
    // Test 2D camera fitting to plot bounds
    let mut camera = Camera::new_2d((-1.0, 1.0, -1.0, 1.0));
    camera.fit_bounds(bounds.min, bounds.max);
    
    // Verify camera adjusted to fit data
    match camera.projection {
        rustmat_plot::core::ProjectionType::Orthographic { left, right, bottom, top, .. } => {
            assert!(left <= -5.0);
            assert!(right >= 5.0);
            assert!(bottom <= -3.0);
            assert!(top >= 3.0);
        }
        _ => panic!("Expected orthographic projection for 2D camera"),
    }
    
    // Test 3D camera
    let mut camera_3d = Camera::new();
    camera_3d.fit_bounds(bounds.min, bounds.max);
    
    // Camera should have adjusted position to show all data
    assert!(camera_3d.position.distance(camera_3d.target) > 0.0);
}

#[test]
fn test_matlab_compatibility() {
    use rustmat_plot::plots::line::matlab_compat as line_matlab;
    use rustmat_plot::plots::scatter::matlab_compat as scatter_matlab;
    
    // Test MATLAB-style line plot creation
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 4.0, 9.0];
    
    let plot1 = line_matlab::plot(x.clone(), y.clone()).unwrap();
    assert_eq!(plot1.len(), 4);
    
    let plot2 = line_matlab::plot_with_color(x.clone(), y.clone(), "r").unwrap();
    assert_eq!(plot2.color, Vec4::new(1.0, 0.0, 0.0, 1.0));
    
    // Test MATLAB-style scatter plot creation
    let scatter1 = scatter_matlab::scatter(x.clone(), y.clone()).unwrap();
    assert_eq!(scatter1.len(), 4);
    
    let scatter2 = scatter_matlab::scatter_with_style(x, y, 5.0, "b").unwrap();
    assert_eq!(scatter2.color, Vec4::new(0.0, 0.0, 1.0, 1.0));
    assert_eq!(scatter2.marker_size, 5.0);
}

#[test]
fn test_performance_characteristics() {
    // Test with larger datasets to ensure good performance
    let n = 10000;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| (x * 0.001).sin()).collect();
    
    // Time plot creation
    let start = std::time::Instant::now();
    let mut plot = LinePlot::new(x_data, y_data).unwrap();
    let creation_time = start.elapsed();
    
    // Should create plot quickly (under 10ms for 10k points)
    assert!(creation_time.as_millis() < 10);
    
    // Time vertex generation
    let start = std::time::Instant::now();
    let vertices = plot.generate_vertices();
    let vertex_time = start.elapsed();
    
    // Should generate vertices quickly (under 50ms for 10k points)
    assert!(vertex_time.as_millis() < 50);
    
    // Verify we have the right number of vertices (19,998 for 9,999 line segments)
    assert_eq!(vertices.len(), (n - 1) * 2);
    
    // Time bounds calculation
    let start = std::time::Instant::now();
    let bounds = plot.bounds();
    let bounds_time = start.elapsed();
    
    // Should calculate bounds quickly (under 5ms)
    assert!(bounds_time.as_millis() < 5);
    
    // Verify bounds are reasonable
    assert!(bounds.min.x >= -1.0);
    assert!(bounds.max.x <= n as f32);
}

#[test]
fn test_memory_management() {
    // Test that plots properly manage memory with data updates
    let initial_x = vec![0.0, 1.0, 2.0];
    let initial_y = vec![0.0, 1.0, 0.0];
    
    let mut plot = LinePlot::new(initial_x, initial_y).unwrap();
    
    // Generate initial vertices
    let initial_vertices = plot.generate_vertices().len();
    let initial_stats = plot.statistics();
    
    // Update with larger dataset
    let new_x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let new_y: Vec<f64> = new_x.iter().map(|x| x.sin()).collect();
    
    plot.update_data(new_x, new_y).unwrap();
    
    // Verify old data is replaced, not accumulated
    let new_vertices = plot.generate_vertices().len();
    let new_stats = plot.statistics();
    
    assert!(new_vertices > initial_vertices);
    assert!(new_stats.memory_usage > initial_stats.memory_usage);
    assert_eq!(new_stats.point_count, 1000);
    
    // Memory usage should be proportional to data size
    let memory_per_point = new_stats.memory_usage / new_stats.point_count;
    assert!(memory_per_point > 0);
    assert!(memory_per_point < 1000); // Reasonable per-point memory usage
}

#[test]
fn test_multi_plot_scenario() {
    // Simulate a complex plotting scenario with multiple data series
    let mut scene = Scene::new();
    let mut plots = Vec::new();
    
    // Create multiple plots with different characteristics
    for i in 0..5 {
        let offset = i as f64 * 2.0;
        let x_data: Vec<f64> = (0..100).map(|j| j as f64 * 0.1 + offset).collect();
        let y_data: Vec<f64> = x_data.iter().map(|x| (x * (i + 1) as f64).sin()).collect();
        
        let color = Vec4::new(
            (i as f32 + 1.0) / 6.0,
            0.5,
            1.0 - (i as f32) / 5.0,
            1.0
        );
        
        let mut plot = LinePlot::new(x_data, y_data).unwrap()
            .with_style(color, 1.0 + i as f32 * 0.5, LineStyle::Solid)
            .with_label(format!("Series {}", i + 1));
        
        // Create scene node
        let node = SceneNode {
            id: 0,
            name: format!("Plot {}", i),
            transform: Mat4::IDENTITY,
            visible: true,
            cast_shadows: false,
            receive_shadows: false,
            parent: None,
            children: Vec::new(),
            render_data: Some(plot.render_data()),
            bounds: plot.bounds(),
            lod_levels: Vec::new(),
            current_lod: 0,
        };
        
        scene.add_node(node);
        plots.push(plot);
    }
    
    // Verify scene contains all plots
    let visible_nodes = scene.get_visible_nodes();
    assert_eq!(visible_nodes.len(), 5);
    
    // Test overall scene bounds
    let world_bounds = scene.world_bounds();
    assert!(world_bounds.max.x - world_bounds.min.x > 10.0); // Should span multiple series
    
    // Test scene statistics
    let stats = scene.statistics();
    assert_eq!(stats.total_nodes, 5);
    assert_eq!(stats.visible_nodes, 5);
    assert!(stats.total_vertices > 500); // Should have lots of vertices from all series
}