//! Tests for Jupyter plotting integration in the RunMat kernel
//!
//! Comprehensive tests covering the plotting manager, protocol integration,
//! and display data generation.

use runmat_kernel::jupyter_plotting::{JupyterPlottingConfig, JupyterPlottingManager};
use runmat_plot::jupyter::OutputFormat;
use runmat_plot::plots::Figure;
use serde_json::{Number, Value as JsonValue};

#[test]
fn test_plotting_manager_creation() {
    let manager = JupyterPlottingManager::new();
    assert_eq!(manager.config().output_format, OutputFormat::HTML);
    assert!(manager.config().auto_display);
    assert_eq!(manager.config().max_plots, 100);
    assert_eq!(manager.list_plots().len(), 0);
}

#[test]
fn test_plotting_manager_with_custom_config() {
    let config = JupyterPlottingConfig {
        output_format: OutputFormat::SVG,
        auto_display: false,
        max_plots: 50,
        inline_display: true,
        image_width: 1024,
        image_height: 768,
    };

    let manager = JupyterPlottingManager::with_config(config.clone());
    assert_eq!(manager.config().output_format, OutputFormat::SVG);
    assert!(!manager.config().auto_display);
    assert_eq!(manager.config().max_plots, 50);
    assert_eq!(manager.config().image_width, 1024);
    assert_eq!(manager.config().image_height, 768);
}

#[test]
fn test_plot_registration() {
    let mut manager = JupyterPlottingManager::new();

    let figure = Figure::new().with_title("Test Plot");
    let display_data = manager.register_plot(figure).unwrap();

    // Should auto-display by default
    assert!(display_data.is_some());
    assert_eq!(manager.list_plots().len(), 1);

    // Check that plot was stored
    let plot_ids = manager.list_plots();
    assert_eq!(plot_ids.len(), 1);
    assert!(plot_ids[0].starts_with("plot_"));
}

#[test]
fn test_plot_registration_no_auto_display() {
    let config = JupyterPlottingConfig {
        output_format: OutputFormat::HTML,
        auto_display: false,
        max_plots: 100,
        inline_display: true,
        image_width: 800,
        image_height: 600,
    };

    let mut manager = JupyterPlottingManager::with_config(config);

    let figure = Figure::new().with_title("Test Plot");
    let display_data = manager.register_plot(figure).unwrap();

    // Should not auto-display
    assert!(display_data.is_none());
    assert_eq!(manager.list_plots().len(), 1);
}

#[test]
fn test_display_data_creation() {
    let mut manager = JupyterPlottingManager::new();

    let mut figure = Figure::new()
        .with_title("Display Test")
        .with_labels("X", "Y");

    let display_data = manager.create_display_data(&mut figure).unwrap();

    // Check structure
    assert!(!display_data.data.is_empty());
    assert!(display_data.data.contains_key("text/html"));
    assert!(display_data.transient.contains_key("runmat_plot_id"));
    assert!(display_data.transient.contains_key("runmat_version"));

    // Check HTML content
    if let Some(JsonValue::String(html)) = display_data.data.get("text/html") {
        assert!(html.contains("RunMat Interactive Plot"));
    } else {
        panic!("Expected HTML content in display data");
    }
}

#[test]
fn test_display_data_svg_format() {
    let config = JupyterPlottingConfig {
        output_format: OutputFormat::SVG,
        auto_display: true,
        max_plots: 100,
        inline_display: true,
        image_width: 800,
        image_height: 600,
    };

    let mut manager = JupyterPlottingManager::with_config(config);
    let mut figure = Figure::new().with_title("SVG Test");

    let display_data = manager.create_display_data(&mut figure).unwrap();

    // Should have SVG content
    assert!(display_data.data.contains_key("image/svg+xml"));

    if let Some(JsonValue::String(svg)) = display_data.data.get("image/svg+xml") {
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    } else {
        panic!("Expected SVG content in display data");
    }
}

#[test]
fn test_plot_function_handling_line() {
    let mut manager = JupyterPlottingManager::new();

    let x_data = JsonValue::Array(vec![
        JsonValue::Number(Number::from(1)),
        JsonValue::Number(Number::from(2)),
        JsonValue::Number(Number::from(3)),
    ]);

    let y_data = JsonValue::Array(vec![
        JsonValue::Number(Number::from(2)),
        JsonValue::Number(Number::from(4)),
        JsonValue::Number(Number::from(6)),
    ]);

    let result = manager
        .handle_plot_function("plot", &[x_data, y_data])
        .unwrap();

    assert!(result.is_some());
    assert_eq!(manager.list_plots().len(), 1);

    let display_data = result.unwrap();
    assert!(display_data.data.contains_key("text/html"));
}

#[test]
fn test_plot_function_handling_scatter() {
    let mut manager = JupyterPlottingManager::new();

    let x_data = JsonValue::Array(vec![
        JsonValue::Number(Number::from(1)),
        JsonValue::Number(Number::from(3)),
        JsonValue::Number(Number::from(5)),
    ]);

    let y_data = JsonValue::Array(vec![
        JsonValue::Number(Number::from(2)),
        JsonValue::Number(Number::from(6)),
        JsonValue::Number(Number::from(10)),
    ]);

    let result = manager
        .handle_plot_function("scatter", &[x_data, y_data])
        .unwrap();

    assert!(result.is_some());
    assert_eq!(manager.list_plots().len(), 1);
}

#[test]
fn test_plot_function_handling_bar() {
    let mut manager = JupyterPlottingManager::new();

    let y_data = JsonValue::Array(vec![
        JsonValue::Number(Number::from(10)),
        JsonValue::Number(Number::from(20)),
        JsonValue::Number(Number::from(15)),
        JsonValue::Number(Number::from(30)),
    ]);

    let result = manager.handle_plot_function("bar", &[y_data]).unwrap();

    assert!(result.is_some());
    assert_eq!(manager.list_plots().len(), 1);
}

#[test]
fn test_plot_function_handling_histogram() {
    let mut manager = JupyterPlottingManager::new();

    let data = JsonValue::Array(vec![
        JsonValue::Number(Number::from(1)),
        JsonValue::Number(Number::from(2)),
        JsonValue::Number(Number::from(2)),
        JsonValue::Number(Number::from(3)),
        JsonValue::Number(Number::from(3)),
        JsonValue::Number(Number::from(3)),
        JsonValue::Number(Number::from(4)),
    ]);

    let bins = JsonValue::Number(Number::from(5));

    let result = manager.handle_plot_function("hist", &[data, bins]).unwrap();

    assert!(result.is_some());
    assert_eq!(manager.list_plots().len(), 1);
}

#[test]
fn test_plot_function_error_handling() {
    let mut manager = JupyterPlottingManager::new();

    // Test unknown function
    let result = manager.handle_plot_function("unknown_plot", &[]);
    assert!(result.is_err());

    // Test mismatched data lengths
    let x_data = JsonValue::Array(vec![JsonValue::Number(Number::from(1))]);
    let y_data = JsonValue::Array(vec![
        JsonValue::Number(Number::from(1)),
        JsonValue::Number(Number::from(2)),
    ]);

    let result = manager.handle_plot_function("plot", &[x_data, y_data]);
    assert!(result.is_err());
}

// Note: extract_numeric_array is a private method, so we test it indirectly through plot functions

// Note: extract_number is also a private method, tested indirectly

#[test]
fn test_config_update() {
    let mut manager = JupyterPlottingManager::new();

    // Initial config
    assert_eq!(manager.config().output_format, OutputFormat::HTML);

    // Update config
    let new_config = JupyterPlottingConfig {
        output_format: OutputFormat::PNG,
        auto_display: false,
        max_plots: 25,
        inline_display: false,
        image_width: 640,
        image_height: 480,
    };

    manager.update_config(new_config.clone());

    // Verify changes
    assert_eq!(manager.config().output_format, OutputFormat::PNG);
    assert!(!manager.config().auto_display);
    assert_eq!(manager.config().max_plots, 25);
    assert_eq!(manager.config().image_width, 640);
}

#[test]
fn test_plot_cleanup() {
    let config = JupyterPlottingConfig {
        output_format: OutputFormat::HTML,
        auto_display: false, // Disable auto-display for easier testing
        max_plots: 3,        // Small limit for testing cleanup
        inline_display: true,
        image_width: 800,
        image_height: 600,
    };

    let mut manager = JupyterPlottingManager::with_config(config);

    // Add plots up to the limit
    for i in 0..5 {
        let figure = Figure::new().with_title(format!("Plot {i}"));
        manager.register_plot(figure).unwrap();
    }

    // Should have cleaned up old plots
    assert!(manager.list_plots().len() <= 3);
}

#[test]
fn test_plot_retrieval() {
    let mut manager = JupyterPlottingManager::new();

    let figure = Figure::new().with_title("Retrievable Plot");
    manager.register_plot(figure).unwrap();

    let plot_ids = manager.list_plots();
    assert_eq!(plot_ids.len(), 1);

    let retrieved_plot = manager.get_plot(&plot_ids[0]);
    assert!(retrieved_plot.is_some());

    // Test non-existent plot
    let non_existent = manager.get_plot("non_existent_id");
    assert!(non_existent.is_none());
}

#[test]
fn test_clear_plots() {
    let mut manager = JupyterPlottingManager::new();

    // Add some plots
    for i in 0..3 {
        let figure = Figure::new().with_title(format!("Plot {i}"));
        manager.register_plot(figure).unwrap();
    }

    assert_eq!(manager.list_plots().len(), 3);

    // Clear all plots
    manager.clear_plots();

    assert_eq!(manager.list_plots().len(), 0);
    // plot_counter is private, can't test directly
}

#[test]
fn test_metadata_in_display_data() {
    let mut manager = JupyterPlottingManager::new();
    let mut figure = Figure::new().with_title("Metadata Test");

    let display_data = manager.create_display_data(&mut figure).unwrap();

    // Check transient metadata
    assert!(display_data.transient.contains_key("runmat_plot_id"));
    assert!(display_data.transient.contains_key("runmat_version"));

    if let Some(JsonValue::String(version)) = display_data.transient.get("runmat_version") {
        assert_eq!(version, "0.0.1");
    }

    if let Some(JsonValue::String(plot_id)) = display_data.transient.get("runmat_plot_id") {
        assert!(plot_id.starts_with("plot_"));
    }
}

#[test]
fn test_concurrent_plotting() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let manager = Arc::new(Mutex::new(JupyterPlottingManager::new()));
    let mut handles = vec![];

    // Create plots from multiple threads
    for i in 0..5 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let figure = Figure::new().with_title(format!("Concurrent Plot {i}"));
            let mut manager_guard = manager_clone.lock().unwrap();
            let result = manager_guard.register_plot(figure);
            assert!(result.is_ok());
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Check that all plots were registered
    let manager_guard = manager.lock().unwrap();
    assert_eq!(manager_guard.list_plots().len(), 5);
}

#[test]
fn test_output_format_switching() {
    let mut manager = JupyterPlottingManager::new();
    let mut figure = Figure::new().with_title("Format Test");

    // Test HTML format
    assert_eq!(manager.config().output_format, OutputFormat::HTML);
    let html_data = manager.create_display_data(&mut figure).unwrap();
    assert!(html_data.data.contains_key("text/html"));

    // Switch to SVG format
    let svg_config = JupyterPlottingConfig {
        output_format: OutputFormat::SVG,
        auto_display: true,
        max_plots: 100,
        inline_display: true,
        image_width: 800,
        image_height: 600,
    };
    manager.update_config(svg_config);

    let svg_data = manager.create_display_data(&mut figure).unwrap();
    assert!(svg_data.data.contains_key("image/svg+xml"));

    // Switch to PNG format
    let png_config = JupyterPlottingConfig {
        output_format: OutputFormat::PNG,
        auto_display: true,
        max_plots: 100,
        inline_display: true,
        image_width: 800,
        image_height: 600,
    };
    manager.update_config(png_config);

    let png_data = manager.create_display_data(&mut figure).unwrap();
    assert!(png_data.data.contains_key("text/html")); // PNG returns HTML img tag
}

#[test]
fn test_edge_case_data() {
    let mut manager = JupyterPlottingManager::new();

    // Test empty arrays
    let empty_x = JsonValue::Array(vec![]);
    let empty_y = JsonValue::Array(vec![]);
    let _empty_result = manager.handle_plot_function("plot", &[empty_x, empty_y]);
    // Should handle gracefully (may succeed with empty plot or fail gracefully)

    // Test very large numbers (use integers for Number::from)
    let large_x = JsonValue::Array(vec![JsonValue::Number(Number::from(1000000))]);
    let large_y = JsonValue::Array(vec![JsonValue::Number(Number::from(2000000))]);
    let large_result = manager.handle_plot_function("plot", &[large_x, large_y]);
    assert!(large_result.is_ok());

    // Test floating point numbers
    let small_x = JsonValue::Array(vec![JsonValue::Number(Number::from_f64(0.001).unwrap())]);
    let small_y = JsonValue::Array(vec![JsonValue::Number(Number::from_f64(0.002).unwrap())]);
    let small_result = manager.handle_plot_function("plot", &[small_x, small_y]);
    assert!(small_result.is_ok());
}

#[test]
fn test_performance_large_datasets() {
    use std::time::Instant;

    let mut manager = JupyterPlottingManager::new();

    // Create large dataset
    let large_data: Vec<JsonValue> = (0..1000)
        .map(|i| JsonValue::Number(Number::from(i)))
        .collect();

    let x_data = JsonValue::Array(large_data.clone());
    let y_data = JsonValue::Array(large_data);

    let start = Instant::now();
    let result = manager.handle_plot_function("plot", &[x_data, y_data]);
    let duration = start.elapsed();

    assert!(result.is_ok());
    println!("Large dataset plot creation took: {duration:?}");

    // Should complete within reasonable time
    assert!(duration.as_millis() < 10000); // 10 seconds max
}
