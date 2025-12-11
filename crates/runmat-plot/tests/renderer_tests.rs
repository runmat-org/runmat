//! WGPU renderer tests for the plotting system
//!
//! This module tests GPU-accelerated rendering infrastructure:
//! - WGPU device and surface setup
//! - Pipeline creation and management
//! - Vertex buffer operations
//! - Uniform buffer management
//! - Rendering command generation

use glam::{Mat4, Vec3, Vec4};
use runmat_plot::core::{vertex_utils, PipelineType, Uniforms};
use runmat_time::Instant;

#[cfg(test)]
mod export_subplot_tests {
    #[test]
    fn test_svg_two_axes_line_scatter() {
        let mut fig = runmat_plot::plots::Figure::new();
        fig.set_subplot_grid(2, 1);
        let x: Vec<f64> = (0..=50).map(|i| i as f64 * 0.2).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let i1 = fig.add_line_plot(runmat_plot::plots::LinePlot::new(x.clone(), y).unwrap());
        let _ = fig.assign_plot_to_axes(i1, 0);
        let y2: Vec<f64> = x.iter().map(|v| v.cos() * 0.5 + 0.5).collect();
        let i2 = fig.add_scatter_plot(runmat_plot::plots::ScatterPlot::new(x.clone(), y2).unwrap());
        let _ = fig.assign_plot_to_axes(i2, 1);
        let exporter = runmat_plot::export::vector::VectorExporter::new();
        let svg = exporter.render_to_svg(&mut fig).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.len() > 500);
    }

    #[test]
    fn test_svg_four_axes_bars_imagesc() {
        let mut fig = runmat_plot::plots::Figure::new();
        fig.set_subplot_grid(2, 2);
        let y = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let labels: Vec<String> = (1..=y.len()).map(|i| i.to_string()).collect();
        let i0 = fig.add_bar_chart(runmat_plot::plots::BarChart::new(labels, y).unwrap());
        let _ = fig.assign_plot_to_axes(i0, 0);
        let rows = 5usize;
        let cols = 5usize;
        let mut grid = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                grid[r][c] = (r as f64) / (rows as f64);
            }
        }
        let xv: Vec<f64> = (1..=cols).map(|i| i as f64).collect();
        let mut yv: Vec<f64> = (1..=rows).map(|i| i as f64).collect();
        yv.reverse();
        let img = runmat_plot::plots::ImagePlot::from_grayscale(
            xv,
            yv,
            grid,
            runmat_plot::plots::surface::ColorMap::Parula,
            None,
        )
        .unwrap();
        let i1 = fig.add_image_plot(img);
        let _ = fig.assign_plot_to_axes(i1, 1);
        let x2: Vec<f64> = (0..=20).map(|i| i as f64 * 0.2).collect();
        let y2: Vec<f64> = x2.iter().map(|v| v.sin()).collect();
        let i2 = fig.add_line_plot(runmat_plot::plots::LinePlot::new(x2, y2).unwrap());
        let _ = fig.assign_plot_to_axes(i2, 2);
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|v| v.cos()).collect();
        let i3 = fig.add_scatter_plot(runmat_plot::plots::ScatterPlot::new(xs, ys).unwrap());
        let _ = fig.assign_plot_to_axes(i3, 3);
        let exporter = runmat_plot::export::vector::VectorExporter::new();
        let svg = exporter.render_to_svg(&mut fig).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.len() > 500);
    }
    #[tokio::test]
    async fn test_export_two_axes_line_scatter() {
        let mut fig = runmat_plot::plots::Figure::new();
        fig.set_subplot_grid(2, 1);
        // Top: line
        let x: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let lp = runmat_plot::plots::LinePlot::new(x.clone(), y).unwrap();
        let i1 = fig.add_line_plot(lp);
        let _ = fig.assign_plot_to_axes(i1, 0);
        // Bottom: scatter
        let y2: Vec<f64> = x.iter().map(|v| v.cos() * 0.5 + 0.5).collect();
        let mut sp = runmat_plot::plots::ScatterPlot::new(x.clone(), y2).unwrap();
        sp.set_marker_style(runmat_plot::plots::scatter::MarkerStyle::Circle);
        let i2 = fig.add_scatter_plot(sp);
        let _ = fig.assign_plot_to_axes(i2, 1);

        let exporter = runmat_plot::export::image::ImageExporter::with_settings(
            runmat_plot::export::image::ImageExportSettings {
                width: 640,
                height: 480,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let tmp = std::env::temp_dir().join("subplot_line_scatter.png");
        exporter.export_png(&mut fig, &tmp).await.unwrap();
        assert!(tmp.exists());
        let meta = std::fs::metadata(&tmp).unwrap();
        assert!(meta.len() > 1_000);
    }

    #[tokio::test]
    async fn test_export_four_axes_bars_imagesc() {
        let mut fig = runmat_plot::plots::Figure::new();
        fig.set_subplot_grid(2, 2);
        // Bars in (0,0)
        let y = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let labels: Vec<String> = (1..=y.len()).map(|i| i.to_string()).collect();
        let b = runmat_plot::plots::BarChart::new(labels, y.clone()).unwrap();
        let i0 = fig.add_bar_chart(b);
        let _ = fig.assign_plot_to_axes(i0, 0);
        // Imagesc in (0,1)
        let rows = 10usize;
        let cols = 10usize;
        let mut grid = vec![vec![0.0f64; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                grid[r][c] = (r as f64) / (rows as f64);
            }
        }
        let x: Vec<f64> = (1..=cols).map(|i| i as f64).collect();
        let mut yv: Vec<f64> = (1..=rows).map(|i| i as f64).collect();
        yv.reverse();
        let img = runmat_plot::plots::ImagePlot::from_grayscale(
            x.clone(),
            yv.clone(),
            grid,
            runmat_plot::plots::surface::ColorMap::Parula,
            None,
        )
        .unwrap();
        let i1 = fig.add_image_plot(img);
        let _ = fig.assign_plot_to_axes(i1, 1);
        // Line in (1,0)
        let x2: Vec<f64> = (0..=50).map(|i| i as f64 * 0.2).collect();
        let y2: Vec<f64> = x2.iter().map(|v| v.sin() * 0.5).collect();
        let i2 = fig.add_line_plot(runmat_plot::plots::LinePlot::new(x2, y2).unwrap());
        let _ = fig.assign_plot_to_axes(i2, 2);
        // Scatter in (1,1)
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|v| v.cos()).collect();
        let i3 = fig.add_scatter_plot(runmat_plot::plots::ScatterPlot::new(xs, ys).unwrap());
        let _ = fig.assign_plot_to_axes(i3, 3);

        let exporter = runmat_plot::export::image::ImageExporter::with_settings(
            runmat_plot::export::image::ImageExportSettings {
                width: 800,
                height: 600,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let tmp = std::env::temp_dir().join("subplot_four_axes.png");
        exporter.export_png(&mut fig, &tmp).await.unwrap();
        assert!(tmp.exists());
        let meta = std::fs::metadata(&tmp).unwrap();
        assert!(meta.len() > 1_000);
    }
}

#[cfg(test)]
mod clipping_tests {
    #[tokio::test]
    async fn test_clipping_out_of_bounds_line() {
        // Top axes has a line; bottom should remain background (white)
        let mut fig = runmat_plot::plots::Figure::new();
        fig.set_subplot_grid(2, 1);
        let x: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let i0 = fig.add_line_plot(runmat_plot::plots::LinePlot::new(x, y).unwrap());
        let _ = fig.assign_plot_to_axes(i0, 0);

        let exporter = runmat_plot::export::image::ImageExporter::with_settings(
            runmat_plot::export::image::ImageExportSettings {
                width: 400,
                height: 300,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let tmp = std::env::temp_dir().join("clip_line.png");
        exporter.export_png(&mut fig, &tmp).await.unwrap();
        assert!(tmp.exists());

        let img = image::open(&tmp).unwrap().to_rgba8();
        // Sample a pixel in bottom center; expect near white background
        let px = img.get_pixel(200, 250);
        assert!(px[0] as f32 >= 240.0 && px[1] as f32 >= 240.0 && px[2] as f32 >= 240.0);
    }

    #[tokio::test]
    async fn test_clipping_out_of_bounds_scatter() {
        // Bottom axes has scatter; top should remain background
        let mut fig = runmat_plot::plots::Figure::new();
        fig.set_subplot_grid(2, 1);
        let x: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v.cos()).collect();
        let i1 = fig.add_scatter_plot(runmat_plot::plots::ScatterPlot::new(x, y).unwrap());
        let _ = fig.assign_plot_to_axes(i1, 1);

        let exporter = runmat_plot::export::image::ImageExporter::with_settings(
            runmat_plot::export::image::ImageExportSettings {
                width: 400,
                height: 300,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let tmp = std::env::temp_dir().join("clip_scatter.png");
        exporter.export_png(&mut fig, &tmp).await.unwrap();
        assert!(tmp.exists());

        let img = image::open(&tmp).unwrap().to_rgba8();
        // Sample a pixel in top center; expect near white background
        let px = img.get_pixel(200, 50);
        assert!(px[0] as f32 >= 240.0 && px[1] as f32 >= 240.0 && px[2] as f32 >= 240.0);
    }
}

#[cfg(test)]
mod aesthetics_line_tests {
    use runmat_plot::plots::line::{LineCap, LineJoin, LinePlot, LineStyle};

    #[tokio::test]
    async fn test_thick_line_caps_png() {
        let mut fig = runmat_plot::plots::Figure::new();
        let x: Vec<f64> = (0..=50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|v| (v * 1.0).sin()).collect();
        let mut lp = LinePlot::new(x.clone(), y.clone()).unwrap();
        lp.set_line_width(8.0);
        lp.set_line_style(LineStyle::Solid);
        lp.set_line_join(LineJoin::Round);
        lp.set_line_cap(LineCap::Round);
        let _ = fig.add_line_plot(lp);

        let exporter = runmat_plot::export::image::ImageExporter::with_settings(
            runmat_plot::export::image::ImageExportSettings {
                width: 320,
                height: 240,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let tmp = std::env::temp_dir().join("thick_line_caps.png");
        exporter.export_png(&mut fig, &tmp).await.unwrap();
        assert!(tmp.exists());
        let img = image::open(&tmp).unwrap().to_rgba8();
        assert!(img.width() == 320 && img.height() == 240);
    }

    #[tokio::test]
    async fn test_dashed_dotted_png() {
        let mut fig = runmat_plot::plots::Figure::new();
        let x: Vec<f64> = (0..=100).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|v| (v * 0.7).sin()).collect();
        let mut dashed = LinePlot::new(x.clone(), y.clone()).unwrap();
        dashed.set_line_style(LineStyle::Dashed);
        dashed.set_line_width(1.0);
        let _ = fig.add_line_plot(dashed);

        let mut dotted = LinePlot::new(x.clone(), y.clone()).unwrap();
        dotted.set_line_style(LineStyle::Dotted);
        dotted.set_line_width(4.0);
        let _ = fig.add_line_plot(dotted);

        let exporter = runmat_plot::export::image::ImageExporter::with_settings(
            runmat_plot::export::image::ImageExportSettings {
                width: 360,
                height: 240,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let tmp = std::env::temp_dir().join("dash_dot_lines.png");
        exporter.export_png(&mut fig, &tmp).await.unwrap();
        assert!(tmp.exists());
    }
}

#[cfg(test)]
mod svg_aesthetics_tests {
    use runmat_plot::export::vector::{VectorExportSettings, VectorExporter};
    use runmat_plot::plots::line::{LineCap, LineJoin, LinePlot, LineStyle};

    #[test]
    fn test_svg_stroke_caps_joins_and_dasharray() {
        let mut fig = runmat_plot::plots::Figure::new();
        let x: Vec<f64> = (0..=20).map(|i| i as f64 * 0.2).collect();
        let y1: Vec<f64> = x.iter().map(|v| (v * 0.7).sin()).collect();
        let y2: Vec<f64> = x.iter().map(|v| (v * 0.7).cos()).collect();

        let mut solid = LinePlot::new(x.clone(), y1).unwrap();
        solid.set_line_style(LineStyle::Solid);
        solid.set_line_width(1.0);
        solid.set_line_join(LineJoin::Bevel);
        solid.set_line_cap(LineCap::Square);
        let _ = fig.add_line_plot(solid);

        let mut dashed = LinePlot::new(x.clone(), y2).unwrap();
        dashed.set_line_style(LineStyle::DashDot);
        dashed.set_line_width(1.0);
        dashed.set_line_join(LineJoin::Round);
        dashed.set_line_cap(LineCap::Round);
        let _ = fig.add_line_plot(dashed);

        let exporter = VectorExporter::with_settings(VectorExportSettings {
            width: 320.0,
            height: 240.0,
            ..Default::default()
        });
        let svg = exporter.render_to_svg(&mut fig).unwrap();
        assert!(svg.contains("stroke-linecap=\"square\""));
        assert!(svg.contains("stroke-linejoin=\"bevel\""));
        assert!(
            svg.contains("stroke-dasharray=\"6,4,1,4\"")
                || svg.contains("stroke-dasharray=\"6,4,1,4\"")
        );
    }

    #[test]
    fn test_svg_dasharray_scales_with_line_width() {
        let mut fig = runmat_plot::plots::Figure::new();
        let x: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| (v * 0.3).sin()).collect();

        // Dashed with lw=0.5 -> 3,3 (lines pipeline)
        let mut dashed = LinePlot::new(x.clone(), y.clone()).unwrap();
        dashed.set_line_style(LineStyle::Dashed);
        dashed.set_line_width(0.5);
        let _ = fig.add_line_plot(dashed);

        // Dotted with lw=0.5 -> 0.5,3
        let mut dotted = LinePlot::new(x.clone(), y.clone()).unwrap();
        dotted.set_line_style(LineStyle::Dotted);
        dotted.set_line_width(0.5);
        let _ = fig.add_line_plot(dotted);

        // DashDot with lw=0.5 -> 3,2,0.5,2
        let mut dashdot = LinePlot::new(x.clone(), y.clone()).unwrap();
        dashdot.set_line_style(LineStyle::DashDot);
        dashdot.set_line_width(0.5);
        let _ = fig.add_line_plot(dashdot);

        let exporter = VectorExporter::with_settings(VectorExportSettings {
            width: 320.0,
            height: 240.0,
            ..Default::default()
        });
        let svg = exporter.render_to_svg(&mut fig).unwrap();

        assert!(
            svg.contains("stroke-dasharray=\"3,3\""),
            "expected dashed 3,3; svg={}",
            svg
        );
        assert!(
            svg.contains("stroke-dasharray=\"0.5,3\""),
            "expected dotted 0.5,3; svg={}",
            svg
        );
        assert!(
            svg.contains("stroke-dasharray=\"3,2,0.5,2\""),
            "expected dashdot 3,2,0.5,2; svg={}",
            svg
        );
    }
}

#[cfg(test)]
mod svg_image_embed_tests {
    use runmat_plot::export::vector::{VectorExportSettings, VectorExporter};
    use runmat_plot::plots::surface::ColorMap;
    use runmat_plot::plots::{image::ImagePlot, Figure};

    #[test]
    fn test_svg_embeds_png_data_uri_for_imagesc() {
        // Simple 2x2 grayscale grid
        let x: Vec<f64> = vec![0.0, 1.0];
        let y: Vec<f64> = vec![0.0, 1.0];
        let z: Vec<Vec<f64>> = vec![vec![0.0, 0.5], vec![0.5, 1.0]];
        let mut fig = Figure::new();
        let img = ImagePlot::from_grayscale(x, y, z, ColorMap::Parula, None).unwrap();
        fig.add_image_plot(img);

        let exporter = VectorExporter::with_settings(VectorExportSettings {
            width: 160.0,
            height: 120.0,
            ..Default::default()
        });
        let svg = exporter.render_to_svg(&mut fig).unwrap();
        assert!(svg.contains("<image "));
        assert!(svg.contains("xlink:href=\"data:image/png;base64,"));
    }
}
#[test]
fn test_uniform_buffer_layout() {
    // Test that uniforms have the expected memory layout for GPU buffers
    let uniforms = Uniforms::new();

    // Check Pod trait (can be safely transmitted to GPU)
    let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
    assert_eq!(bytes.len(), std::mem::size_of::<Uniforms>());

    // Verify expected size - 2 Mat4 (64 bytes each) + 4x3 normal matrix (48 bytes)
    // Should be aligned to 16 bytes for GPU usage
    let expected_size = std::mem::size_of::<[[f32; 4]; 4]>() * 2 + // view_proj + model matrices
                       std::mem::size_of::<[[f32; 4]; 3]>(); // normal matrix (4x3 for alignment)
    assert_eq!(std::mem::size_of::<Uniforms>(), expected_size);
}

#[test]
fn test_uniform_matrix_updates() {
    let mut uniforms = Uniforms::new();

    // Test view-projection matrix update
    let view_proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
    uniforms.update_view_proj(view_proj);
    assert_eq!(uniforms.view_proj, view_proj.to_cols_array_2d());

    // Test model matrix update
    let model = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    uniforms.update_model(model);
    assert_eq!(uniforms.model, model.to_cols_array_2d());

    // Verify normal matrix is computed correctly (inverse transpose of upper 3x3)
    let normal_mat = model.inverse().transpose();
    let expected_normal = [
        [
            normal_mat.x_axis.x,
            normal_mat.x_axis.y,
            normal_mat.x_axis.z,
            0.0,
        ],
        [
            normal_mat.y_axis.x,
            normal_mat.y_axis.y,
            normal_mat.y_axis.z,
            0.0,
        ],
        [
            normal_mat.z_axis.x,
            normal_mat.z_axis.y,
            normal_mat.z_axis.z,
            0.0,
        ],
    ];
    assert_eq!(uniforms.normal_matrix, expected_normal);
}

#[test]
fn test_pipeline_type_completeness() {
    // Ensure all pipeline types are covered
    let types = vec![
        PipelineType::Points,
        PipelineType::Lines,
        PipelineType::Triangles,
        PipelineType::Scatter3,
    ];

    // All types should be Debug-printable
    for pipeline_type in types {
        let debug_str = format!("{pipeline_type:?}");
        assert!(!debug_str.is_empty());

        // Check that PartialEq works
        assert_eq!(pipeline_type, pipeline_type);
    }
}

#[test]
fn test_vertex_utils_line_creation() {
    let start = Vec3::new(0.0, 0.0, 0.0);
    let end = Vec3::new(1.0, 1.0, 0.0);
    let color = Vec4::new(1.0, 0.0, 0.0, 1.0);

    let vertices = vertex_utils::create_line(start, end, color);

    assert_eq!(vertices.len(), 2);
    assert_eq!(vertices[0].position, start.to_array());
    assert_eq!(vertices[1].position, end.to_array());
    assert_eq!(vertices[0].color, color.to_array());
    assert_eq!(vertices[1].color, color.to_array());
}

#[test]
fn test_vertex_utils_triangle_creation() {
    let p1 = Vec3::new(0.0, 0.0, 0.0);
    let p2 = Vec3::new(1.0, 0.0, 0.0);
    let p3 = Vec3::new(0.5, 1.0, 0.0);
    let color = Vec4::new(0.0, 1.0, 0.0, 1.0);

    let vertices = vertex_utils::create_triangle(p1, p2, p3, color);

    assert_eq!(vertices.len(), 3);
    assert_eq!(vertices[0].position, p1.to_array());
    assert_eq!(vertices[1].position, p2.to_array());
    assert_eq!(vertices[2].position, p3.to_array());

    // All vertices should have the same color
    for vertex in &vertices {
        assert_eq!(vertex.color, color.to_array());
    }
}

#[test]
fn test_vertex_utils_point_cloud() {
    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 0.5, 2.0),
    ];
    let colors = vec![
        Vec4::new(1.0, 0.0, 0.0, 1.0), // Red
        Vec4::new(0.0, 1.0, 0.0, 1.0), // Green
        Vec4::new(0.0, 0.0, 1.0, 1.0), // Blue
    ];

    let vertices = vertex_utils::create_point_cloud(&points, &colors);

    assert_eq!(vertices.len(), 3);
    for (i, vertex) in vertices.iter().enumerate() {
        assert_eq!(vertex.position, points[i].to_array());
        assert_eq!(vertex.color, colors[i].to_array());
    }
}

#[test]
fn test_vertex_utils_line_plot() {
    let x_data = vec![0.0, 1.0, 2.0, 3.0];
    let y_data = vec![0.0, 1.0, 0.0, -1.0];
    let color = Vec4::new(0.0, 0.5, 1.0, 1.0);

    let vertices = vertex_utils::create_line_plot(&x_data, &y_data, color);

    // Should create line segments between consecutive points
    let expected_segments = x_data.len() - 1;
    assert_eq!(vertices.len(), expected_segments * 2); // 2 vertices per segment

    // Check first line segment
    assert_eq!(vertices[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(vertices[1].position, [1.0, 1.0, 0.0]);

    // Check second line segment
    assert_eq!(vertices[2].position, [1.0, 1.0, 0.0]);
    assert_eq!(vertices[3].position, [2.0, 0.0, 0.0]);

    // All vertices should have the same color
    for vertex in &vertices {
        assert_eq!(vertex.color, color.to_array());
    }
}

#[test]
fn test_vertex_utils_scatter_plot() {
    let x_data = vec![0.0, 1.5, -0.5, 2.0];
    let y_data = vec![0.0, 2.0, -1.0, 1.0];
    let color = Vec4::new(1.0, 0.5, 0.0, 1.0);

    let vertices = vertex_utils::create_scatter_plot(&x_data, &y_data, color);

    assert_eq!(vertices.len(), x_data.len());

    for (i, vertex) in vertices.iter().enumerate() {
        assert_eq!(vertex.position, [x_data[i] as f32, y_data[i] as f32, 0.0]);
        assert_eq!(vertex.color, color.to_array());
    }
}

#[test]
fn test_vertex_utils_empty_data() {
    let empty_x: Vec<f64> = vec![];
    let empty_y: Vec<f64> = vec![];
    let color = Vec4::ONE;

    // Line plot with empty data should return empty vertices
    let line_vertices = vertex_utils::create_line_plot(&empty_x, &empty_y, color);
    assert_eq!(line_vertices.len(), 0);

    // Scatter plot with empty data should return empty vertices
    let scatter_vertices = vertex_utils::create_scatter_plot(&empty_x, &empty_y, color);
    assert_eq!(scatter_vertices.len(), 0);

    // Point cloud with empty data should return empty vertices
    let empty_points: Vec<Vec3> = vec![];
    let empty_colors: Vec<Vec4> = vec![];
    let cloud_vertices = vertex_utils::create_point_cloud(&empty_points, &empty_colors);
    assert_eq!(cloud_vertices.len(), 0);
}

#[test]
fn test_vertex_utils_single_point_data() {
    let x_data = vec![1.5];
    let y_data = vec![2.5];
    let color = Vec4::new(0.5, 0.5, 0.5, 1.0);

    // Line plot with single point should return empty (no segments possible)
    let line_vertices = vertex_utils::create_line_plot(&x_data, &y_data, color);
    assert_eq!(line_vertices.len(), 0);

    // Scatter plot with single point should return one vertex
    let scatter_vertices = vertex_utils::create_scatter_plot(&x_data, &y_data, color);
    assert_eq!(scatter_vertices.len(), 1);
    assert_eq!(scatter_vertices[0].position, [1.5, 2.5, 0.0]);
}

#[test]
fn test_vertex_large_dataset() {
    // Test with larger dataset to ensure performance is reasonable
    let n = 10000;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| x.sin()).collect();
    let color = Vec4::new(0.0, 0.5, 1.0, 1.0);

    let start = Instant::now();
    let vertices = vertex_utils::create_line_plot(&x_data, &y_data, color);
    let duration = start.elapsed();

    // Should complete quickly (under 1ms for this size)
    assert!(duration.as_millis() < 10);

    // Should have correct number of vertices
    assert_eq!(vertices.len(), (n - 1) * 2);

    // Spot check a few vertices
    assert_eq!(vertices[0].position[0], 0.0);
    assert_eq!(vertices[1].position[0], 0.001);
}

#[cfg(test)]
mod bar_tests {
    use runmat_plot::plots::{BarChart, Figure};

    #[test]
    fn test_bar_vector_bounds() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let values = vec![1.0, -2.0, 3.0];
        let mut bar = BarChart::new(labels, values).unwrap();
        let bounds = bar.bounds();
        assert!(bounds.min.x < 1.0);
        assert!(bounds.max.x > 3.0);
        assert_eq!(bounds.min.y, -2.0);
        assert_eq!(bounds.max.y, 3.0);
    }

    #[test]
    fn test_barh_horizontal_bounds() {
        let labels = vec!["A".to_string(), "B".to_string()];
        let values = vec![4.0, -1.0];
        let mut bar = BarChart::new(labels, values)
            .unwrap()
            .with_orientation(runmat_plot::plots::bar::Orientation::Horizontal);
        let bounds = bar.bounds();
        assert!(bounds.min.y < 1.0);
        assert!(bounds.max.y > 2.0);
        assert!(bounds.min.x <= -1.0);
        assert!(bounds.max.x >= 4.0);
    }

    #[test]
    fn test_grouped_bars_geometry_counts() {
        let labels = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let values1 = vec![1.0, 2.0, 3.0];
        let values2 = vec![2.0, 1.0, 0.0];
        let mut b1 = BarChart::new(labels.clone(), values1)
            .unwrap()
            .with_group(0, 2);
        let mut b2 = BarChart::new(labels, values2).unwrap().with_group(1, 2);
        let (v1, i1) = b1.generate_vertices();
        let (v2, i2) = b2.generate_vertices();
        assert_eq!(v1.len(), 12); // 3 bars * 4 vertices
        assert_eq!(i1.len(), 18); // 3 bars * 6 indices
        assert_eq!(v2.len(), 12);
        assert_eq!(i2.len(), 18);
    }

    #[test]
    fn test_stacked_bars_offsets() {
        let labels = vec!["1".to_string(), "2".to_string()];
        let values_a = vec![1.0, -2.0];
        let values_b = vec![3.0, -1.0];
        let mut bar_a = BarChart::new(labels.clone(), values_a.clone())
            .unwrap()
            .with_stack_offsets(vec![0.0, 0.0]);
        let mut bar_b = BarChart::new(labels, values_b.clone())
            .unwrap()
            .with_stack_offsets(vec![1.0, -2.0]);
        let (_va, ia) = bar_a.generate_vertices();
        let (_vb, ib) = bar_b.generate_vertices();
        assert_eq!(ia.len(), 12); // 2 bars
        assert_eq!(ib.len(), 12);
        let _ = bar_a.bounds();
        let bounds_b = bar_b.bounds();
        assert!(bounds_b.max.y >= (values_a[0] + values_b[0]) as f32);
        assert!(bounds_b.min.y <= (values_a[1] + values_b[1]) as f32);
    }

    #[test]
    fn test_categorical_labels_in_figure() {
        let labels = vec!["Cat".to_string(), "Dog".to_string(), "Bird".to_string()];
        let values = vec![5.0, 2.0, 7.0];
        let bar = BarChart::new(labels.clone(), values).unwrap();
        let mut fig = Figure::new();
        fig.add_bar_chart(bar);
        let cat = fig.categorical_axis_labels();
        assert!(cat.is_some());
        let (is_x, lbls) = cat.unwrap();
        assert!(is_x);
        assert_eq!(lbls, labels);
    }

    #[test]
    fn test_bar_skips_nan_inf() {
        let labels = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let values = vec![1.0, f64::NAN, f64::INFINITY, -2.0];
        let mut chart = BarChart::new(labels, values).unwrap();
        let (vertices, indices) = chart.generate_vertices();
        // Only finite values 1.0 and -2.0 should produce bars => 2 bars
        assert_eq!(vertices.len(), 8);
        assert_eq!(indices.len(), 12);
    }
}

#[cfg(test)]
mod new_plots_tests {
    #![allow(dead_code)]
    use runmat_plot::plots::*;

    #[test]
    fn test_imagesc_surface_flatten() {
        // Simulate imagesc via SurfacePlot with flatten_z=true and colormap mapping
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let z = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.25, 0.75, 0.1],
            vec![0.9, 0.2, 0.4],
        ];
        let mut surf = SurfacePlot::new(x, y, z)
            .unwrap()
            .with_flatten_z(true)
            .with_colormap(surface::ColorMap::Parula);
        let rd = surf.render_data();
        assert_eq!(rd.pipeline_type, runmat_plot::core::PipelineType::Triangles);
        // All Z positions should be 0
        let all_zero_z = rd.vertices.iter().all(|v| v.position[2] == 0.0);
        assert!(all_zero_z);
    }

    #[test]
    fn test_imshow_surface_color_grid() {
        // Simulate imshow via SurfacePlot with color_grid (RGB) and flatten_z=true
        let cols = 2usize;
        let rows = 2usize;
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        let mut grid: Vec<Vec<glam::Vec4>> = vec![vec![glam::Vec4::ZERO; cols]; rows];
        grid[0][0] = glam::Vec4::new(1.0, 0.0, 0.0, 1.0); // red
        grid[0][1] = glam::Vec4::new(0.0, 1.0, 0.0, 1.0); // green
        grid[1][0] = glam::Vec4::new(0.0, 0.0, 1.0, 1.0); // blue
        grid[1][1] = glam::Vec4::new(1.0, 1.0, 1.0, 1.0); // white
        let mut surf = SurfacePlot::new(x, y, vec![vec![0.0; cols]; rows])
            .unwrap()
            .with_flatten_z(true)
            .with_color_grid(grid);
        let rd = surf.render_data();
        assert_eq!(rd.pipeline_type, runmat_plot::core::PipelineType::Triangles);
        assert!(rd.indices.is_some());
        // Expect at least 4 vertices
        assert!(rd.vertices.len() >= 4);
    }
    fn test_quiver_vertices_and_bounds() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let u = vec![1.0, -1.0];
        let v = vec![0.5, 0.5];
        let mut q = QuiverPlot::new(x, y, u, v).unwrap().with_style(
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            1.5,
            1.0,
            0.1,
        );
        let vertices = q.generate_vertices();
        // Each vector at least contributes a shaft of 2 vertices; with heads more
        assert!(vertices.len() >= 4);
        let rd = q.render_data();
        assert_eq!(rd.pipeline_type, runmat_plot::core::PipelineType::Lines);
        let _bounds = q.bounds();
    }

    #[test]
    fn test_pie_indices_and_pipeline() {
        let values = vec![1.0, 2.0, 3.0];
        let mut p = PieChart::new(values, None).unwrap();
        let (_v, i) = p.generate_vertices();
        assert!(i.len() > 0);
        let rd = p.render_data();
        assert_eq!(rd.pipeline_type, runmat_plot::core::PipelineType::Triangles);
        assert!(rd.indices.as_ref().unwrap().len() > 0);
    }
    fn test_errorbar_geometry_with_caps() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 1.5];
        let el = vec![0.2, 0.1, 0.3];
        let eu = vec![0.3, 0.4, 0.2];
        let mut eb = ErrorBar::new(x, y, el, eu).unwrap().with_style(
            glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            1.5,
            0.2,
        );
        let vertices = eb.generate_vertices();
        // For 3 points with caps: 2 (line) + 4 (caps) per point = 6 per point => 18
        assert!(vertices.len() >= 6);
        let rd = eb.render_data();
        assert_eq!(rd.pipeline_type, runmat_plot::core::PipelineType::Lines);
    }

    #[test]
    fn test_stairs_geometry_counts() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 1.5, 2.5];
        let mut st = StairsPlot::new(x, y).unwrap();
        let vertices = st.generate_vertices();
        // For n=4, steps = 3; each step adds 4 vertices (2 horizontal, 2 vertical) => 12
        assert_eq!(vertices.len(), 12);
        let rd = st.render_data();
        assert_eq!(rd.pipeline_type, runmat_plot::core::PipelineType::Lines);
    }

    #[test]
    fn test_stem_geometry_counts() {
        let x = vec![0.0, 1.0];
        let y = vec![1.0, -0.5];
        let mut sm = StemPlot::new(x, y).unwrap();
        let vertices = sm.generate_vertices();
        // Each point: 2 vertices for stem + 4 for cross marker => 6; for 2 points => 12
        assert_eq!(vertices.len(), 12);
        let rd = sm.render_data();
        assert_eq!(rd.pipeline_type, runmat_plot::core::PipelineType::Lines);
    }

    #[test]
    fn test_area_triangles() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 0.5];
        let mut ar = AreaPlot::new(x, y)
            .unwrap()
            .with_style(glam::Vec4::new(0.0, 0.5, 1.0, 0.4), 0.0);
        let (v, i) = ar.generate_vertices();
        assert_eq!(v.len(), 6); // baseline+curve pairs
        assert_eq!(i.len(), 12); // 2 segments * 2 triangles * 3 indices
        let rd = ar.render_data();
        assert_eq!(rd.pipeline_type, runmat_plot::core::PipelineType::Triangles);
        assert!(rd.indices.as_ref().unwrap().len() > 0);
    }
}

#[cfg(test)]
mod image_export_tests {
    #![allow(unused_imports)]
    use runmat_plot::plots::surface::ColorMap;
    use runmat_plot::plots::{image::ImagePlot, Figure};
    use std::path::PathBuf;

    // Helper to write PNG via headless exporter
    async fn export_png(fig: &mut Figure, name: &str) -> Result<PathBuf, String> {
        let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        path.push(format!("{name}.png"));
        let exporter = runmat_plot::export::ImageExporter::new().await?;
        exporter.export_png(fig, &path).await?;
        Ok(path)
    }

    #[tokio::test]
    async fn test_headless_imagesc_grayscale_export() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let z = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.25, 0.75, 0.1],
            vec![0.9, 0.2, 0.4],
        ];
        let img = ImagePlot::from_grayscale(x, y, z, ColorMap::Parula, None).unwrap();
        let mut fig = Figure::new();
        fig.add_image_plot(img);
        let png_path = export_png(&mut fig, "imagesc_gray").await.unwrap();
        assert!(png_path.exists());
        let bytes = std::fs::read(&png_path).unwrap();
        assert!(bytes.len() > 1000); // basic sanity
    }

    #[tokio::test]
    async fn test_headless_imshow_rgb_export() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let mut grid: Vec<Vec<glam::Vec4>> = vec![vec![glam::Vec4::ZERO; 2]; 2];
        grid[0][0] = glam::Vec4::new(1.0, 0.0, 0.0, 1.0);
        grid[0][1] = glam::Vec4::new(0.0, 1.0, 0.0, 1.0);
        grid[1][0] = glam::Vec4::new(0.0, 0.0, 1.0, 1.0);
        grid[1][1] = glam::Vec4::new(1.0, 1.0, 1.0, 1.0);
        let img = ImagePlot::from_color_grid(x, y, grid).unwrap();
        let mut fig = Figure::new();
        fig.add_image_plot(img);
        let png_path = export_png(&mut fig, "imshow_rgb").await.unwrap();
        assert!(png_path.exists());
        let bytes = std::fs::read(&png_path).unwrap();
        assert!(bytes.len() > 1000);
    }
}

#[cfg(test)]
mod scatter_export_tests {
    use runmat_plot::plots::{Figure, ScatterPlot};
    use std::path::PathBuf;

    async fn export_png(fig: &mut Figure, name: &str) -> Result<PathBuf, String> {
        let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        path.push(format!("{name}.png"));
        let exporter = runmat_plot::export::ImageExporter::new().await?;
        exporter.export_png(fig, &path).await?;
        Ok(path)
    }

    #[tokio::test]
    async fn test_scatter_marker_size_headless() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 1.0, 2.0];
        let sp = ScatterPlot::new(x, y).unwrap().with_style(
            glam::Vec4::new(0.2, 0.6, 1.0, 1.0),
            12.0,
            runmat_plot::plots::scatter::MarkerStyle::Circle,
        );
        let mut fig = Figure::new().with_title("Scatter Size");
        fig.add_scatter_plot(sp);
        let path = export_png(&mut fig, "scatter_size").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_scatter_edge_color_headless() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        let mut sp = ScatterPlot::new(x, y).unwrap().with_style(
            glam::Vec4::new(1.0, 1.0, 0.0, 1.0),
            10.0,
            runmat_plot::plots::scatter::MarkerStyle::Circle,
        );
        sp.set_edge_color(glam::Vec4::new(0.0, 0.0, 0.0, 1.0));
        sp.set_edge_thickness(2.0);
        let mut fig = Figure::new().with_title("Scatter Edge");
        fig.add_scatter_plot(sp);
        let path = export_png(&mut fig, "scatter_edge").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_scatter_square_marker_headless() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.5];
        let sp = ScatterPlot::new(x, y).unwrap().with_style(
            glam::Vec4::new(0.9, 0.1, 0.1, 1.0),
            10.0,
            runmat_plot::plots::scatter::MarkerStyle::Square,
        );
        let mut fig = Figure::new().with_title("Scatter Square");
        fig.add_scatter_plot(sp);
        let path = export_png(&mut fig, "scatter_square").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_scatter_sizes_array_headless() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let mut sp = ScatterPlot::new(x, y).unwrap().with_style(
            glam::Vec4::new(0.2, 0.8, 0.2, 1.0),
            5.0,
            runmat_plot::plots::scatter::MarkerStyle::Circle,
        );
        sp.set_sizes(vec![4.0, 8.0, 12.0, 16.0, 20.0]);
        let mut fig = Figure::new().with_title("Scatter Sizes");
        fig.add_scatter_plot(sp);
        let path = export_png(&mut fig, "scatter_sizes").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_scatter_rgb_colors_headless() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.5];
        let mut sp = ScatterPlot::new(x, y).unwrap().with_style(
            glam::Vec4::new(0.5, 0.5, 0.5, 1.0),
            10.0,
            runmat_plot::plots::scatter::MarkerStyle::Square,
        );
        sp.set_colors(vec![
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
        ]);
        sp.set_filled(true);
        let mut fig = Figure::new().with_title("Scatter RGB");
        fig.add_scatter_plot(sp);
        let path = export_png(&mut fig, "scatter_rgb").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_scatter_cdata_scalar_headless() {
        use runmat_plot::plots::surface::ColorMap;
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.1, 0.2, 0.3, 0.4];
        let mut sp = ScatterPlot::new(x, y).unwrap().with_style(
            glam::Vec4::new(0.5, 0.5, 0.5, 1.0),
            10.0,
            runmat_plot::plots::scatter::MarkerStyle::Circle,
        );
        sp.set_color_values(vec![0.0, 0.33, 0.66, 1.0], None);
        sp = sp.with_colormap(ColorMap::Jet);
        let mut fig = Figure::new().with_title("Scatter CData");
        fig.add_scatter_plot(sp);
        let path = export_png(&mut fig, "scatter_cdata").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }
}

#[cfg(test)]
mod export_parity_more_tests {
    use runmat_plot::plots::{AreaPlot, BarChart, ErrorBar, Figure, QuiverPlot};
    use std::path::PathBuf;

    async fn export_png(fig: &mut Figure, name: &str) -> Result<PathBuf, String> {
        let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        path.push(format!("{name}.png"));
        let exporter = runmat_plot::export::ImageExporter::new().await?;
        exporter.export_png(fig, &path).await?;
        Ok(path)
    }

    #[tokio::test]
    async fn test_bar_export_basic() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let values = vec![10.0, 20.0, 15.0];
        let bar = BarChart::new(labels, values).unwrap().with_label("Bars");
        let mut fig = Figure::new().with_title("Bar Basic");
        fig.add_bar_chart(bar);
        let path = export_png(&mut fig, "bar_basic").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_barh_export_basic() {
        let labels = vec!["X".to_string(), "Y".to_string()];
        let values = vec![5.0, 8.0];
        let bar = BarChart::new(labels, values)
            .unwrap()
            .with_orientation(runmat_plot::plots::bar::Orientation::Horizontal)
            .with_label("BarH");
        let mut fig = Figure::new().with_title("BarH Basic");
        fig.add_bar_chart(bar);
        let path = export_png(&mut fig, "barh_basic").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_errorbar_export_basic() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 1.5, 2.2];
        let e = vec![0.1, 0.2, 0.15, 0.1];
        let eb = ErrorBar::new(x, y, e.clone(), e).unwrap().with_label("Err");
        let mut fig = Figure::new().with_title("ErrorBar");
        fig.add_errorbar(eb);
        let path = export_png(&mut fig, "errorbar_basic").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_area_export_stacked() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y1 = vec![1.0, 2.0, 1.0, 2.0];
        let y2 = vec![0.5, 0.5, 0.5, 0.5];
        // Build stacked by adding two area plots with increasing baseline via area semantics
        let acc = vec![0.0f64; x.len()];
        let y1_top: Vec<f64> = (0..x.len()).map(|i| acc[i] + y1[i]).collect();
        let p1 = AreaPlot::new(x.clone(), y1_top.clone())
            .unwrap()
            .with_style(glam::Vec4::new(0.0, 0.5, 1.0, 0.4), 0.0)
            .with_label("A1");
        let mut fig = Figure::new().with_title("Area Stacked");
        fig.add_area_plot(p1);
        let y2_top: Vec<f64> = (0..x.len()).map(|i| y1_top[i] + y2[i]).collect();
        let p2 = AreaPlot::new(x, y2_top)
            .unwrap()
            .with_style(glam::Vec4::new(0.0, 0.8, 0.2, 0.4), 0.0)
            .with_label("A2");
        fig.add_area_plot(p2);
        let path = export_png(&mut fig, "area_stacked").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_quiver_export_basic() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.0];
        let u = vec![1.0, 0.0, -1.0];
        let v = vec![0.0, 1.0, 0.0];
        let q = QuiverPlot::new(x, y, u, v)
            .unwrap()
            .with_style(glam::Vec4::new(0.0, 0.0, 0.0, 1.0), 1.0, 1.0, 0.1)
            .with_label("Q");
        let mut fig = Figure::new().with_title("Quiver");
        fig.add_quiver_plot(q);
        let path = export_png(&mut fig, "quiver_basic").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[tokio::test]
    async fn test_pie_export_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let p = runmat_plot::plots::PieChart::new(vals, None)
            .unwrap()
            .with_label("Pie");
        let mut fig = Figure::new().with_title("Pie");
        fig.add_pie_chart(p);
        let path = export_png(&mut fig, "pie_basic").await.unwrap();
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod gpu_stress {
    use glam::{Vec3, Vec4};
    use once_cell::sync::OnceCell;
    use pollster::FutureExt;
    use runmat_plot::{
        export::image::{ImageExportSettings, ImageExporter},
        install_shared_wgpu_context,
        plots::{scatter::MarkerStyle, Figure, LinePlot, Scatter3Plot, ScatterPlot, SurfacePlot},
        SharedWgpuContext,
    };
    use runmat_time::Instant;
    use std::sync::Arc;

    static CONTEXT_READY: OnceCell<bool> = OnceCell::new();

    struct RenderStats {
        frames: usize,
        avg_ms: f64,
        max_ms: f64,
    }

    fn ensure_shared_context() -> bool {
        *CONTEXT_READY.get_or_init(|| install_context())
    }

    fn install_context() -> bool {
        if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").is_err() {
            return false;
        }
        match build_shared_context() {
            Some(ctx) => {
                install_shared_wgpu_context(ctx);
                true
            }
            None => false,
        }
    }

    fn build_shared_context() -> Option<SharedWgpuContext> {
        let instance = Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor::default()));
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()?;
        let features = adapter.features();
        let limits = adapter.limits();
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("runmat-plot-gpu-stress-device"),
                    required_features: features,
                    required_limits: limits.clone(),
                },
                None,
            )
            .block_on()
            .ok()?;
        Some(SharedWgpuContext {
            instance,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
            adapter_info,
            limits,
            features,
        })
    }

    async fn render_headless(fig: &mut Figure, iterations: usize) -> Result<RenderStats, String> {
        if iterations == 0 {
            return Ok(RenderStats {
                frames: 0,
                avg_ms: 0.0,
                max_ms: 0.0,
            });
        }
        let exporter = ImageExporter::with_settings(ImageExportSettings {
            width: 1280,
            height: 720,
            ..Default::default()
        })
        .await?;
        let mut total = 0.0;
        let mut max_ms = 0.0;
        for _ in 0..iterations {
            let mut clone = fig.clone();
            let start = Instant::now();
            let bytes = exporter.render_png_bytes(&mut clone).await?;
            assert!(
                !bytes.is_empty(),
                "exporter returned an empty PNG buffer during GPU stress test"
            );
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            total += elapsed;
            if elapsed > max_ms {
                max_ms = elapsed;
            }
        }
        Ok(RenderStats {
            frames: iterations,
            avg_ms: total / iterations as f64,
            max_ms,
        })
    }

    fn build_multi_axes_figure() -> Figure {
        let mut fig = Figure::new();
        fig.set_subplot_grid(2, 2);
        fig.set_title("GPU stress – multi-axes");
        fig.set_labels(&["t".into(), "value".into()]);

        let sample_count = 50_000;
        let xs: Vec<f64> = (0..sample_count).map(|i| i as f64 * 0.0005).collect();
        let sin_y: Vec<f64> = xs.iter().map(|t| (t * 6.0).sin()).collect();
        let line = LinePlot::new(xs.clone(), sin_y)
            .expect("line data must be valid")
            .with_style(
                Vec4::new(0.2, 0.6, 0.9, 1.0),
                2.0,
                runmat_plot::plots::LineStyle::Solid,
            )
            .with_label("sin(6t)");
        let line_index = fig.add_line_plot(line);
        fig.assign_plot_to_axes(line_index, 0).unwrap();

        let scatter_y: Vec<f64> = xs.iter().map(|t| (t * 3.0).cos() * 0.5).collect();
        let scatter = ScatterPlot::new(xs.clone(), scatter_y)
            .expect("scatter data must match lengths")
            .with_style(Vec4::new(0.9, 0.3, 0.3, 0.9), 9.0, MarkerStyle::Triangle)
            .with_label("cos(3t)");
        let scatter_index = fig.add_scatter_plot(scatter);
        fig.assign_plot_to_axes(scatter_index, 1).unwrap();

        let bars_x: Vec<f64> = (0..128).map(|i| i as f64).collect();
        let bars_y: Vec<f64> = bars_x.iter().map(|x| (x / 32.0).sin() + 1.5).collect();
        let stairs =
            runmat_plot::plots::StairsPlot::new(bars_x, bars_y).expect("stairs data must be valid");
        let stairs_index = fig.add_stairs_plot(stairs);
        fig.assign_plot_to_axes(stairs_index, 2).unwrap();

        let surface = build_surface_plot(96);
        let surface_index = fig.add_surface_plot(surface);
        fig.assign_plot_to_axes(surface_index, 3).unwrap();

        fig
    }

    fn build_surface_plot(size: usize) -> SurfacePlot {
        let x: Vec<f64> = (0..size).map(|i| i as f64 / 12.0).collect();
        let y: Vec<f64> = (0..size).map(|i| i as f64 / 12.0).collect();
        let mut z = Vec::with_capacity(x.len());
        for (xi_idx, &xi) in x.iter().enumerate() {
            let mut row = Vec::with_capacity(y.len());
            for (yi_idx, &yi) in y.iter().enumerate() {
                let phase = (xi_idx as f64 * 0.05) + (yi_idx as f64 * 0.08);
                row.push((xi * 0.75).sin() * (yi * 0.5).cos() + phase.sin() * 0.1);
            }
            z.push(row);
        }
        SurfacePlot::new(x, y, z).expect("surface grid must be consistent")
    }

    fn build_scatter3_cloud(point_count: usize) -> Figure {
        let mut fig = Figure::new();
        fig.set_title("GPU stress – scatter3 cloud");
        let mut points = Vec::with_capacity(point_count);
        for i in 0..point_count {
            let t = i as f32 * 0.00025;
            let radius = 0.5 + (i as f32 % 1024.0) * 0.0001;
            points.push(Vec3::new(
                (t * 3.1).cos() * radius,
                (t * 2.7).sin() * radius,
                (t * 1.3).sin() * 0.75,
            ));
        }
        let scatter3 = Scatter3Plot::new(points)
            .expect("scatter3 data must be valid")
            .with_point_size(4.0)
            .with_label("helix cloud");
        fig.add_scatter3_plot(scatter3);
        fig
    }

    #[tokio::test]
    async fn headless_multi_axes_renderer_stress() {
        if !ensure_shared_context() {
            eprintln!(
                "skipping GPU stress test; set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to enable headless renders"
            );
            return;
        }
        let mut fig = build_multi_axes_figure();
        let stats = render_headless(&mut fig, 3)
            .await
            .expect("headless multi-axes render failed");
        assert_eq!(stats.frames, 3);
        eprintln!(
            "[gpu-stress] multi-axes figure avg={:.2}ms max={:.2}ms",
            stats.avg_ms, stats.max_ms
        );
    }

    #[tokio::test]
    async fn headless_scatter3_cloud_snapshot() {
        if !ensure_shared_context() {
            eprintln!(
                "skipping GPU stress test; set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to enable headless renders"
            );
            return;
        }
        let mut fig = build_scatter3_cloud(600_000);
        let stats = render_headless(&mut fig, 2)
            .await
            .expect("scatter3 headless render failed");
        assert_eq!(stats.frames, 2);
        eprintln!(
            "[gpu-stress] scatter3 cloud avg={:.2}ms max={:.2}ms",
            stats.avg_ms, stats.max_ms
        );
    }
}
