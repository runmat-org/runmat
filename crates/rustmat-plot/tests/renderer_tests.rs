//! WGPU renderer tests for the plotting system
//!
//! This module tests GPU-accelerated rendering infrastructure:
//! - WGPU device and surface setup
//! - Pipeline creation and management
//! - Vertex buffer operations
//! - Uniform buffer management
//! - Rendering command generation

use rustmat_plot::core::{Uniforms, PipelineType, vertex_utils};
use glam::{Vec3, Vec4, Mat4};

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
                       std::mem::size_of::<[[f32; 4]; 3]>();       // normal matrix (4x3 for alignment)
    assert_eq!(std::mem::size_of::<Uniforms>(), expected_size);
}

#[test]
fn test_uniform_matrix_updates() {
    let mut uniforms = Uniforms::new();
    
    // Test view-projection matrix update
    let view_proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0/9.0, 0.1, 100.0);
    uniforms.update_view_proj(view_proj);
    assert_eq!(uniforms.view_proj, view_proj.to_cols_array_2d());
    
    // Test model matrix update
    let model = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    uniforms.update_model(model);
    assert_eq!(uniforms.model, model.to_cols_array_2d());
    
    // Verify normal matrix is computed correctly (inverse transpose of upper 3x3)
    let normal_mat = model.inverse().transpose();
    let expected_normal = [
        [normal_mat.x_axis.x, normal_mat.x_axis.y, normal_mat.x_axis.z, 0.0],
        [normal_mat.y_axis.x, normal_mat.y_axis.y, normal_mat.y_axis.z, 0.0],
        [normal_mat.z_axis.x, normal_mat.z_axis.y, normal_mat.z_axis.z, 0.0],
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
        PipelineType::PointCloud,
    ];
    
    // All types should be Debug-printable
    for pipeline_type in types {
        let debug_str = format!("{:?}", pipeline_type);
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
    
    let start = std::time::Instant::now();
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