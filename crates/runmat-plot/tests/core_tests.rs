//! Core functionality tests for the runmat-plot library
//!
//! This module tests the fundamental building blocks of the plotting system:
//! - Vertex data structures and layouts
//! - Basic geometric primitives
//! - Core data types and conversions

use glam::{Vec3, Vec4};
use runmat_plot::core::{PipelineType, Vertex};

#[test]
fn test_vertex_creation() {
    let position = Vec3::new(1.0, 2.0, 3.0);
    let color = Vec4::new(1.0, 0.0, 0.0, 1.0); // Red

    let vertex = Vertex::new(position, color);

    assert_eq!(vertex.position, [1.0, 2.0, 3.0]);
    assert_eq!(vertex.color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(vertex.normal, [0.0, 0.0, 1.0]); // Default normal
    assert_eq!(vertex.tex_coords, [0.0, 0.0]); // Default UV
}

#[test]
fn test_vertex_memory_layout() {
    // Ensure vertex has the expected memory layout for GPU usage
    let vertex = Vertex::new(Vec3::ZERO, Vec4::ONE);

    // Check that bytemuck Pod trait is working (can convert to bytes)
    let bytes: &[u8] = bytemuck::bytes_of(&vertex);
    assert_eq!(bytes.len(), std::mem::size_of::<Vertex>());

    // Verify the expected size (3 + 4 + 3 + 2 = 12 f32s = 48 bytes)
    assert_eq!(std::mem::size_of::<Vertex>(), 48);
}

#[test]
fn test_vertex_buffer_layout() {
    let layout = Vertex::desc();

    assert_eq!(layout.array_stride, 48); // 12 * 4 bytes per f32
    assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
    assert_eq!(layout.attributes.len(), 4); // position, color, normal, tex_coords

    // Check position attribute
    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[0].shader_location, 0);
    assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Float32x3);

    // Check color attribute
    assert_eq!(layout.attributes[1].offset, 12); // After 3 f32s
    assert_eq!(layout.attributes[1].shader_location, 1);
    assert_eq!(layout.attributes[1].format, wgpu::VertexFormat::Float32x4);
}

#[test]
fn test_pipeline_types() {
    // Ensure all pipeline types are properly defined
    let point_pipeline = PipelineType::Points;
    let line_pipeline = PipelineType::Lines;
    let triangle_pipeline = PipelineType::Triangles;

    // Test Debug trait
    assert_eq!(format!("{point_pipeline:?}"), "Points");
    assert_eq!(format!("{line_pipeline:?}"), "Lines");
    assert_eq!(format!("{triangle_pipeline:?}"), "Triangles");
}

#[test]
fn test_vertex_creation_batch() {
    // Test creating multiple vertices for typical plotting scenarios
    let vertices = [
        Vertex::new(Vec3::new(0.0, 0.0, 0.0), Vec4::new(1.0, 0.0, 0.0, 1.0)), // Red
        Vertex::new(Vec3::new(1.0, 1.0, 0.0), Vec4::new(0.0, 1.0, 0.0, 1.0)), // Green
        Vertex::new(Vec3::new(2.0, 0.5, 0.0), Vec4::new(0.0, 0.0, 1.0, 1.0)), // Blue
    ];

    assert_eq!(vertices.len(), 3);

    // Verify all vertices are valid
    for vertex in vertices.iter() {
        assert_eq!(vertex.position[2], 0.0); // All in same Z plane
        assert_eq!(vertex.color[3], 1.0); // All fully opaque
        assert_eq!(vertex.normal, [0.0, 0.0, 1.0]); // All facing forward
    }
}

#[test]
fn test_vertex_with_custom_normal_and_uv() {
    let mut vertex = Vertex::new(Vec3::X, Vec4::Y);

    // Modify normal and texture coordinates
    vertex.normal = [1.0, 0.0, 0.0]; // Point along X axis
    vertex.tex_coords = [0.5, 0.5]; // Center of texture

    assert_eq!(vertex.normal, [1.0, 0.0, 0.0]);
    assert_eq!(vertex.tex_coords, [0.5, 0.5]);
}

#[test]
fn test_vertex_zero_and_one_constants() {
    // Test with common Vec3/Vec4 constants
    let zero_vertex = Vertex::new(Vec3::ZERO, Vec4::ZERO);
    let one_vertex = Vertex::new(Vec3::ONE, Vec4::ONE);

    assert_eq!(zero_vertex.position, [0.0, 0.0, 0.0]);
    assert_eq!(zero_vertex.color, [0.0, 0.0, 0.0, 0.0]);

    assert_eq!(one_vertex.position, [1.0, 1.0, 1.0]);
    assert_eq!(one_vertex.color, [1.0, 1.0, 1.0, 1.0]);
}
