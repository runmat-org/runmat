//! The renderer's vertex, uniform and pipeline-kind types.

use super::*;

/// The draw command for a bound vertex buffer, without owning or copying GPU data.
///
/// Indexed commands use `Uint32` indices. Indirect commands are non-indexed and
/// take their vertex and instance ranges from the existing GPU argument buffer.
pub enum VertexDraw<'a> {
    Direct {
        vertex_count: u32,
    },
    Indexed {
        index_buffer: &'a wgpu::Buffer,
        index_count: u32,
    },
    Indirect(&'a GpuIndirectDraw),
}

/// Uniforms for the procedural 3D grid plane.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GridUniforms {
    pub major_step: f32,
    pub minor_step: f32,
    pub fade_start: f32,
    pub fade_end: f32,
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub target_pos: [f32; 3],
    pub _pad1: f32,
    pub major_color: [f32; 4],
    pub minor_color: [f32; 4],
}

impl Default for GridUniforms {
    fn default() -> Self {
        Self {
            major_step: 1.0,
            minor_step: 0.1,
            fade_start: 10.0,
            fade_end: 15.0,
            camera_pos: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            target_pos: [0.0, 0.0, 0.0],
            _pad1: 0.0,
            major_color: [0.90, 0.92, 0.96, 0.30],
            minor_color: [0.82, 0.84, 0.88, 0.18],
        }
    }
}

/// Vertex data for rendering points, lines, and triangles
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    pub fn new(position: Vec3, color: Vec4) -> Self {
        Self {
            position: position.to_array(),
            color: color.to_array(),
            normal: [0.0, 0.0, 1.0], // Default normal
            tex_coords: [0.0, 0.0],  // Default UV
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        let stride = std::mem::size_of::<Vertex>() as wgpu::BufferAddress;
        log::trace!(
            target: "runmat_plot",
            "vertex layout: size={}, stride={}",
            std::mem::size_of::<Vertex>(),
            stride
        );
        wgpu::VertexBufferLayout {
            array_stride: stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Texture coordinates
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 10]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Uniform buffer for camera and transformation matrices
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 3], // Use 4x3 for proper alignment instead of 3x3
    pub cad_view_rotation: [[f32; 4]; 3],
    pub cad_lighting: [u32; 4], // mode + padding; preserves 16-byte uniform alignment
}

/// Optimized uniform buffer for direct coordinate transformation rendering
/// Enables precise viewport-constrained data visualization
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DirectUniforms {
    pub data_min: [f32; 2],     // (x_min, y_min) in data space
    pub data_max: [f32; 2],     // (x_max, y_max) in data space
    pub viewport_min: [f32; 2], // NDC coordinates of viewport bottom-left
    pub viewport_max: [f32; 2], // NDC coordinates of viewport top-right
    pub viewport_px: [f32; 2],  // viewport size in pixels (width, height)
    pub log_flags: [u32; 2],    // (x_log, y_log), 1 when axis uses log10 mapping
    pub _pad: [u32; 2],
}

/// Style uniforms for direct point rendering (scatter markers)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointStyleUniforms {
    pub face_color: [f32; 4],
    pub edge_color: [f32; 4],
    pub edge_thickness_px: f32,
    pub marker_shape: u32,
    pub _pad: [f32; 2],
}

/// Screen-space uniforms for camera-projected marker billboards.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MarkerScreenUniforms {
    pub viewport_px: [f32; 2],
    /// Device pixels per CSS pixel, so screen-space widths given in CSS px keep their weight.
    pub pixels_per_point: f32,
    pub _pad: f32,
}

impl Default for Uniforms {
    fn default() -> Self {
        Self::new()
    }
}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            normal_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            cad_view_rotation: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            cad_lighting: [0; 4],
        }
    }

    pub fn update_view_proj(&mut self, view_proj: Mat4) {
        self.view_proj = view_proj.to_cols_array_2d();
    }

    pub fn update_model(&mut self, model: Mat4) {
        self.model = model.to_cols_array_2d();
        // Update normal matrix (upper 3x3 of inverse transpose) with proper alignment
        let normal_mat = model.inverse().transpose();
        self.normal_matrix = [
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
    }
}

impl DirectUniforms {
    pub fn new(
        data_min: [f32; 2],
        data_max: [f32; 2],
        viewport_min: [f32; 2],
        viewport_max: [f32; 2],
        viewport_px: [f32; 2],
        log_flags: [u32; 2],
    ) -> Self {
        Self {
            data_min,
            data_max,
            viewport_min,
            viewport_max,
            viewport_px,
            log_flags,
            _pad: [0, 0],
        }
    }
}

pub fn marker_shape_code(style: crate::plots::scatter::MarkerStyle) -> u32 {
    match style {
        crate::plots::scatter::MarkerStyle::Circle => 0,
        crate::plots::scatter::MarkerStyle::Square => 1,
        crate::plots::scatter::MarkerStyle::Triangle => 2,
        crate::plots::scatter::MarkerStyle::Diamond => 3,
        crate::plots::scatter::MarkerStyle::Plus => 4,
        crate::plots::scatter::MarkerStyle::Cross => 5,
        crate::plots::scatter::MarkerStyle::Star => 6,
        crate::plots::scatter::MarkerStyle::Hexagon => 7,
    }
}

/// Rendering pipeline types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineType {
    Points,
    Lines,
    LinesNoDepth,
    AnnotationStroke,
    /// Depth-tested screen-space strokes for geometry edges (see `geometry_scene::edge_stroke`).
    EdgeStroke,
    /// Edge strokes drawn over everything, for edge display modes that show hidden lines.
    EdgeStrokeNoDepth,
    Triangles,
    Scatter3,
    Textured,
}

impl PipelineType {
    /// Pipelines whose shaders read the screen uniforms (viewport size, pixels per point) at
    /// bind group 1.
    pub fn binds_screen_uniforms(self) -> bool {
        matches!(
            self,
            PipelineType::AnnotationStroke
                | PipelineType::EdgeStroke
                | PipelineType::EdgeStrokeNoDepth
        )
    }

    /// Pipelines composited after surfaces because they ignore depth.
    pub fn is_depth_free_overlay(self) -> bool {
        matches!(
            self,
            PipelineType::LinesNoDepth
                | PipelineType::AnnotationStroke
                | PipelineType::EdgeStrokeNoDepth
        )
    }
}
