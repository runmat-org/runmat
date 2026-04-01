//! Contour plot implementation (iso-lines on a surface or base plane).

use crate::core::{
    vertex_utils, BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData,
    Vertex,
};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct ContourPlot {
    pub base_z: f32,
    pub label: Option<String>,
    pub visible: bool,
    pub line_width: f32,
    vertices: Option<Vec<Vertex>>,
    gpu_vertices: Option<GpuVertexBuffer>,
    vertex_count: usize,
    bounds: Option<BoundingBox>,
}

impl ContourPlot {
    /// Create a contour plot from CPU vertices.
    pub fn from_vertices(vertices: Vec<Vertex>, base_z: f32, bounds: BoundingBox) -> Self {
        Self {
            base_z,
            label: None,
            visible: true,
            line_width: 1.0,
            vertex_count: vertices.len(),
            vertices: Some(vertices),
            gpu_vertices: None,
            bounds: Some(bounds),
        }
    }

    /// Create a contour plot backed by a GPU vertex buffer.
    pub fn from_gpu_buffer(
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        base_z: f32,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            base_z,
            label: None,
            visible: true,
            line_width: 1.0,
            vertex_count,
            vertices: None,
            gpu_vertices: Some(buffer),
            bounds: Some(bounds),
        }
    }

    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn with_line_width(mut self, line_width: f32) -> Self {
        self.line_width = line_width.max(0.5);
        self
    }

    pub fn vertices(&mut self) -> &Vec<Vertex> {
        if self.vertices.is_none() {
            self.vertices = Some(Vec::new());
        }
        self.vertices.as_ref().unwrap()
    }

    pub fn bounds(&self) -> BoundingBox {
        self.bounds.unwrap_or_default()
    }

    pub fn cpu_vertices(&self) -> Option<&[Vertex]> {
        self.vertices.as_deref()
    }

    pub fn render_data(&mut self) -> RenderData {
        let using_gpu = self.gpu_vertices.is_some();
        let bounds = self.bounds();
        let (vertices, vertex_count, gpu_vertices, pipeline_type) = if using_gpu {
            (
                Vec::new(),
                self.vertex_count,
                self.gpu_vertices.clone(),
                PipelineType::Lines,
            )
        } else {
            let verts = self.vertices().clone();
            if self.line_width > 1.0 {
                let mut thick = Vec::new();
                for segment in verts.chunks_exact(2) {
                    let x = [segment[0].position[0] as f64, segment[1].position[0] as f64];
                    let y = [segment[0].position[1] as f64, segment[1].position[1] as f64];
                    let color = Vec4::from_array(segment[0].color);
                    thick.extend(vertex_utils::create_thick_polyline(
                        &x,
                        &y,
                        color,
                        self.line_width,
                    ));
                }
                let count = thick.len();
                (thick, count, None, PipelineType::Triangles)
            } else {
                let count = verts.len();
                (verts, count, None, PipelineType::Lines)
            }
        };

        let material = Material {
            albedo: Vec4::ONE,
            roughness: self.line_width.max(0.0),
            ..Default::default()
        };

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };

        RenderData {
            pipeline_type,
            vertices,
            indices: None,
            gpu_vertices,
            bounds: Some(bounds),
            material,
            draw_calls: vec![draw_call],
            image: None,
        }
    }

    pub fn estimated_memory_usage(&self) -> usize {
        self.vertices
            .as_ref()
            .map(|v| v.len() * std::mem::size_of::<Vertex>())
            .unwrap_or(0)
    }
}

pub fn contour_bounds(x_min: f32, x_max: f32, y_min: f32, y_max: f32, base_z: f32) -> BoundingBox {
    BoundingBox::new(
        Vec3::new(x_min, y_min, base_z),
        Vec3::new(x_max, y_max, base_z),
    )
}
