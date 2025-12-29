//! Contour plot implementation (iso-lines on a surface or base plane).

use crate::core::{
    BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct ContourPlot {
    pub base_z: f32,
    pub label: Option<String>,
    pub visible: bool,
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

    pub fn vertices(&mut self) -> &Vec<Vertex> {
        if self.vertices.is_none() {
            self.vertices = Some(Vec::new());
        }
        self.vertices.as_ref().unwrap()
    }

    pub fn bounds(&self) -> BoundingBox {
        self.bounds.unwrap_or_default()
    }

    pub fn render_data(&mut self) -> RenderData {
        let using_gpu = self.gpu_vertices.is_some();
        let (vertices, vertex_count, gpu_vertices) = if using_gpu {
            (Vec::new(), self.vertex_count, self.gpu_vertices.clone())
        } else {
            let verts = self.vertices().clone();
            let count = verts.len();
            (verts, count, None)
        };

        let material = Material {
            albedo: Vec4::ONE,
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
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices,
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
