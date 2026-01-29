//! Filled contour plot implementation (triangles on the base plane).

use crate::core::{
    BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use glam::Vec4;

#[derive(Debug, Clone)]
pub struct ContourFillPlot {
    pub label: Option<String>,
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    gpu_vertices: Option<GpuVertexBuffer>,
    vertex_count: usize,
    bounds: BoundingBox,
}

impl ContourFillPlot {
    pub fn from_vertices(vertices: Vec<Vertex>, bounds: BoundingBox) -> Self {
        let vertex_count = vertices.len();
        Self {
            label: None,
            visible: true,
            vertices: Some(vertices),
            gpu_vertices: None,
            vertex_count,
            bounds,
        }
    }

    pub fn from_gpu_buffer(
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            label: None,
            visible: true,
            vertices: None,
            gpu_vertices: Some(buffer),
            vertex_count,
            bounds,
        }
    }

    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn bounds(&self) -> BoundingBox {
        self.bounds
    }

    pub fn render_data(&mut self) -> RenderData {
        let material = Material {
            albedo: Vec4::ONE,
            ..Default::default()
        };

        let (vertices, gpu_vertices) = if let Some(buffer) = &self.gpu_vertices {
            (Vec::new(), Some(buffer.clone()))
        } else {
            (self.vertices.clone().unwrap_or_default(), None)
        };

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count: self.vertex_count,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };

        RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices,
            indices: None,
            gpu_vertices,
            bounds: None,
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
