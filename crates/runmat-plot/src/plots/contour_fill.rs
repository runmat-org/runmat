//! Filled contour plot implementation (triangles on the base plane).

use crate::context::shared_wgpu_context;
use crate::core::{
    BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use crate::gpu::util::copy_readback_bytes;
use bytemuck::cast_slice;
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
    pub async fn export_scene_vertices(&self) -> Result<Vec<Vertex>, String> {
        if let Some(vertices) = &self.vertices {
            return Ok(vertices.clone());
        }

        if let Some(gpu_vertices) = &self.gpu_vertices {
            let context = shared_wgpu_context().ok_or_else(|| {
                "filled contour plot has GPU vertices but no shared WGPU context is installed"
                    .to_string()
            })?;
            let generated_count = if let Some(indirect) = &gpu_vertices.indirect {
                let bytes = copy_readback_bytes(
                    &context.device,
                    &context.queue,
                    &indirect.args,
                    std::mem::size_of::<u32>(),
                )
                .await?;
                u32::from_le_bytes(
                    bytes
                        .get(0..4)
                        .ok_or_else(|| {
                            "filled contour indirect readback returned too few bytes".to_string()
                        })?
                        .try_into()
                        .map_err(|_| "filled contour indirect readback failed".to_string())?,
                ) as usize
            } else {
                self.vertex_count
            };
            let vertex_count = generated_count.min(gpu_vertices.vertex_count);
            let byte_len = vertex_count * std::mem::size_of::<Vertex>();
            let bytes = copy_readback_bytes(
                &context.device,
                &context.queue,
                &gpu_vertices.buffer,
                byte_len,
            )
            .await?;
            let vertices: &[Vertex] = cast_slice(&bytes);
            return Ok(vertices.to_vec());
        }

        Ok(Vec::new())
    }

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

    pub fn cpu_vertices(&self) -> Option<&[Vertex]> {
        self.vertices.as_deref()
    }

    pub fn render_data(&mut self) -> RenderData {
        let bounds = self.bounds();
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
