//! Stairs (step) plot implementation

use crate::core::{
    AlphaMode, BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use crate::plots::line::LineMarkerAppearance;
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct StairsPlot {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub color: Vec4,
    pub line_width: f32,
    pub label: Option<String>,
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
    gpu_vertices: Option<GpuVertexBuffer>,
    gpu_vertex_count: Option<usize>,
    gpu_bounds: Option<BoundingBox>,
    marker: Option<LineMarkerAppearance>,
    marker_vertices: Option<Vec<Vertex>>,
    marker_gpu_vertices: Option<GpuVertexBuffer>,
    marker_dirty: bool,
}

impl StairsPlot {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, String> {
        if x.len() != y.len() || x.is_empty() {
            return Err("stairs: X and Y must be same non-zero length".to_string());
        }
        Ok(Self {
            x,
            y,
            color: Vec4::new(0.0, 0.5, 1.0, 1.0),
            line_width: 1.0,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_bounds: None,
            marker: None,
            marker_vertices: None,
            marker_gpu_vertices: None,
            marker_dirty: true,
        })
    }

    /// Build a stairs plot backed directly by a GPU vertex buffer.
    pub fn from_gpu_buffer(
        color: Vec4,
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            color,
            line_width: 1.0,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: false,
            gpu_vertices: Some(buffer),
            gpu_vertex_count: Some(vertex_count),
            gpu_bounds: Some(bounds),
            marker: None,
            marker_vertices: None,
            marker_gpu_vertices: None,
            marker_dirty: true,
        }
    }

    pub fn with_style(mut self, color: Vec4, line_width: f32) -> Self {
        self.color = color;
        self.line_width = line_width.max(0.5);
        self.dirty = true;
        self.marker_dirty = true;
        self.drop_gpu();
        self
    }
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }
    pub fn set_visible(&mut self, v: bool) {
        self.visible = v;
    }

    pub fn set_marker(&mut self, marker: Option<LineMarkerAppearance>) {
        self.marker = marker;
        self.marker_dirty = true;
        if self.marker.is_none() {
            self.marker_vertices = None;
            self.marker_gpu_vertices = None;
        }
    }

    pub fn set_marker_gpu_vertices(&mut self, buffer: Option<GpuVertexBuffer>) {
        let has_gpu = buffer.is_some();
        self.marker_gpu_vertices = buffer;
        if has_gpu {
            self.marker_vertices = None;
        }
    }

    fn drop_gpu(&mut self) {
        self.gpu_vertices = None;
        self.gpu_vertex_count = None;
        self.gpu_bounds = None;
        self.marker_gpu_vertices = None;
    }
    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.gpu_vertices.is_some() {
            if self.vertices.is_none() {
                self.vertices = Some(Vec::new());
            }
            return self.vertices.as_ref().unwrap();
        }
        if self.dirty || self.vertices.is_none() {
            let mut verts = Vec::new();
            for i in 0..self.x.len().saturating_sub(1) {
                let x0 = self.x[i] as f32;
                let y0 = self.y[i] as f32;
                let x1 = self.x[i + 1] as f32;
                let y1 = self.y[i + 1] as f32;
                if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
                    continue;
                }
                // Horizontal segment
                verts.push(Vertex::new(Vec3::new(x0, y0, 0.0), self.color));
                verts.push(Vertex::new(Vec3::new(x1, y0, 0.0), self.color));
                // Vertical jump
                verts.push(Vertex::new(Vec3::new(x1, y0, 0.0), self.color));
                verts.push(Vertex::new(Vec3::new(x1, y1, 0.0), self.color));
            }
            self.vertices = Some(verts);
            self.dirty = false;
        }
        self.vertices.as_ref().unwrap()
    }
    pub fn bounds(&mut self) -> BoundingBox {
        if let Some(bounds) = self.gpu_bounds {
            return bounds;
        }
        if self.dirty || self.bounds.is_none() {
            let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, 0.0);
            let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0);
            for (&x, &y) in self.x.iter().zip(self.y.iter()) {
                let (x, y) = (x as f32, y as f32);
                if !x.is_finite() || !y.is_finite() {
                    continue;
                }
                min.x = min.x.min(x);
                max.x = max.x.max(x);
                min.y = min.y.min(y);
                max.y = max.y.max(y);
            }
            if !min.x.is_finite() {
                min = Vec3::ZERO;
                max = Vec3::ZERO;
            }
            self.bounds = Some(BoundingBox::new(min, max));
        }
        self.bounds.unwrap()
    }
    pub fn render_data(&mut self) -> RenderData {
        let using_gpu = self.gpu_vertices.is_some();
        let (vertices, vertex_count, gpu_vertices) = if using_gpu {
            (
                Vec::new(),
                self.gpu_vertex_count.unwrap_or(0),
                self.gpu_vertices.clone(),
            )
        } else {
            let verts = self.generate_vertices().clone();
            let count = verts.len();
            (verts, count, None)
        };
        let material = Material {
            albedo: self.color,
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

    pub fn marker_render_data(&mut self) -> Option<RenderData> {
        let marker = self.marker.clone()?;
        if let Some(gpu_vertices) = self.marker_gpu_vertices.clone() {
            let vertex_count = gpu_vertices.vertex_count;
            if vertex_count == 0 {
                return None;
            }
            let draw_call = DrawCall {
                vertex_offset: 0,
                vertex_count,
                index_offset: None,
                index_count: None,
                instance_count: 1,
            };
            let material = Self::marker_material(&marker);
            return Some(RenderData {
                pipeline_type: PipelineType::Points,
                vertices: Vec::new(),
                indices: None,
                gpu_vertices: Some(gpu_vertices),
                material,
                draw_calls: vec![draw_call],
                image: None,
            });
        }

        let vertices = self.marker_vertices_slice(&marker)?;
        if vertices.is_empty() {
            return None;
        }
        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count: vertices.len(),
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };
        let material = Self::marker_material(&marker);
        Some(RenderData {
            pipeline_type: PipelineType::Points,
            vertices: vertices.to_vec(),
            indices: None,
            gpu_vertices: None,
            material,
            draw_calls: vec![draw_call],
            image: None,
        })
    }

    fn marker_material(marker: &LineMarkerAppearance) -> Material {
        let mut material = Material {
            albedo: marker.face_color,
            ..Default::default()
        };
        if !marker.filled {
            material.albedo.w = 0.0;
        }
        material.emissive = marker.edge_color;
        material.roughness = 1.0;
        material.metallic = 0.0;
        material.alpha_mode = AlphaMode::Blend;
        material
    }

    fn marker_vertices_slice(&mut self, marker: &LineMarkerAppearance) -> Option<&[Vertex]> {
        if self.x.len() != self.y.len() || self.x.is_empty() {
            return None;
        }
        if self.marker_vertices.is_none() || self.marker_dirty {
            let mut verts = Vec::with_capacity(self.x.len());
            for (&x, &y) in self.x.iter().zip(self.y.iter()) {
                let mut vertex = Vertex::new(Vec3::new(x as f32, y as f32, 0.0), marker.face_color);
                vertex.normal[2] = marker.size.max(1.0);
                verts.push(vertex);
            }
            self.marker_vertices = Some(verts);
            self.marker_dirty = false;
        }
        self.marker_vertices.as_deref()
    }
    pub fn estimated_memory_usage(&self) -> usize {
        self.vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
    }
}
