//! Quiver plot (vector field) implementation

use crate::context::shared_wgpu_context;
use crate::core::{
    BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use crate::gpu::axis::OwnedAxisData;
use crate::gpu::{util::readback_scalar_buffer_f64, ScalarType};
use glam::{Vec3, Vec4};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct QuiverPlot {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,

    pub color: Vec4,
    pub line_width: f32,
    pub scale: f32,
    pub head_size: f32,

    pub label: Option<String>,
    pub visible: bool,

    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
    gpu_vertices: Option<GpuVertexBuffer>,
    gpu_vertex_count: Option<usize>,
    gpu_bounds: Option<BoundingBox>,
    gpu_source: Option<QuiverGpuSource>,
}

#[derive(Clone, Debug)]
pub struct QuiverGpuSource {
    pub x_data: OwnedAxisData,
    pub y_data: OwnedAxisData,
    pub u_buffer: Arc<wgpu::Buffer>,
    pub v_buffer: Arc<wgpu::Buffer>,
    pub count: usize,
    pub rows: usize,
    pub cols: usize,
    pub xy_mode: u32,
    pub scalar: ScalarType,
}

impl QuiverPlot {
    pub async fn export_scene_vector_data(
        &self,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        if !self.x.is_empty() || !self.y.is_empty() || !self.u.is_empty() || !self.v.is_empty() {
            return Ok((
                self.x.clone(),
                self.y.clone(),
                self.u.clone(),
                self.v.clone(),
            ));
        }

        if let Some(source) = &self.gpu_source {
            let context = shared_wgpu_context().ok_or_else(|| {
                "quiver plot has GPU source data but no shared WGPU context is installed"
                    .to_string()
            })?;
            let u = readback_scalar_buffer_f64(
                &context.device,
                &context.queue,
                &source.u_buffer,
                source.count,
                source.scalar,
            )
            .await?;
            let v = readback_scalar_buffer_f64(
                &context.device,
                &context.queue,
                &source.v_buffer,
                source.count,
                source.scalar,
            )
            .await?;
            let x_axis_len = if source.xy_mode == 0 {
                source.count
            } else {
                source.cols
            };
            let y_axis_len = if source.xy_mode == 0 {
                source.count
            } else {
                source.rows
            };
            let x_axis = source
                .x_data
                .export_f64(&context.device, &context.queue, x_axis_len, source.scalar)
                .await?;
            let y_axis = source
                .y_data
                .export_f64(&context.device, &context.queue, y_axis_len, source.scalar)
                .await?;
            let (x, y) = if source.xy_mode == 0 {
                (x_axis, y_axis)
            } else {
                let mut x = Vec::with_capacity(source.count);
                let mut y = Vec::with_capacity(source.count);
                for i in 0..source.count {
                    let col = i / source.rows;
                    let row = i % source.rows;
                    x.push(*x_axis.get(col).unwrap_or(&0.0));
                    y.push(*y_axis.get(row).unwrap_or(&0.0));
                }
                (x, y)
            };
            return Ok((x, y, u, v));
        }

        if self.gpu_vertices.is_some() {
            return Err(
                "quiver plot has GPU render vertices but no exportable source data".to_string(),
            );
        }

        Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()))
    }

    pub fn new(x: Vec<f64>, y: Vec<f64>, u: Vec<f64>, v: Vec<f64>) -> Result<Self, String> {
        let n = x.len();
        if n == 0 || y.len() != n || u.len() != n || v.len() != n {
            return Err("quiver: X,Y,U,V must have same non-zero length".to_string());
        }
        Ok(Self {
            x,
            y,
            u,
            v,
            color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            line_width: 1.0,
            scale: 1.0,
            head_size: 0.1,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_bounds: None,
            gpu_source: None,
        })
    }
    pub fn from_gpu_buffer(
        color: Vec4,
        line_width: f32,
        scale: f32,
        head_size: f32,
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            color,
            line_width,
            scale,
            head_size,
            label: None,
            visible: true,
            vertices: None,
            bounds: Some(bounds),
            dirty: false,
            gpu_vertices: Some(buffer),
            gpu_vertex_count: Some(vertex_count),
            gpu_bounds: Some(bounds),
            gpu_source: None,
        }
    }
    pub fn with_gpu_source(mut self, source: QuiverGpuSource) -> Self {
        self.gpu_source = Some(source);
        self
    }
    pub fn with_style(mut self, color: Vec4, line_width: f32, scale: f32, head_size: f32) -> Self {
        self.color = color;
        self.line_width = line_width.max(0.5);
        self.scale = scale.max(0.0);
        self.head_size = head_size.max(0.0);
        self.dirty = true;
        self
    }
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }
    pub fn set_visible(&mut self, v: bool) {
        self.visible = v;
    }

    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.dirty || self.vertices.is_none() {
            let mut verts = Vec::new();
            for i in 0..self.x.len() {
                let (x, y, u, v) = (
                    self.x[i] as f32,
                    self.y[i] as f32,
                    self.u[i] as f32,
                    self.v[i] as f32,
                );
                if !x.is_finite() || !y.is_finite() || !u.is_finite() || !v.is_finite() {
                    continue;
                }
                let dx = u * self.scale;
                let dy = v * self.scale;
                // Main shaft
                verts.push(Vertex::new(Vec3::new(x, y, 0.0), self.color));
                verts.push(Vertex::new(Vec3::new(x + dx, y + dy, 0.0), self.color));
                // Arrowhead as two short lines forming a V
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.0 && self.head_size > 0.0 {
                    let hx = dx / len;
                    let hy = dy / len;
                    // Perpendicular
                    let px = -hy;
                    let py = hx;
                    let h = self.head_size.min(len * 0.5);
                    let tipx = x + dx;
                    let tipy = y + dy;
                    let leftx = tipx - h * hx + 0.5 * h * px;
                    let lefty = tipy - h * hy + 0.5 * h * py;
                    let rightx = tipx - h * hx - 0.5 * h * px;
                    let righty = tipy - h * hy - 0.5 * h * py;
                    verts.push(Vertex::new(Vec3::new(tipx, tipy, 0.0), self.color));
                    verts.push(Vertex::new(Vec3::new(leftx, lefty, 0.0), self.color));
                    verts.push(Vertex::new(Vec3::new(tipx, tipy, 0.0), self.color));
                    verts.push(Vertex::new(Vec3::new(rightx, righty, 0.0), self.color));
                }
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
            for i in 0..self.x.len() {
                let x = self.x[i] as f32;
                let y = self.y[i] as f32;
                let dx = (self.u[i] as f32) * self.scale;
                let dy = (self.v[i] as f32) * self.scale;
                if !x.is_finite() || !y.is_finite() || !dx.is_finite() || !dy.is_finite() {
                    continue;
                }
                min.x = min.x.min(x.min(x + dx));
                max.x = max.x.max(x.max(x + dx));
                min.y = min.y.min(y.min(y + dy));
                max.y = max.y.max(y.max(y + dy));
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
        let bounds = self.bounds();
        let vertices = if using_gpu {
            Vec::new()
        } else {
            self.generate_vertices().clone()
        };
        let material = Material {
            albedo: self.color,
            ..Default::default()
        };
        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count: self.gpu_vertex_count.unwrap_or(vertices.len()),
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };
        RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices: self.gpu_vertices.clone(),
            bounds: Some(bounds),
            material,
            draw_calls: vec![draw_call],
            image: None,
        }
    }

    pub fn estimated_memory_usage(&self) -> usize {
        self.vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
    }
}
