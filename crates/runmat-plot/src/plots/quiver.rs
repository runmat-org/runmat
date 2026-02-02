//! Quiver plot (vector field) implementation

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};

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
}

impl QuiverPlot {
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
        })
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
        let vertices = self.generate_vertices().clone();
        let material = Material {
            albedo: self.color,
            ..Default::default()
        };
        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count: vertices.len(),
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };
        RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: None,
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
