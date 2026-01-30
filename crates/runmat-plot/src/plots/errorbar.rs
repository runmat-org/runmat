//! Error bar plot implementation
//!
//! Draws vertical error bars with optional horizontal caps.

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct ErrorBar {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub err_low: Vec<f64>,
    pub err_high: Vec<f64>,

    pub color: Vec4,
    pub line_width: f32,
    pub cap_width: f32,

    pub label: Option<String>,
    pub visible: bool,

    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

impl ErrorBar {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        err_low: Vec<f64>,
        err_high: Vec<f64>,
    ) -> Result<Self, String> {
        let n = x.len();
        if n == 0 || y.len() != n || err_low.len() != n || err_high.len() != n {
            return Err("errorbar: input vectors must have equal non-zero length".to_string());
        }
        Ok(Self {
            x,
            y,
            err_low,
            err_high,
            color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            line_width: 1.0,
            cap_width: 0.02,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
        })
    }

    pub fn with_style(mut self, color: Vec4, line_width: f32, cap_width: f32) -> Self {
        self.color = color;
        self.line_width = line_width.max(0.5);
        self.cap_width = cap_width.max(0.0);
        self.dirty = true;
        self
    }

    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.dirty || self.vertices.is_none() {
            let (min_x, max_x) = self
                .x
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            let cap = ((max_x - min_x).abs() as f32 * self.cap_width).max(0.0);
            let mut verts = Vec::new();
            for i in 0..self.x.len() {
                let xi = self.x[i] as f32;
                let yi = self.y[i] as f32;
                let y0 = (self.y[i] - self.err_low[i]) as f32;
                let y1 = (self.y[i] + self.err_high[i]) as f32;
                if !xi.is_finite() || !yi.is_finite() || !y0.is_finite() || !y1.is_finite() {
                    continue;
                }
                // Vertical line
                verts.push(Vertex::new(Vec3::new(xi, y0, 0.0), self.color));
                verts.push(Vertex::new(Vec3::new(xi, y1, 0.0), self.color));
                if cap > 0.0 {
                    // Bottom cap
                    verts.push(Vertex::new(Vec3::new(xi - cap * 0.5, y0, 0.0), self.color));
                    verts.push(Vertex::new(Vec3::new(xi + cap * 0.5, y0, 0.0), self.color));
                    // Top cap
                    verts.push(Vertex::new(Vec3::new(xi - cap * 0.5, y1, 0.0), self.color));
                    verts.push(Vertex::new(Vec3::new(xi + cap * 0.5, y1, 0.0), self.color));
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
                let xi = self.x[i] as f32;
                let y0 = (self.y[i] - self.err_low[i]) as f32;
                let y1 = (self.y[i] + self.err_high[i]) as f32;
                if !xi.is_finite() || !y0.is_finite() || !y1.is_finite() {
                    continue;
                }
                min.x = min.x.min(xi);
                max.x = max.x.max(xi);
                min.y = min.y.min(y0.min(y1));
                max.y = max.y.max(y0.max(y1));
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
