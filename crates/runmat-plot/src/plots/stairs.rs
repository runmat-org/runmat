//! Stairs (step) plot implementation

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
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
}

impl StairsPlot {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, String> {
        if x.len() != y.len() || x.is_empty() { return Err("stairs: X and Y must be same non-zero length".to_string()); }
        Ok(Self { x, y, color: Vec4::new(0.0, 0.5, 1.0, 1.0), line_width: 1.0, label: None, visible: true, vertices: None, bounds: None, dirty: true })
    }
    pub fn with_style(mut self, color: Vec4, line_width: f32) -> Self { self.color = color; self.line_width = line_width.max(0.5); self.dirty = true; self }
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self { self.label = Some(label.into()); self }
    pub fn set_visible(&mut self, v: bool) { self.visible = v; }
    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.dirty || self.vertices.is_none() {
            let mut verts = Vec::new();
            for i in 0..self.x.len().saturating_sub(1) {
                let x0 = self.x[i] as f32; let y0 = self.y[i] as f32;
                let x1 = self.x[i+1] as f32; let y1 = self.y[i+1] as f32;
                if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() { continue; }
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
        if self.dirty || self.bounds.is_none() {
            let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, 0.0);
            let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0);
            for (&x, &y) in self.x.iter().zip(self.y.iter()) { let (x, y) = (x as f32, y as f32); if !x.is_finite() || !y.is_finite() { continue; } min.x = min.x.min(x); max.x = max.x.max(x); min.y = min.y.min(y); max.y = max.y.max(y); }
            if !min.x.is_finite() { min = Vec3::ZERO; max = Vec3::ZERO; }
            self.bounds = Some(BoundingBox::new(min, max));
        }
        self.bounds.unwrap()
    }
    pub fn render_data(&mut self) -> RenderData {
        let vertices = self.generate_vertices().clone();
        let material = Material { albedo: self.color, ..Default::default() };
        let draw_call = DrawCall { vertex_offset: 0, vertex_count: vertices.len(), index_offset: None, index_count: None, instance_count: 1 };
        RenderData { pipeline_type: PipelineType::Lines, vertices, indices: None, material, draw_calls: vec![draw_call], image: None }
    }
    pub fn estimated_memory_usage(&self) -> usize { self.vertices.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<Vertex>()) }
}


