//! Stem plot implementation (lines from baseline to points)

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct StemPlot {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub baseline: f64,
    pub color: Vec4,
    pub marker_color: Vec4,
    pub label: Option<String>,
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

impl StemPlot {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, String> {
        if x.len() != y.len() || x.is_empty() { return Err("stem: X and Y must be same non-zero length".to_string()); }
        Ok(Self { x, y, baseline: 0.0, color: Vec4::new(0.0, 0.0, 0.0, 1.0), marker_color: Vec4::new(0.0, 0.5, 1.0, 1.0), label: None, visible: true, vertices: None, bounds: None, dirty: true })
    }
    pub fn with_style(mut self, color: Vec4, marker_color: Vec4, baseline: f64) -> Self { self.color = color; self.marker_color = marker_color; self.baseline = baseline; self.dirty = true; self }
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self { self.label = Some(label.into()); self }
    pub fn set_visible(&mut self, v: bool) { self.visible = v; }
    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.dirty || self.vertices.is_none() {
            let mut verts = Vec::new();
            for i in 0..self.x.len() {
                let x = self.x[i] as f32; let y = self.y[i] as f32; let b = self.baseline as f32;
                if !x.is_finite() || !y.is_finite() { continue; }
                // Stem line
                verts.push(Vertex::new(Vec3::new(x, b, 0.0), self.color));
                verts.push(Vertex::new(Vec3::new(x, y, 0.0), self.color));
                // Marker as short cross
                let s = 0.01f32.max(0.01);
                verts.push(Vertex::new(Vec3::new(x - s, y, 0.0), self.marker_color));
                verts.push(Vertex::new(Vec3::new(x + s, y, 0.0), self.marker_color));
                verts.push(Vertex::new(Vec3::new(x, y - s, 0.0), self.marker_color));
                verts.push(Vertex::new(Vec3::new(x, y + s, 0.0), self.marker_color));
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
            for (&x, &y) in self.x.iter().zip(self.y.iter()) { let (x, y) = (x as f32, y as f32); if !x.is_finite() || !y.is_finite() { continue; } min.x = min.x.min(x); max.x = max.x.max(x); min.y = min.y.min(y.min(self.baseline as f32)); max.y = max.y.max(y.max(self.baseline as f32)); }
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


