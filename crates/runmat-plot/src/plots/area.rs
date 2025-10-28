//! Area plot implementation (filled area under curve)

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct AreaPlot {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub baseline: f64,
    pub color: Vec4,
    pub label: Option<String>,
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    indices: Option<Vec<u32>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

impl AreaPlot {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, String> {
        if x.len() != y.len() || x.is_empty() { return Err("area: X and Y must be same non-zero length".to_string()); }
        Ok(Self { x, y, baseline: 0.0, color: Vec4::new(0.0, 0.5, 1.0, 0.4), label: None, visible: true, vertices: None, indices: None, bounds: None, dirty: true })
    }
    pub fn with_style(mut self, color: Vec4, baseline: f64) -> Self { self.color = color; self.baseline = baseline; self.dirty = true; self }
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self { self.label = Some(label.into()); self }
    pub fn set_visible(&mut self, v: bool) { self.visible = v; }
    pub fn generate_vertices(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.dirty || self.vertices.is_none() {
            let mut verts = Vec::new();
            let mut inds = Vec::new();
            // Build a triangle strip-like mesh: baseline to curve segments
            for i in 0..self.x.len() {
                let xi = self.x[i] as f32; let yi = self.y[i] as f32; let b = self.baseline as f32;
                if !xi.is_finite() || !yi.is_finite() { continue; }
                verts.push(Vertex::new(Vec3::new(xi, b, 0.0), self.color));
                verts.push(Vertex::new(Vec3::new(xi, yi, 0.0), self.color));
            }
            // Triangles between successive pairs
            for i in 0..(verts.len()/2 - 1) {
                let base = (i * 2) as u32;
                inds.extend_from_slice(&[base, base+1, base+3, base, base+3, base+2]);
            }
            self.vertices = Some(verts);
            self.indices = Some(inds);
            self.dirty = false;
        }
        (self.vertices.as_ref().unwrap(), self.indices.as_ref().unwrap())
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
        let (v, i) = self.generate_vertices();
        let vertices = v.clone(); let indices = i.clone();
        let material = Material { albedo: self.color, ..Default::default() };
        let draw_call = DrawCall { vertex_offset: 0, vertex_count: vertices.len(), index_offset: Some(0), index_count: Some(indices.len()), instance_count: 1 };
        RenderData { pipeline_type: PipelineType::Triangles, vertices, indices: Some(indices), material, draw_calls: vec![draw_call], image: None }
    }
    pub fn estimated_memory_usage(&self) -> usize {
        self.vertices.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<Vertex>()) + self.indices.as_ref().map_or(0, |i| i.len() * std::mem::size_of::<u32>())
    }
}


