//! Pie chart (2D) implementation using triangle fan

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};
use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct PieChart {
    pub values: Vec<f64>,
    pub colors: Vec<Vec4>,
    pub label: Option<String>,
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    indices: Option<Vec<u32>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

impl PieChart {
    pub fn new(values: Vec<f64>, colors: Option<Vec<Vec4>>) -> Result<Self, String> {
        if values.is_empty() {
            return Err("pie: empty values".to_string());
        }
        let mut v = Self {
            values,
            colors: colors.unwrap_or_default(),
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: None,
            dirty: true,
        };
        if v.colors.is_empty() {
            // simple color cycle
            let palette = [
                Vec4::new(0.0, 0.447, 0.741, 1.0),
                Vec4::new(0.85, 0.325, 0.098, 1.0),
                Vec4::new(0.929, 0.694, 0.125, 1.0),
                Vec4::new(0.494, 0.184, 0.556, 1.0),
                Vec4::new(0.466, 0.674, 0.188, 1.0),
                Vec4::new(0.301, 0.745, 0.933, 1.0),
                Vec4::new(0.635, 0.078, 0.184, 1.0),
            ];
            v.colors = (0..v.values.len())
                .map(|i| palette[i % palette.len()])
                .collect();
        }
        Ok(v)
    }
    pub fn with_label<S: Into<String>>(mut self, s: S) -> Self {
        self.label = Some(s.into());
        self
    }
    pub fn set_visible(&mut self, v: bool) {
        self.visible = v;
    }
    pub fn generate_vertices(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.dirty || self.vertices.is_none() {
            let mut verts = vec![Vertex::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec4::new(1.0, 1.0, 1.0, 1.0),
            )];
            let mut inds: Vec<u32> = Vec::new();
            let sum: f64 = self
                .values
                .iter()
                .filter(|v| v.is_finite() && **v >= 0.0)
                .sum();
            let mut angle = 0.0f32;
            let mut acc_index = 1u32;
            for (i, &val) in self.values.iter().enumerate() {
                if !val.is_finite() || val <= 0.0 || sum == 0.0 {
                    continue;
                }
                let frac = (val / sum) as f32;
                let theta = frac * 2.0 * PI;
                let steps = (theta * 20.0).ceil().max(1.0) as u32; // adaptive tessellation
                let color = self.colors[i % self.colors.len()];
                let start = angle;
                for s in 0..=steps {
                    let a = start + (theta * (s as f32 / steps as f32));
                    verts.push(Vertex::new(Vec3::new(a.cos(), a.sin(), 0.0), color));
                    if s > 0 {
                        inds.extend_from_slice(&[0, acc_index + s - 1, acc_index + s]);
                    }
                }
                acc_index += steps + 1;
                angle += theta;
            }
            self.vertices = Some(verts);
            self.indices = Some(inds);
            self.dirty = false;
        }
        (
            self.vertices.as_ref().unwrap(),
            self.indices.as_ref().unwrap(),
        )
    }
    pub fn bounds(&mut self) -> BoundingBox {
        if self.bounds.is_none() || self.dirty {
            self.bounds = Some(BoundingBox::new(
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ));
        }
        self.bounds.unwrap()
    }
    pub fn render_data(&mut self) -> RenderData {
        let (v, i) = self.generate_vertices();
        let vertices = v.clone();
        let indices = i.clone();
        let material = Material {
            albedo: Vec4::new(1.0, 1.0, 1.0, 1.0),
            ..Default::default()
        };
        let dc = DrawCall {
            vertex_offset: 0,
            vertex_count: vertices.len(),
            index_offset: Some(0),
            index_count: Some(indices.len()),
            instance_count: 1,
        };
        RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices,
            indices: Some(indices),
            material,
            draw_calls: vec![dc],
            gpu_vertices: None,
            image: None,
        }
    }
    pub fn estimated_memory_usage(&self) -> usize {
        self.vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
            + self
                .indices
                .as_ref()
                .map_or(0, |i| i.len() * std::mem::size_of::<u32>())
    }
}
