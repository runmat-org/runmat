//! Pie chart (2D) implementation using triangle fan

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};
use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct PieSliceMeta {
    pub label: String,
    pub color: Vec4,
    pub mid_angle: f32,
    pub offset: Vec3,
    pub fraction: f32,
}

#[derive(Debug, Clone)]
pub struct PieChart {
    pub values: Vec<f64>,
    pub colors: Vec<Vec4>,
    pub label: Option<String>,
    pub slice_labels: Vec<String>,
    pub label_format: Option<String>,
    pub explode: Vec<bool>,
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    indices: Option<Vec<u32>>,
    bounds: Option<BoundingBox>,
    slices: Option<Vec<PieSliceMeta>>,
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
            slice_labels: Vec::new(),
            label_format: None,
            explode: Vec::new(),
            visible: true,
            vertices: None,
            indices: None,
            bounds: None,
            slices: None,
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
    pub fn with_slice_labels(mut self, labels: Vec<String>) -> Self {
        self.slice_labels = labels;
        self.dirty = true;
        self
    }
    pub fn set_slice_labels(&mut self, labels: Vec<String>) {
        self.slice_labels = labels;
        self.dirty = true;
    }
    pub fn with_label_format<S: Into<String>>(mut self, format: S) -> Self {
        self.label_format = Some(format.into());
        self.dirty = true;
        self
    }
    pub fn with_explode(mut self, explode: Vec<bool>) -> Self {
        self.explode = explode;
        self.dirty = true;
        self
    }
    pub fn set_visible(&mut self, v: bool) {
        self.visible = v;
    }
    pub fn slice_labels(&self) -> Vec<String> {
        self.slice_meta()
            .into_iter()
            .map(|slice| slice.label)
            .collect()
    }
    pub fn slice_meta(&self) -> Vec<PieSliceMeta> {
        self.slices
            .clone()
            .unwrap_or_else(|| self.compute_slice_meta())
    }
    fn compute_slice_meta(&self) -> Vec<PieSliceMeta> {
        let positive_sum: f64 = self
            .values
            .iter()
            .filter(|v| v.is_finite() && **v >= 0.0)
            .sum();
        let full_circle = positive_sum > 1.0 || positive_sum == 0.0;
        let mut angle = 0.0f32;
        let mut out = Vec::new();
        for (i, &val) in self.values.iter().enumerate() {
            if !val.is_finite() || val < 0.0 {
                continue;
            }
            let frac = if full_circle {
                if positive_sum == 0.0 {
                    0.0
                } else {
                    (val / positive_sum) as f32
                }
            } else {
                val as f32
            };
            let theta = frac * 2.0 * PI;
            let mid = angle + theta * 0.5;
            let exploded = self.explode.get(i).copied().unwrap_or(false);
            let offset = if exploded {
                Vec3::new(mid.cos() * 0.12, mid.sin() * 0.12, 0.0)
            } else {
                Vec3::ZERO
            };
            let label = self
                .slice_labels
                .get(i)
                .cloned()
                .unwrap_or_else(|| format_percentage_label(self.label_format.as_deref(), frac));
            out.push(PieSliceMeta {
                label,
                color: self.colors[i % self.colors.len()],
                mid_angle: mid,
                offset,
                fraction: frac,
            });
            angle += theta;
        }
        out
    }
    pub fn generate_vertices(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.dirty || self.vertices.is_none() {
            let mut verts = Vec::new();
            let mut inds: Vec<u32> = Vec::new();
            let mut angle = 0.0f32;
            let mut acc_index = 0u32;
            let slices = self.compute_slice_meta();
            for (i, &val) in self.values.iter().enumerate() {
                if !val.is_finite() || val < 0.0 {
                    continue;
                }
                let Some(slice) = slices.get(i) else {
                    continue;
                };
                let theta = slice.fraction * 2.0 * PI;
                let steps = (theta * 20.0).ceil().max(1.0) as u32; // adaptive tessellation
                let color = slice.color;
                let start = angle;
                let offset = slice.offset;
                let center_index = acc_index;
                verts.push(Vertex::new(offset, Vec4::new(1.0, 1.0, 1.0, 1.0)));
                acc_index += 1;
                for s in 0..=steps {
                    let a = start + (theta * (s as f32 / steps as f32));
                    verts.push(Vertex::new(
                        Vec3::new(a.cos(), a.sin(), 0.0) + offset,
                        color,
                    ));
                    if s > 0 {
                        inds.extend_from_slice(&[center_index, acc_index + s - 1, acc_index + s]);
                    }
                }
                acc_index += steps + 1;
                angle += theta;
            }
            self.vertices = Some(verts);
            self.indices = Some(inds);
            self.slices = Some(slices);
            self.dirty = false;
        }
        (
            self.vertices.as_ref().unwrap(),
            self.indices.as_ref().unwrap(),
        )
    }
    pub fn bounds(&mut self) -> BoundingBox {
        if self.bounds.is_none() || self.dirty {
            let slices = self.compute_slice_meta();
            let mut min = Vec3::new(-1.0, -1.0, 0.0);
            let mut max = Vec3::new(1.0, 1.0, 0.0);
            for slice in &slices {
                min.x = min.x.min(-1.0 + slice.offset.x);
                min.y = min.y.min(-1.0 + slice.offset.y);
                max.x = max.x.max(1.0 + slice.offset.x);
                max.y = max.y.max(1.0 + slice.offset.y);
            }
            self.bounds = Some(BoundingBox::new(min, max));
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
            bounds: None,
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

fn format_percentage_label(fmt: Option<&str>, frac: f32) -> String {
    match fmt.unwrap_or("%.0f%%") {
        "%.0f%%" => format!("{:.0}%", frac * 100.0),
        "%.1f%%" => format!("{:.1}%", frac * 100.0),
        "%.2f%%" => format!("{:.2}%", frac * 100.0),
        other => {
            if other.contains("%") {
                other
                    .replace("%.0f", &format!("{:.0}", frac * 100.0))
                    .replace("%.1f", &format!("{:.1}", frac * 100.0))
                    .replace("%.2f", &format!("{:.2}", frac * 100.0))
            } else {
                other.to_string()
            }
        }
    }
}
