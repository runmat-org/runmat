//! Stem plot implementation.

use crate::core::{
    AlphaMode, BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use crate::plots::line::{LineMarkerAppearance, LineStyle};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct StemPlot {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub baseline: f64,
    pub color: Vec4,
    pub line_width: f32,
    pub line_style: LineStyle,
    pub baseline_color: Vec4,
    pub baseline_visible: bool,
    pub marker: Option<LineMarkerAppearance>,
    pub label: Option<String>,
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
    gpu_vertices: Option<GpuVertexBuffer>,
    gpu_vertex_count: Option<usize>,
    gpu_bounds: Option<BoundingBox>,
    marker_vertices: Option<Vec<Vertex>>,
    marker_gpu_vertices: Option<GpuVertexBuffer>,
    marker_dirty: bool,
}

impl StemPlot {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, String> {
        if x.len() != y.len() || x.is_empty() {
            return Err("stem: X and Y must be same non-zero length".to_string());
        }
        Ok(Self {
            x,
            y,
            baseline: 0.0,
            color: Vec4::new(0.0, 0.447, 0.741, 1.0),
            line_width: 1.0,
            line_style: LineStyle::Solid,
            baseline_color: Vec4::new(0.15, 0.15, 0.15, 1.0),
            baseline_visible: true,
            marker: Some(LineMarkerAppearance {
                kind: crate::plots::scatter::MarkerStyle::Circle,
                size: 6.0,
                edge_color: Vec4::new(0.0, 0.447, 0.741, 1.0),
                face_color: Vec4::new(0.0, 0.447, 0.741, 1.0),
                filled: false,
            }),
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_bounds: None,
            marker_vertices: None,
            marker_gpu_vertices: None,
            marker_dirty: true,
        })
    }

    pub fn from_gpu_buffer(
        color: Vec4,
        line_width: f32,
        line_style: LineStyle,
        baseline: f64,
        baseline_color: Vec4,
        baseline_visible: bool,
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            baseline,
            color,
            line_width,
            line_style,
            baseline_color,
            baseline_visible,
            marker: None,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: false,
            gpu_vertices: Some(buffer),
            gpu_vertex_count: Some(vertex_count),
            gpu_bounds: Some(bounds),
            marker_vertices: None,
            marker_gpu_vertices: None,
            marker_dirty: true,
        }
    }

    pub fn with_style(
        mut self,
        color: Vec4,
        line_width: f32,
        line_style: LineStyle,
        baseline: f64,
    ) -> Self {
        self.color = color;
        self.line_width = line_width.max(0.5);
        self.line_style = line_style;
        self.baseline = baseline;
        self.dirty = true;
        self.marker_dirty = true;
        self.gpu_vertices = None;
        self.gpu_vertex_count = None;
        self.gpu_bounds = None;
        self.marker_gpu_vertices = None;
        self
    }

    pub fn with_baseline_style(mut self, color: Vec4, visible: bool) -> Self {
        self.baseline_color = color;
        self.baseline_visible = visible;
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

    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.gpu_vertices.is_some() {
            if self.vertices.is_none() {
                self.vertices = Some(Vec::new());
            }
            return self.vertices.as_ref().unwrap();
        }
        if self.dirty || self.vertices.is_none() {
            let mut vertices = Vec::new();
            let finite_x: Vec<f32> = self
                .x
                .iter()
                .map(|v| *v as f32)
                .filter(|v| v.is_finite())
                .collect();
            if self.baseline_visible && !finite_x.is_empty() {
                let min_x = finite_x.iter().copied().fold(f32::INFINITY, f32::min);
                let max_x = finite_x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                vertices.push(Vertex::new(
                    Vec3::new(min_x, self.baseline as f32, 0.0),
                    self.baseline_color,
                ));
                vertices.push(Vertex::new(
                    Vec3::new(max_x, self.baseline as f32, 0.0),
                    self.baseline_color,
                ));
            }
            for i in 0..self.x.len() {
                let x = self.x[i] as f32;
                let y = self.y[i] as f32;
                let b = self.baseline as f32;
                if !x.is_finite() || !y.is_finite() {
                    continue;
                }
                if include_segment(i, self.line_style) {
                    vertices.push(Vertex::new(Vec3::new(x, b, 0.0), self.color));
                    vertices.push(Vertex::new(Vec3::new(x, y, 0.0), self.color));
                }
            }
            self.vertices = Some(vertices);
            self.dirty = false;
        }
        self.vertices.as_ref().unwrap()
    }

    pub fn marker_render_data(&mut self) -> Option<RenderData> {
        let marker = self.marker.clone()?;
        if let Some(gpu_vertices) = self.marker_gpu_vertices.clone() {
            let vertex_count = gpu_vertices.vertex_count;
            if vertex_count == 0 {
                return None;
            }
            return Some(RenderData {
                pipeline_type: PipelineType::Triangles,
                vertices: Vec::new(),
                indices: None,
                gpu_vertices: Some(gpu_vertices),
                bounds: None,
                material: Material {
                    albedo: marker.face_color,
                    alpha_mode: if marker.face_color.w < 0.999 {
                        AlphaMode::Blend
                    } else {
                        AlphaMode::Opaque
                    },
                    ..Default::default()
                },
                draw_calls: vec![DrawCall {
                    vertex_offset: 0,
                    vertex_count,
                    index_offset: None,
                    index_count: None,
                    instance_count: 1,
                }],
                image: None,
            });
        }
        if self.marker_dirty || self.marker_vertices.is_none() {
            let size = (marker.size.max(1.0) * 0.005).max(0.005);
            let mut vertices = Vec::new();
            for (&x, &y) in self.x.iter().zip(self.y.iter()) {
                let x = x as f32;
                let y = y as f32;
                if !x.is_finite() || !y.is_finite() {
                    continue;
                }
                vertices.extend(square_marker(x, y, size, marker.face_color));
            }
            self.marker_vertices = Some(vertices);
            self.marker_dirty = false;
        }
        let vertices = self.marker_vertices.as_ref()?;
        if vertices.is_empty() {
            return None;
        }
        Some(RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices: vertices.clone(),
            indices: None,
            gpu_vertices: None,
            bounds: None,
            material: Material {
                albedo: marker.face_color,
                alpha_mode: if marker.face_color.w < 0.999 {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                },
                ..Default::default()
            },
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count: vertices.len(),
                index_offset: None,
                index_count: None,
                instance_count: 1,
            }],
            image: None,
        })
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
                min.y = min.y.min(y.min(self.baseline as f32));
                max.y = max.y.max(y.max(self.baseline as f32));
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
        let (vertices, vertex_count, gpu_vertices) = if self.gpu_vertices.is_some() {
            (
                Vec::new(),
                self.gpu_vertex_count.unwrap_or(0),
                self.gpu_vertices.clone(),
            )
        } else {
            let vertices = self.generate_vertices().clone();
            let count = vertices.len();
            (vertices, count, None)
        };
        RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices,
            bounds: None,
            material: Material {
                albedo: self.color,
                roughness: self.line_width,
                ..Default::default()
            },
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count,
                index_offset: None,
                index_count: None,
                instance_count: 1,
            }],
            image: None,
        }
    }

    pub fn estimated_memory_usage(&self) -> usize {
        self.vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
            + self
                .marker_vertices
                .as_ref()
                .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
    }
}

fn include_segment(index: usize, style: LineStyle) -> bool {
    match style {
        LineStyle::Solid => true,
        LineStyle::Dashed => (index % 4) < 2,
        LineStyle::Dotted => (index % 4) == 0,
        LineStyle::DashDot => {
            let m = index % 6;
            m < 2 || m == 3
        }
    }
}

fn square_marker(x: f32, y: f32, half: f32, color: Vec4) -> [Vertex; 6] {
    [
        Vertex::new(Vec3::new(x - half, y - half, 0.0), color),
        Vertex::new(Vec3::new(x + half, y - half, 0.0), color),
        Vertex::new(Vec3::new(x + half, y + half, 0.0), color),
        Vertex::new(Vec3::new(x - half, y - half, 0.0), color),
        Vertex::new(Vec3::new(x + half, y + half, 0.0), color),
        Vertex::new(Vec3::new(x - half, y + half, 0.0), color),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_bounds_include_baseline() {
        let mut plot = StemPlot::new(vec![0.0, 1.0], vec![1.0, -2.0])
            .unwrap()
            .with_style(Vec4::ONE, 1.0, LineStyle::Solid, -1.0);
        let bounds = plot.bounds();
        assert_eq!(bounds.min.y, -2.0);
        assert_eq!(bounds.max.y, 1.0);
    }
}
