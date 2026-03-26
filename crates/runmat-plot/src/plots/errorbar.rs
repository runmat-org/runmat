//! Error bar plot implementation.

use crate::core::{
    marker_shape_code, AlphaMode, BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType,
    RenderData, Vertex,
};
use crate::plots::line::{LineMarkerAppearance, LineStyle};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorBarOrientation {
    Vertical,
    Horizontal,
    Both,
}

#[derive(Debug, Clone)]
pub struct ErrorBar {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub y_neg: Vec<f64>,
    pub y_pos: Vec<f64>,
    pub x_neg: Vec<f64>,
    pub x_pos: Vec<f64>,
    pub orientation: ErrorBarOrientation,

    pub color: Vec4,
    pub line_width: f32,
    pub line_style: LineStyle,
    pub cap_size: f32,
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

impl ErrorBar {
    pub fn new_vertical(
        x: Vec<f64>,
        y: Vec<f64>,
        y_neg: Vec<f64>,
        y_pos: Vec<f64>,
    ) -> Result<Self, String> {
        let n = x.len();
        if n == 0 || y.len() != n || y_neg.len() != n || y_pos.len() != n {
            return Err("errorbar: input vectors must have equal non-zero length".to_string());
        }
        Ok(Self {
            x,
            y,
            y_neg,
            y_pos,
            x_neg: vec![0.0; n],
            x_pos: vec![0.0; n],
            orientation: ErrorBarOrientation::Vertical,
            color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            line_width: 1.0,
            line_style: LineStyle::Solid,
            cap_size: 6.0,
            marker: Some(LineMarkerAppearance {
                kind: crate::plots::scatter::MarkerStyle::Circle,
                size: 6.0,
                edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
                face_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
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

    pub fn new_both(
        x: Vec<f64>,
        y: Vec<f64>,
        x_neg: Vec<f64>,
        x_pos: Vec<f64>,
        y_neg: Vec<f64>,
        y_pos: Vec<f64>,
    ) -> Result<Self, String> {
        let n = x.len();
        if n == 0
            || y.len() != n
            || x_neg.len() != n
            || x_pos.len() != n
            || y_neg.len() != n
            || y_pos.len() != n
        {
            return Err("errorbar: input vectors must have equal non-zero length".to_string());
        }
        let mut plot = Self::new_vertical(x, y, y_neg, y_pos)?;
        plot.x_neg = x_neg;
        plot.x_pos = x_pos;
        plot.orientation = ErrorBarOrientation::Both;
        Ok(plot)
    }

    pub fn with_style(
        mut self,
        color: Vec4,
        line_width: f32,
        line_style: LineStyle,
        cap_size: f32,
    ) -> Self {
        self.color = color;
        self.line_width = line_width.max(0.5);
        self.line_style = line_style;
        self.cap_size = cap_size.max(0.0);
        self.dirty = true;
        self.gpu_vertices = None;
        self.gpu_vertex_count = None;
        self.gpu_bounds = None;
        self.marker_gpu_vertices = None;
        self.marker_dirty = true;
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

    #[allow(clippy::too_many_arguments)]
    pub fn from_gpu_buffer(
        color: Vec4,
        line_width: f32,
        line_style: LineStyle,
        cap_size: f32,
        orientation: ErrorBarOrientation,
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            y_neg: Vec::new(),
            y_pos: Vec::new(),
            x_neg: Vec::new(),
            x_pos: Vec::new(),
            orientation,
            color,
            line_width,
            line_style,
            cap_size,
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

    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.gpu_vertices.is_some() {
            if self.vertices.is_none() {
                self.vertices = Some(Vec::new());
            }
            return self.vertices.as_ref().unwrap();
        }
        if self.dirty || self.vertices.is_none() {
            let mut verts = Vec::new();
            for i in 0..self.x.len() {
                let xi = self.x[i] as f32;
                let yi = self.y[i] as f32;
                if !xi.is_finite() || !yi.is_finite() {
                    continue;
                }
                if matches!(
                    self.orientation,
                    ErrorBarOrientation::Vertical | ErrorBarOrientation::Both
                ) {
                    let y0 = (self.y[i] - self.y_neg[i]) as f32;
                    let y1 = (self.y[i] + self.y_pos[i]) as f32;
                    if y0.is_finite() && y1.is_finite() && include_segment(i, self.line_style) {
                        verts.push(Vertex::new(Vec3::new(xi, y0, 0.0), self.color));
                        verts.push(Vertex::new(Vec3::new(xi, y1, 0.0), self.color));
                        if self.cap_size > 0.0 {
                            let half = self.cap_size * 0.005;
                            verts.push(Vertex::new(Vec3::new(xi - half, y0, 0.0), self.color));
                            verts.push(Vertex::new(Vec3::new(xi + half, y0, 0.0), self.color));
                            verts.push(Vertex::new(Vec3::new(xi - half, y1, 0.0), self.color));
                            verts.push(Vertex::new(Vec3::new(xi + half, y1, 0.0), self.color));
                        }
                    }
                }
                if matches!(
                    self.orientation,
                    ErrorBarOrientation::Horizontal | ErrorBarOrientation::Both
                ) {
                    let x0 = (self.x[i] - self.x_neg[i]) as f32;
                    let x1 = (self.x[i] + self.x_pos[i]) as f32;
                    if x0.is_finite() && x1.is_finite() && include_segment(i, self.line_style) {
                        verts.push(Vertex::new(Vec3::new(x0, yi, 0.0), self.color));
                        verts.push(Vertex::new(Vec3::new(x1, yi, 0.0), self.color));
                        if self.cap_size > 0.0 {
                            let half = self.cap_size * 0.005;
                            verts.push(Vertex::new(Vec3::new(x0, yi - half, 0.0), self.color));
                            verts.push(Vertex::new(Vec3::new(x0, yi + half, 0.0), self.color));
                            verts.push(Vertex::new(Vec3::new(x1, yi - half, 0.0), self.color));
                            verts.push(Vertex::new(Vec3::new(x1, yi + half, 0.0), self.color));
                        }
                    }
                }
            }
            self.vertices = Some(verts);
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
                pipeline_type: PipelineType::Points,
                vertices: Vec::new(),
                indices: None,
                gpu_vertices: Some(gpu_vertices),
                bounds: None,
                material: Material {
                    albedo: marker.face_color,
                    emissive: marker.edge_color,
                    roughness: 1.0,
                    metallic: marker_shape_code(marker.kind) as f32,
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
            let mut vertices = Vec::new();
            for (&x, &y) in self.x.iter().zip(self.y.iter()) {
                let x = x as f32;
                let y = y as f32;
                if !x.is_finite() || !y.is_finite() {
                    continue;
                }
                vertices.push(Vertex {
                    position: [x, y, 0.0],
                    color: marker.face_color.to_array(),
                    normal: [0.0, 0.0, marker.size],
                    tex_coords: [0.0, 0.0],
                });
            }
            self.marker_vertices = Some(vertices);
            self.marker_dirty = false;
        }
        let vertices = self.marker_vertices.as_ref()?;
        if vertices.is_empty() {
            return None;
        }
        Some(RenderData {
            pipeline_type: PipelineType::Points,
            vertices: vertices.clone(),
            indices: None,
            gpu_vertices: None,
            bounds: None,
            material: Material {
                albedo: marker.face_color,
                emissive: marker.edge_color,
                roughness: 1.0,
                metallic: marker_shape_code(marker.kind) as f32,
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
            for i in 0..self.x.len() {
                let xi = self.x[i] as f32;
                let yi = self.y[i] as f32;
                if !xi.is_finite() || !yi.is_finite() {
                    continue;
                }
                min.x = min
                    .x
                    .min(xi - self.x_neg.get(i).copied().unwrap_or(0.0) as f32);
                max.x = max
                    .x
                    .max(xi + self.x_pos.get(i).copied().unwrap_or(0.0) as f32);
                min.y = min
                    .y
                    .min(yi - self.y_neg.get(i).copied().unwrap_or(0.0) as f32);
                max.y = max
                    .y
                    .max(yi + self.y_pos.get(i).copied().unwrap_or(0.0) as f32);
            }
            if self.cap_size > 0.0 {
                let half = self.cap_size * 0.005;
                min.x -= half;
                max.x += half;
                min.y -= half;
                max.y += half;
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
        LineStyle::Dotted => index.is_multiple_of(4),
        LineStyle::DashDot => {
            let m = index % 6;
            m < 2 || m == 3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errorbar_bounds_include_caps() {
        let mut plot = ErrorBar::new_vertical(
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        )
        .unwrap()
        .with_style(Vec4::ONE, 1.0, LineStyle::Solid, 10.0);
        let bounds = plot.bounds();
        assert!(bounds.max.x > 1.0);
        assert!(bounds.min.x < 0.0);
    }
}
