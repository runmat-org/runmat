use crate::core::{
    BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct Line3Plot {
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
    pub z_data: Vec<f64>,
    pub color: Vec4,
    pub line_width: f32,
    pub line_style: crate::plots::line::LineStyle,
    pub label: Option<String>,
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
    pub gpu_vertices: Option<GpuVertexBuffer>,
    pub gpu_vertex_count: Option<usize>,
}

impl Line3Plot {
    pub fn new(x_data: Vec<f64>, y_data: Vec<f64>, z_data: Vec<f64>) -> Result<Self, String> {
        if x_data.len() != y_data.len() || x_data.len() != z_data.len() {
            return Err("Data length mismatch for plot3".to_string());
        }
        if x_data.is_empty() {
            return Err("plot3 requires at least one point".to_string());
        }
        Ok(Self {
            x_data,
            y_data,
            z_data,
            color: Vec4::new(0.0, 0.5, 1.0, 1.0),
            line_width: 1.0,
            line_style: crate::plots::line::LineStyle::Solid,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
            gpu_vertices: None,
            gpu_vertex_count: None,
        })
    }

    pub fn from_gpu_buffer(
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        color: Vec4,
        line_width: f32,
        line_style: crate::plots::line::LineStyle,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            x_data: Vec::new(),
            y_data: Vec::new(),
            z_data: Vec::new(),
            color,
            line_width,
            line_style,
            label: None,
            visible: true,
            vertices: None,
            bounds: Some(bounds),
            dirty: false,
            gpu_vertices: Some(buffer),
            gpu_vertex_count: Some(vertex_count),
        }
    }

    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_style(
        mut self,
        color: Vec4,
        line_width: f32,
        line_style: crate::plots::line::LineStyle,
    ) -> Self {
        self.color = color;
        self.line_width = line_width;
        self.line_style = line_style;
        self.dirty = true;
        self.gpu_vertices = None;
        self.gpu_vertex_count = None;
        self
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.gpu_vertices.is_some() {
            if self.vertices.is_none() {
                self.vertices = Some(Vec::new());
            }
            return self.vertices.as_ref().unwrap();
        }
        if self.dirty || self.vertices.is_none() {
            let points: Vec<Vec3> = self
                .x_data
                .iter()
                .zip(self.y_data.iter())
                .zip(self.z_data.iter())
                .map(|((&x, &y), &z)| Vec3::new(x as f32, y as f32, z as f32))
                .collect();
            let vertices = if points.len() == 1 {
                let mut vertex = Vertex::new(points[0], self.color);
                vertex.normal[2] = (self.line_width.max(1.0) * 4.0).max(6.0);
                vec![vertex]
            } else if self.line_width > 1.0 {
                create_thick_polyline3_dashed(&points, self.color, self.line_width, self.line_style)
            } else {
                create_line3_vertices_dashed(&points, self.color, self.line_style)
            };
            self.vertices = Some(vertices);
            self.dirty = false;
        }
        self.vertices.as_ref().unwrap()
    }

    pub fn bounds(&mut self) -> BoundingBox {
        if self.bounds.is_some() && self.x_data.is_empty() {
            return self.bounds.unwrap();
        }
        if self.bounds.is_none() || self.dirty {
            let points: Vec<Vec3> = self
                .x_data
                .iter()
                .zip(self.y_data.iter())
                .zip(self.z_data.iter())
                .map(|((&x, &y), &z)| Vec3::new(x as f32, y as f32, z as f32))
                .collect();
            self.bounds = Some(BoundingBox::from_points(&points));
        }
        self.bounds.unwrap()
    }

    pub fn render_data(&mut self) -> RenderData {
        let single_point = self.x_data.len() == 1 || self.gpu_vertex_count == Some(1);
        let vertex_count = self
            .gpu_vertex_count
            .unwrap_or_else(|| self.generate_vertices().len());
        let thick = self.line_width > 1.0 && !single_point;
        RenderData {
            pipeline_type: if single_point {
                PipelineType::Scatter3
            } else if thick {
                PipelineType::Triangles
            } else {
                PipelineType::Lines
            },
            vertices: if self.gpu_vertices.is_some() {
                Vec::new()
            } else {
                self.generate_vertices().clone()
            },
            indices: None,
            gpu_vertices: self.gpu_vertices.clone(),
            bounds: Some(self.bounds()),
            material: Material {
                albedo: self.color,
                roughness: self.line_width.max(0.5),
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
            .map(|v| v.len() * std::mem::size_of::<Vertex>())
            .unwrap_or(0)
    }
}

fn create_line3_vertices_dashed(
    points: &[Vec3],
    color: Vec4,
    style: crate::plots::line::LineStyle,
) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    for i in 1..points.len() {
        let include = match style {
            crate::plots::line::LineStyle::Solid => true,
            crate::plots::line::LineStyle::Dashed => (i % 4) < 2,
            crate::plots::line::LineStyle::Dotted => (i % 4) == 0,
            crate::plots::line::LineStyle::DashDot => {
                let m = i % 6;
                m < 2 || m == 3
            }
        };
        if include {
            vertices.push(Vertex::new(points[i - 1], color));
            vertices.push(Vertex::new(points[i], color));
        }
    }
    vertices
}

fn create_thick_polyline3_dashed(
    points: &[Vec3],
    color: Vec4,
    width: f32,
    style: crate::plots::line::LineStyle,
) -> Vec<Vertex> {
    let mut out = Vec::new();
    for i in 0..points.len().saturating_sub(1) {
        let include = match style {
            crate::plots::line::LineStyle::Solid => true,
            crate::plots::line::LineStyle::Dashed => (i % 4) < 2,
            crate::plots::line::LineStyle::Dotted => (i % 4) == 0,
            crate::plots::line::LineStyle::DashDot => {
                let m = i % 6;
                m < 2 || m == 3
            }
        };
        if !include {
            continue;
        }
        out.extend(extrude_segment_3d(
            points[i],
            points[i + 1],
            color,
            width.max(0.5) * 0.01,
        ));
    }
    out
}

fn extrude_segment_3d(start: Vec3, end: Vec3, color: Vec4, half_width: f32) -> Vec<Vertex> {
    let dir = (end - start).normalize_or_zero();
    let mut normal = dir.cross(Vec3::Z);
    if normal.length_squared() < 1e-6 {
        normal = dir.cross(Vec3::X);
    }
    let normal = normal.normalize_or_zero() * half_width;
    let v0 = start + normal;
    let v1 = end + normal;
    let v2 = end - normal;
    let v3 = start - normal;
    vec![
        Vertex::new(v0, color),
        Vertex::new(v1, color),
        Vertex::new(v2, color),
        Vertex::new(v0, color),
        Vertex::new(v2, color),
        Vertex::new(v3, color),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line3_creation_and_bounds() {
        let mut plot = Line3Plot::new(vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0]).unwrap();
        let bounds = plot.bounds();
        assert_eq!(bounds.min, Vec3::new(0.0, 1.0, 2.0));
        assert_eq!(bounds.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn line3_dashed_and_thick_generate_geometry() {
        let mut plot = Line3Plot::new(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        )
        .unwrap()
        .with_style(Vec4::ONE, 3.0, crate::plots::line::LineStyle::Dashed);
        let render = plot.render_data();
        assert_eq!(render.pipeline_type, PipelineType::Triangles);
        assert!(!render.vertices.is_empty());
        assert!(render.draw_calls[0].vertex_count >= 2);
    }

    #[test]
    fn line3_single_point_uses_scatter_pipeline() {
        let mut plot = Line3Plot::new(vec![1.0], vec![2.0], vec![3.0])
            .unwrap()
            .with_style(Vec4::ONE, 2.0, crate::plots::line::LineStyle::Solid);
        let render = plot.render_data();
        assert_eq!(render.pipeline_type, PipelineType::Scatter3);
        assert_eq!(render.vertices.len(), 1);
        assert!(render.vertices[0].normal[2] >= 6.0);
    }
}
