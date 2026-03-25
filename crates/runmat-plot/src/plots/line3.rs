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
        if x_data.len() < 2 {
            return Err("plot3 requires at least two points".to_string());
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
            let mut vertices = Vec::with_capacity((self.x_data.len() - 1) * 2);
            for i in 0..self.x_data.len() - 1 {
                vertices.push(Vertex::new(
                    Vec3::new(
                        self.x_data[i] as f32,
                        self.y_data[i] as f32,
                        self.z_data[i] as f32,
                    ),
                    self.color,
                ));
                vertices.push(Vertex::new(
                    Vec3::new(
                        self.x_data[i + 1] as f32,
                        self.y_data[i + 1] as f32,
                        self.z_data[i + 1] as f32,
                    ),
                    self.color,
                ));
            }
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
        let vertex_count = self
            .gpu_vertex_count
            .unwrap_or_else(|| self.generate_vertices().len());
        RenderData {
            pipeline_type: PipelineType::Lines,
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
}
