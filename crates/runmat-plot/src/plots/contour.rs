//! Contour plot implementation (iso-lines on a surface or base plane).

use crate::core::{
    vertex_utils, BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData,
    Vertex,
};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct ContourPlot {
    pub base_z: f32,
    pub force_3d: bool,
    pub label: Option<String>,
    pub visible: bool,
    pub line_width: f32,
    vertices: Option<Vec<Vertex>>,
    gpu_vertices: Option<GpuVertexBuffer>,
    vertex_count: usize,
    bounds: Option<BoundingBox>,
}

impl ContourPlot {
    /// Create a contour plot from CPU vertices.
    pub fn from_vertices(vertices: Vec<Vertex>, base_z: f32, bounds: BoundingBox) -> Self {
        Self {
            base_z,
            force_3d: false,
            label: None,
            visible: true,
            line_width: 1.0,
            vertex_count: vertices.len(),
            vertices: Some(vertices),
            gpu_vertices: None,
            bounds: Some(bounds),
        }
    }

    /// Create a contour plot backed by a GPU vertex buffer.
    pub fn from_gpu_buffer(
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        base_z: f32,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            base_z,
            force_3d: false,
            label: None,
            visible: true,
            line_width: 1.0,
            vertex_count,
            vertices: None,
            gpu_vertices: Some(buffer),
            bounds: Some(bounds),
        }
    }

    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_force_3d(mut self, force_3d: bool) -> Self {
        self.force_3d = force_3d;
        self
    }

    pub fn is_3d(&self) -> bool {
        self.force_3d || (self.bounds().max.z - self.bounds().min.z).abs() > f32::EPSILON
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn with_line_width(mut self, line_width: f32) -> Self {
        self.line_width = line_width.max(0.5);
        self
    }

    pub fn vertices(&mut self) -> &Vec<Vertex> {
        if self.vertices.is_none() {
            self.vertices = Some(Vec::new());
        }
        self.vertices.as_ref().unwrap()
    }

    pub fn bounds(&self) -> BoundingBox {
        self.bounds.unwrap_or_default()
    }

    pub fn cpu_vertices(&self) -> Option<&[Vertex]> {
        self.vertices.as_deref()
    }

    pub fn render_data_with_viewport(&mut self, viewport_px: Option<(u32, u32)>) -> RenderData {
        if self.gpu_vertices.is_some() {
            return self.render_data();
        }

        let bounds = self.bounds();
        let (vertices, vertex_count, pipeline_type, render_bounds) = if self.line_width > 1.0 {
            let Some(viewport_px) = viewport_px else {
                return self.render_data();
            };
            let data_per_px = crate::core::data_units_per_px(&bounds, viewport_px);
            let width_data = self.line_width.max(0.1) * data_per_px;
            let verts = self.vertices().clone();
            let mut thick = Vec::new();
            for segment in verts.chunks_exact(2) {
                let color = Vec4::from_array(segment[0].color);
                if self.is_3d() {
                    thick.extend(create_thick_segment_3d(
                        Vec3::from_array(segment[0].position),
                        Vec3::from_array(segment[1].position),
                        color,
                        width_data * 0.5,
                    ));
                } else {
                    let x = [segment[0].position[0] as f64, segment[1].position[0] as f64];
                    let y = [segment[0].position[1] as f64, segment[1].position[1] as f64];
                    thick.extend(vertex_utils::create_thick_polyline(
                        &x, &y, color, width_data,
                    ));
                }
            }
            let count = thick.len();
            let render_bounds = expanded_bounds_for_vertices(bounds, &thick);
            (thick, count, PipelineType::Triangles, render_bounds)
        } else {
            let verts = self.vertices().clone();
            let count = verts.len();
            (verts, count, PipelineType::Lines, bounds)
        };

        let material = Material {
            albedo: Vec4::ONE,
            roughness: self.line_width.max(0.0),
            ..Default::default()
        };

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };

        RenderData {
            pipeline_type,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: Some(render_bounds),
            material,
            draw_calls: vec![draw_call],
            image: None,
        }
    }

    pub fn render_data(&mut self) -> RenderData {
        let using_gpu = self.gpu_vertices.is_some();
        let bounds = self.bounds();
        let (vertices, vertex_count, gpu_vertices, pipeline_type, render_bounds) = if using_gpu {
            (
                Vec::new(),
                self.vertex_count,
                self.gpu_vertices.clone(),
                PipelineType::Lines,
                bounds,
            )
        } else {
            let verts = self.vertices().clone();
            if self.line_width > 1.0 {
                let mut thick = Vec::new();
                for segment in verts.chunks_exact(2) {
                    let color = Vec4::from_array(segment[0].color);
                    if self.is_3d() {
                        thick.extend(create_thick_segment_3d(
                            Vec3::from_array(segment[0].position),
                            Vec3::from_array(segment[1].position),
                            color,
                            self.line_width.max(0.5) * 0.01,
                        ));
                    } else {
                        let x = [segment[0].position[0] as f64, segment[1].position[0] as f64];
                        let y = [segment[0].position[1] as f64, segment[1].position[1] as f64];
                        thick.extend(vertex_utils::create_thick_polyline(
                            &x,
                            &y,
                            color,
                            self.line_width,
                        ));
                    }
                }
                let count = thick.len();
                let render_bounds = expanded_bounds_for_vertices(bounds, &thick);
                (thick, count, None, PipelineType::Triangles, render_bounds)
            } else {
                let count = verts.len();
                (verts, count, None, PipelineType::Lines, bounds)
            }
        };

        let material = Material {
            albedo: Vec4::ONE,
            roughness: self.line_width.max(0.0),
            ..Default::default()
        };

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };

        RenderData {
            pipeline_type,
            vertices,
            indices: None,
            gpu_vertices,
            bounds: Some(render_bounds),
            material,
            draw_calls: vec![draw_call],
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

pub fn contour_bounds(x_min: f32, x_max: f32, y_min: f32, y_max: f32, base_z: f32) -> BoundingBox {
    BoundingBox::new(
        Vec3::new(x_min, y_min, base_z),
        Vec3::new(x_max, y_max, base_z),
    )
}

pub fn contour_bounds_3d(
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    z_min: f32,
    z_max: f32,
) -> BoundingBox {
    BoundingBox::new(
        Vec3::new(x_min, y_min, z_min),
        Vec3::new(x_max, y_max, z_max),
    )
}

fn expanded_bounds_for_vertices(mut bounds: BoundingBox, vertices: &[Vertex]) -> BoundingBox {
    for vertex in vertices {
        bounds.expand(Vec3::from_array(vertex.position));
    }
    bounds
}

fn create_thick_segment_3d(start: Vec3, end: Vec3, color: Vec4, half_width: f32) -> Vec<Vertex> {
    let dir = (end - start).normalize_or_zero();
    let mut normal = dir.cross(Vec3::Z);
    if normal.length_squared() < 1e-6 {
        normal = dir.cross(Vec3::X);
    }
    let normal = normal.normalize_or_zero() * half_width.max(0.0001);
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

    fn test_vertex(x: f32, y: f32, z: f32) -> Vertex {
        Vertex::new(Vec3::new(x, y, z), Vec4::ONE)
    }

    #[test]
    fn viewport_thick_contour_bounds_include_extruded_geometry() {
        let bounds = BoundingBox::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let mut contour = ContourPlot::from_vertices(
            vec![test_vertex(0.0, 0.0, 0.0), test_vertex(1.0, 0.0, 0.0)],
            0.0,
            bounds,
        )
        .with_line_width(2.0);

        let render_data = contour.render_data_with_viewport(Some((100, 100)));
        let render_bounds = render_data.bounds.expect("bounds");

        assert!(render_bounds.min.y < bounds.min.y);
        assert!(render_bounds.max.y > bounds.max.y);
    }

    #[test]
    fn non_viewport_thick_3d_contour_bounds_include_extruded_geometry() {
        let bounds = BoundingBox::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0));
        let mut contour = ContourPlot::from_vertices(
            vec![test_vertex(0.0, 0.0, 1.0), test_vertex(0.0, 1.0, 1.0)],
            0.0,
            bounds,
        )
        .with_force_3d(true)
        .with_line_width(2.0);

        let render_data = contour.render_data();
        let render_bounds = render_data.bounds.expect("bounds");

        assert!(render_bounds.min.x < bounds.min.x);
        assert!(render_bounds.max.x > bounds.max.x);
    }

    #[test]
    fn viewport_thick_3d_contour_uses_half_width_data() {
        let bounds = BoundingBox::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0));
        let mut contour = ContourPlot::from_vertices(
            vec![test_vertex(0.0, 0.0, 1.0), test_vertex(1.0, 0.0, 1.0)],
            0.0,
            bounds,
        )
        .with_force_3d(true)
        .with_line_width(4.0);

        let render_data = contour.render_data_with_viewport(Some((100, 100)));
        let render_bounds = render_data.bounds.expect("bounds");

        assert!((render_bounds.min.y - -0.02).abs() < 1e-6);
    }
}
