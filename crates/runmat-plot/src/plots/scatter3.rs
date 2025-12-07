//! 3D scatter plot implementation for MATLAB's `scatter3`.

use crate::core::{
    vertex_utils, BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData,
    Vertex,
};
use glam::{Vec3, Vec4};

/// GPU-accelerated scatter3 plot for MATLAB semantics.
#[derive(Debug, Clone)]
pub struct Scatter3Plot {
    /// Point positions in 3D space.
    pub points: Vec<Vec3>,
    /// Per-point RGBA colors.
    pub colors: Vec<Vec4>,
    /// Marker size in pixels.
    pub point_size: f32,
    /// Legend label.
    pub label: Option<String>,
    /// Visibility flag.
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    gpu_vertices: Option<GpuVertexBuffer>,
    gpu_point_count: Option<usize>,
}

impl Scatter3Plot {
    /// Create a new scatter3 plot. Colors default to a blue colormap.
    pub fn new(points: Vec<Vec3>) -> Result<Self, String> {
        let default_color = Vec4::new(0.1, 0.7, 0.3, 1.0);
        let colors = vec![default_color; points.len()];
        Ok(Self {
            points,
            colors,
            point_size: 8.0,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            gpu_vertices: None,
            gpu_point_count: None,
        })
    }

    /// Build a scatter plot directly from a GPU vertex buffer, bypassing CPU copies.
    pub fn from_gpu_buffer(
        buffer: GpuVertexBuffer,
        point_count: usize,
        color: Vec4,
        point_size: f32,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            points: Vec::new(),
            colors: vec![color],
            point_size,
            label: None,
            visible: true,
            vertices: None,
            bounds: Some(bounds),
            gpu_vertices: Some(buffer),
            gpu_point_count: Some(point_count),
        }
    }

    /// Override all point colors with a single RGBA value.
    pub fn with_color(mut self, color: Vec4) -> Self {
        self.colors = vec![color; self.points.len()];
        self.vertices = None;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self
    }

    /// Supply per-point colors. Length must match the number of points.
    pub fn with_colors(mut self, colors: Vec<Vec4>) -> Result<Self, String> {
        if colors.len() != self.points.len() {
            return Err(format!(
                "Point cloud color count ({}) must match point count ({})",
                colors.len(),
                self.points.len()
            ));
        }
        self.colors = colors;
        self.vertices = None;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        Ok(self)
    }

    /// Set the legend label.
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set marker size in pixels.
    pub fn with_point_size(mut self, size: f32) -> Self {
        self.point_size = size.max(1.0);
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self
    }

    /// Enable or disable visibility.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Attach a GPU-resident vertex buffer that already encodes this point cloud in the renderer's vertex format.
    /// When provided, the renderer can skip per-frame uploads and reuse the supplied buffer directly.
    pub fn with_gpu_vertices(mut self, buffer: GpuVertexBuffer, point_count: usize) -> Self {
        self.gpu_vertices = Some(buffer);
        self.gpu_point_count = Some(point_count);
        self.vertices = None;
        self
    }

    fn ensure_vertices(&mut self) {
        if self.vertices.is_none() {
            self.vertices = Some(vertex_utils::create_point_cloud(&self.points, &self.colors));
        }
    }

    fn ensure_bounds(&mut self) {
        if self.bounds.is_none() {
            self.bounds = Some(BoundingBox::from_points(&self.points));
        }
    }

    /// Estimate memory required for this plot.
    pub fn estimated_memory_usage(&self) -> usize {
        let gpu_bytes = self
            .gpu_point_count
            .map(|count| count * std::mem::size_of::<Vertex>())
            .unwrap_or(0);
        self.points.len() * std::mem::size_of::<Vec3>()
            + self.colors.len() * std::mem::size_of::<Vec4>()
            + gpu_bytes
    }

    /// Generate render data for the renderer.
    pub fn render_data(&mut self) -> RenderData {
        let vertex_count = self.gpu_point_count.unwrap_or_else(|| {
            self.ensure_vertices();
            self.vertices
                .as_ref()
                .map(|v| v.len())
                .unwrap_or(self.points.len())
        });

        let vertices = if self.gpu_vertices.is_some() {
            Vec::new()
        } else {
            self.ensure_vertices();
            self.vertices.clone().unwrap_or_default()
        };

        RenderData {
            pipeline_type: PipelineType::Scatter3,
            vertices,
            indices: None,
            gpu_vertices: self.gpu_vertices.clone(),
            material: Material {
                albedo: Vec4::ONE,
                roughness: 0.0,
                metallic: 0.0,
                emissive: Vec4::ZERO,
                alpha_mode: crate::core::scene::AlphaMode::Blend,
                double_sided: true,
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

    /// Compute the axis-aligned bounding box.
    pub fn bounds(&mut self) -> BoundingBox {
        self.ensure_bounds();
        self.bounds.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter3_defaults() {
        let points = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0)];
        let cloud = Scatter3Plot::new(points.clone()).unwrap();
        assert_eq!(cloud.points.len(), points.len());
        assert_eq!(cloud.colors.len(), points.len());
        assert!(cloud.visible);
    }

    #[test]
    fn scatter3_custom_colors() {
        let points = vec![Vec3::new(0.0, 0.0, 0.0)];
        let colors = vec![Vec4::new(1.0, 0.0, 0.0, 1.0)];
        let cloud = Scatter3Plot::new(points)
            .unwrap()
            .with_colors(colors)
            .unwrap();
        assert_eq!(cloud.colors[0], Vec4::new(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn scatter3_render_data_contains_vertices() {
        let points = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)];
        let mut cloud = Scatter3Plot::new(points).unwrap();
        let render_data = cloud.render_data();
        assert_eq!(render_data.vertices.len(), 2);
        assert_eq!(render_data.pipeline_type, PipelineType::Scatter3);
    }
}
