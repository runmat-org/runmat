//! 3D scatter plot implementation for MATLAB's `scatter3`.

use crate::context::shared_wgpu_context;
use crate::core::{
    vertex_utils, BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData,
    Vertex,
};
use crate::gpu::scatter2::ScatterColorBuffer;
use crate::gpu::scatter3::Scatter3GpuInputs;
use crate::gpu::util::{copy_readback_bytes, readback_scalar_buffer_f64};
use crate::plots::scatter::MarkerStyle;
use glam::{Vec3, Vec4};

#[derive(Clone, Copy, Debug)]
pub struct Scatter3GpuStyle {
    pub color: Vec4,
    pub edge_color: Vec4,
    pub edge_thickness: f32,
    pub marker_style: MarkerStyle,
    pub filled: bool,
    pub has_per_point_colors: bool,
    pub edge_from_vertex_colors: bool,
}

/// GPU-accelerated scatter3 plot for MATLAB semantics.
#[derive(Debug, Clone)]
pub struct Scatter3Plot {
    /// Point positions in 3D space.
    pub points: Vec<Vec3>,
    /// Per-point RGBA colors.
    pub colors: Vec<Vec4>,
    /// Marker size in pixels.
    pub point_size: f32,
    /// Optional per-point marker sizes.
    pub point_sizes: Option<Vec<f32>>,
    /// Marker edge color.
    pub edge_color: Vec4,
    /// Marker edge thickness in pixels.
    pub edge_thickness: f32,
    /// Marker shape.
    pub marker_style: MarkerStyle,
    /// Whether marker faces are filled.
    pub filled: bool,
    /// Whether edge color should come from per-vertex colors.
    pub edge_color_from_vertex_colors: bool,
    /// Legend label.
    pub label: Option<String>,
    /// Visibility flag.
    pub visible: bool,
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    gpu_vertices: Option<GpuVertexBuffer>,
    gpu_point_count: Option<usize>,
    gpu_inputs: Option<Scatter3GpuInputs>,
    gpu_has_per_point_colors: bool,
}

impl Scatter3Plot {
    pub async fn export_scene_points(&self) -> Result<Vec<Vec3>, String> {
        if !self.points.is_empty() {
            return Ok(self.points.clone());
        }

        if let Some(inputs) = &self.gpu_inputs {
            let context = shared_wgpu_context().ok_or_else(|| {
                "scatter3 plot has GPU source data but no shared WGPU context is installed"
                    .to_string()
            })?;
            let len = inputs.len as usize;
            let x = readback_scalar_buffer_f64(
                &context.device,
                &context.queue,
                &inputs.x_buffer,
                len,
                inputs.scalar,
            )
            .await?;
            let y = readback_scalar_buffer_f64(
                &context.device,
                &context.queue,
                &inputs.y_buffer,
                len,
                inputs.scalar,
            )
            .await?;
            let z = readback_scalar_buffer_f64(
                &context.device,
                &context.queue,
                &inputs.z_buffer,
                len,
                inputs.scalar,
            )
            .await?;
            let points = x
                .into_iter()
                .zip(y)
                .zip(z)
                .map(|((x, y), z)| Vec3::new(x as f32, y as f32, z as f32))
                .collect();
            return Ok(points);
        }

        if self.gpu_vertices.is_some() {
            return Err(
                "scatter3 plot has GPU render vertices but no exportable source data".to_string(),
            );
        }

        Ok(Vec::new())
    }

    pub async fn export_scene_colors(&self, point_count: usize) -> Result<Vec<Vec4>, String> {
        if self.colors.len() == point_count {
            return Ok(self.colors.clone());
        }
        if self.colors.len() == 1 && !self.gpu_has_per_point_colors {
            return Ok(vec![self.colors[0]; point_count]);
        }

        if let Some(inputs) = &self.gpu_inputs {
            match &inputs.colors {
                ScatterColorBuffer::None => {
                    let color = self.colors.first().copied().unwrap_or(Vec4::ONE);
                    return Ok(vec![color; point_count]);
                }
                ScatterColorBuffer::Host(colors) => {
                    if colors.len() != point_count {
                        return Err(format!(
                            "scatter3 color count ({}) does not match point count ({point_count})",
                            colors.len()
                        ));
                    }
                    return Ok(colors
                        .iter()
                        .map(|color| Vec4::from_array(*color))
                        .collect());
                }
                ScatterColorBuffer::Gpu { buffer, components } => {
                    let context = shared_wgpu_context().ok_or_else(|| {
                        "scatter3 plot has GPU color data but no shared WGPU context is installed"
                            .to_string()
                    })?;
                    let components = *components as usize;
                    if components != 3 && components != 4 {
                        return Err(format!(
                            "scatter3 GPU color source has unsupported component count {components}"
                        ));
                    }
                    let value_count = point_count
                        .checked_mul(components)
                        .ok_or_else(|| "scatter3 GPU color source size overflowed".to_string())?;
                    let byte_len = value_count
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| {
                            "scatter3 GPU color source byte size overflowed".to_string()
                        })?;
                    let bytes =
                        copy_readback_bytes(&context.device, &context.queue, buffer, byte_len)
                            .await?;
                    let values: &[f32] = bytemuck::try_cast_slice(&bytes)
                        .map_err(|err| format!("scatter3 GPU color readback failed: {err}"))?;
                    if values.len() != value_count {
                        return Err(format!(
                            "scatter3 GPU color readback returned {} values, expected {value_count}",
                            values.len()
                        ));
                    }
                    let mut colors = Vec::with_capacity(point_count);
                    for chunk in values.chunks_exact(components) {
                        let alpha = if components == 4 { chunk[3] } else { 1.0 };
                        colors.push(Vec4::new(chunk[0], chunk[1], chunk[2], alpha));
                    }
                    return Ok(colors);
                }
            }
        }

        if self.gpu_has_per_point_colors {
            return Err(
                "scatter3 plot has GPU per-point colors but no exportable color source".to_string(),
            );
        }
        if self.colors.is_empty() {
            return Ok(vec![Vec4::ONE; point_count]);
        }
        Err(format!(
            "scatter3 color count ({}) does not match point count {point_count}",
            self.colors.len()
        ))
    }

    /// Create a new scatter3 plot. Colors default to a blue colormap.
    pub fn new(points: Vec<Vec3>) -> Result<Self, String> {
        let default_color = Vec4::new(0.1, 0.7, 0.3, 1.0);
        let colors = vec![default_color; points.len()];
        Ok(Self {
            points,
            colors,
            point_size: 8.0,
            point_sizes: None,
            edge_color: default_color,
            edge_thickness: 1.0,
            marker_style: MarkerStyle::Circle,
            filled: true,
            edge_color_from_vertex_colors: false,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            gpu_vertices: None,
            gpu_point_count: None,
            gpu_inputs: None,
            gpu_has_per_point_colors: false,
        })
    }

    /// Build a scatter plot directly from a GPU vertex buffer, bypassing CPU copies.
    pub fn from_gpu_buffer(
        buffer: GpuVertexBuffer,
        point_count: usize,
        style: Scatter3GpuStyle,
        point_size: f32,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            points: Vec::new(),
            colors: vec![style.color],
            point_size,
            point_sizes: None,
            edge_color: style.edge_color,
            edge_thickness: style.edge_thickness,
            marker_style: style.marker_style,
            filled: style.filled,
            edge_color_from_vertex_colors: style.edge_from_vertex_colors,
            label: None,
            visible: true,
            vertices: None,
            bounds: Some(bounds),
            gpu_vertices: Some(buffer),
            gpu_point_count: Some(point_count),
            gpu_inputs: None,
            gpu_has_per_point_colors: style.has_per_point_colors,
        }
    }

    pub fn with_gpu_source_inputs(mut self, inputs: Scatter3GpuInputs) -> Self {
        self.gpu_inputs = Some(inputs);
        self
    }

    /// Override all point colors with a single RGBA value.
    pub fn with_color(mut self, color: Vec4) -> Self {
        self.colors = vec![color; self.points.len()];
        self.vertices = None;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
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
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
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
        self.point_sizes = None;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
        self
    }

    pub fn set_marker_style(&mut self, style: MarkerStyle) {
        self.marker_style = style;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
    }

    pub fn set_filled(&mut self, filled: bool) {
        self.filled = filled;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
    }

    pub fn set_edge_color(&mut self, color: Vec4) {
        self.edge_color = color;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
    }

    pub fn set_edge_thickness(&mut self, px: f32) {
        self.edge_thickness = px.max(0.0);
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
    }

    pub fn set_edge_color_from_vertex(&mut self, enabled: bool) {
        self.edge_color_from_vertex_colors = enabled;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self.gpu_has_per_point_colors = false;
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
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
        self
    }

    /// Supply per-point sizes in pixels.
    pub fn set_point_sizes(&mut self, sizes: Vec<f32>) {
        self.point_sizes = Some(sizes);
        self.vertices = None;
        self.gpu_vertices = None;
        self.gpu_point_count = None;
        self.gpu_inputs = None;
        self.gpu_has_per_point_colors = false;
    }

    fn ensure_vertices(&mut self) {
        if self.vertices.is_none() {
            let mut verts = vertex_utils::create_point_cloud(&self.points, &self.colors);
            if let Some(sizes) = self.point_sizes.as_ref() {
                for (idx, vertex) in verts.iter_mut().enumerate() {
                    let size = sizes.get(idx).copied().unwrap_or(self.point_size);
                    vertex.normal[2] = size;
                }
            } else {
                for vertex in &mut verts {
                    vertex.normal[2] = self.point_size;
                }
            }
            self.vertices = Some(verts);
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
            + self
                .point_sizes
                .as_ref()
                .map(|sizes| sizes.len() * std::mem::size_of::<f32>())
                .unwrap_or(0)
            + gpu_bytes
    }

    /// Generate render data for the renderer.
    pub fn render_data(&mut self) -> RenderData {
        let bounds = self.bounds();
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

        let is_multi_color = if self.gpu_vertices.is_some() {
            self.gpu_has_per_point_colors || self.colors.len() > 1
        } else if vertices.is_empty() {
            false
        } else {
            let first = vertices[0].color;
            vertices.iter().any(|v| v.color != first)
        };
        let has_vertex_colors = if self.gpu_vertices.is_some() {
            self.gpu_has_per_point_colors
        } else {
            self.colors.len() > 1
        };
        let use_vertex_edge_color = self.edge_color_from_vertex_colors && has_vertex_colors;
        let mut material = Material {
            albedo: self.colors.first().copied().unwrap_or(Vec4::ONE),
            roughness: self.edge_thickness,
            metallic: match self.marker_style {
                MarkerStyle::Circle => 0.0,
                MarkerStyle::Square => 1.0,
                MarkerStyle::Triangle => 2.0,
                MarkerStyle::Diamond => 3.0,
                MarkerStyle::Plus => 4.0,
                MarkerStyle::Cross => 5.0,
                MarkerStyle::Star => 6.0,
                MarkerStyle::Hexagon => 7.0,
            },
            emissive: self.edge_color,
            alpha_mode: crate::core::scene::AlphaMode::Blend,
            double_sided: true,
        };
        if is_multi_color {
            material.albedo.w = 0.0;
        } else if self.filled {
            material.albedo.w = 1.0;
        }
        material.emissive.w = if use_vertex_edge_color { 0.0 } else { 1.0 };

        RenderData {
            pipeline_type: PipelineType::Scatter3,
            vertices,
            indices: None,
            gpu_vertices: self.gpu_vertices.clone(),
            bounds: Some(bounds),
            material,
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

    #[test]
    fn scatter3_marker_style_encodes_material_shape_channel() {
        let points = vec![Vec3::new(0.0, 0.0, 0.0)];
        let mut cloud = Scatter3Plot::new(points).unwrap();
        cloud.set_marker_style(MarkerStyle::Diamond);
        let render_data = cloud.render_data();
        assert_eq!(render_data.material.metallic, 3.0);
    }

    #[test]
    fn scatter3_default_material_uses_plot_color_not_white_override() {
        let points = vec![Vec3::new(0.0, 0.0, 0.0)];
        let mut cloud = Scatter3Plot::new(points).unwrap();
        let render_data = cloud.render_data();
        assert_ne!(render_data.material.albedo.truncate(), Vec4::ONE.truncate());
    }
}
