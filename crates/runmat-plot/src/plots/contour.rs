//! Contour plot implementation (iso-lines on a surface or base plane).

use crate::context::shared_wgpu_context;
use crate::core::{
    vertex_utils, BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData,
    Vertex,
};
use crate::gpu::util::copy_readback_bytes;
use bytemuck::cast_slice;
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub struct ContourPlot {
    pub base_z: f32,
    pub force_3d: bool,
    pub label: Option<String>,
    pub visible: bool,
    /// The data range the contour colors span, for the colorbar.
    pub value_range: Option<(f64, f64)>,
    /// Each CPU vertex's position in the colormap (0 to 1), so a new colormap can recolor it.
    colormap_positions: Option<Vec<f32>>,
    pub line_width: f32,
    vertices: Option<Vec<Vertex>>,
    gpu_vertices: Option<GpuVertexBuffer>,
    vertex_count: usize,
    bounds: Option<BoundingBox>,
}

impl ContourPlot {
    pub async fn export_scene_vertices(&self) -> Result<Vec<Vertex>, String> {
        if let Some(vertices) = &self.vertices {
            return Ok(vertices.clone());
        }

        if let Some(gpu_vertices) = &self.gpu_vertices {
            let context = shared_wgpu_context().ok_or_else(|| {
                "contour plot has GPU vertices but no shared WGPU context is installed".to_string()
            })?;
            let vertex_count = self.vertex_count.min(gpu_vertices.vertex_count);
            let byte_len = vertex_count * std::mem::size_of::<Vertex>();
            let bytes = copy_readback_bytes(
                &context.access,
                &crate::gpu::GpuInputBuffer::from(gpu_vertices.buffer.clone()),
                byte_len,
            )
            .await?;
            let vertices: &[Vertex] = cast_slice(&bytes);
            return Ok(vertices.to_vec());
        }

        Ok(Vec::new())
    }

    /// Create a contour plot from CPU vertices.
    pub fn from_vertices(vertices: Vec<Vertex>, base_z: f32, bounds: BoundingBox) -> Self {
        Self {
            base_z,
            force_3d: false,
            label: None,
            visible: true,
            value_range: None,
            colormap_positions: None,
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
            value_range: None,
            colormap_positions: None,
            line_width: 1.0,
            vertex_count,
            vertices: None,
            gpu_vertices: Some(buffer),
            bounds: Some(bounds),
        }
    }

    /// Record each vertex's colormap position (0 to 1) so the colors can follow the colormap.
    pub fn with_colormap_positions(mut self, positions: Vec<f32>) -> Self {
        let matches = self
            .vertices
            .as_ref()
            .is_some_and(|v| v.len() == positions.len());
        self.colormap_positions = matches.then_some(positions);
        self
    }

    /// Recolor the vertices through `cmap`; plots without recorded positions keep their colors.
    pub fn recolor_with_colormap(&mut self, cmap: &crate::plots::surface::ColorMap) {
        let (Some(vertices), Some(positions)) =
            (self.vertices.as_mut(), self.colormap_positions.as_ref())
        else {
            return;
        };
        for (vertex, &t) in vertices.iter_mut().zip(positions) {
            let rgb = cmap.map_value(t);
            vertex.color = [rgb.x, rgb.y, rgb.z, vertex.color[3]];
        }
    }

    pub fn with_value_range(mut self, range: Option<(f64, f64)>) -> Self {
        self.value_range = range;
        self
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

    /// The color the legend shows for this contour: its line color when set (every vertex has
    /// it), otherwise a level from the middle of its colormap. Mid-gray when only GPU vertices exist.
    pub fn legend_color(&self) -> Vec4 {
        self.vertices
            .as_ref()
            .and_then(|v| v.get(v.len() / 2))
            .map(|v| Vec4::from_array(v.color))
            .unwrap_or(Vec4::new(0.5, 0.5, 0.5, 1.0))
    }

    pub fn cpu_vertices(&self) -> Option<&[Vertex]> {
        self.vertices.as_deref()
    }

    /// Preserve decorated auto-limit extents without generating a second vertex buffer.
    /// Uses exactly the same segment corners as the rendering kernels.
    pub(crate) fn prepared_bounds(&self, viewport: Option<(u32, u32)>) -> BoundingBox {
        let mut bounds = self.bounds();
        if self.gpu_vertices.is_some() || self.line_width <= 1.0 {
            return bounds;
        }
        let width = viewport
            .map(|vp| self.line_width.max(0.1) * crate::core::data_units_per_px(&bounds, vp))
            .unwrap_or(self.line_width);
        for segment in self.vertices.as_deref().unwrap_or_default().chunks_exact(2) {
            let start = Vec3::from_array(segment[0].position);
            let end = Vec3::from_array(segment[1].position);
            let corners = if self.is_3d() {
                let half = if viewport.is_some() {
                    width * 0.5
                } else {
                    self.line_width.max(0.5) * 0.01
                };
                thick_segment_3d_corners(start, end, half)
            } else {
                vertex_utils::extruded_segment_corners(
                    Vec3::new(start.x, start.y, 0.0),
                    Vec3::new(end.x, end.y, 0.0),
                    width,
                )
            };
            for corner in corners {
                bounds.expand(corner);
            }
        }
        bounds
    }

    pub fn render_data_with_viewport(&mut self, viewport_px: Option<(u32, u32)>) -> RenderData {
        self.render_data_with_resolved_viewport(crate::core::ResolvedViewport {
            pixels: viewport_px,
            view: None,
        })
    }

    pub(crate) fn render_data_with_resolved_viewport(
        &mut self,
        viewport: crate::core::ResolvedViewport,
    ) -> RenderData {
        let viewport_px = viewport.pixels;
        if self.gpu_vertices.is_some() {
            return self.render_data();
        }

        let bounds = self.bounds();
        let (vertices, vertex_count, pipeline_type, render_bounds) = if self.line_width > 1.0 {
            let Some(viewport_px) = viewport_px else {
                return self.render_data();
            };
            let data_per_px = crate::core::data_units_per_px(&viewport.bounds(bounds), viewport_px);
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

    pub(crate) fn retained_snapshot_bytes(&self) -> Option<usize> {
        let mut total = crate::plots::clone_budget::optional_string(self.label.as_ref());
        crate::plots::clone_budget::add(
            &mut total,
            crate::plots::clone_budget::vec::<Vertex>(
                self.vertices.as_ref().map_or(0, Vec::capacity),
            )?,
        )?;
        Some(total)
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
    let [v0, v1, v2, v3] = thick_segment_3d_corners(start, end, half_width);
    vec![
        Vertex::new(v0, color),
        Vertex::new(v1, color),
        Vertex::new(v2, color),
        Vertex::new(v0, color),
        Vertex::new(v2, color),
        Vertex::new(v3, color),
    ]
}

fn thick_segment_3d_corners(start: Vec3, end: Vec3, half_width: f32) -> [Vec3; 4] {
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
    [v0, v1, v2, v3]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vertex(x: f32, y: f32, z: f32) -> Vertex {
        Vertex::new(Vec3::new(x, y, z), Vec4::ONE)
    }

    #[test]
    fn prepared_extents_match_rendered_2d_and_3d_decoration_bounds() {
        for is_3d in [false, true] {
            for viewport in [None, Some((600, 400))] {
                let start = Vec3::new(0., 0., if is_3d { 2. } else { 0. });
                let end = Vec3::new(4., 3., if is_3d { 4. } else { 0. });
                let mut plot = ContourPlot::from_vertices(
                    vec![Vertex::new(start, Vec4::ONE), Vertex::new(end, Vec4::ONE)],
                    0.,
                    BoundingBox::new(start, end),
                )
                .with_line_width(5.);
                plot.force_3d = is_3d;
                let prepared = plot.prepared_bounds(viewport);
                let rendered = plot.render_data_with_viewport(viewport).bounds.unwrap();
                assert_eq!(prepared.min, rendered.min);
                assert_eq!(prepared.max, rendered.max);
            }
        }
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
