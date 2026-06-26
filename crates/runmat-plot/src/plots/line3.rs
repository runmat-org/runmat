use crate::context::shared_wgpu_context;
use crate::core::{
    BoundingBox, DrawCall, GpuPackContext, GpuVertexBuffer, Material, PipelineType, RenderData,
    Vertex,
};
use crate::geometry::stroke3d::{
    create_line_vertices_dashed, tessellate_polyline_tube, StrokeCap3D, StrokeStyle3D,
};
use crate::gpu::line3::{Line3GpuInputs, Line3GpuParams};
use crate::gpu::util::readback_scalar_buffer_f64;
use glam::{Vec3, Vec4};
use log::warn;

const POINTS_TO_PX: f32 = 96.0 / 72.0;
const TUBE_RADIAL_SEGMENTS: usize = 8;

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
    gpu_line_inputs: Option<Line3GpuInputs>,
}

impl Line3Plot {
    #[inline]
    fn line_width_px(&self) -> f32 {
        (self.line_width.max(0.1)) * POINTS_TO_PX
    }

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
            gpu_line_inputs: None,
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
            gpu_line_inputs: None,
        }
    }

    pub fn from_gpu_xyz(
        inputs: Line3GpuInputs,
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
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_line_inputs: Some(inputs),
        }
    }

    pub fn with_gpu_xyz_inputs(mut self, inputs: Line3GpuInputs, bounds: BoundingBox) -> Self {
        self.gpu_line_inputs = Some(inputs);
        self.bounds = Some(bounds);
        self
    }

    pub async fn export_scene_xyz_data(&self) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
        if !self.x_data.is_empty()
            && self.x_data.len() == self.y_data.len()
            && self.x_data.len() == self.z_data.len()
        {
            return Ok((
                self.x_data.clone(),
                self.y_data.clone(),
                self.z_data.clone(),
            ));
        }
        if !self.x_data.is_empty() || !self.y_data.is_empty() || !self.z_data.is_empty() {
            return Err(format!(
                "plot3 line has incomplete CPU data: x={}, y={}, z={}",
                self.x_data.len(),
                self.y_data.len(),
                self.z_data.len()
            ));
        }

        if let Some(inputs) = &self.gpu_line_inputs {
            let context = shared_wgpu_context().ok_or_else(|| {
                "plot3 line has GPU source data but no shared WGPU context is installed".to_string()
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
            return Ok((x, y, z));
        }

        if self.gpu_vertices.is_some() {
            return Err(
                "plot3 line has GPU render vertices but no exportable source data".to_string(),
            );
        }

        Ok((Vec::new(), Vec::new(), Vec::new()))
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
                vertex.normal[2] = (self.line_width_px().max(1.0) * 4.0).max(6.0);
                vec![vertex]
            } else if self.line_width_px() > 1.0 {
                // No viewport hint: interpret width in data units for legacy/non-viewport paths.
                let fallback_half_width_data = self.line_width_px() * 0.5;
                tessellate_polyline_tube(
                    &points,
                    self.color,
                    StrokeStyle3D::new(
                        fallback_half_width_data,
                        self.line_style,
                        StrokeCap3D::Square,
                    ),
                    TUBE_RADIAL_SEGMENTS,
                )
            } else {
                create_line_vertices_dashed(&points, self.color, self.line_style)
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
        let width_px = self.line_width_px();
        let thick = width_px > 1.0 && !single_point;
        let indices = if self.gpu_vertices.is_none() && thick {
            Some((0..vertex_count as u32).collect::<Vec<u32>>())
        } else {
            None
        };
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
            indices,
            gpu_vertices: self.gpu_vertices.clone(),
            bounds: Some(self.bounds()),
            material: Material {
                albedo: self.color,
                roughness: width_px.max(0.5),
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

    fn pack_gpu_vertices_if_needed(
        &mut self,
        gpu: &GpuPackContext<'_>,
        viewport_px: (u32, u32),
    ) -> Result<(), String> {
        if self.gpu_vertices.is_some() {
            return Ok(());
        }
        let Some(inputs) = self.gpu_line_inputs.as_ref() else {
            return Ok(());
        };
        let bounds = self
            .bounds
            .as_ref()
            .ok_or_else(|| "plot3: missing bounds for GPU packing".to_string())?;
        let width_px = self.line_width_px();
        let thick_px = width_px > 1.0;
        let data_per_px = crate::core::data_units_per_px_3d(bounds, viewport_px);
        let half_width_data = if thick_px {
            (width_px * 0.5) * data_per_px
        } else {
            0.0
        };
        let packed = crate::gpu::line3::pack_vertices_from_xyz(
            gpu.device,
            gpu.queue,
            inputs,
            &Line3GpuParams {
                color: self.color,
                half_width_data,
                thick: thick_px,
                line_style: self.line_style,
            },
        )?;
        self.gpu_vertex_count =
            Some((inputs.len.saturating_sub(1) as usize) * if thick_px { 6 } else { 2 });
        self.gpu_vertices = Some(packed);
        Ok(())
    }

    pub fn render_data_with_viewport_gpu(
        &mut self,
        viewport_px: Option<(u32, u32)>,
        view_angles_deg: Option<(f32, f32)>,
        gpu: Option<&GpuPackContext<'_>>,
    ) -> RenderData {
        let can_gpu_pack = self.line_width_px() <= 1.0;
        if can_gpu_pack && self.gpu_line_inputs.is_some() && self.gpu_vertices.is_none() {
            if let (Some(gpu), Some(vp)) = (gpu, viewport_px) {
                if let Err(err) = self.pack_gpu_vertices_if_needed(gpu, vp) {
                    warn!("plot3 gpu pack failed: {err}");
                }
            }
        }
        self.render_data_with_viewport_and_view(viewport_px, view_angles_deg)
    }

    pub fn render_data_with_viewport(&mut self, viewport_px: Option<(u32, u32)>) -> RenderData {
        self.render_data_with_viewport_and_view(viewport_px, None)
    }

    pub fn render_data_with_viewport_and_view(
        &mut self,
        viewport_px: Option<(u32, u32)>,
        view_angles_deg: Option<(f32, f32)>,
    ) -> RenderData {
        if self.gpu_vertices.is_some() {
            return self.render_data();
        }

        let single_point = self.x_data.len() == 1;
        let width_px = self.line_width_px();
        let (vertices, vertex_count, pipeline) = if !single_point && width_px > 1.0 {
            let Some(vp) = viewport_px else {
                return self.render_data();
            };
            let points: Vec<Vec3> = self
                .x_data
                .iter()
                .zip(self.y_data.iter())
                .zip(self.z_data.iter())
                .map(|((&x, &y), &z)| Vec3::new(x as f32, y as f32, z as f32))
                .collect();
            let bounds = self.bounds();
            let data_per_px =
                crate::core::data_units_per_px_3d_camera(&bounds, vp, view_angles_deg);
            let half_width_data = (width_px * 0.5) * data_per_px;
            let tris = tessellate_polyline_tube(
                &points,
                self.color,
                StrokeStyle3D::new(half_width_data, self.line_style, StrokeCap3D::Square),
                TUBE_RADIAL_SEGMENTS,
            );
            let count = tris.len();
            (tris, count, PipelineType::Triangles)
        } else {
            let verts = self.generate_vertices().clone();
            let count = verts.len();
            let pipeline = if single_point {
                PipelineType::Scatter3
            } else {
                PipelineType::Lines
            };
            (verts, count, pipeline)
        };

        let indices = if pipeline == PipelineType::Triangles {
            Some((0..vertex_count as u32).collect::<Vec<u32>>())
        } else {
            None
        };

        RenderData {
            pipeline_type: pipeline,
            vertices,
            indices,
            gpu_vertices: None,
            bounds: Some(self.bounds()),
            material: Material {
                albedo: self.color,
                roughness: width_px.max(0.5),
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
