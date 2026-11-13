//! Error bar plot implementation.

use crate::context::shared_wgpu_context;
use crate::core::{
    marker_shape_code, AlphaMode, BoundingBox, DrawCall, GpuPackContext, GpuVertexBuffer, Material,
    PipelineType, RenderData, Vertex,
};
use crate::gpu::errorbar::{ErrorBarGpuInputs, ErrorBarGpuParams};
use crate::gpu::util::readback_scalar_buffer_f64;
use crate::plots::line::{LineMarkerAppearance, LineStyle};
use crate::plots::stroke;
use glam::{Vec3, Vec4};
use log::warn;

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
    gpu_inputs: Option<ErrorBarGpuInputs>,
    gpu_cap_scale: Option<f32>,
    marker_vertices: Option<Vec<Vertex>>,
    marker_gpu_vertices: Option<GpuVertexBuffer>,
    marker_dirty: bool,
}

impl ErrorBar {
    pub fn new_vertical(
        x: Vec<f64>,
        y: Vec<f64>,
        mut y_neg: Vec<f64>,
        mut y_pos: Vec<f64>,
    ) -> Result<Self, String> {
        let n = x.len();
        if y_neg.is_empty() {
            y_neg.resize(n, 0.0);
        }
        if y_pos.is_empty() {
            y_pos.resize(n, 0.0);
        }
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
            marker: None,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_bounds: None,
            gpu_inputs: None,
            gpu_cap_scale: None,
            marker_vertices: None,
            marker_gpu_vertices: None,
            marker_dirty: true,
        })
    }

    pub fn new_both(
        x: Vec<f64>,
        y: Vec<f64>,
        mut x_neg: Vec<f64>,
        mut x_pos: Vec<f64>,
        y_neg: Vec<f64>,
        y_pos: Vec<f64>,
    ) -> Result<Self, String> {
        let n = x.len();
        if x_neg.is_empty() {
            x_neg.resize(n, 0.0);
        }
        if x_pos.is_empty() {
            x_pos.resize(n, 0.0);
        }
        if n == 0 || y.len() != n || x_neg.len() != n || x_pos.len() != n {
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
            gpu_inputs: None,
            gpu_cap_scale: None,
            marker_vertices: None,
            marker_gpu_vertices: None,
            marker_dirty: true,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_gpu_inputs(
        color: Vec4,
        line_width: f32,
        line_style: LineStyle,
        cap_size: f32,
        orientation: ErrorBarOrientation,
        inputs: ErrorBarGpuInputs,
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
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_bounds: Some(bounds),
            gpu_inputs: Some(inputs),
            gpu_cap_scale: None,
            marker_vertices: None,
            marker_gpu_vertices: None,
            marker_dirty: true,
        }
    }

    pub async fn export_scene_data(
        &self,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        if !self.x.is_empty()
            && self.x.len() == self.y.len()
            && self.x.len() == self.y_neg.len()
            && self.x.len() == self.y_pos.len()
            && self.x.len() == self.x_neg.len()
            && self.x.len() == self.x_pos.len()
        {
            return Ok((
                self.x.clone(),
                self.y.clone(),
                self.y_neg.clone(),
                self.y_pos.clone(),
                self.x_neg.clone(),
                self.x_pos.clone(),
            ));
        }
        if !self.x.is_empty()
            || !self.y.is_empty()
            || !self.y_neg.is_empty()
            || !self.y_pos.is_empty()
            || !self.x_neg.is_empty()
            || !self.x_pos.is_empty()
        {
            return Err(format!(
                "errorbar plot has partial CPU source data: x={}, y={}, y_neg={}, y_pos={}, x_neg={}, x_pos={}",
                self.x.len(),
                self.y.len(),
                self.y_neg.len(),
                self.y_pos.len(),
                self.x_neg.len(),
                self.x_pos.len()
            ));
        }

        if let Some(inputs) = &self.gpu_inputs {
            let context = shared_wgpu_context().ok_or_else(|| {
                "errorbar plot has GPU source data but no shared WGPU context is installed"
                    .to_string()
            })?;
            let len = inputs.len as usize;
            let x =
                readback_scalar_buffer_f64(&context.access, &inputs.x_buffer, len, inputs.scalar)
                    .await?;
            let y =
                readback_scalar_buffer_f64(&context.access, &inputs.y_buffer, len, inputs.scalar)
                    .await?;
            let y_neg = readback_scalar_buffer_f64(
                &context.access,
                &inputs.y_neg_buffer,
                len,
                inputs.scalar,
            )
            .await?;
            let y_pos = readback_scalar_buffer_f64(
                &context.access,
                &inputs.y_pos_buffer,
                len,
                inputs.scalar,
            )
            .await?;
            let x_neg = if let Some(buffer) = &inputs.x_neg_buffer {
                readback_scalar_buffer_f64(&context.access, buffer, len, inputs.scalar).await?
            } else {
                vec![0.0; len]
            };
            let x_pos = if let Some(buffer) = &inputs.x_pos_buffer {
                readback_scalar_buffer_f64(&context.access, buffer, len, inputs.scalar).await?
            } else {
                vec![0.0; len]
            };
            return Ok((x, y, y_neg, y_pos, x_neg, x_pos));
        }

        if self.gpu_vertices.is_some() {
            return Err(
                "errorbar plot has GPU render vertices but no exportable source data".to_string(),
            );
        }

        Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
    }

    fn cap_half_width_data(&self, bounds: &BoundingBox, viewport_px: Option<(u32, u32)>) -> f32 {
        match viewport_px {
            Some(vp) => {
                let data_per_px = crate::core::data_units_per_px(bounds, vp);
                self.cap_size.max(0.0) * 0.5 * data_per_px
            }
            None => self.cap_size.max(0.0) * 0.5,
        }
    }

    fn generate_line_vertices_with_cap_half(&self, cap_half_data: f32) -> Vec<Vertex> {
        let identity = |x: f64, y: f64| {
            let p = [x as f32 as f64, y as f32 as f64];
            (p[0].is_finite() && p[1].is_finite()).then_some(p)
        };
        let mut segments = self.bar_segments(identity, cap_half_data as f64);
        segments.extend(self.data_line_segments(identity, &[]));
        segments
            .into_iter()
            .flat_map(|(a, b)| {
                [a, b].map(|p| Vertex::new(Vec3::new(p[0] as f32, p[1] as f32, 0.0), self.color))
            })
            .collect()
    }

    /// Bars and caps as segments in the space `map` projects into: data values, or viewport
    /// pixels. Caps extend `cap_half` either side of a bar end, in that same space.
    fn bar_segments(
        &self,
        map: impl Fn(f64, f64) -> Option<[f64; 2]>,
        cap_half: f64,
    ) -> Vec<([f64; 2], [f64; 2])> {
        let mut segments = Vec::new();
        let offset = |p: [f64; 2], dx: f64, dy: f64| [p[0] + dx, p[1] + dy];
        for i in 0..self.x.len() {
            if !self.x[i].is_finite() || !self.y[i].is_finite() {
                continue;
            }
            if matches!(
                self.orientation,
                ErrorBarOrientation::Vertical | ErrorBarOrientation::Both
            ) {
                let ends = (
                    map(self.x[i], self.y[i] - self.y_neg[i]),
                    map(self.x[i], self.y[i] + self.y_pos[i]),
                );
                if let (Some(lo), Some(hi)) = ends {
                    segments.push((lo, hi));
                    if cap_half > 0.0 {
                        for end in [lo, hi] {
                            segments
                                .push((offset(end, -cap_half, 0.0), offset(end, cap_half, 0.0)));
                        }
                    }
                }
            }
            if matches!(
                self.orientation,
                ErrorBarOrientation::Horizontal | ErrorBarOrientation::Both
            ) {
                let ends = (
                    map(self.x[i] - self.x_neg[i], self.y[i]),
                    map(self.x[i] + self.x_pos[i], self.y[i]),
                );
                if let (Some(lo), Some(hi)) = ends {
                    segments.push((lo, hi));
                    if cap_half > 0.0 {
                        for end in [lo, hi] {
                            segments
                                .push((offset(end, 0.0, -cap_half), offset(end, 0.0, cap_half)));
                        }
                    }
                }
            }
        }
        segments
    }

    /// The line through the data points, in the space `map` projects into, cut into dashes
    /// of `pattern` (on, off, ... lengths in that space) when the style is not solid. Error
    /// bars are always solid; `LineStyle` is the style of this line, as in MATLAB.
    fn data_line_segments(
        &self,
        map: impl Fn(f64, f64) -> Option<[f64; 2]>,
        pattern: &[f64],
    ) -> Vec<([f64; 2], [f64; 2])> {
        let mut segments = Vec::new();
        if matches!(self.line_style, LineStyle::None) {
            return segments;
        }
        let points: Vec<Option<[f64; 2]>> = self
            .x
            .iter()
            .zip(&self.y)
            .map(|(&x, &y)| {
                (x.is_finite() && y.is_finite())
                    .then(|| map(x, y))
                    .flatten()
            })
            .collect();
        let dashed = !matches!(self.line_style, LineStyle::Solid) && !pattern.is_empty();
        let (mut phase, mut remaining) = (0usize, pattern.first().copied().unwrap_or(0.0));
        for pair in points.windows(2) {
            let (Some(a), Some(b)) = (pair[0], pair[1]) else {
                continue;
            };
            if !dashed {
                segments.push((a, b));
                continue;
            }
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let at = |t: f64| {
                [
                    a[0] + (b[0] - a[0]) * t / length,
                    a[1] + (b[1] - a[1]) * t / length,
                ]
            };
            let mut t = 0.0;
            while length > 0.0 && t < length {
                let step = remaining.min(length - t);
                if phase % 2 == 0 {
                    segments.push((at(t), at(t + step)));
                }
                t += step;
                remaining -= step;
                if remaining <= 1.0e-9 {
                    phase = (phase + 1) % pattern.len();
                    remaining = pattern[phase];
                }
            }
        }
        segments
    }

    /// Dash pattern in pixels for the data line, scaled with its width.
    fn dash_pattern_px(&self) -> Vec<f64> {
        let w = self.line_width.max(1.0) as f64;
        match self.line_style {
            LineStyle::Dashed => vec![6.0 + 2.0 * w, 4.0 + w],
            LineStyle::Dotted => vec![w, 2.0 + 1.5 * w],
            LineStyle::DashDot => vec![6.0 + 2.0 * w, 3.0 + w, w, 3.0 + w],
            LineStyle::Solid | LineStyle::None => Vec::new(),
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
            let bounds = self.bounds();
            let cap_half_data = self.cap_half_width_data(&bounds, None);
            let verts = self.generate_line_vertices_with_cap_half(cap_half_data);
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
            // Caps are drawn in viewport pixels, so they add nothing to the data bounds.
            if !min.x.is_finite() {
                min = Vec3::ZERO;
                max = Vec3::ZERO;
            }
            self.bounds = Some(BoundingBox::new(min, max));
        }
        self.bounds.unwrap()
    }

    pub fn render_data(&mut self) -> RenderData {
        let bounds = self.bounds();
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
            bounds: Some(bounds),
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

    fn orientation_code(&self) -> u32 {
        match self.orientation {
            ErrorBarOrientation::Vertical => 0,
            ErrorBarOrientation::Horizontal => 1,
            ErrorBarOrientation::Both => 2,
        }
    }

    fn pack_gpu_vertices_if_needed(
        &mut self,
        gpu: &GpuPackContext<'_>,
        viewport: crate::core::ResolvedViewport,
        viewport_px: (u32, u32),
    ) -> Result<(), String> {
        let Some(inputs) = self.gpu_inputs.as_ref() else {
            return Ok(());
        };
        let bounds = self
            .gpu_bounds
            .ok_or_else(|| "errorbar: missing GPU bounds".to_string())?;
        let data_per_px = crate::core::data_units_per_px(&viewport.bounds(bounds), viewport_px);
        let cap_size_data = self.cap_size.max(0.0) * data_per_px;
        if self.gpu_vertices.is_some() && self.gpu_cap_scale == Some(cap_size_data) {
            return Ok(());
        }
        let orientation = self.orientation_code();
        let packed = crate::gpu::errorbar::pack_vertical_vertices(
            gpu.emission,
            inputs,
            &ErrorBarGpuParams {
                color: self.color,
                cap_size_data,
                line_style: self.line_style,
                orientation,
            },
        )?;
        let segments_per_point = match orientation {
            0 | 1 => 6usize,
            _ => 12usize,
        };
        self.gpu_vertex_count = Some(inputs.len as usize * segments_per_point);
        self.gpu_vertices = Some(packed);
        self.gpu_cap_scale = Some(cap_size_data);
        Ok(())
    }

    pub fn render_data_with_viewport_gpu(
        &mut self,
        viewport_px: Option<(u32, u32)>,
        view_bounds: Option<(f64, f64, f64, f64)>,
        log_axes: (bool, bool),
        gpu: Option<&GpuPackContext<'_>>,
    ) -> RenderData {
        if self.gpu_inputs.is_some() {
            if let (Some(gpu), Some(vp)) = (gpu, viewport_px) {
                let viewport = crate::core::ResolvedViewport {
                    pixels: viewport_px,
                    view: view_bounds,
                };
                if let Err(err) = self.pack_gpu_vertices_if_needed(gpu, viewport, vp) {
                    warn!("errorbar gpu pack failed: {err}");
                }
            }
        }
        self.render_data_on_axes(viewport_px, view_bounds, log_axes)
    }

    pub fn render_data_with_viewport(&mut self, viewport_px: Option<(u32, u32)>) -> RenderData {
        self.render_data_on_axes(viewport_px, None, (false, false))
    }

    /// Render data sized in pixels for axes with the given visible bounds and log scales.
    /// Bars, caps and widths are built in viewport pixels, so caps are `cap_size` pixels wide.
    pub fn render_data_on_axes(
        &mut self,
        viewport_px: Option<(u32, u32)>,
        view_bounds: Option<(f64, f64, f64, f64)>,
        log_axes: (bool, bool),
    ) -> RenderData {
        if self.gpu_vertices.is_some() {
            return self.render_data();
        }

        let bounds = self.bounds();
        let Some(viewport_px) = viewport_px else {
            return self.render_data();
        };
        let map = stroke::PixelMap::new(
            stroke::stroke_bounds(bounds, view_bounds),
            viewport_px,
            log_axes,
            &self.x,
            &self.y,
        );
        let mut px_segments =
            self.bar_segments(|x, y| map.to_px(x, y), self.cap_size.max(0.0) as f64 * 0.5);
        px_segments
            .extend(self.data_line_segments(|x, y| map.to_px(x, y), &self.dash_pattern_px()));
        let segments: Vec<stroke::PixelSegment> = px_segments
            .into_iter()
            .map(|(a, b)| (a, b, self.color))
            .collect();
        let (vertices, pipeline_type) =
            stroke::segment_geometry(&map, &segments, self.line_width.max(0.1));
        let vertex_count = vertices.len();
        RenderData {
            pipeline_type,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: Some(bounds),
            material: Material {
                albedo: self.color,
                roughness: self.line_width.max(0.0),
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

    pub(crate) fn retained_snapshot_bytes(&self) -> Option<usize> {
        let mut total = self.label.as_ref().map_or(0, String::capacity);
        for values in [
            &self.x,
            &self.y,
            &self.y_neg,
            &self.y_pos,
            &self.x_neg,
            &self.x_pos,
        ] {
            crate::plots::clone_budget::add(
                &mut total,
                crate::plots::clone_budget::vec::<f64>(values.capacity())?,
            )?;
        }
        for vertices in [&self.vertices, &self.marker_vertices]
            .into_iter()
            .flatten()
        {
            crate::plots::clone_budget::add(
                &mut total,
                crate::plots::clone_budget::vec::<Vertex>(vertices.capacity())?,
            )?;
        }
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_are_sized_from_display_limits_for_thin_and_thick_lines() {
        for width in [1., 3.] {
            let mut plot = ErrorBar::new_vertical(vec![0.], vec![0.], vec![1.], vec![1.]).unwrap();
            plot.line_width = width;
            let data = plot.render_data_on_axes(
                Some((100, 100)),
                Some((-10., 10., -10., 10.)),
                (false, false),
            );
            let min = data
                .vertices
                .iter()
                .map(|v| v.position[0])
                .fold(f32::INFINITY, f32::min);
            let max = data
                .vertices
                .iter()
                .map(|v| v.position[0])
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (max - min - 1.2).abs() < 1e-5,
                "width={width}: {min}..{max}"
            );
        }
    }

    #[test]
    fn errorbar_bounds_fit_bars_not_caps() {
        // Caps are drawn in pixels, so they add nothing to the data bounds.
        let mut plot = ErrorBar::new_vertical(
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        )
        .unwrap()
        .with_style(Vec4::ONE, 1.0, LineStyle::Solid, 10.0);
        let bounds = plot.bounds();
        assert_eq!((bounds.min.x, bounds.max.x), (0.0, 1.0));
        assert_eq!((bounds.min.y, bounds.max.y), (0.5, 2.5));
    }

    #[test]
    fn errorbar_cap_padding_keeps_positive_bounds_positive() {
        let mut plot = ErrorBar::new_vertical(
            vec![1.0, 10.0],
            vec![1.0, 2.0],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        )
        .unwrap()
        .with_style(Vec4::ONE, 1.0, LineStyle::Solid, 10.0);
        let bounds = plot.bounds();
        assert_eq!(bounds.min.x, 1.0);
        assert!(bounds.min.y > 0.0);
    }

    #[test]
    fn thick_errorbar_use_viewport_aware_triangles() {
        let mut plot = ErrorBar::new_vertical(
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        )
        .unwrap()
        .with_style(Vec4::ONE, 2.0, LineStyle::Solid, 6.0);
        let render = plot.render_data_with_viewport(Some((600, 400)));
        assert_eq!(render.pipeline_type, PipelineType::Triangles);
        assert!(!render.vertices.is_empty());
    }

    #[test]
    fn empty_directional_deltas_are_omitted_and_default_marker_is_none() {
        let plot = ErrorBar::new_both(
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            Vec::new(),
            Vec::new(),
            vec![0.1, 0.2],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(plot.x_neg, vec![0.0, 0.0]);
        assert_eq!(plot.x_pos, vec![0.0, 0.0]);
        assert_eq!(plot.y_pos, vec![0.0, 0.0]);
        assert!(plot.marker.is_none());
    }
}
