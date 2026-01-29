//! Line plot implementation
//!
//! High-performance line plotting with GPU acceleration.

use crate::core::{
    vertex_utils, AlphaMode, BoundingBox, DrawCall, GpuPackContext, GpuVertexBuffer, Material,
    PipelineType, RenderData, Vertex,
};
use crate::gpu::line::LineGpuInputs;
use crate::plots::scatter::MarkerStyle as ScatterMarkerStyle;
use glam::{Vec3, Vec4};
use log::trace;

/// High-performance GPU-accelerated line plot
#[derive(Debug, Clone)]
pub struct LinePlot {
    /// Raw data points (x, y coordinates)
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,

    /// Visual styling
    pub color: Vec4,
    pub line_width: f32,
    pub line_style: LineStyle,
    pub line_join: LineJoin,
    pub line_cap: LineCap,
    pub marker: Option<LineMarkerAppearance>,

    /// Metadata
    pub label: Option<String>,
    pub visible: bool,

    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
    gpu_vertices: Option<GpuVertexBuffer>,
    gpu_vertex_count: Option<usize>,
    gpu_line_inputs: Option<LineGpuInputs>,
    marker_vertices: Option<Vec<Vertex>>,
    marker_gpu_vertices: Option<GpuVertexBuffer>,
    marker_dirty: bool,
    gpu_topology: Option<PipelineType>,
}

#[derive(Debug, Clone)]
pub struct LineMarkerAppearance {
    pub kind: ScatterMarkerStyle,
    pub size: f32,
    pub edge_color: Vec4,
    pub face_color: Vec4,
    pub filled: bool,
}

#[derive(Debug, Clone)]
pub struct LineGpuStyle {
    pub color: Vec4,
    pub line_width: f32,
    pub line_style: LineStyle,
    pub marker: Option<LineMarkerAppearance>,
}

/// Line rendering styles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

/// Line join style for thick polylines
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineJoin {
    Miter,
    Bevel,
    Round,
}

impl Default for LineJoin {
    fn default() -> Self {
        Self::Miter
    }
}

/// Line cap style for thick polylines
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineCap {
    Butt,
    Square,
    Round,
}

impl Default for LineCap {
    fn default() -> Self {
        Self::Butt
    }
}

impl Default for LineStyle {
    fn default() -> Self {
        Self::Solid
    }
}

impl LinePlot {
    /// Create a new line plot with data
    pub fn new(x_data: Vec<f64>, y_data: Vec<f64>) -> Result<Self, String> {
        if x_data.len() != y_data.len() {
            return Err(format!(
                "Data length mismatch: x_data has {} points, y_data has {} points",
                x_data.len(),
                y_data.len()
            ));
        }

        if x_data.is_empty() {
            return Err("Cannot create line plot with empty data".to_string());
        }

        Ok(Self {
            x_data,
            y_data,
            color: Vec4::new(0.0, 0.5, 1.0, 1.0), // Default blue
            line_width: 1.0,
            line_style: LineStyle::default(),
            line_join: LineJoin::default(),
            line_cap: LineCap::default(),
            marker: None,
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_line_inputs: None,
            marker_vertices: None,
            marker_gpu_vertices: None,
            marker_dirty: true,
            gpu_topology: None,
        })
    }

    /// Build a line plot directly from a GPU vertex buffer.
    pub fn from_gpu_buffer(
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        style: LineGpuStyle,
        bounds: BoundingBox,
        pipeline: PipelineType,
        marker_buffer: Option<GpuVertexBuffer>,
    ) -> Self {
        Self {
            x_data: Vec::new(),
            y_data: Vec::new(),
            color: style.color,
            line_width: style.line_width,
            line_style: style.line_style,
            line_join: LineJoin::Miter,
            line_cap: LineCap::Butt,
            marker: style.marker,
            label: None,
            visible: true,
            vertices: None,
            bounds: Some(bounds),
            dirty: false,
            gpu_vertices: Some(buffer),
            gpu_vertex_count: Some(vertex_count),
            gpu_line_inputs: None,
            marker_vertices: None,
            marker_gpu_vertices: marker_buffer,
            marker_dirty: true,
            gpu_topology: Some(pipeline),
        }
    }

    /// Create a GPU-backed line plot from X/Y device buffers.
    ///
    /// Geometry is packed at render-time when a viewport size is available so that pixel-based
    /// widths can be converted into data units.
    pub fn from_gpu_xy(
        inputs: LineGpuInputs,
        style: LineGpuStyle,
        bounds: BoundingBox,
        marker_buffer: Option<GpuVertexBuffer>,
    ) -> Self {
        Self {
            x_data: Vec::new(),
            y_data: Vec::new(),
            color: style.color,
            line_width: style.line_width,
            line_style: style.line_style,
            line_join: LineJoin::Miter,
            line_cap: LineCap::Butt,
            marker: style.marker,
            label: None,
            visible: true,
            vertices: None,
            bounds: Some(bounds),
            dirty: false,
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_line_inputs: Some(inputs),
            marker_vertices: None,
            marker_gpu_vertices: marker_buffer,
            marker_dirty: true,
            gpu_topology: None,
        }
    }

    fn invalidate_gpu_data(&mut self) {
        self.gpu_vertices = None;
        self.gpu_vertex_count = None;
        self.bounds = None;
        self.gpu_line_inputs = None;
        self.marker_gpu_vertices = None;
        self.marker_dirty = true;
        self.gpu_topology = None;
    }

    fn invalidate_marker_data(&mut self) {
        self.marker_vertices = None;
        self.marker_dirty = true;
        if self.gpu_vertices.is_none() {
            self.marker_gpu_vertices = None;
        }
    }

    /// Create a line plot with custom styling
    pub fn with_style(mut self, color: Vec4, line_width: f32, line_style: LineStyle) -> Self {
        self.color = color;
        self.line_width = line_width;
        self.line_style = line_style;
        self.dirty = true;
        self.invalidate_gpu_data();
        self
    }

    /// Set the plot label for legends
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Update the data points
    pub fn update_data(&mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> Result<(), String> {
        if x_data.len() != y_data.len() {
            return Err(format!(
                "Data length mismatch: x_data has {} points, y_data has {} points",
                x_data.len(),
                y_data.len()
            ));
        }

        if x_data.is_empty() {
            return Err("Cannot update with empty data".to_string());
        }

        self.x_data = x_data;
        self.y_data = y_data;
        self.dirty = true;
        self.invalidate_marker_data();
        Ok(())
    }

    /// Set the color of the line
    pub fn set_color(&mut self, color: Vec4) {
        self.color = color;
        self.dirty = true;
        self.invalidate_gpu_data();
        self.invalidate_marker_data();
    }

    /// Set the line width
    pub fn set_line_width(&mut self, width: f32) {
        self.line_width = width.max(0.1); // Minimum line width
        self.dirty = true;
        self.invalidate_gpu_data();
    }

    /// Set the line style
    pub fn set_line_style(&mut self, style: LineStyle) {
        self.line_style = style;
        self.dirty = true;
        self.invalidate_gpu_data();
    }

    /// Attach marker metadata so renderers can emit hybrid line+marker plots.
    pub fn set_marker(&mut self, marker: Option<LineMarkerAppearance>) {
        self.marker = marker;
        self.invalidate_marker_data();
    }

    /// Set the line join style for thick lines
    pub fn set_line_join(&mut self, join: LineJoin) {
        self.line_join = join;
        self.dirty = true;
        self.invalidate_gpu_data();
    }

    /// Set the line cap style for thick lines
    pub fn set_line_cap(&mut self, cap: LineCap) {
        self.line_cap = cap;
        self.dirty = true;
        self.invalidate_gpu_data();
    }

    /// Show or hide the plot
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        if !self.x_data.is_empty() {
            self.x_data.len()
        } else {
            self.gpu_vertex_count.unwrap_or(0)
        }
    }

    /// Check if the plot has no data
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generate vertices for GPU rendering
    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.gpu_vertices.is_some() {
            if self.vertices.is_none() {
                self.vertices = Some(Vec::new());
            }
            return self.vertices.as_ref().unwrap();
        }
        if self.dirty || self.vertices.is_none() {
            if self.line_width > 1.0 {
                // Use triangle extrusion for thicker lines; switch pipeline in render_data
                let base_tris = match self.line_cap {
                    LineCap::Butt => vertex_utils::create_thick_polyline_with_join(
                        &self.x_data,
                        &self.y_data,
                        self.color,
                        self.line_width,
                        self.line_join,
                    ),
                    LineCap::Square => vertex_utils::create_thick_polyline_square_caps(
                        &self.x_data,
                        &self.y_data,
                        self.color,
                        self.line_width,
                    ),
                    LineCap::Round => vertex_utils::create_thick_polyline_round_caps(
                        &self.x_data,
                        &self.y_data,
                        self.color,
                        self.line_width,
                        12,
                    ),
                };
                let tris = match self.line_style {
                    LineStyle::Solid => base_tris,
                    LineStyle::Dashed | LineStyle::DashDot | LineStyle::Dotted => {
                        vertex_utils::create_thick_polyline_dashed(
                            &self.x_data,
                            &self.y_data,
                            self.color,
                            self.line_width,
                            self.line_style,
                        )
                    }
                };
                self.vertices = Some(tris);
            } else {
                let verts = match self.line_style {
                    LineStyle::Solid => {
                        vertex_utils::create_line_plot(&self.x_data, &self.y_data, self.color)
                    }
                    LineStyle::Dashed | LineStyle::DashDot => {
                        vertex_utils::create_line_plot_dashed(
                            &self.x_data,
                            &self.y_data,
                            self.color,
                            self.line_style,
                        )
                    }
                    LineStyle::Dotted => {
                        // Render as a sequence of tiny dashes to approximate dots
                        vertex_utils::create_line_plot_dashed(
                            &self.x_data,
                            &self.y_data,
                            self.color,
                            LineStyle::Dashed,
                        )
                    }
                };
                self.vertices = Some(verts);
            }
            self.dirty = false;
        }
        self.vertices.as_ref().unwrap()
    }

    /// Get the bounding box of the data
    pub fn bounds(&mut self) -> BoundingBox {
        if self.bounds.is_some() && self.x_data.is_empty() && self.y_data.is_empty() {
            return self.bounds.unwrap_or_default();
        }
        if self.dirty || self.bounds.is_none() {
            let points: Vec<Vec3> = self
                .x_data
                .iter()
                .zip(self.y_data.iter())
                .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
                .collect();
            self.bounds = Some(BoundingBox::from_points(&points));
        }
        self.bounds.unwrap()
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
        let bounds = self.bounds.as_ref().ok_or_else(|| "missing line bounds".to_string())?;

        let thick_px = self.line_width > 1.0;
        let data_per_px = crate::core::data_units_per_px(bounds, viewport_px);
        let half_width_data = if thick_px {
            ((self.line_width.max(0.1)) * 0.5) * data_per_px
        } else {
            0.0
        };
        trace!(
            target: "runmat_plot",
            "line-pack: begin len={} line_width_px={} thick={} half_width_data={} viewport_px={:?} bounds=({:?}..{:?})",
            inputs.len,
            self.line_width,
            thick_px,
            half_width_data,
            viewport_px,
            bounds.min,
            bounds.max
        );

        let params = crate::gpu::line::LineGpuParams {
            color: self.color,
            half_width_data,
            thick: thick_px,
            line_style: self.line_style,
            marker_size: 1.0,
        };
        let packed = crate::gpu::line::pack_vertices_from_xy(gpu.device, gpu.queue, inputs, &params)
            .map_err(|e| format!("gpu line packing failed: {e}"))?;
        trace!(
            target: "runmat_plot",
            "line-pack: complete max_vertices={} indirect_present={}",
            packed.vertex_count,
            packed.indirect.is_some()
        );

        self.gpu_vertices = Some(packed);
        self.gpu_topology = Some(if thick_px {
            PipelineType::Triangles
        } else {
            PipelineType::Lines
        });
        Ok(())
    }

    pub fn render_data_with_viewport_gpu(
        &mut self,
        viewport_px: Option<(u32, u32)>,
        gpu: Option<&GpuPackContext<'_>>,
    ) -> RenderData {
        if self.gpu_line_inputs.is_some() && self.gpu_vertices.is_none() {
            if let (Some(gpu), Some(vp)) = (gpu, viewport_px) {
                // Best-effort: if packing fails, fall through and let the caller handle
                // missing geometry (typically via a plotting error upstream).
                let _ = self.pack_gpu_vertices_if_needed(gpu, vp);
            }
        }
        self.render_data_with_viewport(viewport_px)
    }

    /// Generate complete render data for the graphics pipeline
    pub fn render_data(&mut self) -> RenderData {
        let using_gpu = self.gpu_vertices.is_some();
        let gpu_vertices = self.gpu_vertices.clone();
        let (vertices, vertex_count) = if using_gpu {
            (Vec::new(), self.gpu_vertex_count.unwrap_or(0))
        } else {
            let verts = self.generate_vertices().clone();
            let count = verts.len();
            (verts, count)
        };

        // Encode width/style/cap/join into material for exporters:
        // - roughness: line width
        // - metallic: line style code (0 solid,1 dashed,2 dotted,3 dashdot)
        // - emissive.x: cap (0 butt,1 square,2 round)
        // - emissive.y: join (0 miter,1 bevel,2 round)
        let style_code = match self.line_style {
            LineStyle::Solid => 0.0,
            LineStyle::Dashed => 1.0,
            LineStyle::Dotted => 2.0,
            LineStyle::DashDot => 3.0,
        };
        let cap_code = match self.line_cap {
            LineCap::Butt => 0.0,
            LineCap::Square => 1.0,
            LineCap::Round => 2.0,
        };
        let join_code = match self.line_join {
            LineJoin::Miter => 0.0,
            LineJoin::Bevel => 1.0,
            LineJoin::Round => 2.0,
        };
        let mut material = Material {
            albedo: self.color,
            ..Default::default()
        };
        material.roughness = self.line_width.max(0.0);
        material.metallic = style_code;
        material.emissive = Vec4::new(cap_code, join_code, -1.0, 0.0);

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };

        // If thick polyline was generated, we must render as triangles
        let pipeline = if using_gpu {
            self.gpu_topology.unwrap_or(if self.line_width > 1.0 {
                PipelineType::Triangles
            } else {
                PipelineType::Lines
            })
        } else if self.line_width > 1.0 {
            PipelineType::Triangles
        } else {
            PipelineType::Lines
        };
        RenderData {
            pipeline_type: pipeline,
            vertices,
            indices: None,
            gpu_vertices,
            bounds: Some(self.bounds()),
            material,
            draw_calls: vec![draw_call],
            image: None,
        }
    }

    /// Generate render data, using an optional viewport size hint (width, height in pixels).
    ///
    /// For thick 2D lines we build triangle geometry. The user-facing `line_width` is
    /// expressed in *pixels*, but triangle extrusion operates in data space. When a viewport
    /// is supplied we convert pixels → data-units using the current data range and target size.
    pub fn render_data_with_viewport(&mut self, viewport_px: Option<(u32, u32)>) -> RenderData {
        if self.gpu_vertices.is_some() {
            // GPU paths already handle sizing via pipeline/state; keep existing behavior.
            return self.render_data();
        }

        let (vertices, vertex_count, pipeline) = if self.line_width > 1.0 {
            let bounds = self.bounds();
            let viewport_px = viewport_px.unwrap_or((600, 400));
            let data_per_px = crate::core::data_units_per_px(&bounds, viewport_px);
            let width_data = (self.line_width.max(0.1)) * data_per_px;

            let base_tris = match self.line_cap {
                LineCap::Butt => vertex_utils::create_thick_polyline_with_join(
                    &self.x_data,
                    &self.y_data,
                    self.color,
                    width_data,
                    self.line_join,
                ),
                LineCap::Square => vertex_utils::create_thick_polyline_square_caps(
                    &self.x_data,
                    &self.y_data,
                    self.color,
                    width_data,
                ),
                LineCap::Round => vertex_utils::create_thick_polyline_round_caps(
                    &self.x_data,
                    &self.y_data,
                    self.color,
                    width_data,
                    12,
                ),
            };
            let tris = match self.line_style {
                LineStyle::Solid => base_tris,
                LineStyle::Dashed | LineStyle::DashDot | LineStyle::Dotted => {
                    vertex_utils::create_thick_polyline_dashed(
                        &self.x_data,
                        &self.y_data,
                        self.color,
                        width_data,
                        self.line_style,
                    )
                }
            };
            let count = tris.len();
            (tris, count, PipelineType::Triangles)
        } else {
            let verts = self.generate_vertices().clone();
            let count = verts.len();
            (verts, count, PipelineType::Lines)
        };

        let style_code = match self.line_style {
            LineStyle::Solid => 0.0,
            LineStyle::Dashed => 1.0,
            LineStyle::Dotted => 2.0,
            LineStyle::DashDot => 3.0,
        };
        let cap_code = match self.line_cap {
            LineCap::Butt => 0.0,
            LineCap::Square => 1.0,
            LineCap::Round => 2.0,
        };
        let join_code = match self.line_join {
            LineJoin::Miter => 0.0,
            LineJoin::Bevel => 1.0,
            LineJoin::Round => 2.0,
        };
        let mut material = Material {
            albedo: self.color,
            ..Default::default()
        };
        // Keep the user-facing width in pixels for exporters/metadata.
        material.roughness = self.line_width.max(0.0);
        material.metallic = style_code;
        material.emissive = Vec4::new(cap_code, join_code, -1.0, 0.0);

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };

        RenderData {
            pipeline_type: pipeline,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: Some(self.bounds()),
            material,
            draw_calls: vec![draw_call],
            image: None,
        }
    }

    /// Generate render data representing the markers for this line plot.
    pub fn marker_render_data(&mut self) -> Option<RenderData> {
        let marker = self.marker.clone()?;
        let material = Self::build_marker_material(&marker);

        if let Some(gpu_vertices) = self.marker_gpu_vertices.clone() {
            let vertex_count = gpu_vertices.vertex_count;
            if vertex_count == 0 {
                return None;
            }
            let draw_call = DrawCall {
                vertex_offset: 0,
                vertex_count,
                index_offset: None,
                index_count: None,
                instance_count: 1,
            };
            return Some(RenderData {
                pipeline_type: PipelineType::Points,
                vertices: Vec::new(),
                indices: None,
                gpu_vertices: Some(gpu_vertices),
                bounds: None,
                material,
                draw_calls: vec![draw_call],
                image: None,
            });
        }

        let vertices = self.marker_vertices_slice(&marker)?;
        if vertices.is_empty() {
            return None;
        }
        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count: vertices.len(),
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };

        Some(RenderData {
            pipeline_type: PipelineType::Points,
            vertices: vertices.to_vec(),
            indices: None,
            gpu_vertices: None,
            bounds: None,
            material,
            draw_calls: vec![draw_call],
            image: None,
        })
    }

    fn build_marker_material(marker: &LineMarkerAppearance) -> Material {
        let mut material = Material {
            albedo: marker.face_color,
            ..Default::default()
        };
        if !marker.filled {
            material.albedo.w = 0.0;
        }
        material.emissive = marker.edge_color;
        material.roughness = 1.0;
        material.metallic = marker_style_code(marker.kind);
        material.alpha_mode = AlphaMode::Blend;
        material
    }

    fn marker_vertices_slice(&mut self, marker: &LineMarkerAppearance) -> Option<&[Vertex]> {
        if self.x_data.len() != self.y_data.len() || self.x_data.is_empty() {
            return None;
        }

        if self.marker_vertices.is_none() || self.marker_dirty {
            let mut verts = Vec::with_capacity(self.x_data.len());
            for (&x, &y) in self.x_data.iter().zip(self.y_data.iter()) {
                let mut vertex = Vertex::new(Vec3::new(x as f32, y as f32, 0.0), marker.face_color);
                vertex.normal[2] = marker.size.max(1.0);
                verts.push(vertex);
            }
            self.marker_vertices = Some(verts);
            self.marker_dirty = false;
        }
        self.marker_vertices.as_deref()
    }

    /// Get plot statistics for debugging
    pub fn statistics(&self) -> PlotStatistics {
        let (min_x, max_x) = self
            .x_data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });
        let (min_y, max_y) = self
            .y_data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &y| {
                (min.min(y), max.max(y))
            });

        PlotStatistics {
            point_count: self.x_data.len(),
            x_range: (min_x, max_x),
            y_range: (min_y, max_y),
            memory_usage: self.estimated_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        std::mem::size_of::<f64>() * (self.x_data.len() + self.y_data.len())
            + self
                .vertices
                .as_ref()
                .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
            + self.gpu_vertex_count.unwrap_or(0) * std::mem::size_of::<Vertex>()
    }
}

fn marker_style_code(kind: ScatterMarkerStyle) -> f32 {
    match kind {
        ScatterMarkerStyle::Circle => 0.0,
        ScatterMarkerStyle::Square => 1.0,
        ScatterMarkerStyle::Triangle => 2.0,
        ScatterMarkerStyle::Diamond => 3.0,
        ScatterMarkerStyle::Plus => 4.0,
        ScatterMarkerStyle::Cross => 5.0,
        ScatterMarkerStyle::Star => 6.0,
        ScatterMarkerStyle::Hexagon => 7.0,
    }
}

/// Plot performance and data statistics
#[derive(Debug, Clone)]
pub struct PlotStatistics {
    pub point_count: usize,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub memory_usage: usize,
}

/// MATLAB-compatible line plot creation utilities
pub mod matlab_compat {
    use super::*;

    /// Create a simple line plot (equivalent to MATLAB's `plot(x, y)`)
    pub fn plot(x: Vec<f64>, y: Vec<f64>) -> Result<LinePlot, String> {
        LinePlot::new(x, y)
    }

    /// Create a line plot with specified color (`plot(x, y, 'r')`)
    pub fn plot_with_color(x: Vec<f64>, y: Vec<f64>, color: &str) -> Result<LinePlot, String> {
        let color_vec = parse_matlab_color(color)?;
        Ok(LinePlot::new(x, y)?.with_style(color_vec, 1.0, LineStyle::Solid))
    }

    /// Parse MATLAB color specifications
    fn parse_matlab_color(color: &str) -> Result<Vec4, String> {
        match color {
            "r" | "red" => Ok(Vec4::new(1.0, 0.0, 0.0, 1.0)),
            "g" | "green" => Ok(Vec4::new(0.0, 1.0, 0.0, 1.0)),
            "b" | "blue" => Ok(Vec4::new(0.0, 0.0, 1.0, 1.0)),
            "c" | "cyan" => Ok(Vec4::new(0.0, 1.0, 1.0, 1.0)),
            "m" | "magenta" => Ok(Vec4::new(1.0, 0.0, 1.0, 1.0)),
            "y" | "yellow" => Ok(Vec4::new(1.0, 1.0, 0.0, 1.0)),
            "k" | "black" => Ok(Vec4::new(0.0, 0.0, 0.0, 1.0)),
            "w" | "white" => Ok(Vec4::new(1.0, 1.0, 1.0, 1.0)),
            _ => Err(format!("Unknown color: {color}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_plot_creation() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.0, 1.0];

        let plot = LinePlot::new(x.clone(), y.clone()).unwrap();

        assert_eq!(plot.x_data, x);
        assert_eq!(plot.y_data, y);
        assert_eq!(plot.len(), 4);
        assert!(!plot.is_empty());
        assert!(plot.visible);
    }

    #[test]
    fn test_line_plot_data_validation() {
        // Mismatched lengths should fail
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0];
        assert!(LinePlot::new(x, y).is_err());

        // Empty data should fail
        let empty_x: Vec<f64> = vec![];
        let empty_y: Vec<f64> = vec![];
        assert!(LinePlot::new(empty_x, empty_y).is_err());
    }

    #[test]
    fn test_line_plot_styling() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 1.5];
        let color = Vec4::new(1.0, 0.0, 0.0, 1.0);

        let plot = LinePlot::new(x, y)
            .unwrap()
            .with_style(color, 2.0, LineStyle::Dashed)
            .with_label("Test Line");

        assert_eq!(plot.color, color);
        assert_eq!(plot.line_width, 2.0);
        assert_eq!(plot.line_style, LineStyle::Dashed);
        assert_eq!(plot.label, Some("Test Line".to_string()));
    }

    #[test]
    fn test_line_plot_data_update() {
        let mut plot = LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();

        let new_x = vec![0.0, 0.5, 1.0, 1.5];
        let new_y = vec![0.0, 0.25, 1.0, 2.25];

        plot.update_data(new_x.clone(), new_y.clone()).unwrap();

        assert_eq!(plot.x_data, new_x);
        assert_eq!(plot.y_data, new_y);
        assert_eq!(plot.len(), 4);
    }

    #[test]
    fn test_line_plot_bounds() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let y = vec![-2.0, 0.0, 1.0, 3.0];

        let mut plot = LinePlot::new(x, y).unwrap();
        let bounds = plot.bounds();

        assert_eq!(bounds.min.x, -1.0);
        assert_eq!(bounds.max.x, 2.0);
        assert_eq!(bounds.min.y, -2.0);
        assert_eq!(bounds.max.y, 3.0);
    }

    #[test]
    fn test_line_plot_vertex_generation() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.0];

        let mut plot = LinePlot::new(x, y).unwrap();
        let vertices = plot.generate_vertices();

        // Should have 2 line segments (4 vertices total)
        assert_eq!(vertices.len(), 4);

        // Check first line segment
        assert_eq!(vertices[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(vertices[1].position, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_line_plot_render_data() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 1.0];

        let mut plot = LinePlot::new(x, y).unwrap();
        let render_data = plot.render_data();

        assert_eq!(render_data.pipeline_type, PipelineType::Lines);
        assert_eq!(render_data.vertices.len(), 4); // 2 line segments
        assert!(render_data.indices.is_none());
        assert_eq!(render_data.draw_calls.len(), 1);
    }

    #[test]
    fn test_line_plot_statistics() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![-1.0, 0.0, 1.0, 2.0];

        let plot = LinePlot::new(x, y).unwrap();
        let stats = plot.statistics();

        assert_eq!(stats.point_count, 4);
        assert_eq!(stats.x_range, (0.0, 3.0));
        assert_eq!(stats.y_range, (-1.0, 2.0));
        assert!(stats.memory_usage > 0);
    }

    #[test]
    fn test_matlab_compat_colors() {
        use super::matlab_compat::*;

        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];

        let red_plot = plot_with_color(x.clone(), y.clone(), "r").unwrap();
        assert_eq!(red_plot.color, Vec4::new(1.0, 0.0, 0.0, 1.0));

        let blue_plot = plot_with_color(x.clone(), y.clone(), "blue").unwrap();
        assert_eq!(blue_plot.color, Vec4::new(0.0, 0.0, 1.0, 1.0));

        // Invalid color should fail
        assert!(plot_with_color(x, y, "invalid").is_err());
    }

    #[test]
    fn marker_render_data_produces_point_draw_call() {
        let mut plot = LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        plot.set_marker(Some(LineMarkerAppearance {
            kind: ScatterMarkerStyle::Circle,
            size: 8.0,
            edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            face_color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            filled: true,
        }));
        let marker_data = plot.marker_render_data().expect("marker render data");
        assert_eq!(marker_data.pipeline_type, PipelineType::Points);
        assert_eq!(marker_data.draw_calls[0].vertex_count, 2);
    }

    #[test]
    fn line_plot_handles_large_trace() {
        let n = 50_000;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
        let mut plot = LinePlot::new(x, y).unwrap();
        let render_data = plot.render_data();
        assert_eq!(render_data.vertices.len(), (n - 1) * 2);
    }
}
