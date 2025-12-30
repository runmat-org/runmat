//! Bar chart implementation
//!
//! High-performance bar charts with GPU acceleration and MATLAB-compatible styling.

use crate::core::{
    BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Vertical,
    Horizontal,
}

/// High-performance GPU-accelerated bar chart
#[derive(Debug, Clone)]
pub struct BarChart {
    /// Category labels and values
    pub labels: Vec<String>,
    values: Option<Vec<f64>>,
    value_count: usize,

    /// Visual styling
    pub color: Vec4,
    pub bar_width: f32,
    pub outline_color: Option<Vec4>,
    pub outline_width: f32,
    per_bar_colors: Option<Vec<Vec4>>,

    /// Orientation (vertical = default bar, horizontal = barh)
    pub orientation: Orientation,

    /// Grouped bar configuration: index within group and total group count
    /// Example: for 3-series grouped bars, group_count=3 and each series has group_index 0,1,2
    pub group_index: usize,
    pub group_count: usize,

    /// Stacked bar offsets per category (bottom/base for each bar)
    /// When provided, bars are drawn starting at offset[i] and extending by values[i]
    pub stack_offsets: Option<Vec<f64>>,

    /// Metadata
    pub label: Option<String>,
    pub visible: bool,

    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    indices: Option<Vec<u32>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
    gpu_vertices: Option<GpuVertexBuffer>,
    gpu_vertex_count: Option<usize>,
    gpu_bounds: Option<BoundingBox>,
}

impl BarChart {
    /// Create a new bar chart with labels and values
    pub fn new(labels: Vec<String>, values: Vec<f64>) -> Result<Self, String> {
        if labels.len() != values.len() {
            return Err(format!(
                "Data length mismatch: {} labels, {} values",
                labels.len(),
                values.len()
            ));
        }

        if labels.is_empty() {
            return Err("Cannot create bar chart with empty data".to_string());
        }

        let count = values.len();
        Ok(Self {
            labels,
            values: Some(values),
            value_count: count,
            color: Vec4::new(0.0, 0.5, 1.0, 1.0), // Default blue
            bar_width: 0.8,                       // 80% of available space
            outline_color: None,
            outline_width: 1.0,
            orientation: Orientation::Vertical,
            group_index: 0,
            group_count: 1,
            stack_offsets: None,
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: None,
            dirty: true,
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_bounds: None,
            per_bar_colors: None,
        })
    }

    /// Build a bar chart backed by a GPU vertex buffer.
    pub fn from_gpu_buffer(
        labels: Vec<String>,
        value_count: usize,
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        bounds: BoundingBox,
        color: Vec4,
        bar_width: f32,
    ) -> Self {
        Self {
            labels,
            values: None,
            value_count,
            color,
            bar_width,
            outline_color: None,
            outline_width: 1.0,
            orientation: Orientation::Vertical,
            group_index: 0,
            group_count: 1,
            stack_offsets: None,
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: Some(bounds),
            dirty: false,
            gpu_vertices: Some(buffer),
            gpu_vertex_count: Some(vertex_count),
            gpu_bounds: Some(bounds),
            per_bar_colors: None,
        }
    }

    fn invalidate_gpu_data(&mut self) {
        self.gpu_vertices = None;
        self.gpu_vertex_count = None;
        self.gpu_bounds = None;
    }

    /// Create a bar chart with custom styling
    pub fn with_style(mut self, color: Vec4, bar_width: f32) -> Self {
        self.color = color;
        self.bar_width = bar_width.clamp(0.1, 1.0);
        self.dirty = true;
        self
    }

    /// Add outline to bars
    pub fn with_outline(mut self, outline_color: Vec4, outline_width: f32) -> Self {
        self.outline_color = Some(outline_color);
        self.outline_width = outline_width.max(0.1);
        self.dirty = true;
        self
    }

    /// Set orientation (vertical/horizontal)
    pub fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self.dirty = true;
        self
    }

    pub fn bar_count(&self) -> usize {
        self.value_count
    }

    pub fn set_per_bar_colors(&mut self, colors: Vec<Vec4>) {
        if colors.is_empty() {
            self.per_bar_colors = None;
        } else {
            self.per_bar_colors = Some(colors);
        }
        self.dirty = true;
        self.invalidate_gpu_data();
    }

    pub fn clear_per_bar_colors(&mut self) {
        if self.per_bar_colors.is_some() {
            self.per_bar_colors = None;
            self.dirty = true;
        }
    }

    /// Configure grouped bars (index within group and total group count)
    pub fn with_group(mut self, group_index: usize, group_count: usize) -> Self {
        self.group_index = group_index.min(group_count.saturating_sub(1));
        self.group_count = group_count.max(1);
        self.dirty = true;
        self
    }

    /// Configure stacked bars with per-category offsets (base values)
    pub fn with_stack_offsets(mut self, offsets: Vec<f64>) -> Self {
        if self
            .values
            .as_ref()
            .is_some_and(|v| offsets.len() == v.len())
            || offsets.len() == self.value_count
        {
            self.stack_offsets = Some(offsets);
            self.dirty = true;
            self.invalidate_gpu_data();
        }
        self
    }

    /// Set the chart label for legends
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Update the data
    pub fn update_data(&mut self, labels: Vec<String>, values: Vec<f64>) -> Result<(), String> {
        if labels.len() != values.len() {
            return Err(format!(
                "Data length mismatch: {} labels, {} values",
                labels.len(),
                values.len()
            ));
        }

        if labels.is_empty() {
            return Err("Cannot update with empty data".to_string());
        }

        self.labels = labels;
        self.value_count = values.len();
        self.values = Some(values);
        self.dirty = true;
        self.vertices = None;
        self.indices = None;
        self.bounds = None;
        self.invalidate_gpu_data();
        Ok(())
    }

    /// Set the bar color
    pub fn set_color(&mut self, color: Vec4) {
        self.color = color;
        self.per_bar_colors = None;
        self.dirty = true;
    }

    /// Set the bar width (0.1 to 1.0)
    pub fn set_bar_width(&mut self, width: f32) {
        self.bar_width = width.clamp(0.1, 1.0);
        self.dirty = true;
    }

    /// Override the face color and width without invalidating GPU data.
    pub fn apply_face_style(&mut self, color: Vec4, width: f32) {
        if self.gpu_vertices.is_some() {
            self.color = color;
            self.bar_width = width.clamp(0.1, 1.0);
        } else {
            self.set_color(color);
            self.set_bar_width(width);
        }
    }

    /// Set the outline color (enables outline if not present)
    pub fn set_outline_color(&mut self, color: Vec4) {
        if self.outline_color.is_none() {
            self.outline_width = self.outline_width.max(1.0);
        }
        self.outline_color = Some(color);
        self.dirty = true;
    }

    /// Set the outline width (enables outline if not present)
    pub fn set_outline_width(&mut self, width: f32) {
        self.outline_width = width.max(0.1);
        if self.outline_color.is_none() {
            // Default to black if no color set
            self.outline_color = Some(Vec4::new(0.0, 0.0, 0.0, 1.0));
        }
        self.dirty = true;
        self.invalidate_gpu_data();
    }

    /// Override outline styling while preserving GPU geometry when possible.
    pub fn apply_outline_style(&mut self, color: Option<Vec4>, width: f32) {
        match color {
            Some(color) => {
                if self.gpu_vertices.is_some() {
                    self.outline_color = Some(color);
                    self.outline_width = width.max(0.1);
                } else {
                    self.set_outline_color(color);
                    self.set_outline_width(width);
                }
            }
            None => {
                self.outline_color = None;
            }
        }
    }

    /// Show or hide the chart
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Get the number of bars
    pub fn len(&self) -> usize {
        self.value_count
    }

    /// Check if the chart has no data
    pub fn is_empty(&self) -> bool {
        self.value_count == 0
    }

    /// Generate vertices for GPU rendering
    pub fn generate_vertices(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
        if self.gpu_vertices.is_some() {
            if self.vertices.is_none() {
                self.vertices = Some(Vec::new());
            }
            if self.indices.is_none() {
                self.indices = Some(Vec::new());
            }
            return (
                self.vertices.as_ref().unwrap(),
                self.indices.as_ref().unwrap(),
            );
        }

        if self.dirty || self.vertices.is_none() {
            let (vertices, indices) = self.create_bar_geometry();
            self.vertices = Some(vertices);
            self.indices = Some(indices);
            self.dirty = false;
        }
        (
            self.vertices.as_ref().unwrap(),
            self.indices.as_ref().unwrap(),
        )
    }

    /// Create the geometry for all bars
    fn create_bar_geometry(&self) -> (Vec<Vertex>, Vec<u32>) {
        let values = self
            .values
            .as_ref()
            .expect("CPU bar geometry requested without host values");
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let group_count = self.group_count.max(1) as f32;
        let per_group_width = (self.bar_width / group_count).max(0.01);
        let group_offset_start = -self.bar_width * 0.5;
        let local_offset = group_offset_start
            + per_group_width * (self.group_index as f32)
            + per_group_width * 0.5;

        match self.orientation {
            Orientation::Vertical => {
                for (i, &value) in values.iter().enumerate() {
                    if !value.is_finite() {
                        continue;
                    }
                    let color = self.color_for_bar(i);
                    let x_center = (i as f32) + 1.0;
                    let center = x_center + local_offset;
                    let half = per_group_width * 0.5;
                    let left = center - half;
                    let right = center + half;
                    let base = self
                        .stack_offsets
                        .as_ref()
                        .map(|v| v[i] as f32)
                        .unwrap_or(0.0);
                    let bottom = base;
                    let top = base + value as f32;

                    let vertex_offset = vertices.len() as u32;
                    vertices.push(Vertex::new(Vec3::new(left, bottom, 0.0), color));
                    vertices.push(Vertex::new(Vec3::new(right, bottom, 0.0), color));
                    vertices.push(Vertex::new(Vec3::new(right, top, 0.0), color));
                    vertices.push(Vertex::new(Vec3::new(left, top, 0.0), color));
                    indices.push(vertex_offset);
                    indices.push(vertex_offset + 1);
                    indices.push(vertex_offset + 2);
                    indices.push(vertex_offset);
                    indices.push(vertex_offset + 2);
                    indices.push(vertex_offset + 3);
                }
            }
            Orientation::Horizontal => {
                for (i, &value) in values.iter().enumerate() {
                    if !value.is_finite() {
                        continue;
                    }
                    let color = self.color_for_bar(i);
                    let y_center = (i as f32) + 1.0;
                    let center = y_center + local_offset;
                    let half = per_group_width * 0.5;
                    let bottom = center - half;
                    let top = center + half;
                    let base = self
                        .stack_offsets
                        .as_ref()
                        .map(|v| v[i] as f32)
                        .unwrap_or(0.0);
                    let left = base;
                    let right = base + value as f32;

                    let vertex_offset = vertices.len() as u32;
                    vertices.push(Vertex::new(Vec3::new(left, bottom, 0.0), color));
                    vertices.push(Vertex::new(Vec3::new(right, bottom, 0.0), color));
                    vertices.push(Vertex::new(Vec3::new(right, top, 0.0), color));
                    vertices.push(Vertex::new(Vec3::new(left, top, 0.0), color));
                    indices.push(vertex_offset);
                    indices.push(vertex_offset + 1);
                    indices.push(vertex_offset + 2);
                    indices.push(vertex_offset);
                    indices.push(vertex_offset + 2);
                    indices.push(vertex_offset + 3);
                }
            }
        }

        (vertices, indices)
    }

    fn color_for_bar(&self, index: usize) -> Vec4 {
        if let Some(colors) = &self.per_bar_colors {
            if let Some(color) = colors.get(index) {
                return *color;
            }
        }
        self.color
    }

    /// Get the bounding box of the chart
    pub fn bounds(&mut self) -> BoundingBox {
        if let Some(bounds) = self.gpu_bounds {
            self.bounds = Some(bounds);
            return bounds;
        }

        if self.dirty || self.bounds.is_none() {
            let values = self
                .values
                .as_ref()
                .expect("CPU bar bounds requested without host values");
            let num_bars = values.len();
            if num_bars == 0 {
                self.bounds = Some(BoundingBox::default());
                return self.bounds.unwrap();
            }

            match self.orientation {
                Orientation::Vertical => {
                    // X spans category centers at 1..n with half-bar padding
                    let min_x = 1.0 - self.bar_width * 0.5;
                    let max_x = num_bars as f32 + self.bar_width * 0.5;
                    // Y spans min/max of values and optional stack offsets
                    let (mut min_y, mut max_y) = (0.0f32, 0.0f32);
                    if let Some(offsets) = &self.stack_offsets {
                        for i in 0..num_bars {
                            let base = offsets[i] as f32;
                            let v = values[i];
                            if !v.is_finite() {
                                continue;
                            }
                            let top = base + v as f32;
                            min_y = min_y.min(base.min(top));
                            max_y = max_y.max(base.max(top));
                        }
                    } else {
                        for &v in values {
                            if !v.is_finite() {
                                continue;
                            }
                            min_y = min_y.min(v as f32);
                            max_y = max_y.max(v as f32);
                        }
                    }
                    self.bounds = Some(BoundingBox::new(
                        Vec3::new(min_x, min_y, 0.0),
                        Vec3::new(max_x, max_y, 0.0),
                    ));
                }
                Orientation::Horizontal => {
                    // Y spans category centers at 1..n with half-bar padding
                    let min_y = 1.0 - self.bar_width * 0.5;
                    let max_y = num_bars as f32 + self.bar_width * 0.5;
                    // X spans min/max of values and optional stack offsets
                    let (mut min_x, mut max_x) = (0.0f32, 0.0f32);
                    if let Some(offsets) = &self.stack_offsets {
                        for i in 0..num_bars {
                            let base = offsets[i] as f32;
                            let v = values[i];
                            if !v.is_finite() {
                                continue;
                            }
                            let right = base + v as f32;
                            min_x = min_x.min(base.min(right));
                            max_x = max_x.max(base.max(right));
                        }
                    } else {
                        for &v in values {
                            if !v.is_finite() {
                                continue;
                            }
                            min_x = min_x.min(v as f32);
                            max_x = max_x.max(v as f32);
                        }
                    }
                    self.bounds = Some(BoundingBox::new(
                        Vec3::new(min_x, min_y, 0.0),
                        Vec3::new(max_x, max_y, 0.0),
                    ));
                }
            }
        }
        self.bounds.unwrap()
    }

    /// Generate complete render data for the graphics pipeline
    pub fn render_data(&mut self) -> RenderData {
        let using_gpu = self.gpu_vertices.is_some();
        let gpu_vertices = self.gpu_vertices.clone();
        let (vertices, indices, vertex_count) = if using_gpu {
            let count = self
                .gpu_vertex_count
                .or_else(|| gpu_vertices.as_ref().map(|buf| buf.vertex_count))
                .unwrap_or(0);
            (Vec::new(), None, count)
        } else {
            let (verts, inds) = self.generate_vertices();
            (verts.clone(), Some(inds.clone()), verts.len())
        };

        let material = Material {
            albedo: self.color,
            ..Default::default()
        };

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: indices.as_ref().map(|_| 0),
            index_count: indices.as_ref().map(|ind| ind.len()),
            instance_count: 1,
        };

        RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices,
            indices,
            gpu_vertices,
            material,
            draw_calls: vec![draw_call],
            image: None,
        }
    }

    /// Get chart statistics for debugging
    pub fn statistics(&self) -> BarChartStatistics {
        let (bar_count, value_range) = if let Some(values) = &self.values {
            if values.is_empty() {
                (0, (0.0, 0.0))
            } else {
                let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                (values.len(), (min_val, max_val))
            }
        } else if let Some(bounds) = self.gpu_bounds.or(self.bounds) {
            (self.value_count, (bounds.min.y as f64, bounds.max.y as f64))
        } else {
            (self.value_count, (0.0, 0.0))
        };

        BarChartStatistics {
            bar_count,
            value_range,
            memory_usage: self.estimated_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let labels_size: usize = self.labels.iter().map(|s| s.len()).sum();
        let values_size = self
            .values
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<f64>());
        let vertices_size = self
            .vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>());
        let indices_size = self
            .indices
            .as_ref()
            .map_or(0, |i| i.len() * std::mem::size_of::<u32>());

        labels_size + values_size + vertices_size + indices_size
    }
}

/// Bar chart statistics
#[derive(Debug, Clone)]
pub struct BarChartStatistics {
    pub bar_count: usize,
    pub value_range: (f64, f64),
    pub memory_usage: usize,
}

/// MATLAB-compatible bar chart creation utilities
pub mod matlab_compat {
    use super::*;

    /// Create a simple bar chart (equivalent to MATLAB's `bar(values)`)
    pub fn bar(values: Vec<f64>) -> Result<BarChart, String> {
        let labels: Vec<String> = (1..=values.len()).map(|i| i.to_string()).collect();
        BarChart::new(labels, values)
    }

    /// Create a bar chart with custom labels (`bar(labels, values)`)
    pub fn bar_with_labels(labels: Vec<String>, values: Vec<f64>) -> Result<BarChart, String> {
        BarChart::new(labels, values)
    }

    /// Create a bar chart with specified color
    pub fn bar_with_color(values: Vec<f64>, color: &str) -> Result<BarChart, String> {
        let color_vec = parse_matlab_color(color)?;
        let labels: Vec<String> = (1..=values.len()).map(|i| i.to_string()).collect();
        Ok(BarChart::new(labels, values)?.with_style(color_vec, 0.8))
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
    fn test_bar_chart_creation() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let values = vec![10.0, 25.0, 15.0];

        let chart = BarChart::new(labels.clone(), values.clone()).unwrap();

        assert_eq!(chart.labels, labels);
        assert_eq!(chart.values.as_ref(), Some(&values));
        assert_eq!(chart.len(), 3);
        assert!(!chart.is_empty());
        assert!(chart.visible);
    }

    #[test]
    fn test_bar_chart_data_validation() {
        // Mismatched lengths should fail
        let labels = vec!["A".to_string(), "B".to_string()];
        let values = vec![10.0, 25.0, 15.0];
        assert!(BarChart::new(labels, values).is_err());

        // Empty data should fail
        let empty_labels: Vec<String> = vec![];
        let empty_values: Vec<f64> = vec![];
        assert!(BarChart::new(empty_labels, empty_values).is_err());
    }

    #[test]
    fn test_bar_chart_styling() {
        let labels = vec!["X".to_string(), "Y".to_string()];
        let values = vec![5.0, 10.0];
        let color = Vec4::new(1.0, 0.0, 0.0, 1.0);

        let chart = BarChart::new(labels, values)
            .unwrap()
            .with_style(color, 0.6)
            .with_outline(Vec4::new(0.0, 0.0, 0.0, 1.0), 2.0)
            .with_label("Test Chart");

        assert_eq!(chart.color, color);
        assert_eq!(chart.bar_width, 0.6);
        assert_eq!(chart.outline_color, Some(Vec4::new(0.0, 0.0, 0.0, 1.0)));
        assert_eq!(chart.outline_width, 2.0);
        assert_eq!(chart.label, Some("Test Chart".to_string()));
    }

    #[test]
    fn test_bar_chart_bounds() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let values = vec![5.0, -2.0, 8.0];

        let mut chart = BarChart::new(labels, values).unwrap();
        let bounds = chart.bounds();

        // X bounds should span all bars (centers are 1-based)
        assert!(bounds.min.x < 1.0);
        assert!(bounds.max.x > 3.0);

        // Y bounds should include negative and positive values
        assert_eq!(bounds.min.y, -2.0);
        assert_eq!(bounds.max.y, 8.0);
    }

    #[test]
    fn test_bar_chart_vertex_generation() {
        let labels = vec!["A".to_string(), "B".to_string()];
        let values = vec![3.0, 5.0];

        let mut chart = BarChart::new(labels, values).unwrap();
        let (vertices, indices) = chart.generate_vertices();

        // Should have 4 vertices per bar (rectangle)
        assert_eq!(vertices.len(), 8);

        // Should have 6 indices per bar (2 triangles)
        assert_eq!(indices.len(), 12);

        // Check first bar vertices are reasonable
        assert_eq!(vertices[0].position[1], 0.0); // Bottom
        assert_eq!(vertices[2].position[1], 3.0); // Top of first bar
    }

    #[test]
    fn test_bar_chart_render_data() {
        let labels = vec!["Test".to_string()];
        let values = vec![10.0];

        let mut chart = BarChart::new(labels, values).unwrap();
        let render_data = chart.render_data();

        assert_eq!(render_data.pipeline_type, PipelineType::Triangles);
        assert_eq!(render_data.vertices.len(), 4); // One rectangle
        assert!(render_data.indices.is_some());
        assert_eq!(render_data.indices.as_ref().unwrap().len(), 6); // Two triangles
    }

    #[test]
    fn test_bar_chart_statistics() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let values = vec![1.0, 5.0, 3.0];

        let chart = BarChart::new(labels, values).unwrap();
        let stats = chart.statistics();

        assert_eq!(stats.bar_count, 3);
        assert_eq!(stats.value_range, (1.0, 5.0));
        assert!(stats.memory_usage > 0);
    }

    #[test]
    fn test_matlab_compat_bar() {
        use super::matlab_compat::*;

        let values = vec![1.0, 3.0, 2.0];

        let chart1 = bar(values.clone()).unwrap();
        assert_eq!(chart1.len(), 3);
        assert_eq!(chart1.labels, vec!["1", "2", "3"]);

        let labels = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let chart2 = bar_with_labels(labels.clone(), values.clone()).unwrap();
        assert_eq!(chart2.labels, labels);

        let chart3 = bar_with_color(values, "g").unwrap();
        assert_eq!(chart3.color, Vec4::new(0.0, 1.0, 0.0, 1.0));
    }
}
