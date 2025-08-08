//! Bar chart implementation
//!
//! High-performance bar charts with GPU acceleration and MATLAB-compatible styling.

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};

/// High-performance GPU-accelerated bar chart
#[derive(Debug, Clone)]
pub struct BarChart {
    /// Category labels and values
    pub labels: Vec<String>,
    pub values: Vec<f64>,

    /// Visual styling
    pub color: Vec4,
    pub bar_width: f32,
    pub outline_color: Option<Vec4>,
    pub outline_width: f32,

    /// Metadata
    pub label: Option<String>,
    pub visible: bool,

    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    indices: Option<Vec<u32>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
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

        Ok(Self {
            labels,
            values,
            color: Vec4::new(0.0, 0.5, 1.0, 1.0), // Default blue
            bar_width: 0.8,                       // 80% of available space
            outline_color: None,
            outline_width: 1.0,
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: None,
            dirty: true,
        })
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
        self.values = values;
        self.dirty = true;
        Ok(())
    }

    /// Set the bar color
    pub fn set_color(&mut self, color: Vec4) {
        self.color = color;
        self.dirty = true;
    }

    /// Set the bar width (0.1 to 1.0)
    pub fn set_bar_width(&mut self, width: f32) {
        self.bar_width = width.clamp(0.1, 1.0);
        self.dirty = true;
    }

    /// Show or hide the chart
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Get the number of bars
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Check if the chart has no data
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Generate vertices for GPU rendering
    pub fn generate_vertices(&mut self) -> (&Vec<Vertex>, &Vec<u32>) {
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
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let _bar_spacing = 1.0; // Space between bar centers
        let half_width = self.bar_width * 0.5;

        for (i, &value) in self.values.iter().enumerate() {
            let x_center = i as f32; // Bar position along X axis
            let left = x_center - half_width;
            let right = x_center + half_width;
            let bottom = 0.0; // Baseline
            let top = value as f32;

            // Create rectangle vertices for this bar (4 vertices per bar)
            let vertex_offset = vertices.len() as u32;

            // Bottom left
            vertices.push(Vertex::new(Vec3::new(left, bottom, 0.0), self.color));

            // Bottom right
            vertices.push(Vertex::new(Vec3::new(right, bottom, 0.0), self.color));

            // Top right
            vertices.push(Vertex::new(Vec3::new(right, top, 0.0), self.color));

            // Top left
            vertices.push(Vertex::new(Vec3::new(left, top, 0.0), self.color));

            // Create indices for two triangles per bar (6 indices per bar)
            // Triangle 1: bottom-left, bottom-right, top-right
            indices.push(vertex_offset);
            indices.push(vertex_offset + 1);
            indices.push(vertex_offset + 2);

            // Triangle 2: bottom-left, top-right, top-left
            indices.push(vertex_offset);
            indices.push(vertex_offset + 2);
            indices.push(vertex_offset + 3);
        }

        (vertices, indices)
    }

    /// Get the bounding box of the chart
    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            let num_bars = self.values.len();
            if num_bars == 0 {
                self.bounds = Some(BoundingBox::default());
                return self.bounds.unwrap();
            }

            let min_x = -self.bar_width * 0.5;
            let max_x = (num_bars - 1) as f32 + self.bar_width * 0.5;

            let min_y = self.values.iter().fold(0.0f64, |acc, &val| acc.min(val)) as f32;
            let max_y = self.values.iter().fold(0.0f64, |acc, &val| acc.max(val)) as f32;

            self.bounds = Some(BoundingBox::new(
                Vec3::new(min_x, min_y, 0.0),
                Vec3::new(max_x, max_y, 0.0),
            ));
        }
        self.bounds.unwrap()
    }

    /// Generate complete render data for the graphics pipeline
    pub fn render_data(&mut self) -> RenderData {
        let (vertices, indices) = self.generate_vertices();
        let vertices = vertices.clone();
        let indices = indices.clone();

        let material = Material { albedo: self.color, ..Default::default() };

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count: vertices.len(),
            index_offset: Some(0),
            index_count: Some(indices.len()),
            instance_count: 1,
        };

        RenderData {
            pipeline_type: PipelineType::Triangles,
            vertices,
            indices: Some(indices),
            material,
            draw_calls: vec![draw_call],
        }
    }

    /// Get chart statistics for debugging
    pub fn statistics(&self) -> BarChartStatistics {
        let value_range = if self.values.is_empty() {
            (0.0, 0.0)
        } else {
            let min_val = self.values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = self.values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            (min_val, max_val)
        };

        BarChartStatistics {
            bar_count: self.values.len(),
            value_range,
            memory_usage: self.estimated_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let labels_size: usize = self.labels.iter().map(|s| s.len()).sum();
        let values_size = self.values.len() * std::mem::size_of::<f64>();
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
        assert_eq!(chart.values, values);
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

        // X bounds should span all bars
        assert!(bounds.min.x < 0.0);
        assert!(bounds.max.x > 2.0);

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
