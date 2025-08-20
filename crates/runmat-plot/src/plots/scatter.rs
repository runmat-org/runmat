//! Scatter plot implementation
//!
//! High-performance scatter plotting with GPU acceleration.

use crate::core::{
    vertex_utils, BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex,
};
use glam::{Vec3, Vec4};

/// High-performance GPU-accelerated scatter plot
#[derive(Debug, Clone)]
pub struct ScatterPlot {
    /// Raw data points (x, y coordinates)
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,

    /// Visual styling
    pub color: Vec4,
    pub marker_size: f32,
    pub marker_style: MarkerStyle,

    /// Metadata
    pub label: Option<String>,
    pub visible: bool,

    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

/// Marker styles for scatter plots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerStyle {
    Circle,
    Square,
    Triangle,
    Diamond,
    Plus,
    Cross,
    Star,
    Hexagon,
}

impl Default for MarkerStyle {
    fn default() -> Self {
        Self::Circle
    }
}

impl ScatterPlot {
    /// Create a new scatter plot with data
    pub fn new(x_data: Vec<f64>, y_data: Vec<f64>) -> Result<Self, String> {
        if x_data.len() != y_data.len() {
            return Err(format!(
                "Data length mismatch: x_data has {} points, y_data has {} points",
                x_data.len(),
                y_data.len()
            ));
        }

        if x_data.is_empty() {
            return Err("Cannot create scatter plot with empty data".to_string());
        }

        Ok(Self {
            x_data,
            y_data,
            color: Vec4::new(1.0, 0.0, 0.0, 1.0), // Default red
            marker_size: 3.0,
            marker_style: MarkerStyle::default(),
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
        })
    }

    /// Create a scatter plot with custom styling
    pub fn with_style(mut self, color: Vec4, marker_size: f32, marker_style: MarkerStyle) -> Self {
        self.color = color;
        self.marker_size = marker_size;
        self.marker_style = marker_style;
        self.dirty = true;
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
        Ok(())
    }

    /// Set the color of the markers
    pub fn set_color(&mut self, color: Vec4) {
        self.color = color;
        self.dirty = true;
    }

    /// Set the marker size
    pub fn set_marker_size(&mut self, size: f32) {
        self.marker_size = size.max(0.1); // Minimum marker size
        self.dirty = true;
    }

    /// Set the marker style
    pub fn set_marker_style(&mut self, style: MarkerStyle) {
        self.marker_style = style;
        self.dirty = true;
    }

    /// Show or hide the plot
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.x_data.len()
    }

    /// Check if the plot has no data
    pub fn is_empty(&self) -> bool {
        self.x_data.is_empty()
    }

    /// Generate vertices for GPU rendering
    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.dirty || self.vertices.is_none() {
            self.vertices = Some(vertex_utils::create_scatter_plot(
                &self.x_data,
                &self.y_data,
                self.color,
            ));
            self.dirty = false;
        }
        self.vertices.as_ref().unwrap()
    }

    /// Get the bounding box of the data
    pub fn bounds(&mut self) -> BoundingBox {
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

    /// Generate complete render data for the graphics pipeline
    pub fn render_data(&mut self) -> RenderData {
        let vertices = self.generate_vertices().clone();
        let vertex_count = vertices.len();

        let material = Material {
            albedo: self.color,
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
            pipeline_type: PipelineType::Points,
            vertices,
            indices: None,
            material,
            draw_calls: vec![draw_call],
        }
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

/// MATLAB-compatible scatter plot creation utilities
pub mod matlab_compat {
    use super::*;

    /// Create a simple scatter plot (equivalent to MATLAB's `scatter(x, y)`)
    pub fn scatter(x: Vec<f64>, y: Vec<f64>) -> Result<ScatterPlot, String> {
        ScatterPlot::new(x, y)
    }

    /// Create a scatter plot with specified color and size (`scatter(x, y, size, color)`)
    pub fn scatter_with_style(
        x: Vec<f64>,
        y: Vec<f64>,
        size: f32,
        color: &str,
    ) -> Result<ScatterPlot, String> {
        let color_vec = parse_matlab_color(color)?;
        Ok(ScatterPlot::new(x, y)?.with_style(color_vec, size, MarkerStyle::Circle))
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
    fn test_scatter_plot_creation() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];

        let plot = ScatterPlot::new(x.clone(), y.clone()).unwrap();

        assert_eq!(plot.x_data, x);
        assert_eq!(plot.y_data, y);
        assert_eq!(plot.len(), 4);
        assert!(!plot.is_empty());
        assert!(plot.visible);
    }

    #[test]
    fn test_scatter_plot_styling() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 1.5];
        let color = Vec4::new(0.0, 1.0, 0.0, 1.0);

        let plot = ScatterPlot::new(x, y)
            .unwrap()
            .with_style(color, 5.0, MarkerStyle::Square)
            .with_label("Test Scatter");

        assert_eq!(plot.color, color);
        assert_eq!(plot.marker_size, 5.0);
        assert_eq!(plot.marker_style, MarkerStyle::Square);
        assert_eq!(plot.label, Some("Test Scatter".to_string()));
    }

    #[test]
    fn test_scatter_plot_render_data() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 1.0];

        let mut plot = ScatterPlot::new(x, y).unwrap();
        let render_data = plot.render_data();

        assert_eq!(render_data.pipeline_type, PipelineType::Points);
        assert_eq!(render_data.vertices.len(), 3); // One vertex per point
        assert!(render_data.indices.is_none());
        assert_eq!(render_data.draw_calls.len(), 1);
    }

    #[test]
    fn test_matlab_compat_scatter() {
        use super::matlab_compat::*;

        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];

        let basic_scatter = scatter(x.clone(), y.clone()).unwrap();
        assert_eq!(basic_scatter.len(), 2);

        let styled_scatter = scatter_with_style(x.clone(), y.clone(), 5.0, "g").unwrap();
        assert_eq!(styled_scatter.color, Vec4::new(0.0, 1.0, 0.0, 1.0));
        assert_eq!(styled_scatter.marker_size, 5.0);
    }
}
