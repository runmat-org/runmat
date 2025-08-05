//! Line plot implementation
//! 
//! High-performance line plotting with GPU acceleration and MATLAB-compatible styling.

use crate::core::{vertex_utils, Vertex, PipelineType, RenderData, Material, DrawCall, BoundingBox};
use glam::{Vec3, Vec4};

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
    
    /// Metadata
    pub label: Option<String>,
    pub visible: bool,
    
    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

/// Line rendering styles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
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
                x_data.len(), y_data.len()
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
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
        })
    }
    
    /// Create a line plot with custom styling
    pub fn with_style(mut self, color: Vec4, line_width: f32, line_style: LineStyle) -> Self {
        self.color = color;
        self.line_width = line_width;
        self.line_style = line_style;
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
                x_data.len(), y_data.len()
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
    
    /// Set the color of the line
    pub fn set_color(&mut self, color: Vec4) {
        self.color = color;
        self.dirty = true;
    }
    
    /// Set the line width
    pub fn set_line_width(&mut self, width: f32) {
        self.line_width = width.max(0.1); // Minimum line width
        self.dirty = true;
    }
    
    /// Set the line style
    pub fn set_line_style(&mut self, style: LineStyle) {
        self.line_style = style;
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
            self.vertices = Some(vertex_utils::create_line_plot(&self.x_data, &self.y_data, self.color));
            self.dirty = false;
        }
        self.vertices.as_ref().unwrap()
    }
    
    /// Get the bounding box of the data
    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            let points: Vec<Vec3> = self.x_data.iter()
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
        
        let mut material = Material::default();
        material.albedo = self.color;
        
        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        };
        
        RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            material,
            draw_calls: vec![draw_call],
        }
    }
    
    /// Get plot statistics for debugging
    pub fn statistics(&self) -> PlotStatistics {
        let (min_x, max_x) = self.x_data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
            (min.min(x), max.max(x))
        });
        let (min_y, max_y) = self.y_data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &y| {
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
        std::mem::size_of::<f64>() * (self.x_data.len() + self.y_data.len()) +
        self.vertices.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<Vertex>())
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
            _ => Err(format!("Unknown color: {}", color)),
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
        
        let plot = LinePlot::new(x, y).unwrap()
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
}