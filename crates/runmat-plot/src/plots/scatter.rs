//! Scatter plot implementation
//!
//! High-performance scatter plotting with GPU acceleration.

use crate::core::{
    vertex_utils, BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex,
};
use crate::plots::surface::ColorMap;
use glam::{Vec3, Vec4};

/// High-performance GPU-accelerated scatter plot
#[derive(Debug, Clone)]
pub struct ScatterPlot {
    /// Raw data points (x, y coordinates)
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,

    /// Visual styling
    pub color: Vec4,
    pub edge_color: Vec4,
    pub edge_thickness: f32,
    pub marker_size: f32,
    pub marker_style: MarkerStyle,
    pub per_point_sizes: Option<Vec<f32>>,      // pixel diameters per point
    pub per_point_colors: Option<Vec<Vec4>>,    // per-point RGBA
    pub color_values: Option<Vec<f64>>,         // scalar values mapped by colormap
    pub color_limits: Option<(f64, f64)>,
    pub colormap: ColorMap,
    pub filled: bool,

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
            color: Vec4::new(1.0, 0.2, 0.2, 1.0), // Brighter red
            edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            edge_thickness: 1.0,
            marker_size: 12.0,
            marker_style: MarkerStyle::default(),
            per_point_sizes: None,
            per_point_colors: None,
            color_values: None,
            color_limits: None,
            colormap: ColorMap::Parula,
            filled: false,
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

    /// Set marker face color
    pub fn set_face_color(&mut self, color: Vec4) { self.color = color; self.dirty = true; }
    /// Set marker edge color
    pub fn set_edge_color(&mut self, color: Vec4) { self.edge_color = color; self.dirty = true; }
    /// Set marker edge thickness (pixels)
    pub fn set_edge_thickness(&mut self, px: f32) { self.edge_thickness = px.max(0.0); self.dirty = true; }
    pub fn set_sizes(&mut self, sizes: Vec<f32>) { self.per_point_sizes = Some(sizes); self.dirty = true; }
    pub fn set_colors(&mut self, colors: Vec<Vec4>) { self.per_point_colors = Some(colors); self.dirty = true; }
    pub fn set_color_values(&mut self, values: Vec<f64>, limits: Option<(f64, f64)>) { self.color_values = Some(values); self.color_limits = limits; self.dirty = true; }
    pub fn with_colormap(mut self, cmap: ColorMap) -> Self { self.colormap = cmap; self }
    pub fn set_filled(&mut self, filled: bool) { self.filled = filled; self.dirty = true; }

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
            let base_color = self.color;
            if self.per_point_colors.is_some() || self.color_values.is_some() { /* vertex color takes precedence; shader blends by face alpha */ }
            let mut verts = vertex_utils::create_scatter_plot(&self.x_data, &self.y_data, base_color);
            // per-point colors
            if let Some(ref colors) = self.per_point_colors {
                let m = colors.len().min(verts.len());
                for i in 0..m { verts[i].color = colors[i].to_array(); }
            } else if let Some(ref vals) = self.color_values {
                let n = verts.len();
                let (mut cmin, mut cmax) = if let Some(lims) = self.color_limits { lims } else {
                    let mut lo = f64::INFINITY; let mut hi = f64::NEG_INFINITY;
                    for &v in vals { if v.is_finite() { if v < lo { lo = v; } if v > hi { hi = v; } } }
                    if !lo.is_finite() || !hi.is_finite() || hi <= lo { (0.0, 1.0) } else { (lo, hi) }
                };
                if !(cmin.is_finite() && cmax.is_finite()) || cmax <= cmin { cmin = 0.0; cmax = 1.0; }
                let denom = (cmax - cmin).max(std::f64::EPSILON);
                for i in 0..n {
                    let t = ((vals[i] - cmin) / denom) as f32;
                    let rgb = self.colormap.map_value(t);
                    verts[i].color = [rgb.x, rgb.y, rgb.z, 1.0];
                }
            }
            // Store marker size in normal.z for direct point expansion
            if let Some(ref sizes) = self.per_point_sizes {
                for i in 0..verts.len() { let s = sizes.get(i).copied().unwrap_or(self.marker_size); verts[i].normal[2] = s.max(1.0); }
            } else {
                for v in &mut verts { v.normal[2] = self.marker_size.max(1.0); }
            }
            self.vertices = Some(verts);
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

        let mut material = Material { albedo: self.color, ..Default::default() };
        // If vertex colors vary across points, prefer per-vertex colors (alpha=0)
        let is_multi_color = {
            if vertices.is_empty() { false } else {
                let first = vertices[0].color;
                vertices.iter().any(|v| v.color != first)
            }
        };
        if is_multi_color || self.per_point_colors.is_some() || self.color_values.is_some() {
            material.albedo.w = 0.0;
        } else if self.filled {
            material.albedo.w = 1.0;
        }
        material.emissive = self.edge_color; // stash edge color
        material.roughness = self.edge_thickness; // stash thickness in roughness
        material.metallic = match self.marker_style {
            MarkerStyle::Circle => 0.0,
            MarkerStyle::Square => 1.0,
            MarkerStyle::Triangle => 2.0,
            MarkerStyle::Diamond => 3.0,
            MarkerStyle::Plus => 4.0,
            MarkerStyle::Cross => 5.0,
            MarkerStyle::Star => 6.0,
            MarkerStyle::Hexagon => 7.0,
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
            image: None,
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
