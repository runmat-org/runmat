//! 3D point cloud visualization
//! 
//! High-performance GPU-accelerated 3D point cloud rendering for scatter data.

use crate::core::{Vertex, BoundingBox};
use crate::plots::surface::ColorMap;
use glam::{Vec3, Vec4};

/// Point cloud plot for 3D scatter data
#[derive(Debug, Clone)]
pub struct PointCloudPlot {
    /// Point positions
    pub positions: Vec<Vec3>,
    
    /// Per-point data
    pub values: Option<Vec<f64>>,
    pub colors: Option<Vec<Vec4>>,
    pub sizes: Option<Vec<f32>>,
    
    /// Global styling
    pub default_color: Vec4,
    pub default_size: f32,
    pub colormap: ColorMap,
    
    /// Point rendering
    pub point_style: PointStyle,
    pub size_mode: SizeMode,
    
    /// Metadata
    pub label: Option<String>,
    pub visible: bool,
    
    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

/// Point rendering styles
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointStyle {
    /// Circular points
    Circle,
    /// Square points
    Square,
    /// 3D spheres (higher quality)
    Sphere,
    /// Custom mesh
    Custom,
}

/// Point size modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeMode {
    /// Fixed size for all points
    Fixed,
    /// Size proportional to value
    Proportional,
    /// Size scaled by distance from camera
    Perspective,
}

impl Default for PointStyle {
    fn default() -> Self {
        Self::Circle
    }
}

impl Default for SizeMode {
    fn default() -> Self {
        Self::Fixed
    }
}

impl PointCloudPlot {
    /// Create a new point cloud from positions
    pub fn new(positions: Vec<Vec3>) -> Self {
        Self {
            positions,
            values: None,
            colors: None,
            sizes: None,
            default_color: Vec4::new(0.0, 0.5, 1.0, 1.0), // Blue
            default_size: 3.0,
            colormap: ColorMap::Viridis,
            point_style: PointStyle::default(),
            size_mode: SizeMode::default(),
            label: None,
            visible: true,
            vertices: None,
            bounds: None,
            dirty: true,
        }
    }
    
    /// Create point cloud with values for color mapping
    pub fn with_values(mut self, values: Vec<f64>) -> Result<Self, String> {
        if values.len() != self.positions.len() {
            return Err(format!(
                "Values length ({}) must match positions length ({})",
                values.len(), self.positions.len()
            ));
        }
        self.values = Some(values);
        self.dirty = true;
        Ok(self)
    }
    
    /// Create point cloud with explicit colors
    pub fn with_colors(mut self, colors: Vec<Vec4>) -> Result<Self, String> {
        if colors.len() != self.positions.len() {
            return Err(format!(
                "Colors length ({}) must match positions length ({})",
                colors.len(), self.positions.len()
            ));
        }
        self.colors = Some(colors);
        self.dirty = true;
        Ok(self)
    }
    
    /// Create point cloud with variable sizes
    pub fn with_sizes(mut self, sizes: Vec<f32>) -> Result<Self, String> {
        if sizes.len() != self.positions.len() {
            return Err(format!(
                "Sizes length ({}) must match positions length ({})",
                sizes.len(), self.positions.len()
            ));
        }
        self.sizes = Some(sizes);
        self.dirty = true;
        Ok(self)
    }
    
    /// Set default color for all points
    pub fn with_default_color(mut self, color: Vec4) -> Self {
        self.default_color = color;
        self.dirty = true;
        self
    }
    
    /// Set default size for all points
    pub fn with_default_size(mut self, size: f32) -> Self {
        self.default_size = size.max(0.1);
        self.dirty = true;
        self
    }
    
    /// Set colormap for value-based coloring
    pub fn with_colormap(mut self, colormap: ColorMap) -> Self {
        self.colormap = colormap;
        self.dirty = true;
        self
    }
    
    /// Set point rendering style
    pub fn with_point_style(mut self, style: PointStyle) -> Self {
        self.point_style = style;
        self.dirty = true;
        self
    }
    
    /// Set size mode
    pub fn with_size_mode(mut self, mode: SizeMode) -> Self {
        self.size_mode = mode;
        self.dirty = true;
        self
    }
    
    /// Set plot label for legends
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }
    
    /// Get the number of points
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    
    /// Check if the point cloud has no data
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
    
    /// Generate vertices for GPU rendering
    pub fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.dirty || self.vertices.is_none() {
            self.compute_vertices();
            self.dirty = false;
        }
        self.vertices.as_ref().unwrap()
    }
    
    /// Get the bounding box of the point cloud
    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            self.compute_bounds();
        }
        self.bounds.unwrap()
    }
    
    /// Compute vertices
    fn compute_vertices(&mut self) {
        let mut vertices = Vec::with_capacity(self.positions.len());
        
        // Compute value range for color mapping
        let (value_min, value_max) = if let Some(ref values) = self.values {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        } else {
            (0.0, 1.0)
        };
        
        for (i, &position) in self.positions.iter().enumerate() {
            // Determine color
            let color = if let Some(ref colors) = self.colors {
                colors[i]
            } else if let Some(ref values) = self.values {
                let normalized = if value_max > value_min {
                    ((values[i] - value_min) / (value_max - value_min)).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let rgb = self.colormap.map_value(normalized as f32);
                Vec4::new(rgb.x, rgb.y, rgb.z, self.default_color.w)
            } else {
                self.default_color
            };
            
            // Determine size (encoded in normal.x for now)
            let size = if let Some(ref sizes) = self.sizes {
                sizes[i]
            } else {
                match self.size_mode {
                    SizeMode::Fixed => self.default_size,
                    SizeMode::Proportional => {
                        if let Some(ref values) = self.values {
                            let normalized = if value_max > value_min {
                                ((values[i] - value_min) / (value_max - value_min)).clamp(0.0, 1.0)
                            } else {
                                0.5
                            };
                            self.default_size * (0.5 + normalized as f32)
                        } else {
                            self.default_size
                        }
                    },
                    SizeMode::Perspective => self.default_size, // Camera distance handled in shader
                }
            };
            
            vertices.push(Vertex {
                position: position.to_array(),
                color: color.to_array(),
                normal: [size, 0.0, 0.0], // Store size in normal.x
                tex_coords: [i as f32, 0.0], // Point index for potential lookup
            });
        }
        
        self.vertices = Some(vertices);
    }
    
    /// Compute bounding box
    fn compute_bounds(&mut self) {
        if self.positions.is_empty() {
            self.bounds = Some(BoundingBox::new(Vec3::ZERO, Vec3::ZERO));
            return;
        }
        
        let mut min = self.positions[0];
        let mut max = self.positions[0];
        
        for &pos in &self.positions[1..] {
            min = min.min(pos);
            max = max.max(pos);
        }
        
        // Expand bounds slightly to account for point size
        let expansion = Vec3::splat(self.default_size * 0.01);
        min -= expansion;
        max += expansion;
        
        self.bounds = Some(BoundingBox::new(min, max));
    }
    
    // TODO: Implement render_data once core rendering types are available
    
    /// Get plot statistics for debugging
    pub fn statistics(&self) -> PointCloudStatistics {
        PointCloudStatistics {
            point_count: self.positions.len(),
            has_values: self.values.is_some(),
            has_colors: self.colors.is_some(),
            has_sizes: self.sizes.is_some(),
            memory_usage: self.estimated_memory_usage(),
        }
    }
    
    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let positions_size = self.positions.len() * std::mem::size_of::<Vec3>();
        let values_size = self.values.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<f64>());
        let colors_size = self.colors.as_ref().map_or(0, |c| c.len() * std::mem::size_of::<Vec4>());
        let sizes_size = self.sizes.as_ref().map_or(0, |s| s.len() * std::mem::size_of::<f32>());
        let vertices_size = self.vertices.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<Vertex>());
        
        positions_size + values_size + colors_size + sizes_size + vertices_size
    }
}

/// Point cloud performance and data statistics
#[derive(Debug, Clone)]
pub struct PointCloudStatistics {
    pub point_count: usize,
    pub has_values: bool,
    pub has_colors: bool,
    pub has_sizes: bool,
    pub memory_usage: usize,
}

/// MATLAB-compatible point cloud creation utilities
pub mod matlab_compat {
    use super::*;
    
    /// Create a 3D scatter plot (equivalent to MATLAB's `scatter3(x, y, z)`)
    pub fn scatter3(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Result<PointCloudPlot, String> {
        if x.len() != y.len() || y.len() != z.len() {
            return Err("X, Y, and Z vectors must have the same length".to_string());
        }
        
        let positions: Vec<Vec3> = x.into_iter()
            .zip(y.into_iter())
            .zip(z.into_iter())
            .map(|((x, y), z)| Vec3::new(x as f32, y as f32, z as f32))
            .collect();
        
        Ok(PointCloudPlot::new(positions))
    }
    
    /// Create scatter3 with colors
    pub fn scatter3_with_colors(
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        colors: Vec<Vec4>,
    ) -> Result<PointCloudPlot, String> {
        scatter3(x, y, z)?.with_colors(colors)
    }
    
    /// Create scatter3 with values for color mapping
    pub fn scatter3_with_values(
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        values: Vec<f64>,
        colormap: &str,
    ) -> Result<PointCloudPlot, String> {
        let cmap = match colormap {
            "jet" => ColorMap::Jet,
            "hot" => ColorMap::Hot,
            "cool" => ColorMap::Cool,
            "viridis" => ColorMap::Viridis,
            "plasma" => ColorMap::Plasma,
            "gray" | "grey" => ColorMap::Gray,
            _ => return Err(format!("Unknown colormap: {}", colormap)),
        };
        
        Ok(scatter3(x, y, z)?.with_values(values)?.with_colormap(cmap))
    }
    
    /// Create point cloud from matrix data
    pub fn point_cloud_from_matrix(points: Vec<Vec<f64>>) -> Result<PointCloudPlot, String> {
        if points.is_empty() {
            return Err("Points matrix cannot be empty".to_string());
        }
        
        let dim = points[0].len();
        if dim < 3 {
            return Err("Points must have at least 3 dimensions (X, Y, Z)".to_string());
        }
        
        let positions: Vec<Vec3> = points.into_iter()
            .map(|point| {
                if point.len() != dim {
                    Vec3::ZERO // Handle inconsistent dimensions gracefully
                } else {
                    Vec3::new(point[0] as f32, point[1] as f32, point[2] as f32)
                }
            })
            .collect();
        
        Ok(PointCloudPlot::new(positions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_point_cloud_creation() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 2.0, 2.0),
        ];
        
        let cloud = PointCloudPlot::new(positions.clone());
        
        assert_eq!(cloud.positions, positions);
        assert_eq!(cloud.len(), 3);
        assert!(!cloud.is_empty());
        assert!(cloud.visible);
        assert!(cloud.values.is_none());
        assert!(cloud.colors.is_none());
        assert!(cloud.sizes.is_none());
    }
    
    #[test]
    fn test_point_cloud_with_values() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let values = vec![0.5, 1.5];
        
        let cloud = PointCloudPlot::new(positions)
            .with_values(values.clone())
            .unwrap();
        
        assert_eq!(cloud.values, Some(values));
    }
    
    #[test]
    fn test_point_cloud_with_colors() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let colors = vec![
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
        ];
        
        let cloud = PointCloudPlot::new(positions)
            .with_colors(colors.clone())
            .unwrap();
        
        assert_eq!(cloud.colors, Some(colors));
    }
    
    #[test]
    fn test_point_cloud_validation() {
        let positions = vec![Vec3::new(0.0, 0.0, 0.0)];
        let wrong_values = vec![1.0, 2.0]; // Length mismatch
        
        let result = PointCloudPlot::new(positions).with_values(wrong_values);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_point_cloud_styling() {
        let positions = vec![Vec3::new(0.0, 0.0, 0.0)];
        
        let cloud = PointCloudPlot::new(positions)
            .with_default_color(Vec4::new(1.0, 0.0, 0.0, 1.0))
            .with_default_size(5.0)
            .with_colormap(ColorMap::Hot)
            .with_point_style(PointStyle::Sphere)
            .with_size_mode(SizeMode::Proportional)
            .with_label("Test Cloud");
        
        assert_eq!(cloud.default_color, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(cloud.default_size, 5.0);
        assert_eq!(cloud.colormap, ColorMap::Hot);
        assert_eq!(cloud.point_style, PointStyle::Sphere);
        assert_eq!(cloud.size_mode, SizeMode::Proportional);
        assert_eq!(cloud.label, Some("Test Cloud".to_string()));
    }
    
    #[test]
    fn test_point_cloud_bounds() {
        let positions = vec![
            Vec3::new(-1.0, -2.0, -3.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        
        let mut cloud = PointCloudPlot::new(positions);
        let bounds = cloud.bounds();
        
        // Should be slightly expanded from actual bounds
        assert!(bounds.min.x <= -1.0);
        assert!(bounds.min.y <= -2.0);
        assert!(bounds.min.z <= -3.0);
        assert!(bounds.max.x >= 1.0);
        assert!(bounds.max.y >= 2.0);
        assert!(bounds.max.z >= 3.0);
    }
    
    #[test]
    fn test_point_cloud_vertex_generation() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        
        let mut cloud = PointCloudPlot::new(positions);
        let vertices = cloud.generate_vertices();
        
        assert_eq!(vertices.len(), 2);
        assert_eq!(vertices[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(vertices[1].position, [1.0, 1.0, 1.0]);
    }
    
    #[test]
    fn test_point_cloud_statistics() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 2.0, 2.0),
        ];
        let values = vec![0.0, 1.0, 2.0];
        
        let cloud = PointCloudPlot::new(positions)
            .with_values(values)
            .unwrap();
        
        let stats = cloud.statistics();
        
        assert_eq!(stats.point_count, 3);
        assert!(stats.has_values);
        assert!(!stats.has_colors);
        assert!(!stats.has_sizes);
        assert!(stats.memory_usage > 0);
    }
    
    #[test]
    fn test_matlab_compat() {
        use super::matlab_compat::*;
        
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let z = vec![0.0, 1.0, 2.0];
        
        let cloud = scatter3(x.clone(), y.clone(), z.clone()).unwrap();
        assert_eq!(cloud.len(), 3);
        
        let values = vec![0.0, 0.5, 1.0];
        let cloud_with_values = scatter3_with_values(x, y, z, values, "viridis").unwrap();
        assert!(cloud_with_values.values.is_some());
        assert_eq!(cloud_with_values.colormap, ColorMap::Viridis);
    }
}