//! 3D surface plot implementation
//!
//! High-performance GPU-accelerated 3D surface rendering with MATLAB-compatible styling.

use crate::core::{BoundingBox, Vertex};
use glam::{Vec3, Vec4};

/// High-performance GPU-accelerated 3D surface plot
#[derive(Debug, Clone)]
pub struct SurfacePlot {
    /// Grid data (Z values at X,Y coordinates)
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
    pub z_data: Vec<Vec<f64>>, // z_data[i][j] = Z value at (x_data[i], y_data[j])

    /// Surface properties
    pub colormap: ColorMap,
    pub shading_mode: ShadingMode,
    pub wireframe: bool,
    pub alpha: f32,

    /// Lighting and material
    pub lighting_enabled: bool,
    pub ambient_strength: f32,
    pub diffuse_strength: f32,
    pub specular_strength: f32,
    pub shininess: f32,

    /// Metadata
    pub label: Option<String>,
    pub visible: bool,

    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    indices: Option<Vec<u32>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

/// Color mapping schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorMap {
    /// MATLAB-compatible colormaps
    Jet,
    Hot,
    Cool,
    Spring,
    Summer,
    Autumn,
    Winter,
    Gray,
    Bone,
    Copper,
    Pink,
    Lines,

    /// Scientific colormaps
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Turbo,

    /// Perceptually uniform
    Parula,

    /// Custom color ranges
    Custom(Vec4, Vec4), // (min_color, max_color)
}

/// Surface shading modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShadingMode {
    /// Flat shading (per-face normals)
    Flat,
    /// Smooth shading (interpolated normals)
    Smooth,
    /// Faceted (flat with visible edges)
    Faceted,
    /// No shading (just color mapping)
    None,
}

impl Default for ColorMap {
    fn default() -> Self {
        Self::Viridis
    }
}

impl Default for ShadingMode {
    fn default() -> Self {
        Self::Smooth
    }
}

impl SurfacePlot {
    /// Create a new surface plot from meshgrid data
    pub fn new(x_data: Vec<f64>, y_data: Vec<f64>, z_data: Vec<Vec<f64>>) -> Result<Self, String> {
        // Validate dimensions
        if z_data.len() != x_data.len() {
            return Err(format!(
                "Z data rows ({}) must match X data length ({})",
                z_data.len(),
                x_data.len()
            ));
        }

        for (i, row) in z_data.iter().enumerate() {
            if row.len() != y_data.len() {
                return Err(format!(
                    "Z data row {} length ({}) must match Y data length ({})",
                    i,
                    row.len(),
                    y_data.len()
                ));
            }
        }

        Ok(Self {
            x_data,
            y_data,
            z_data,
            colormap: ColorMap::default(),
            shading_mode: ShadingMode::default(),
            wireframe: false,
            alpha: 1.0,
            lighting_enabled: true,
            ambient_strength: 0.2,
            diffuse_strength: 0.8,
            specular_strength: 0.5,
            shininess: 32.0,
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: None,
            dirty: true,
        })
    }

    /// Create surface from a function
    pub fn from_function<F>(
        x_range: (f64, f64),
        y_range: (f64, f64),
        resolution: (usize, usize),
        func: F,
    ) -> Result<Self, String>
    where
        F: Fn(f64, f64) -> f64,
    {
        let (x_res, y_res) = resolution;
        if x_res < 2 || y_res < 2 {
            return Err("Resolution must be at least 2x2".to_string());
        }

        let x_data: Vec<f64> = (0..x_res)
            .map(|i| x_range.0 + (x_range.1 - x_range.0) * i as f64 / (x_res - 1) as f64)
            .collect();

        let y_data: Vec<f64> = (0..y_res)
            .map(|j| y_range.0 + (y_range.1 - y_range.0) * j as f64 / (y_res - 1) as f64)
            .collect();

        let z_data: Vec<Vec<f64>> = x_data
            .iter()
            .map(|&x| y_data.iter().map(|&y| func(x, y)).collect())
            .collect();

        Self::new(x_data, y_data, z_data)
    }

    /// Set color mapping
    pub fn with_colormap(mut self, colormap: ColorMap) -> Self {
        self.colormap = colormap;
        self.dirty = true;
        self
    }

    /// Set shading mode
    pub fn with_shading(mut self, shading: ShadingMode) -> Self {
        self.shading_mode = shading;
        self.dirty = true;
        self
    }

    /// Enable/disable wireframe
    pub fn with_wireframe(mut self, enabled: bool) -> Self {
        self.wireframe = enabled;
        self.dirty = true;
        self
    }

    /// Set transparency
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self.dirty = true;
        self
    }

    /// Set plot label for legends
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the number of grid points
    pub fn len(&self) -> usize {
        self.x_data.len() * self.y_data.len()
    }

    /// Check if the surface has no data
    pub fn is_empty(&self) -> bool {
        self.x_data.is_empty() || self.y_data.is_empty()
    }

    /// Get the bounding box of the surface
    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            self.compute_bounds();
        }
        self.bounds.unwrap()
    }

    /// Compute bounding box
    fn compute_bounds(&mut self) {
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for &x in &self.x_data {
            min_x = min_x.min(x as f32);
            max_x = max_x.max(x as f32);
        }

        for &y in &self.y_data {
            min_y = min_y.min(y as f32);
            max_y = max_y.max(y as f32);
        }

        for row in &self.z_data {
            for &z in row {
                if z.is_finite() {
                    min_z = min_z.min(z as f32);
                    max_z = max_z.max(z as f32);
                }
            }
        }

        self.bounds = Some(BoundingBox::new(
            Vec3::new(min_x, min_y, min_z),
            Vec3::new(max_x, max_y, max_z),
        ));
    }

    /// Get plot statistics for debugging
    pub fn statistics(&self) -> SurfaceStatistics {
        let grid_size = self.x_data.len() * self.y_data.len();
        let triangle_count = if self.x_data.len() > 1 && self.y_data.len() > 1 {
            (self.x_data.len() - 1) * (self.y_data.len() - 1) * 2
        } else {
            0
        };

        SurfaceStatistics {
            grid_points: grid_size,
            triangle_count,
            x_resolution: self.x_data.len(),
            y_resolution: self.y_data.len(),
            memory_usage: self.estimated_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let data_size = std::mem::size_of::<f64>()
            * (self.x_data.len() + self.y_data.len() + self.z_data.len() * self.y_data.len());

        let vertices_size = self
            .vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>());

        let indices_size = self
            .indices
            .as_ref()
            .map_or(0, |i| i.len() * std::mem::size_of::<u32>());

        data_size + vertices_size + indices_size
    }
}

/// Surface plot performance and data statistics
#[derive(Debug, Clone)]
pub struct SurfaceStatistics {
    pub grid_points: usize,
    pub triangle_count: usize,
    pub x_resolution: usize,
    pub y_resolution: usize,
    pub memory_usage: usize,
}

impl ColorMap {
    /// Map a normalized value [0,1] to a color
    pub fn map_value(&self, t: f32) -> Vec3 {
        let t = t.clamp(0.0, 1.0);

        match self {
            ColorMap::Jet => self.jet_colormap(t),
            ColorMap::Hot => self.hot_colormap(t),
            ColorMap::Cool => self.cool_colormap(t),
            ColorMap::Viridis => self.viridis_colormap(t),
            ColorMap::Plasma => self.plasma_colormap(t),
            ColorMap::Gray => Vec3::splat(t),
            ColorMap::Custom(min_color, max_color) => {
                min_color.truncate().lerp(max_color.truncate(), t)
            }
            _ => self.default_colormap(t), // Fallback for unimplemented maps
        }
    }

    /// MATLAB Jet colormap
    fn jet_colormap(&self, t: f32) -> Vec3 {
        let r = (1.5 - 4.0 * (t - 0.75).abs()).clamp(0.0, 1.0);
        let g = (1.5 - 4.0 * (t - 0.5).abs()).clamp(0.0, 1.0);
        let b = (1.5 - 4.0 * (t - 0.25).abs()).clamp(0.0, 1.0);
        Vec3::new(r, g, b)
    }

    /// Hot colormap (black -> red -> yellow -> white)
    fn hot_colormap(&self, t: f32) -> Vec3 {
        if t < 1.0 / 3.0 {
            Vec3::new(3.0 * t, 0.0, 0.0)
        } else if t < 2.0 / 3.0 {
            Vec3::new(1.0, 3.0 * t - 1.0, 0.0)
        } else {
            Vec3::new(1.0, 1.0, 3.0 * t - 2.0)
        }
    }

    /// Cool colormap (cyan -> magenta)
    fn cool_colormap(&self, t: f32) -> Vec3 {
        Vec3::new(t, 1.0 - t, 1.0)
    }

    /// Viridis colormap (perceptually uniform)
    fn viridis_colormap(&self, t: f32) -> Vec3 {
        // Simplified Viridis approximation
        let r = (0.267004 + t * (0.993248 - 0.267004)).clamp(0.0, 1.0);
        let g = (0.004874 + t * (0.906157 - 0.004874)).clamp(0.0, 1.0);
        let b = (0.329415 + t * (0.143936 - 0.329415) + t * t * 0.5).clamp(0.0, 1.0);
        Vec3::new(r, g, b)
    }

    /// Plasma colormap (perceptually uniform)
    fn plasma_colormap(&self, t: f32) -> Vec3 {
        // Simplified Plasma approximation
        let r = (0.050383 + t * (0.940015 - 0.050383)).clamp(0.0, 1.0);
        let g = (0.029803 + t * (0.975158 - 0.029803) * (1.0 - t)).clamp(0.0, 1.0);
        let b = (0.527975 + t * (0.131326 - 0.527975)).clamp(0.0, 1.0);
        Vec3::new(r, g, b)
    }

    /// Default colormap fallback
    fn default_colormap(&self, t: f32) -> Vec3 {
        // Use a simple RGB transition as fallback
        if t < 0.5 {
            Vec3::new(0.0, 2.0 * t, 1.0 - 2.0 * t)
        } else {
            Vec3::new(2.0 * (t - 0.5), 1.0 - 2.0 * (t - 0.5), 0.0)
        }
    }
}

/// MATLAB-compatible surface plot creation utilities
pub mod matlab_compat {
    use super::*;

    /// Create a surface plot (equivalent to MATLAB's `surf(X, Y, Z)`)
    pub fn surf(x: Vec<f64>, y: Vec<f64>, z: Vec<Vec<f64>>) -> Result<SurfacePlot, String> {
        SurfacePlot::new(x, y, z)
    }

    /// Create a mesh plot (wireframe surface)
    pub fn mesh(x: Vec<f64>, y: Vec<f64>, z: Vec<Vec<f64>>) -> Result<SurfacePlot, String> {
        Ok(SurfacePlot::new(x, y, z)?
            .with_wireframe(true)
            .with_shading(ShadingMode::None))
    }

    /// Create surface from meshgrid
    pub fn meshgrid_surf(
        x_range: (f64, f64),
        y_range: (f64, f64),
        resolution: (usize, usize),
        func: impl Fn(f64, f64) -> f64,
    ) -> Result<SurfacePlot, String> {
        SurfacePlot::from_function(x_range, y_range, resolution, func)
    }

    /// Create surface with specific colormap
    pub fn surf_with_colormap(
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<Vec<f64>>,
        colormap: &str,
    ) -> Result<SurfacePlot, String> {
        let cmap = match colormap {
            "jet" => ColorMap::Jet,
            "hot" => ColorMap::Hot,
            "cool" => ColorMap::Cool,
            "viridis" => ColorMap::Viridis,
            "plasma" => ColorMap::Plasma,
            "gray" | "grey" => ColorMap::Gray,
            _ => return Err(format!("Unknown colormap: {}", colormap)),
        };

        Ok(SurfacePlot::new(x, y, z)?.with_colormap(cmap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_plot_creation() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0];
        let z = vec![vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0]];

        let surface = SurfacePlot::new(x, y, z).unwrap();

        assert_eq!(surface.x_data.len(), 3);
        assert_eq!(surface.y_data.len(), 2);
        assert_eq!(surface.z_data.len(), 3);
        assert_eq!(surface.z_data[0].len(), 2);
        assert!(surface.visible);
    }

    #[test]
    fn test_surface_from_function() {
        let surface =
            SurfacePlot::from_function((-2.0, 2.0), (-2.0, 2.0), (10, 10), |x, y| x * x + y * y)
                .unwrap();

        assert_eq!(surface.x_data.len(), 10);
        assert_eq!(surface.y_data.len(), 10);
        assert_eq!(surface.z_data.len(), 10);

        // Check that function is evaluated correctly
        assert_eq!(surface.z_data[0][0], 8.0); // (-2)^2 + (-2)^2 = 8
    }

    #[test]
    fn test_surface_validation() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0, 2.0];
        let z = vec![
            vec![0.0, 1.0], // Wrong: should have 3 elements to match y
            vec![1.0, 2.0],
        ];

        assert!(SurfacePlot::new(x, y, z).is_err());
    }

    #[test]
    fn test_surface_styling() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let z = vec![vec![0.0, 1.0], vec![1.0, 2.0]];

        let surface = SurfacePlot::new(x, y, z)
            .unwrap()
            .with_colormap(ColorMap::Hot)
            .with_wireframe(true)
            .with_alpha(0.8)
            .with_label("Test Surface");

        assert_eq!(surface.colormap, ColorMap::Hot);
        assert!(surface.wireframe);
        assert_eq!(surface.alpha, 0.8);
        assert_eq!(surface.label, Some("Test Surface".to_string()));
    }

    #[test]
    fn test_colormap_mapping() {
        let jet = ColorMap::Jet;

        // Test boundary values
        let color_0 = jet.map_value(0.0);
        let color_1 = jet.map_value(1.0);

        assert!(color_0.x >= 0.0 && color_0.x <= 1.0);
        assert!(color_1.x >= 0.0 && color_1.x <= 1.0);

        // Test that different values give different colors
        let color_mid = jet.map_value(0.5);
        assert_ne!(color_0, color_mid);
        assert_ne!(color_mid, color_1);
    }

    #[test]
    fn test_surface_statistics() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0];
        let z = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let surface = SurfacePlot::new(x, y, z).unwrap();
        let stats = surface.statistics();

        assert_eq!(stats.grid_points, 12); // 4 * 3
        assert_eq!(stats.triangle_count, 12); // (4-1) * (3-1) * 2
        assert_eq!(stats.x_resolution, 4);
        assert_eq!(stats.y_resolution, 3);
        assert!(stats.memory_usage > 0);
    }

    #[test]
    fn test_matlab_compat() {
        use super::matlab_compat::*;

        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let z = vec![vec![0.0, 1.0], vec![1.0, 2.0]];

        let surface = surf(x.clone(), y.clone(), z.clone()).unwrap();
        assert!(!surface.wireframe);

        let mesh_plot = mesh(x.clone(), y.clone(), z.clone()).unwrap();
        assert!(mesh_plot.wireframe);

        let colormap_surface = surf_with_colormap(x, y, z, "viridis").unwrap();
        assert_eq!(colormap_surface.colormap, ColorMap::Viridis);
    }
}
