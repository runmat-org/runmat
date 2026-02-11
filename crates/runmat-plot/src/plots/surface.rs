//! 3D surface plot implementation
//!
//! High-performance GPU-accelerated 3D surface rendering.

use crate::core::{
    BoundingBox, DrawCall, GpuVertexBuffer, Material, PipelineType, RenderData, Vertex,
};
use glam::{Vec3, Vec4};

/// High-performance GPU-accelerated 3D surface plot
#[derive(Debug, Clone)]
pub struct SurfacePlot {
    /// Grid data (Z values at X,Y coordinates)
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
    pub z_data: Option<Vec<Vec<f64>>>, // Host data when available
    /// Grid resolution for rendering/index generation (kept even for GPU-backed plots).
    x_len: usize,
    y_len: usize,

    /// Surface properties
    pub colormap: ColorMap,
    pub shading_mode: ShadingMode,
    pub wireframe: bool,
    pub alpha: f32,
    /// If true, render Z at 0 (flat), but color-map using Z values
    pub flatten_z: bool,

    /// Optional color limits override for mapping Z -> color (caxis)
    pub color_limits: Option<(f64, f64)>,

    /// Optional per-vertex color grid (for RGB images); if set, overrides colormap mapping
    pub color_grid: Option<Vec<Vec<Vec4>>>, // [x_index][y_index] -> RGBA

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
    gpu_vertices: Option<GpuVertexBuffer>,
    gpu_vertex_count: Option<usize>,
    gpu_bounds: Option<BoundingBox>,
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
            x_len: x_data.len(),
            y_len: y_data.len(),
            x_data,
            y_data,
            z_data: Some(z_data),
            colormap: ColorMap::default(),
            shading_mode: ShadingMode::default(),
            wireframe: false,
            alpha: 1.0,
            flatten_z: false,
            color_limits: None,
            color_grid: None,
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
            gpu_vertices: None,
            gpu_vertex_count: None,
            gpu_bounds: None,
        })
    }

    /// Create a surface plot backed by a GPU vertex buffer.
    pub fn from_gpu_buffer(
        x_len: usize,
        y_len: usize,
        buffer: GpuVertexBuffer,
        vertex_count: usize,
        bounds: BoundingBox,
    ) -> Self {
        Self {
            x_data: Vec::new(),
            y_data: Vec::new(),
            z_data: None,
            x_len,
            y_len,
            colormap: ColorMap::default(),
            shading_mode: ShadingMode::default(),
            wireframe: false,
            alpha: 1.0,
            flatten_z: false,
            color_limits: None,
            color_grid: None,
            lighting_enabled: true,
            ambient_strength: 0.2,
            diffuse_strength: 0.8,
            specular_strength: 0.5,
            shininess: 32.0,
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: Some(bounds),
            dirty: false,
            gpu_vertices: Some(buffer),
            gpu_vertex_count: Some(vertex_count),
            gpu_bounds: Some(bounds),
        }
    }

    fn drop_gpu_if_possible(&mut self) {
        if self.gpu_vertices.is_some() && self.z_data.is_some() {
            self.invalidate_gpu_data();
        }
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

    fn invalidate_gpu_data(&mut self) {
        self.gpu_vertices = None;
        self.gpu_vertex_count = None;
        self.gpu_bounds = None;
    }

    /// Set color mapping
    pub fn with_colormap(mut self, colormap: ColorMap) -> Self {
        self.colormap = colormap;
        self.dirty = true;
        self.drop_gpu_if_possible();
        self
    }

    /// Set shading mode
    pub fn with_shading(mut self, shading: ShadingMode) -> Self {
        self.shading_mode = shading;
        self.dirty = true;
        self.drop_gpu_if_possible();
        self
    }

    /// Enable/disable wireframe
    pub fn with_wireframe(mut self, enabled: bool) -> Self {
        self.wireframe = enabled;
        self.dirty = true;
        self.drop_gpu_if_possible();
        self
    }

    /// Set transparency
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self.dirty = true;
        self.drop_gpu_if_possible();
        self
    }

    /// Render surface flat in Z while mapping colors from Z values (for imagesc/imshow)
    pub fn with_flatten_z(mut self, enabled: bool) -> Self {
        self.flatten_z = enabled;
        self.dirty = true;
        self.drop_gpu_if_possible();
        self
    }

    /// Override color mapping limits (caxis)
    pub fn with_color_limits(mut self, limits: Option<(f64, f64)>) -> Self {
        self.color_limits = limits;
        self.dirty = true;
        self.drop_gpu_if_possible();
        self
    }

    /// Mutably set color mapping limits (caxis)
    pub fn set_color_limits(&mut self, limits: Option<(f64, f64)>) {
        self.color_limits = limits;
        self.dirty = true;
        self.drop_gpu_if_possible();
    }

    /// Provide explicit per-vertex colors (RGB[A])
    pub fn with_color_grid(mut self, grid: Vec<Vec<Vec4>>) -> Self {
        self.color_grid = Some(grid);
        self.dirty = true;
        self.drop_gpu_if_possible();
        self
    }

    /// Set plot label for legends
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the number of grid points
    pub fn len(&self) -> usize {
        self.x_len * self.y_len
    }

    /// Check if the surface has no data
    pub fn is_empty(&self) -> bool {
        self.x_len == 0 || self.y_len == 0
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
        if let Some(bounds) = self.gpu_bounds {
            self.bounds = Some(bounds);
            return;
        }

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

        if let Some(rows) = &self.z_data {
            for row in rows {
                for &z in row {
                    if z.is_finite() {
                        min_z = min_z.min(z as f32);
                        max_z = max_z.max(z as f32);
                    }
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
        let grid_size = self.x_len * self.y_len;
        let triangle_count = if self.x_len > 1 && self.y_len > 1 {
            (self.x_len - 1) * (self.y_len - 1) * 2
        } else {
            0
        };

        SurfaceStatistics {
            grid_points: grid_size,
            triangle_count,
            x_resolution: self.x_len,
            y_resolution: self.y_len,
            memory_usage: self.estimated_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let data_size = std::mem::size_of::<f64>()
            * (self.x_data.len()
                + self.y_data.len()
                + self
                    .z_data
                    .as_ref()
                    .map_or(0, |z| z.len() * self.y_data.len()));

        let vertices_size = self
            .vertices
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<Vertex>());

        let indices_size = self
            .indices
            .as_ref()
            .map_or(0, |i| i.len() * std::mem::size_of::<u32>());

        let gpu_size = self.gpu_vertex_count.unwrap_or(0) * std::mem::size_of::<Vertex>();

        data_size + vertices_size + indices_size + gpu_size
    }

    /// Generate vertices for surface mesh
    fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.gpu_vertices.is_some() {
            if self.vertices.is_none() {
                self.vertices = Some(Vec::new());
            }
            return self.vertices.as_ref().unwrap();
        }

        if self.dirty || self.vertices.is_none() {
            log::trace!(
                target: "runmat_plot",
                "surface gen vertices {} x {}",
                self.x_data.len(),
                self.y_data.len()
            );

            let mut vertices = Vec::new();

            // Determine color mapping range
            let z_rows = self
                .z_data
                .as_ref()
                .expect("CPU surface data missing during vertex generation");
            let (min_z, max_z) = if let Some((lo, hi)) = self.color_limits {
                (lo, hi)
            } else {
                let mut min_z = f64::INFINITY;
                let mut max_z = f64::NEG_INFINITY;
                for row in z_rows {
                    for &z in row {
                        if z.is_finite() {
                            min_z = min_z.min(z);
                            max_z = max_z.max(z);
                        }
                    }
                }
                (min_z, max_z)
            };
            let z_range = (max_z - min_z).max(f64::MIN_POSITIVE);

            // Generate vertices for each grid point
            for (i, &x) in self.x_data.iter().enumerate() {
                for (j, &y) in self.y_data.iter().enumerate() {
                    let z = z_rows[i][j];
                    let z_pos = if self.flatten_z { 0.0 } else { z as f32 };
                    let position = Vec3::new(x as f32, y as f32, z_pos);

                    // Simple normal calculation (can be improved with proper gradients)
                    let normal = Vec3::new(0.0, 0.0, 1.0); // Placeholder

                    // Determine color: explicit grid (RGB) or colormap from Z
                    let color = if let Some(grid) = &self.color_grid {
                        let c = grid[i][j];
                        Vec4::new(c.x, c.y, c.z, c.w)
                    } else {
                        let t = ((z - min_z) / z_range) as f32;
                        let color_rgb = self.colormap.map_value(t.clamp(0.0, 1.0));
                        Vec4::new(color_rgb.x, color_rgb.y, color_rgb.z, self.alpha)
                    };

                    vertices.push(Vertex {
                        position: position.to_array(),
                        normal: normal.to_array(),
                        color: color.to_array(),
                        tex_coords: [
                            i as f32 / (self.x_data.len() - 1).max(1) as f32,
                            j as f32 / (self.y_data.len() - 1).max(1) as f32,
                        ],
                    });
                }
            }

            log::trace!(target: "runmat_plot", "surface vertices={}", vertices.len());
            self.vertices = Some(vertices);
        }
        self.vertices.as_ref().unwrap()
    }

    /// Generate indices for surface triangulation
    fn generate_indices(&mut self) -> &Vec<u32> {
        if self.dirty || self.indices.is_none() {
            log::trace!(target: "runmat_plot", "surface generating indices");

            let mut indices = Vec::new();
            let x_res = self.x_len;
            let y_res = self.y_len;

            // Generate triangle indices for surface mesh
            for i in 0..x_res - 1 {
                for j in 0..y_res - 1 {
                    let base = (i * y_res + j) as u32;
                    let next_row = base + y_res as u32;

                    // Two triangles per quad
                    // Triangle 1: (i,j), (i+1,j), (i,j+1)
                    indices.push(base);
                    indices.push(next_row);
                    indices.push(base + 1);

                    // Triangle 2: (i+1,j), (i+1,j+1), (i,j+1)
                    indices.push(next_row);
                    indices.push(next_row + 1);
                    indices.push(base + 1);
                }
            }

            log::trace!(target: "runmat_plot", "surface indices={}", indices.len());
            self.indices = Some(indices);
            self.dirty = false;
        }
        self.indices.as_ref().unwrap()
    }

    /// Generate complete render data for the graphics pipeline
    pub fn render_data(&mut self) -> RenderData {
        log::debug!(
            target: "runmat_plot",
            "surface render_data start: {} x {}",
            self.x_len,
            self.y_len
        );

        let using_gpu = self.gpu_vertices.is_some();
        let vertices = if using_gpu {
            Vec::new()
        } else {
            self.generate_vertices().clone()
        };
        let indices = self.generate_indices().clone();

        let material = Material {
            albedo: Vec4::new(1.0, 1.0, 1.0, self.alpha),
            ..Default::default()
        };

        let vertex_count = if using_gpu {
            self.gpu_vertex_count.unwrap_or(0)
        } else {
            vertices.len()
        };

        log::debug!(
            target: "runmat_plot",
            "surface render_data generated: vertex_count={} (gpu={}), indices={}",
            vertex_count,
            using_gpu,
            indices.len()
        );

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: Some(0),
            index_count: Some(indices.len()),
            instance_count: 1,
        };

        log::trace!(target: "runmat_plot", "surface render_data done");

        RenderData {
            pipeline_type: if self.wireframe {
                PipelineType::Lines
            } else {
                PipelineType::Triangles
            },
            vertices,
            indices: Some(indices),

            gpu_vertices: self.gpu_vertices.clone(),
            bounds: None,
            material,
            draw_calls: vec![draw_call],
            image: None,
        }
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
            ColorMap::Spring => self.spring_colormap(t),
            ColorMap::Summer => self.summer_colormap(t),
            ColorMap::Autumn => self.autumn_colormap(t),
            ColorMap::Winter => self.winter_colormap(t),
            ColorMap::Gray => Vec3::splat(t),
            ColorMap::Bone => self.bone_colormap(t),
            ColorMap::Copper => self.copper_colormap(t),
            ColorMap::Pink => self.pink_colormap(t),
            ColorMap::Lines => self.lines_colormap(t),
            ColorMap::Viridis => self.viridis_colormap(t),
            ColorMap::Plasma => self.plasma_colormap(t),
            ColorMap::Inferno => self.inferno_colormap(t),
            ColorMap::Magma => self.magma_colormap(t),
            ColorMap::Turbo => self.turbo_colormap(t),
            ColorMap::Parula => self.parula_colormap(t),
            ColorMap::Custom(min_color, max_color) => {
                min_color.truncate().lerp(max_color.truncate(), t)
            }
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

    /// Spring colormap (magenta -> yellow)
    fn spring_colormap(&self, t: f32) -> Vec3 {
        Vec3::new(1.0, t, 1.0 - t)
    }

    /// Summer colormap (green -> yellow)
    fn summer_colormap(&self, t: f32) -> Vec3 {
        Vec3::new(t, 0.5 + 0.5 * t, 0.4)
    }

    /// Autumn colormap (red -> yellow)
    fn autumn_colormap(&self, t: f32) -> Vec3 {
        Vec3::new(1.0, t, 0.0)
    }

    /// Winter colormap (blue -> green)
    fn winter_colormap(&self, t: f32) -> Vec3 {
        Vec3::new(0.0, t, 1.0 - 0.5 * t)
    }

    /// Bone colormap (black -> white with blue tint)
    fn bone_colormap(&self, t: f32) -> Vec3 {
        if t < 3.0 / 8.0 {
            Vec3::new(7.0 / 8.0 * t, 7.0 / 8.0 * t, 29.0 / 24.0 * t)
        } else {
            Vec3::new(
                (29.0 + 7.0 * t) / 24.0,
                (29.0 + 7.0 * t) / 24.0,
                (29.0 + 7.0 * t) / 24.0,
            )
        }
    }

    /// Copper colormap (black -> copper)
    fn copper_colormap(&self, t: f32) -> Vec3 {
        Vec3::new((1.25 * t).min(1.0), 0.7812 * t, 0.4975 * t)
    }

    /// Pink colormap (black -> pink -> white)
    fn pink_colormap(&self, t: f32) -> Vec3 {
        let sqrt_t = t.sqrt();
        if t < 3.0 / 8.0 {
            Vec3::new(14.0 / 9.0 * sqrt_t, 2.0 / 3.0 * sqrt_t, 2.0 / 3.0 * sqrt_t)
        } else {
            Vec3::new(
                2.0 * sqrt_t - 1.0 / 3.0,
                8.0 / 9.0 * sqrt_t + 1.0 / 3.0,
                8.0 / 9.0 * sqrt_t + 1.0 / 3.0,
            )
        }
    }

    /// Lines colormap (cycling through basic colors)
    fn lines_colormap(&self, t: f32) -> Vec3 {
        let _phase = (t * 7.0) % 1.0; // For future use in color transitions
        let index = (t * 7.0) as usize % 7;
        match index {
            0 => Vec3::new(0.0, 0.0, 1.0),    // Blue
            1 => Vec3::new(0.0, 0.5, 0.0),    // Green
            2 => Vec3::new(1.0, 0.0, 0.0),    // Red
            3 => Vec3::new(0.0, 0.75, 0.75),  // Cyan
            4 => Vec3::new(0.75, 0.0, 0.75),  // Magenta
            5 => Vec3::new(0.75, 0.75, 0.0),  // Yellow
            _ => Vec3::new(0.25, 0.25, 0.25), // Dark gray
        }
    }

    /// Inferno colormap (perceptually uniform)
    fn inferno_colormap(&self, t: f32) -> Vec3 {
        // Simplified Inferno approximation
        let r = (0.001462 + t * (0.988362 - 0.001462)).clamp(0.0, 1.0);
        let g = (0.000466 + t * t * (0.982895 - 0.000466)).clamp(0.0, 1.0);
        let b = (0.013866 + t * (1.0 - t) * (0.416065 - 0.013866)).clamp(0.0, 1.0);
        Vec3::new(r, g, b)
    }

    /// Magma colormap (perceptually uniform)
    fn magma_colormap(&self, t: f32) -> Vec3 {
        // Simplified Magma approximation
        let r = (0.001462 + t * (0.987053 - 0.001462)).clamp(0.0, 1.0);
        let g = (0.000466 + t * t * (0.991438 - 0.000466)).clamp(0.0, 1.0);
        let b = (0.013866 + t * (0.644237 - 0.013866) * (1.0 - t)).clamp(0.0, 1.0);
        Vec3::new(r, g, b)
    }

    /// Turbo colormap (improved rainbow)
    fn turbo_colormap(&self, t: f32) -> Vec3 {
        // Simplified Turbo approximation (Google's improved rainbow)
        let r = if t < 0.5 {
            (0.13 + 0.87 * (2.0 * t).powf(0.25)).clamp(0.0, 1.0)
        } else {
            (0.8685 + 0.1315 * (2.0 * (1.0 - t)).powf(0.25)).clamp(0.0, 1.0)
        };

        let g = if t < 0.25 {
            4.0 * t
        } else if t < 0.75 {
            1.0
        } else {
            1.0 - 4.0 * (t - 0.75)
        }
        .clamp(0.0, 1.0);

        let b = if t < 0.5 {
            (0.8 * (1.0 - 2.0 * t).powf(0.25)).clamp(0.0, 1.0)
        } else {
            (0.1 + 0.9 * (2.0 * t - 1.0).powf(0.25)).clamp(0.0, 1.0)
        };

        Vec3::new(r, g, b)
    }

    /// Parula colormap (MATLAB's default)
    fn parula_colormap(&self, t: f32) -> Vec3 {
        // Simplified Parula approximation
        let r = if t < 0.25 {
            0.2081 * (1.0 - t)
        } else if t < 0.5 {
            t - 0.25
        } else if t < 0.75 {
            1.0
        } else {
            1.0 - 0.5 * (t - 0.75)
        }
        .clamp(0.0, 1.0);

        let g = if t < 0.125 {
            0.1663 * t / 0.125
        } else if t < 0.375 {
            0.1663 + (0.7079 - 0.1663) * (t - 0.125) / 0.25
        } else if t < 0.625 {
            0.7079 + (0.9839 - 0.7079) * (t - 0.375) / 0.25
        } else {
            0.9839 * (1.0 - (t - 0.625) / 0.375)
        }
        .clamp(0.0, 1.0);

        let b = if t < 0.25 {
            0.5 + 0.5 * t / 0.25
        } else if t < 0.5 {
            1.0
        } else {
            1.0 - 2.0 * (t - 0.5)
        }
        .clamp(0.0, 1.0);

        Vec3::new(r, g, b)
    }

    /// Default colormap fallback
    #[allow(dead_code)] // Fallback method for colormap errors
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
            _ => return Err(format!("Unknown colormap: {colormap}")),
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
        let rows = surface.z_data.as_ref().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].len(), 2);
        assert!(surface.visible);
    }

    #[test]
    fn test_surface_from_function() {
        let surface =
            SurfacePlot::from_function((-2.0, 2.0), (-2.0, 2.0), (10, 10), |x, y| x * x + y * y)
                .unwrap();

        assert_eq!(surface.x_data.len(), 10);
        assert_eq!(surface.y_data.len(), 10);
        let rows = surface.z_data.as_ref().unwrap();
        assert_eq!(rows.len(), 10);

        // Check that function is evaluated correctly
        assert_eq!(rows[0][0], 8.0); // (-2)^2 + (-2)^2 = 8
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
