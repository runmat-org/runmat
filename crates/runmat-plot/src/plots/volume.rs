//! Volume rendering implementation
//!
//! GPU-accelerated volume visualization with raycasting and texture-based rendering.

use crate::core::{BoundingBox, DrawCall, Material, PipelineType, RenderData, Vertex};
use glam::{Vec3, Vec4};

/// High-performance GPU-accelerated volume plot
#[derive(Debug, Clone)]
pub struct VolumePlot {
    /// 3D volume data (values at 3D grid points)
    pub volume_data: Vec<Vec<Vec<f64>>>, // volume_data[x][y][z]

    /// Grid dimensions and spacing
    pub dimensions: (usize, usize, usize), // (nx, ny, nz)
    pub spacing: Vec3, // Grid spacing in world coordinates
    pub origin: Vec3,  // Origin of the volume in world coordinates

    /// Volume rendering properties
    pub opacity: f32,
    pub color_map: VolumeColorMap,
    pub iso_value: Option<f64>, // For isosurface extraction

    /// Transfer function for opacity mapping
    pub opacity_transfer: Vec<(f64, f32)>, // (value, opacity) pairs
    pub color_transfer: Vec<(f64, Vec4)>, // (value, color) pairs

    /// Rendering settings
    pub ray_step_size: f32,
    pub max_steps: u32,
    pub lighting_enabled: bool,

    /// Metadata
    pub label: Option<String>,
    pub visible: bool,

    /// Generated rendering data (cached)
    vertices: Option<Vec<Vertex>>,
    indices: Option<Vec<u32>>,
    bounds: Option<BoundingBox>,
    dirty: bool,
}

/// Volume color mapping schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolumeColorMap {
    /// Grayscale
    Grayscale,
    /// Hot (black -> red -> orange -> yellow -> white)
    Hot,
    /// Jet (blue -> cyan -> green -> yellow -> red)
    Jet,
    /// Viridis (purple -> blue -> green -> yellow)
    Viridis,
    /// Alpha-blended RGB
    RGB,
    /// Custom transfer function
    Custom,
}

/// Volume rendering statistics
#[derive(Debug, Clone)]
pub struct VolumeStatistics {
    pub voxel_count: usize,
    pub memory_usage: usize,
    pub data_range: (f64, f64),
    pub dimensions: (usize, usize, usize),
}

impl Default for VolumeColorMap {
    fn default() -> Self {
        Self::Viridis
    }
}

impl VolumePlot {
    /// Create a new volume plot from 3D data
    pub fn new(volume_data: Vec<Vec<Vec<f64>>>) -> Result<Self, String> {
        if volume_data.is_empty() || volume_data[0].is_empty() || volume_data[0][0].is_empty() {
            return Err("Volume data cannot be empty".to_string());
        }

        let dimensions = (
            volume_data.len(),
            volume_data[0].len(),
            volume_data[0][0].len(),
        );

        // Validate consistent dimensions
        for (x, plane) in volume_data.iter().enumerate() {
            if plane.len() != dimensions.1 {
                return Err(format!("Inconsistent Y dimension at X={x}"));
            }
            for (y, row) in plane.iter().enumerate() {
                if row.len() != dimensions.2 {
                    return Err(format!("Inconsistent Z dimension at X={x}, Y={y}"));
                }
            }
        }

        Ok(Self {
            volume_data,
            dimensions,
            spacing: Vec3::new(1.0, 1.0, 1.0),
            origin: Vec3::ZERO,
            opacity: 0.5,
            color_map: VolumeColorMap::default(),
            iso_value: None,
            opacity_transfer: vec![(0.0, 0.0), (1.0, 1.0)],
            color_transfer: vec![
                (0.0, Vec4::new(0.0, 0.0, 0.5, 1.0)),
                (0.5, Vec4::new(0.0, 1.0, 0.0, 1.0)),
                (1.0, Vec4::new(1.0, 0.0, 0.0, 1.0)),
            ],
            ray_step_size: 0.01,
            max_steps: 1000,
            lighting_enabled: true,
            label: None,
            visible: true,
            vertices: None,
            indices: None,
            bounds: None,
            dirty: true,
        })
    }

    /// Set volume rendering properties
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self.dirty = true;
        self
    }

    /// Set color mapping scheme
    pub fn with_colormap(mut self, color_map: VolumeColorMap) -> Self {
        self.color_map = color_map;
        self.dirty = true;
        self
    }

    /// Set isosurface value for surface extraction
    pub fn with_isosurface(mut self, iso_value: f64) -> Self {
        self.iso_value = Some(iso_value);
        self.dirty = true;
        self
    }

    /// Set the plot label for legends
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the bounding box of the volume
    pub fn bounds(&mut self) -> BoundingBox {
        if self.dirty || self.bounds.is_none() {
            let max_coord = Vec3::new(
                (self.dimensions.0 - 1) as f32 * self.spacing.x,
                (self.dimensions.1 - 1) as f32 * self.spacing.y,
                (self.dimensions.2 - 1) as f32 * self.spacing.z,
            ) + self.origin;

            self.bounds = Some(BoundingBox::new(self.origin, max_coord));
        }
        self.bounds.unwrap()
    }

    /// Generate vertices for volume rendering (bounding box for raycasting)
    fn generate_vertices(&mut self) -> &Vec<Vertex> {
        if self.dirty || self.vertices.is_none() {
            println!(
                "DEBUG: Generating volume vertices for {} x {} x {} grid",
                self.dimensions.0, self.dimensions.1, self.dimensions.2
            );

            let mut vertices = Vec::new();
            let bounds = self.bounds();

            // Generate bounding box vertices for ray casting entry/exit points
            let min = bounds.min;
            let max = bounds.max;

            // 8 vertices of the bounding box
            let positions = [
                Vec3::new(min.x, min.y, min.z), // 0
                Vec3::new(max.x, min.y, min.z), // 1
                Vec3::new(max.x, max.y, min.z), // 2
                Vec3::new(min.x, max.y, min.z), // 3
                Vec3::new(min.x, min.y, max.z), // 4
                Vec3::new(max.x, min.y, max.z), // 5
                Vec3::new(max.x, max.y, max.z), // 6
                Vec3::new(min.x, max.y, max.z), // 7
            ];

            for pos in positions.iter() {
                vertices.push(Vertex {
                    position: pos.to_array(),
                    normal: [0.0, 0.0, 1.0], // Will be computed in shader
                    color: [1.0, 1.0, 1.0, self.opacity],
                    tex_coords: [pos.x / (max.x - min.x), pos.y / (max.y - min.y)],
                });
            }

            log::trace!(
                target: "runmat_plot",
                "volume bbox vertices={}",
                vertices.len()
            );
            self.vertices = Some(vertices);
            self.dirty = false;
        }
        self.vertices.as_ref().unwrap()
    }

    /// Generate indices for volume bounding box (12 triangles forming a cube)
    fn generate_indices(&mut self) -> &Vec<u32> {
        if self.dirty || self.indices.is_none() {
            log::trace!(target: "runmat_plot", "volume generating indices");

            // Cube faces (2 triangles per face)
            let indices = vec![
                // Front face
                0, 1, 2, 0, 2, 3, // Back face
                4, 6, 5, 4, 7, 6, // Left face
                0, 3, 7, 0, 7, 4, // Right face
                1, 5, 6, 1, 6, 2, // Bottom face
                0, 4, 5, 0, 5, 1, // Top face
                3, 2, 6, 3, 6, 7,
            ];

            log::trace!(target: "runmat_plot", "volume indices={}", indices.len());
            self.indices = Some(indices);
        }
        self.indices.as_ref().unwrap()
    }

    /// Generate complete render data for the graphics pipeline
    pub fn render_data(&mut self) -> RenderData {
        log::debug!(
            target: "runmat_plot",
            "volume render_data: dims=({},{},{})",
            self.dimensions.0, self.dimensions.1, self.dimensions.2
        );

        let vertices = self.generate_vertices().clone();
        let indices = self.generate_indices().clone();

        log::debug!(
            target: "runmat_plot",
            "volume render_data generated: vertices={}, indices={}",
            vertices.len(),
            indices.len()
        );

        let material = Material {
            albedo: Vec4::new(1.0, 1.0, 1.0, self.opacity),
            ..Default::default()
        };

        let draw_call = DrawCall {
            vertex_offset: 0,
            vertex_count: vertices.len(),
            index_offset: Some(0),
            index_count: Some(indices.len()),
            instance_count: 1,
        };

        log::trace!(target: "runmat_plot", "volume render_data done");

        RenderData {
            pipeline_type: PipelineType::Triangles, // For volume bounding box
            vertices,
            indices: Some(indices),
            material,
            draw_calls: vec![draw_call],
            gpu_vertices: None,
            bounds: None,
            image: None,
        }
    }

    /// Get volume statistics for debugging
    pub fn statistics(&self) -> VolumeStatistics {
        let voxel_count = self.dimensions.0 * self.dimensions.1 * self.dimensions.2;

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for plane in &self.volume_data {
            for row in plane {
                for &val in row {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
        }

        VolumeStatistics {
            voxel_count,
            memory_usage: self.estimated_memory_usage(),
            data_range: (min_val, max_val),
            dimensions: self.dimensions,
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let data_size =
            self.dimensions.0 * self.dimensions.1 * self.dimensions.2 * std::mem::size_of::<f64>();
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
