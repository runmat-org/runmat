//! Image plot implemented via a flattened SurfacePlot under the hood.

use crate::core::renderer::Vertex;
use crate::core::scene::ImageData;
use crate::core::{BoundingBox, Material, PipelineType, RenderData};
use crate::plots::surface::{ColorMap, SurfacePlot};
use glam::Vec4;

#[derive(Debug, Clone)]
pub struct ImagePlot {
    surface: SurfacePlot,
    pub label: Option<String>,
    pub visible: bool,
}

impl ImagePlot {
    pub fn from_grayscale(
        x: Vec<f64>,
        y: Vec<f64>,
        z_grid: Vec<Vec<f64>>, // rows x cols
        colormap: ColorMap,
        color_limits: Option<(f64, f64)>,
    ) -> Result<Self, String> {
        let surface = SurfacePlot::new(x, y, z_grid)?
            .with_flatten_z(true)
            .with_colormap(colormap)
            .with_color_limits(color_limits);
        Ok(Self {
            surface,
            label: None,
            visible: true,
        })
    }

    pub fn from_color_grid(
        x: Vec<f64>,
        y: Vec<f64>,
        color_grid: Vec<Vec<Vec4>>, // rows x cols -> RGBA
    ) -> Result<Self, String> {
        let cols = if color_grid.is_empty() {
            0
        } else {
            color_grid[0].len()
        };
        let rows = color_grid.len();
        let surface = SurfacePlot::new(x, y, vec![vec![0.0; cols]; rows])?
            .with_flatten_z(true)
            .with_color_grid(color_grid);
        Ok(Self {
            surface,
            label: None,
            visible: true,
        })
    }

    pub fn with_label<S: Into<String>>(mut self, s: S) -> Self {
        self.label = Some(s.into());
        self
    }
    pub fn set_visible(&mut self, v: bool) {
        self.visible = v;
    }

    pub fn bounds(&mut self) -> BoundingBox {
        self.surface.bounds()
    }
    pub fn render_data(&mut self) -> RenderData {
        // Build a single textured quad using the surface's bounds and texcoords.
        // For now, convert any SurfacePlot content into an RGBA image on CPU via color_grid or colormap mapping.
        // Fast path: if color_grid exists, pack to RGBA8; if grayscale, map via colormap and color_limits.

        // Use the surface to compute vertices grid (to get sizes)
        let bounds = self.surface.bounds();
        let min = bounds.min;
        let max = bounds.max;

        // Determine image size from surface grid resolution
        let x_len = self.surface.x_data.len().max(2);
        let y_len = self.surface.y_data.len().max(2);
        let width = x_len as u32;
        let height = y_len as u32;

        // Create CPU RGBA8 buffer
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        let z_rows = match &self.surface.z_data {
            Some(rows) => rows,
            None => {
                return RenderData {
                    pipeline_type: PipelineType::Triangles,
                    vertices: Vec::new(),
                    indices: None,
                    gpu_vertices: None,
                    bounds: None,
                    material: Material::default(),
                    draw_calls: Vec::new(),
                    image: None,
                };
            }
        };

        if let Some(grid) = &self.surface.color_grid {
            // RGB[A] provided
            for row in grid.iter().take(x_len) {
                for c in row.iter().take(y_len).copied() {
                    rgba.push((c.x.clamp(0.0, 1.0) * 255.0) as u8);
                    rgba.push((c.y.clamp(0.0, 1.0) * 255.0) as u8);
                    rgba.push((c.z.clamp(0.0, 1.0) * 255.0) as u8);
                    rgba.push((c.w.clamp(0.0, 1.0) * 255.0) as u8);
                }
            }
        } else {
            // Grayscale via colormap; reuse surface z and colormap
            // Determine z range
            let (min_z, max_z) = if let Some((lo, hi)) = self.surface.color_limits {
                (lo, hi)
            } else {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for row in z_rows {
                    for &z in row {
                        if z.is_finite() {
                            lo = lo.min(z);
                            hi = hi.max(z);
                        }
                    }
                }
                (lo, hi)
            };
            let zr = (max_z - min_z).max(f64::MIN_POSITIVE);
            for row in z_rows.iter().take(x_len) {
                for z in row.iter().take(y_len).copied() {
                    let t = (((z - min_z) / zr) as f32).clamp(0.0, 1.0);
                    let rgb = self.surface.colormap.map_value(t);
                    rgba.push((rgb.x * 255.0) as u8);
                    rgba.push((rgb.y * 255.0) as u8);
                    rgba.push((rgb.z * 255.0) as u8);
                    rgba.push((self.surface.alpha.clamp(0.0, 1.0) * 255.0) as u8);
                }
            }
        }

        // Create quad vertices (two triangles) covering [min,max] in XY, Z=0
        let color = Vec4::new(1.0, 1.0, 1.0, 1.0);
        let vertices = vec![
            Vertex {
                position: [min.x, min.y, 0.0],
                color: color.to_array(),
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: [max.x, min.y, 0.0],
                color: color.to_array(),
                normal: [0.0, 0.0, 1.0],
                tex_coords: [1.0, 0.0],
            },
            Vertex {
                position: [max.x, max.y, 0.0],
                color: color.to_array(),
                normal: [0.0, 0.0, 1.0],
                tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: [min.x, max.y, 0.0],
                color: color.to_array(),
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.0, 1.0],
            },
        ];
        let indices = vec![0u32, 1, 2, 0, 2, 3];

        let material = crate::core::Material::default();
        let draw_call = crate::core::DrawCall {
            vertex_offset: 0,
            vertex_count: vertices.len(),
            index_offset: Some(0),
            index_count: Some(indices.len()),
            instance_count: 1,
        };
        RenderData {
            pipeline_type: crate::core::renderer::PipelineType::Textured,
            vertices,
            indices: Some(indices),
            gpu_vertices: None,
            bounds: None,
            material,
            draw_calls: vec![draw_call],
            image: Some(ImageData::Rgba8 {
                width,
                height,
                data: rgba,
            }),
        }
    }
    pub fn estimated_memory_usage(&self) -> usize {
        self.surface.estimated_memory_usage()
    }
}
