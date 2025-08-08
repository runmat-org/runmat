//! Unified plot rendering pipeline for both interactive GUI and static export
//!
//! This module provides the core rendering logic that is shared between
//! interactive plotting windows and static file exports, ensuring consistent
//! high-quality output across all use cases.

use crate::core::{Camera, Scene, WgpuRenderer};
use crate::plots::Figure;
use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;

/// Unified plot renderer that handles both interactive and static rendering
pub struct PlotRenderer {
    /// WGPU renderer for GPU-accelerated rendering
    pub wgpu_renderer: WgpuRenderer,

    /// Current scene being rendered
    pub scene: Scene,

    /// Camera for view transformations
    pub camera: Camera,

    /// Current theme configuration  
    pub theme: crate::styling::PlotThemeConfig,

    /// Cached rendering state
    data_bounds: Option<(f64, f64, f64, f64)>,
    needs_update: bool,
}

/// Configuration for plot rendering
#[derive(Debug, Clone)]
pub struct PlotRenderConfig {
    /// Output dimensions
    pub width: u32,
    pub height: u32,

    /// Background color
    pub background_color: Vec4,

    /// Whether to draw grid
    pub show_grid: bool,

    /// Whether to draw axes
    pub show_axes: bool,

    /// Whether to draw title
    pub show_title: bool,

    /// Anti-aliasing samples
    pub msaa_samples: u32,

    /// Theme to use
    pub theme: crate::styling::PlotThemeConfig,
}

impl Default for PlotRenderConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            background_color: Vec4::new(0.08, 0.09, 0.11, 1.0), // Dark theme background
            show_grid: true,
            show_axes: true,
            show_title: true,
            msaa_samples: 4,
            theme: crate::styling::PlotThemeConfig::default(),
        }
    }
}

/// Result of rendering operation
#[derive(Debug)]
pub struct RenderResult {
    /// Whether rendering was successful
    pub success: bool,

    /// Rendered data bounds
    pub data_bounds: Option<(f64, f64, f64, f64)>,

    /// Performance metrics
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub render_time_ms: f64,
}

impl PlotRenderer {
    /// Create a new plot renderer
    pub async fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_config: wgpu::SurfaceConfiguration,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let wgpu_renderer = WgpuRenderer::new(device, queue, surface_config).await;
        let scene = Scene::new();
        let camera = Self::create_default_camera();
        let theme = crate::styling::PlotThemeConfig::default();

        Ok(Self {
            wgpu_renderer,
            scene,
            camera,
            theme,
            data_bounds: None,
            needs_update: true,
        })
    }

    /// Set the figure to render
    pub fn set_figure(&mut self, figure: Figure) {
        // Clear existing scene
        self.scene.clear();

        // Convert figure to scene nodes
        self.add_figure_to_scene(figure);

        // Mark for update
        self.needs_update = true;
    }

    /// Add a figure to the current scene
    fn add_figure_to_scene(&mut self, mut figure: Figure) {
        use crate::core::SceneNode;

        let mut node_id_counter = 0u64;

        // Convert figure to render data first, then create scene nodes
        let render_data_list = figure.render_data();

        for render_data in render_data_list.into_iter() {
            // Create scene node for this plot element
            let node = SceneNode {
                id: node_id_counter,
                name: format!("Plot {}", node_id_counter),
                transform: Mat4::IDENTITY,
                visible: true,
                cast_shadows: false,
                receive_shadows: false,
                parent: None,
                children: Vec::new(),
                render_data: Some(render_data),
                bounds: crate::core::BoundingBox::default(),
                lod_levels: Vec::new(),
                current_lod: 0,
            };

            self.scene.add_node(node);
            node_id_counter += 1;
        }

        // Update camera to fit data
        // println!("Scene now has {} visible nodes", self.scene.get_visible_nodes().len());
        self.fit_camera_to_data();
    }

    /// Calculate data bounds from scene
    pub fn calculate_data_bounds(&mut self) -> Option<(f64, f64, f64, f64)> {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                for vertex in &render_data.vertices {
                    let x = vertex.position[0] as f64;
                    let y = vertex.position[1] as f64;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        if min_x != f64::INFINITY && max_x != f64::NEG_INFINITY {
            // Add 10% margin around data for better visualization
            let x_range = (max_x - min_x).max(0.1);
            let y_range = (max_y - min_y).max(0.1);
            let x_margin = x_range * 0.1;
            let y_margin = y_range * 0.1;

            let bounds = (
                min_x - x_margin,
                max_x + x_margin,
                min_y - y_margin,
                max_y + y_margin,
            );

            // println!("Calculated data bounds: {:?}", bounds); // Too noisy
            self.data_bounds = Some(bounds);
            Some(bounds)
        } else {
            self.data_bounds = None;
            None
        }
    }

    /// Fit camera to show all data
    pub fn fit_camera_to_data(&mut self) {
        if let Some((x_min, x_max, y_min, y_max)) = self.calculate_data_bounds() {
            // Update camera projection to match data bounds
            if let crate::core::camera::ProjectionType::Orthographic {
                ref mut left,
                ref mut right,
                ref mut bottom,
                ref mut top,
                ..
            } = self.camera.projection
            {
                // TEMP: Use fixed bounds to test projection matrix
                *left = -2.0;
                *right = 4.0;
                *bottom = -2.0;
                *top = 4.0;

                println!(
                    "CAMERA: Set orthographic bounds: left={}, right={}, bottom={}, top={}",
                    *left, *right, *bottom, *top
                );
            }

            // Center camera to look at data center
            let center_x = (x_min + x_max) / 2.0;
            let center_y = (y_min + y_max) / 2.0;
            self.camera.position = Vec3::new(center_x as f32, center_y as f32, 5.0);
            self.camera.target = Vec3::new(center_x as f32, center_y as f32, 0.0);
        }
    }

    /// Render the current scene to a specific viewport within a texture/surface
    pub fn render_to_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        _viewport: (f32, f32, f32, f32), // (x, y, width, height) in framebuffer coordinates
        clear_background: bool,
        background_color: Option<glam::Vec4>,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();

        // Collect render data and create buffers first
        let mut render_items = Vec::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if !render_data.vertices.is_empty() {
                    // Ensure pipeline exists
                    self.wgpu_renderer
                        .ensure_pipeline(render_data.pipeline_type);

                    // Create vertex buffer
                    let vertex_buffer = self
                        .wgpu_renderer
                        .create_vertex_buffer(&render_data.vertices);

                    // Debug: Count vertices being sent to GPU
                    if render_data.vertices.len() == 12 {
                        println!(
                            "CRITICAL: {} vertices -> GPU, draw calls: {}",
                            render_data.vertices.len(),
                            render_data.draw_calls.len()
                        );
                        for (i, call) in render_data.draw_calls.iter().enumerate() {
                            println!(
                                "  Call {}: offset={}, count={}",
                                i, call.vertex_offset, call.vertex_count
                            );
                        }
                    }

                    render_items.push((render_data, vertex_buffer));
                    total_vertices += render_data.vertices.len();

                    // Count triangles based on pipeline type
                    match render_data.pipeline_type {
                        crate::core::PipelineType::Triangles => {
                            total_triangles += render_data.vertices.len() / 3;
                        }
                        _ => {
                            // Other pipeline types don't count as triangles
                        }
                    }
                }
            }
        }

        // Update uniforms
        let view_proj_matrix = self.camera.view_proj_matrix();

        self.wgpu_renderer
            .update_uniforms(view_proj_matrix, Mat4::IDENTITY);

        // Create render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Plot Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear_background {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color.map_or(0.08, |c| c.x as f64),
                            g: background_color.map_or(0.09, |c| c.y as f64),
                            b: background_color.map_or(0.11, |c| c.z as f64),
                            a: background_color.map_or(1.0, |c| c.w as f64),
                        })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // TEMP: Disable viewport to test if that's causing triangle collapse
        // let (viewport_x, viewport_y, viewport_width, viewport_height) = _viewport;
        // render_pass.set_viewport(viewport_x, viewport_y, viewport_width, viewport_height, 0.0, 1.0);

        // Render all items
        for (render_data, vertex_buffer) in &render_items {
            let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
            println!(
                "RENDER: Using {:?} pipeline for {} vertices",
                render_data.pipeline_type,
                render_data.vertices.len()
            );
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, self.wgpu_renderer.get_uniform_bind_group(), &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

            // Render using draw calls from render_data
            for draw_call in &render_data.draw_calls {
                render_pass.draw(
                    draw_call.vertex_offset as u32
                        ..(draw_call.vertex_offset + draw_call.vertex_count) as u32,
                    0..draw_call.instance_count as u32,
                );
            }
        }

        drop(render_pass);

        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: render_time,
        })
    }

    /// High-performance direct viewport rendering with optimized coordinate transformation
    /// Provides precise data-to-screen mapping for interactive plot windows
    pub fn render_direct_to_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        viewport: (f32, f32, f32, f32), // (x, y, width, height) in framebuffer coordinates
        data_bounds: (f64, f64, f64, f64), // (x_min, y_min, x_max, y_max) in data space
        clear_background: bool,
        background_color: Option<glam::Vec4>,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();

        // Ensure direct line pipeline exists
        self.wgpu_renderer.ensure_direct_line_pipeline();

        // Calculate viewport NDC bounds
        let window_width = self.wgpu_renderer.surface_config.width as f32;
        let window_height = self.wgpu_renderer.surface_config.height as f32;

        let (viewport_x, viewport_y, viewport_width, viewport_height) = viewport;

        // Convert viewport to NDC coordinates
        let ndc_left = (viewport_x / window_width) * 2.0 - 1.0;
        let ndc_right = ((viewport_x + viewport_width) / window_width) * 2.0 - 1.0;
        let ndc_top = 1.0 - (viewport_y / window_height) * 2.0;
        let ndc_bottom = 1.0 - ((viewport_y + viewport_height) / window_height) * 2.0;

        // Configure shader uniforms for direct coordinate transformation
        self.wgpu_renderer.update_direct_uniforms(
            [data_bounds.0 as f32, data_bounds.2 as f32], // data_min (x_min, y_min)
            [data_bounds.1 as f32, data_bounds.3 as f32], // data_max (x_max, y_max)
            [ndc_left, ndc_bottom],                       // viewport_min (NDC)
            [ndc_right, ndc_top],                         // viewport_max (NDC)
        );

        // Collect render data
        let mut render_items = Vec::new();
        let mut total_vertices = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if !render_data.vertices.is_empty() {
                    let vertex_buffer = self
                        .wgpu_renderer
                        .create_vertex_buffer(&render_data.vertices);
                    render_items.push((render_data, vertex_buffer));
                    total_vertices += render_data.vertices.len();
                }
            }
        }

        // Create render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Direct Viewport Plot Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear_background {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color.map_or(0.08, |c| c.x as f64),
                            g: background_color.map_or(0.09, |c| c.y as f64),
                            b: background_color.map_or(0.11, |c| c.z as f64),
                            a: background_color.map_or(1.0, |c| c.w as f64),
                        })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Execute optimized rendering pipeline with pre-transformed coordinates
        for (render_data, vertex_buffer) in &render_items {
            // Use direct line pipeline for all line rendering
            if let Some(pipeline) = &self.wgpu_renderer.direct_line_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &self.wgpu_renderer.direct_uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

                // Draw all vertices as lines
                for draw_call in &render_data.draw_calls {
                    render_pass.draw(
                        draw_call.vertex_offset as u32
                            ..(draw_call.vertex_offset + draw_call.vertex_count) as u32,
                        0..1,
                    );
                }
            }
        }

        drop(render_pass);

        let render_time = start_time.elapsed().as_millis() as f64;

        Ok(RenderResult {
            success: true,
            data_bounds: Some(data_bounds),
            vertex_count: total_vertices,
            triangle_count: 0,
            render_time_ms: render_time,
        })
    }

    /// Render the current scene to a texture/surface
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        config: &PlotRenderConfig,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();

        // Update camera aspect ratio
        let aspect_ratio = config.width as f32 / config.height as f32;
        self.camera.update_aspect_ratio(aspect_ratio);

        // Update WGPU uniforms
        let view_proj_matrix = self.camera.view_proj_matrix();
        let model_matrix = Mat4::IDENTITY;
        self.wgpu_renderer
            .update_uniforms(view_proj_matrix, model_matrix);

        // Collect all render data and create vertex buffers first (outside render pass)
        let mut render_items = Vec::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if !render_data.vertices.is_empty() {
                    // Ensure pipeline exists for this render data
                    self.wgpu_renderer
                        .ensure_pipeline(render_data.pipeline_type);

                    // Create vertex buffer for this node
                    let vertex_buffer = self
                        .wgpu_renderer
                        .create_vertex_buffer(&render_data.vertices);

                    // Create index buffer if needed
                    let index_buffer = if let Some(indices) = &render_data.indices {
                        Some(self.wgpu_renderer.create_index_buffer(indices))
                    } else {
                        None
                    };

                    render_items.push((render_data, vertex_buffer, index_buffer));

                    total_vertices += render_data.vertices.len();
                    if let Some(indices) = &render_data.indices {
                        total_triangles += indices.len() / 3;
                    }
                }
            }
        }

        // Create render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Plot Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: config.background_color.x as f64,
                        g: config.background_color.y as f64,
                        b: config.background_color.z as f64,
                        a: config.background_color.w as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Now render all items with proper bind group setup
        for (render_data, vertex_buffer, index_buffer) in &render_items {
            // Get the appropriate pipeline for this render data (pipeline ensured above)
            let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
            render_pass.set_pipeline(pipeline);

            // Set the uniform bind group (required by shaders)
            render_pass.set_bind_group(0, self.wgpu_renderer.get_uniform_bind_group(), &[]);

            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

            if let Some(index_buffer) = index_buffer {
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                if let Some(indices) = &render_data.indices {
                    println!(
                        "RENDER: Drawing {} indices with triangle pipeline",
                        indices.len()
                    );
                    render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                }
            } else {
                println!("RENDER: Drawing direct vertices - no index buffer");
                // Use draw_calls from render_data for proper vertex range handling
                for draw_call in &render_data.draw_calls {
                    println!("RENDER: Direct draw - vertex_offset={}, vertex_count={}, instance_count={}", 
                             draw_call.vertex_offset, draw_call.vertex_count, draw_call.instance_count);
                    render_pass.draw(
                        draw_call.vertex_offset as u32
                            ..(draw_call.vertex_offset + draw_call.vertex_count) as u32,
                        0..draw_call.instance_count as u32,
                    );
                }
            }
        }

        drop(render_pass);

        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: render_time,
        })
    }

    /// Create default 2D camera for plotting
    fn create_default_camera() -> Camera {
        let mut camera = Camera::new();
        camera.projection = crate::core::camera::ProjectionType::Orthographic {
            left: -5.0,
            right: 5.0,
            bottom: -5.0,
            top: 5.0,
            near: 0.1,
            far: 100.0,
        };
        camera.position = Vec3::new(0.0, 0.0, 5.0);
        camera.target = Vec3::new(0.0, 0.0, 0.0);
        camera.up = Vec3::new(0.0, 1.0, 0.0);
        camera
    }

    /// Get current data bounds
    pub fn data_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        self.data_bounds
    }

    /// Get camera reference
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Get mutable camera reference
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Get scene reference
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Get scene statistics
    pub fn scene_statistics(&self) -> crate::core::SceneStatistics {
        self.scene.statistics()
    }
}

/// High-level plotting utilities that use the unified renderer
pub mod plot_utils {

    /// Calculate nice tick intervals for axis labeling
    pub fn calculate_tick_interval(range: f64) -> f64 {
        let magnitude = 10.0_f64.powf(range.log10().floor());
        let normalized = range / magnitude;

        let nice_interval = if normalized <= 1.0 {
            0.2
        } else if normalized <= 2.0 {
            0.5
        } else if normalized <= 5.0 {
            1.0
        } else {
            2.0
        };

        nice_interval * magnitude
    }

    /// Format a tick label value for display
    pub fn format_tick_label(value: f64) -> String {
        if value.abs() < 0.001 {
            "0".to_string()
        } else if value.abs() >= 1000.0 {
            format!("{:.0}", value)
        } else if value.fract().abs() < 0.001 {
            format!("{:.0}", value)
        } else {
            format!("{:.1}", value)
        }
    }

    /// Generate grid lines for plotting
    pub fn generate_grid_lines(
        bounds: (f64, f64, f64, f64),
        plot_rect: (f32, f32, f32, f32), // (left, right, bottom, top)
    ) -> Vec<(f32, f32, f32, f32)> {
        // Vector of (x1, y1, x2, y2) line segments
        let (x_min, x_max, y_min, y_max) = bounds;
        let (left, right, bottom, top) = plot_rect;

        let mut lines = Vec::new();

        // X-axis grid lines
        let x_range = x_max - x_min;
        let x_interval = calculate_tick_interval(x_range);
        let mut x_val = (x_min / x_interval).ceil() * x_interval;

        while x_val <= x_max {
            let x_screen = left + ((x_val - x_min) / x_range) as f32 * (right - left);
            lines.push((x_screen, bottom, x_screen, top));
            x_val += x_interval;
        }

        // Y-axis grid lines
        let y_range = y_max - y_min;
        let y_interval = calculate_tick_interval(y_range);
        let mut y_val = (y_min / y_interval).ceil() * y_interval;

        while y_val <= y_max {
            let y_screen = bottom + ((y_val - y_min) / y_range) as f32 * (top - bottom);
            lines.push((left, y_screen, right, y_screen));
            y_val += y_interval;
        }

        lines
    }
}
