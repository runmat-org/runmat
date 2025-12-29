//! Unified plot rendering pipeline for both interactive GUI and static export
//!
//! This module provides the core rendering logic that is shared between
//! interactive plotting windows and static file exports, ensuring consistent
//! high-quality output across all use cases.

use crate::core::renderer::Vertex;
use crate::core::{Camera, Scene, WgpuRenderer};
use crate::plots::figure::LegendEntry;
use crate::plots::surface::ColorMap;
use crate::plots::Figure;
use glam::{Mat4, Vec3, Vec4};
use runmat_time::Instant;
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

    // Cached figure metadata for overlay
    figure_title: Option<String>,
    figure_x_label: Option<String>,
    figure_y_label: Option<String>,
    figure_show_grid: bool,
    figure_show_legend: bool,
    figure_x_limits: Option<(f64, f64)>,
    figure_y_limits: Option<(f64, f64)>,
    legend_entries: Vec<LegendEntry>,
    figure_x_log: bool,
    figure_y_log: bool,
    figure_axis_equal: bool,
    figure_colormap: ColorMap,
    figure_colorbar_enabled: bool,
    // Categorical axis cache
    figure_categorical_is_x: Option<bool>,
    figure_categorical_labels: Option<Vec<String>>,
    /// Per-axes cameras (for subplots). If empty, use `camera` for single axes.
    axes_cameras: Vec<Camera>,
    /// Keep a clone of the last figure set for export/UX operations
    pub(crate) last_figure: Option<crate::plots::Figure>,
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
    fn prepare_buffers_for_render_data(
        &self,
        render_data: &crate::core::RenderData,
    ) -> Option<(Arc<wgpu::Buffer>, Option<wgpu::Buffer>)> {
        let vertex_buffer = self
            .wgpu_renderer
            .vertex_buffer_from_sources(render_data.gpu_vertices.as_ref(), &render_data.vertices)?;
        let index_buffer = render_data
            .indices
            .as_ref()
            .map(|indices| self.wgpu_renderer.create_index_buffer(indices));
        Some((vertex_buffer, index_buffer))
    }

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
            figure_title: None,
            figure_x_label: None,
            figure_y_label: None,
            figure_show_grid: true,
            figure_show_legend: true,
            figure_x_limits: None,
            figure_y_limits: None,
            legend_entries: Vec::new(),
            figure_x_log: false,
            figure_y_log: false,
            figure_axis_equal: false,
            figure_colormap: ColorMap::Parula,
            figure_colorbar_enabled: false,
            figure_categorical_is_x: None,
            figure_categorical_labels: None,
            axes_cameras: Vec::new(),
            last_figure: None,
        })
    }

    /// Set the figure to render
    pub fn set_figure(&mut self, figure: Figure) {
        // Clear existing scene
        self.scene.clear();

        // Convert figure to scene nodes
        self.cache_figure_meta(&figure);
        self.last_figure = Some(figure.clone());
        // Initialize axes cameras for subplot grid
        let (rows, cols) = figure.axes_grid();
        let num_axes = rows.max(1) * cols.max(1);
        if self.axes_cameras.len() != num_axes {
            self.axes_cameras = (0..num_axes)
                .map(|_| Self::create_default_camera())
                .collect();
        }

        self.add_figure_to_scene(figure);

        // Mark for update
        self.needs_update = true;

        // Recompute bounds and fit camera immediately
        self.fit_camera_to_data();
    }

    /// Add a figure to the current scene
    fn add_figure_to_scene(&mut self, mut figure: Figure) {
        use crate::core::SceneNode;

        // Convert figure to render data first, then create scene nodes
        let render_data_list = figure.render_data();
        let axes_map: Vec<usize> = figure.plot_axes_indices().to_vec();
        let (rows, cols) = figure.axes_grid();

        for (node_id_counter, render_data) in render_data_list.into_iter().enumerate() {
            let axes_index = axes_map
                .get(node_id_counter)
                .copied()
                .unwrap_or(0)
                .min(rows * cols - 1);
            // Create scene node for this plot element
            let node = SceneNode {
                id: node_id_counter as u64,
                name: format!("Plot {node_id_counter} @axes {axes_index}"),
                transform: Mat4::IDENTITY,
                visible: true,
                cast_shadows: false,
                receive_shadows: false,
                axes_index,
                parent: None,
                children: Vec::new(),
                render_data: Some(render_data),
                bounds: crate::core::BoundingBox::default(),
                lod_levels: Vec::new(),
                current_lod: 0,
            };

            let nid = self.scene.add_node(node);
            // Tag node with axes index via a no-op mechanism for now (could extend SceneNode in future)
            let _ = nid;
            let _ = axes_index;
            let _ = rows;
            let _ = cols;
        }

        // Update camera to fit data
        // println!("Scene now has {} visible nodes", self.scene.get_visible_nodes().len());
        self.fit_camera_to_data();
    }

    /// Cache figure metadata for overlay consumption
    fn cache_figure_meta(&mut self, figure: &Figure) {
        self.figure_title = figure.title.clone();
        self.figure_x_label = figure.x_label.clone();
        self.figure_y_label = figure.y_label.clone();
        self.figure_show_grid = figure.grid_enabled;
        self.figure_show_legend = figure.legend_enabled;
        self.figure_x_limits = figure.x_limits;
        self.figure_y_limits = figure.y_limits;
        self.legend_entries = figure.legend_entries();
        self.figure_x_log = figure.x_log;
        self.figure_y_log = figure.y_log;
        self.figure_axis_equal = figure.axis_equal;
        self.figure_colormap = figure.colormap;
        self.figure_colorbar_enabled = figure.colorbar_enabled;
        // Cache categorical labels for overlay
        if let Some((is_x, labels)) = figure.categorical_axis_labels() {
            self.figure_categorical_is_x = Some(is_x);
            self.figure_categorical_labels = Some(labels);
        } else {
            self.figure_categorical_is_x = None;
            self.figure_categorical_labels = None;
        }
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
            // Match the camera bounds exactly to data bounds to align with overlay grid
            if let crate::core::camera::ProjectionType::Orthographic { .. } = self.camera.projection
            {
                let mut l = x_min as f32;
                let mut r = x_max as f32;
                let mut b = y_min as f32;
                let mut t = y_max as f32;
                if self.figure_axis_equal {
                    let cx = (l + r) * 0.5;
                    let cy = (b + t) * 0.5;
                    let width = (r - l).abs();
                    let height = (t - b).abs();
                    let size = width.max(height);
                    l = cx - size * 0.5;
                    r = cx + size * 0.5;
                    b = cy - size * 0.5;
                    t = cy + size * 0.5;
                }
                if let crate::core::camera::ProjectionType::Orthographic {
                    ref mut left,
                    ref mut right,
                    ref mut bottom,
                    ref mut top,
                    ..
                } = self.camera.projection
                {
                    *left = l;
                    *right = r;
                    *bottom = b;
                    *top = t;
                }
                // Center camera at data center
                let cx = (l + r) * 0.5;
                let cy = (b + t) * 0.5;
                self.camera.position.x = cx;
                self.camera.position.y = cy;
                self.camera.target.x = cx;
                self.camera.target.y = cy;
                self.camera.mark_dirty();
            }
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
        let start_time = Instant::now();

        // Collect render data and create buffers first
        let mut render_items = Vec::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some(vertex_buffer) = self.wgpu_renderer.vertex_buffer_from_sources(
                    render_data.gpu_vertices.as_ref(),
                    &render_data.vertices,
                ) {
                    self.wgpu_renderer
                        .ensure_pipeline(render_data.pipeline_type);

                    log::trace!(
                        target: "runmat_plot",
                        "upload vertices={}, draw_calls={}",
                        render_data.vertex_count(),
                        render_data.draw_calls.len()
                    );

                    render_items.push((render_data, vertex_buffer));
                    total_vertices += render_data.vertex_count();

                    if render_data.pipeline_type == crate::core::PipelineType::Triangles {
                        total_triangles += render_data.vertex_count() / 3;
                    }
                }
            }
        }

        // Update uniforms
        let view_proj_matrix = self.camera.view_proj_matrix();

        self.wgpu_renderer
            .update_uniforms(view_proj_matrix, Mat4::IDENTITY);

        // Create render pass (respect MSAA)
        let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
        let msaa_view_opt = if use_msaa {
            let tex = self
                .wgpu_renderer
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("runmat_msaa_color_camera"),
                    size: wgpu::Extent3d {
                        width: self.wgpu_renderer.surface_config.width,
                        height: self.wgpu_renderer.surface_config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: self.wgpu_renderer.msaa_sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.wgpu_renderer.surface_config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
            Some(tex.create_view(&wgpu::TextureViewDescriptor::default()))
        } else {
            None
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Plot Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view_opt.as_ref().unwrap_or(target_view),
                resolve_target: if use_msaa { Some(target_view) } else { None },
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

        // Apply viewport scissor to match overlay plot rect
        let (vx, vy, vw, vh) = _viewport;
        render_pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);

        // Configure direct-uniforms for precise data-to-NDC mapping within this viewport
        let sw = self.wgpu_renderer.surface_config.width as f32;
        let sh = self.wgpu_renderer.surface_config.height as f32;
        let ndc_left = (vx / sw) * 2.0 - 1.0;
        let ndc_right = ((vx + vw) / sw) * 2.0 - 1.0;
        let ndc_top = 1.0 - (vy / sh) * 2.0;
        let ndc_bottom = 1.0 - ((vy + vh) / sh) * 2.0;

        // data_bounds passed in from caller: (x_min, y_min, x_max, y_max)
        let (x_min, y_min, x_max, y_max) = (0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64);
        self.wgpu_renderer.update_direct_uniforms(
            [x_min as f32, y_min as f32],
            [x_max as f32, y_max as f32],
            [ndc_left, ndc_bottom],
            [ndc_right, ndc_top],
            [sw, sh],
        );

        // Continue with specific pipelines below (implementation omitted here)
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
        let start_time = Instant::now();

        // Ensure direct pipelines exist
        self.wgpu_renderer.ensure_direct_line_pipeline();
        self.wgpu_renderer.ensure_direct_triangle_pipeline();

        // Update camera uniforms for standard rendering path
        let view_proj_matrix = self.camera.view_proj_matrix();
        self.wgpu_renderer
            .update_uniforms(view_proj_matrix, Mat4::IDENTITY);

        // Collect render data and buffers (include optional index buffers)
        let mut render_items: Vec<(
            &crate::core::RenderData,
            Arc<wgpu::Buffer>,
            Option<wgpu::Buffer>,
        )> = Vec::new();
        let mut total_vertices = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some((vertex_buffer, index_buffer)) =
                    self.prepare_buffers_for_render_data(render_data)
                {
                    render_items.push((render_data, vertex_buffer, index_buffer));
                    total_vertices += render_data.vertex_count();
                }
            }
        }

        // (render pass will be created after precreating resources)

        // Update direct uniforms for line/triangle rendering (data -> viewport mapping)
        // IMPORTANT: When a viewport is set, clip-space coordinates are interpreted
        // relative to that viewport. Therefore, map data directly to [-1,1] in both axes.
        // The viewport transformation will place the geometry into (viewport_x, viewport_y, viewport_width, viewport_height).
        let ndc_left = -1.0;
        let ndc_right = 1.0;
        let ndc_bottom = -1.0;
        let ndc_top = 1.0;

        let (x_min, y_min, x_max, y_max) = data_bounds; // already reordered by caller
        self.wgpu_renderer.update_direct_uniforms(
            [x_min as f32, y_min as f32],
            [x_max as f32, y_max as f32],
            [ndc_left, ndc_bottom],
            [ndc_right, ndc_top],
            [viewport.2, viewport.3],
        );

        // Ensure pipelines are ready to avoid borrow conflicts inside the loop
        self.wgpu_renderer
            .ensure_pipeline(crate::core::PipelineType::Points);
        self.wgpu_renderer
            .ensure_pipeline(crate::core::PipelineType::Lines);
        self.wgpu_renderer
            .ensure_pipeline(crate::core::PipelineType::Triangles);
        self.wgpu_renderer.ensure_direct_line_pipeline();
        self.wgpu_renderer.ensure_direct_triangle_pipeline();
        self.wgpu_renderer.ensure_direct_point_pipeline();
        self.wgpu_renderer.ensure_image_pipeline();

        // Pre-create image bind groups to avoid lifetimes inside pass
        let mut image_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        let mut point_style_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in &render_items {
            if render_data.pipeline_type == crate::core::PipelineType::Textured {
                if let Some(crate::core::scene::ImageData::Rgba8 {
                    width,
                    height,
                    data,
                }) = &render_data.image
                {
                    let (_t, _v, img_bg) = self
                        .wgpu_renderer
                        .create_image_texture_and_bind_group(*width, *height, data);
                    image_bind_groups.push(Some(img_bg));
                } else {
                    image_bind_groups.push(None);
                }
            } else {
                image_bind_groups.push(None);
            }
            if render_data.pipeline_type == crate::core::PipelineType::Points {
                let style = crate::core::renderer::PointStyleUniforms {
                    face_color: render_data.material.albedo.to_array(),
                    edge_color: render_data.material.emissive.to_array(),
                    edge_thickness_px: render_data.material.roughness,
                    marker_shape: render_data.material.metallic as u32,
                    _pad: [0.0, 0.0],
                };
                let (_buf, bg) = self.wgpu_renderer.create_point_style_bind_group(style);
                point_style_bind_groups.push(Some(bg));
            } else {
                point_style_bind_groups.push(None);
            }
        }

        // Begin pass with Load (preserve egui), after precreating image bind groups
        {
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
            // Constrain drawing to the plot viewport
            let (viewport_x, viewport_y, viewport_width, viewport_height) = viewport;
            render_pass.set_viewport(
                viewport_x,
                viewport_y,
                viewport_width,
                viewport_height,
                0.0,
                1.0,
            );

            // Execute rendering within viewport mapping
            let mut __temp_point_buffers_cam: Vec<wgpu::Buffer> = Vec::new();
            for (idx, (render_data, vertex_buffer, index_buffer)) in render_items.iter().enumerate()
            {
                match render_data.pipeline_type {
                    crate::core::PipelineType::Lines => {
                        let pipeline = self.wgpu_renderer.direct_line_pipeline.as_ref().unwrap();
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(
                            0,
                            &self.wgpu_renderer.direct_uniform_bind_group,
                            &[],
                        );
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        for draw_call in &render_data.draw_calls {
                            render_pass.draw(
                                draw_call.vertex_offset as u32
                                    ..(draw_call.vertex_offset + draw_call.vertex_count) as u32,
                                0..draw_call.instance_count as u32,
                            );
                        }
                    }
                    crate::core::PipelineType::Triangles => {
                        // Use direct triangle pipeline so triangles map to the viewport rect
                        let pipeline = self
                            .wgpu_renderer
                            .direct_triangle_pipeline
                            .as_ref()
                            .unwrap();
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(
                            0,
                            &self.wgpu_renderer.direct_uniform_bind_group,
                            &[],
                        );
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        if let Some(idx) = index_buffer {
                            render_pass.set_index_buffer(idx.slice(..), wgpu::IndexFormat::Uint32);
                            if let Some(indices) = &render_data.indices {
                                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                            }
                        }
                    }
                    crate::core::PipelineType::Textured => {
                        // Use image pipeline with direct uniforms + image bind group
                        let pipeline = self
                            .wgpu_renderer
                            .get_pipeline(crate::core::PipelineType::Textured);
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(
                            0,
                            &self.wgpu_renderer.direct_uniform_bind_group,
                            &[],
                        );
                        if let Some(ref bg) = image_bind_groups[idx] {
                            render_pass.set_bind_group(1, bg, &[]);
                        }
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        if let Some(idx) = index_buffer {
                            render_pass.set_index_buffer(idx.slice(..), wgpu::IndexFormat::Uint32);
                            if let Some(indices) = &render_data.indices {
                                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                            }
                        } else {
                            for dc in &render_data.draw_calls {
                                render_pass.draw(
                                    dc.vertex_offset as u32
                                        ..(dc.vertex_offset + dc.vertex_count) as u32,
                                    0..dc.instance_count as u32,
                                );
                            }
                        }
                    }
                    _ => {
                        // Fallback to standard pipeline for other types
                        let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(
                            0,
                            self.wgpu_renderer.get_uniform_bind_group(),
                            &[],
                        );
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        if render_data.indices.is_some() {
                            if let Some(idx) = index_buffer {
                                render_pass
                                    .set_index_buffer(idx.slice(..), wgpu::IndexFormat::Uint32);
                                if let Some(indices) = &render_data.indices {
                                    render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                                }
                            }
                        } else {
                            // Points: bind direct uniforms and optional style for markers
                            if matches!(
                                render_data.pipeline_type,
                                crate::core::PipelineType::Points
                            ) {
                                // Use direct pipeline for points for pixel-stable markers
                                let pipeline =
                                    self.wgpu_renderer.direct_point_pipeline.as_ref().unwrap();
                                render_pass.set_pipeline(pipeline);
                                render_pass.set_bind_group(
                                    0,
                                    &self.wgpu_renderer.direct_uniform_bind_group,
                                    &[],
                                );
                                if let Some(ref bg) = point_style_bind_groups[idx] {
                                    render_pass.set_bind_group(1, bg, &[]);
                                }
                            }
                            for dc in &render_data.draw_calls {
                                render_pass.draw(
                                    dc.vertex_offset as u32
                                        ..(dc.vertex_offset + dc.vertex_count) as u32,
                                    0..dc.instance_count as u32,
                                );
                            }
                        }
                    }
                }
            }

            // render_pass dropped at end of scope
        }

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
        let start_time = Instant::now();

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
                if let Some((vertex_buffer, index_buffer)) =
                    self.prepare_buffers_for_render_data(render_data)
                {
                    self.wgpu_renderer
                        .ensure_pipeline(render_data.pipeline_type);
                    render_items.push((render_data, vertex_buffer, index_buffer));

                    total_vertices += render_data.vertex_count();
                    if let Some(indices) = &render_data.indices {
                        total_triangles += indices.len() / 3;
                    }
                }
            }
        }

        // Pre-create image bind groups and set direct uniforms once (for textured items)
        let mut image_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        // Ensure image pipeline once to avoid mutable borrow during pass
        self.wgpu_renderer.ensure_image_pipeline();
        if let Some((x_min, x_max, y_min, y_max)) = self.data_bounds {
            self.wgpu_renderer.update_direct_uniforms(
                [x_min as f32, y_min as f32],
                [x_max as f32, y_max as f32],
                [-1.0, -1.0],
                [1.0, 1.0],
                [config.width as f32, config.height as f32],
            );
        }
        for (render_data, _vb, _ib) in &render_items {
            if render_data.pipeline_type == crate::core::PipelineType::Textured {
                if let Some(crate::core::scene::ImageData::Rgba8 {
                    width,
                    height,
                    data,
                }) = &render_data.image
                {
                    let (_tex, _view, img_bg) = self
                        .wgpu_renderer
                        .create_image_texture_and_bind_group(*width, *height, data);
                    image_bind_groups.push(Some(img_bg));
                } else {
                    image_bind_groups.push(None);
                }
            } else {
                image_bind_groups.push(None);
            }
        }
        let mut point_style_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in &render_items {
            if render_data.pipeline_type == crate::core::PipelineType::Points {
                let style = crate::core::renderer::PointStyleUniforms {
                    face_color: render_data.material.albedo.to_array(),
                    edge_color: render_data.material.emissive.to_array(),
                    edge_thickness_px: render_data.material.roughness,
                    marker_shape: render_data.material.metallic as u32,
                    _pad: [0.0, 0.0],
                };
                let (_buf, bg) = self.wgpu_renderer.create_point_style_bind_group(style);
                point_style_bind_groups.push(Some(bg));
            } else {
                point_style_bind_groups.push(None);
            }
        }

        // Create render pass
        {
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
            for (i, (render_data, vertex_buffer, index_buffer)) in render_items.iter().enumerate() {
                // Get the appropriate pipeline for this render data (pipeline ensured above)
                if render_data.pipeline_type == crate::core::PipelineType::Textured {
                    // Ensure image pipeline
                    let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
                    render_pass.set_pipeline(pipeline);
                    // Bind direct uniforms at set(0)
                    // Use data bounds for image mapping
                    render_pass.set_bind_group(
                        0,
                        &self.wgpu_renderer.direct_uniform_bind_group,
                        &[],
                    );
                    if let Some(ref img_bg) = image_bind_groups[i] {
                        render_pass.set_bind_group(1, img_bg, &[]);
                    }
                } else {
                    let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
                    render_pass.set_pipeline(pipeline);
                    // Set the uniform bind group (required by shaders)
                    render_pass.set_bind_group(0, self.wgpu_renderer.get_uniform_bind_group(), &[]);
                }

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

                if let Some(index_buffer) = index_buffer {
                    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    if let Some(indices) = &render_data.indices {
                        log::trace!(target: "runmat_plot", "draw indexed count={}", indices.len());
                        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                    }
                } else {
                    log::trace!(target: "runmat_plot", "draw direct vertices");
                    // Use draw_calls from render_data for proper vertex range handling
                    for draw_call in &render_data.draw_calls {
                        log::trace!(
                            target: "runmat_plot",
                            "draw vertices offset={} count={} instances={}",
                            draw_call.vertex_offset,
                            draw_call.vertex_count,
                            draw_call.instance_count
                        );
                        render_pass.draw(
                            draw_call.vertex_offset as u32
                                ..(draw_call.vertex_offset + draw_call.vertex_count) as u32,
                            0..draw_call.instance_count as u32,
                        );
                    }
                }
            }
            // drop render_pass at end of scope
        }

        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: render_time,
        })
    }

    /// Render using the camera-based pipeline into a viewport region with a scissor rectangle.
    /// This preserves existing contents (Load) and draws only inside the viewport rectangle.
    pub fn render_camera_to_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        viewport_scissor: (u32, u32, u32, u32),
        config: &PlotRenderConfig,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        // Apply MSAA preference into pipelines
        self.wgpu_renderer.ensure_msaa(config.msaa_samples);

        // Update aspect ratio based on provided config (caller should pass plot area aspect)
        let aspect_ratio = (config.width.max(1)) as f32 / (config.height.max(1)) as f32;
        self.camera.update_aspect_ratio(aspect_ratio);

        // Update standard uniforms
        let view_proj_matrix = self.camera.view_proj_matrix();
        self.wgpu_renderer
            .update_uniforms(view_proj_matrix, Mat4::IDENTITY);

        // Prepare render items outside the pass
        let mut render_items = Vec::new();
        let mut total_vertices = 0usize;
        let mut total_triangles = 0usize;
        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some((vb, ib)) = self.prepare_buffers_for_render_data(render_data) {
                    self.wgpu_renderer
                        .ensure_pipeline(render_data.pipeline_type);
                    total_vertices += render_data.vertex_count();
                    if let Some(indices) = &render_data.indices {
                        total_triangles += indices.len() / 3;
                    }
                    render_items.push((render_data, vb, ib));
                }
            }
        }

        // Precompute expanded point buffers to keep them alive across the render pass
        let mut point_buffers: Vec<Option<(wgpu::Buffer, usize)>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in render_items.iter() {
            if matches!(render_data.pipeline_type, crate::core::PipelineType::Points) {
                let expanded = self
                    .wgpu_renderer
                    // size_px=0.0 => use per-vertex normal.z sizes
                    .create_direct_point_vertices(&render_data.vertices, 0.0);
                let buf = self.wgpu_renderer.create_vertex_buffer(&expanded);
                point_buffers.push(Some((buf, expanded.len())));
            } else {
                point_buffers.push(None);
            }
        }
        // Precreate image bind groups for textured items to avoid lifetime issues
        self.wgpu_renderer.ensure_image_pipeline();
        let mut image_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());

        for (render_data, _vb, _ib) in render_items.iter() {
            if render_data.pipeline_type == crate::core::PipelineType::Textured {
                if let Some(crate::core::scene::ImageData::Rgba8 {
                    width,
                    height,
                    data,
                }) = &render_data.image
                {
                    let (_t, _v, bg) = self
                        .wgpu_renderer
                        .create_image_texture_and_bind_group(*width, *height, data);
                    image_bind_groups.push(Some(bg));
                } else {
                    image_bind_groups.push(None);
                }
            } else {
                image_bind_groups.push(None);
            }
        }
        // Precreate point style bind groups for points to match pipeline layout [direct uniforms, point style]
        let mut point_style_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in render_items.iter() {
            if matches!(render_data.pipeline_type, crate::core::PipelineType::Points) {
                let style = crate::core::renderer::PointStyleUniforms {
                    face_color: render_data.material.albedo.to_array(),
                    edge_color: render_data.material.emissive.to_array(),
                    edge_thickness_px: render_data.material.roughness,
                    marker_shape: render_data.material.metallic as u32,
                    _pad: [0.0, 0.0],
                };
                let (_buf, bg) = self.wgpu_renderer.create_point_style_bind_group(style);
                point_style_bind_groups.push(Some(bg));
            } else {
                point_style_bind_groups.push(None);
            }
        }

        // Precompute optional grid geometry and uniforms so we can draw it under data
        let (sx, sy, sw, sh) = viewport_scissor;
        // Grid is drawn only when enabled and in 2D orthographic
        let mut grid_vb_opt: Option<wgpu::Buffer> = None;
        if self.figure_show_grid {
            if let Some((l, r, b, t)) = self.view_bounds() {
                // Update direct uniforms mapping for viewport
                self.wgpu_renderer.update_direct_uniforms(
                    [l as f32, b as f32],
                    [r as f32, t as f32],
                    [-1.0, -1.0],
                    [1.0, 1.0],
                    [sw.max(1) as f32, sh.max(1) as f32],
                );
                self.wgpu_renderer.ensure_direct_line_pipeline();

                let x_range = (r - l).max(1e-6);
                let y_range = (t - b).max(1e-6);
                let x_step = plot_utils::calculate_tick_interval(x_range);
                let y_step = plot_utils::calculate_tick_interval(y_range);
                let mut grid_vertices: Vec<Vertex> = Vec::new();
                let g = 80.0_f32 / 255.0_f32;
                let col = Vec4::new(g, g, g, 1.0);
                if x_step.is_finite() && x_step > 0.0 {
                    let mut x = ((l / x_step).ceil() * x_step) as f32;
                    let b_f = b as f32;
                    let t_f = t as f32;
                    while (x as f64) <= r {
                        grid_vertices.push(Vertex::new(Vec3::new(x, b_f, 0.0), col));
                        grid_vertices.push(Vertex::new(Vec3::new(x, t_f, 0.0), col));
                        x += x_step as f32;
                    }
                }
                if y_step.is_finite() && y_step > 0.0 {
                    let mut y = ((b / y_step).ceil() * y_step) as f32;
                    let l_f = l as f32;
                    let r_f = r as f32;
                    while (y as f64) <= t {
                        grid_vertices.push(Vertex::new(Vec3::new(l_f, y, 0.0), col));
                        grid_vertices.push(Vertex::new(Vec3::new(r_f, y, 0.0), col));
                        y += y_step as f32;
                    }
                }
                if !grid_vertices.is_empty() {
                    grid_vb_opt = Some(self.wgpu_renderer.create_vertex_buffer(&grid_vertices));
                }
            }
        }

        // Before the pass: configure direct uniforms and ensure pipelines
        let bounds_opt = self.data_bounds;
        if let Some((l, r, b, t)) = self.view_bounds() {
            self.wgpu_renderer.update_direct_uniforms(
                [l as f32, b as f32],
                [r as f32, t as f32],
                [-1.0, -1.0],
                [1.0, 1.0],
                [sw.max(1) as f32, sh.max(1) as f32],
            );
        } else if let Some((x_min, x_max, y_min, y_max)) = bounds_opt {
            self.wgpu_renderer.update_direct_uniforms(
                [x_min as f32, y_min as f32],
                [x_max as f32, y_max as f32],
                [-1.0, -1.0],
                [1.0, 1.0],
                [sw.max(1) as f32, sh.max(1) as f32],
            );
        }
        self.wgpu_renderer.ensure_direct_triangle_pipeline();
        self.wgpu_renderer.ensure_direct_line_pipeline();
        self.wgpu_renderer.ensure_direct_point_pipeline();

        // Begin pass with Load (preserve egui)
        {
            // Prepare MSAA render target if enabled
            let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
            let msaa_view_opt = if use_msaa {
                let tex = self
                    .wgpu_renderer
                    .device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("runmat_msaa_color_direct"),
                        size: wgpu::Extent3d {
                            width: self.wgpu_renderer.surface_config.width,
                            height: self.wgpu_renderer.surface_config.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: self.wgpu_renderer.msaa_sample_count,
                        dimension: wgpu::TextureDimension::D2,
                        format: self.wgpu_renderer.surface_config.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    });
                Some(tex.create_view(&wgpu::TextureViewDescriptor::default()))
            } else {
                None
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Plot Camera Viewport Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: msaa_view_opt.as_ref().unwrap_or(target_view),
                    resolve_target: if use_msaa { Some(target_view) } else { None },
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Apply viewport and scissor rectangle to draw into the plot rect
            render_pass.set_viewport(
                sx as f32,
                sy as f32,
                sw.max(1) as f32,
                sh.max(1) as f32,
                0.0,
                1.0,
            );
            render_pass.set_scissor_rect(sx, sy, sw.max(1), sh.max(1));
            if let Some(ref vb_grid) = grid_vb_opt {
                if let Some(ref pipeline) = self.wgpu_renderer.direct_line_pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(
                        0,
                        &self.wgpu_renderer.direct_uniform_bind_group,
                        &[],
                    );
                    render_pass.set_vertex_buffer(0, vb_grid.slice(..));
                    // Each grid line is two vertices (LineList)
                    // Draw full buffer
                    // Note: vertex count equals number of vertices
                    // wgpu will interpret as lines via pipeline topology
                    render_pass.draw(
                        0..(vb_grid.size() / std::mem::size_of::<Vertex>() as u64) as u32,
                        0..1,
                    );
                }
            }

            // Use direct pipelines for precise 2D mapping inside the viewport
            let use_direct_for_triangles = true;
            let use_direct_for_lines = true;
            let direct_tri_pipeline = if use_direct_for_triangles && bounds_opt.is_some() {
                self.wgpu_renderer
                    .direct_triangle_pipeline
                    .as_ref()
                    .map(|p| p as *const wgpu::RenderPipeline)
            } else {
                None
            };
            let direct_line_pipeline = if use_direct_for_lines && bounds_opt.is_some() {
                self.wgpu_renderer
                    .direct_line_pipeline
                    .as_ref()
                    .map(|p| p as *const wgpu::RenderPipeline)
            } else {
                None
            };
            let direct_point_pipeline = if bounds_opt.is_some() {
                self.wgpu_renderer
                    .direct_point_pipeline
                    .as_ref()
                    .map(|p| p as *const wgpu::RenderPipeline)
            } else {
                None
            };

            // Keep transient point buffers alive during this pass
            let mut __temp_point_buffers_cam: Vec<wgpu::Buffer> = Vec::new();
            for (idx, (render_data, vertex_buffer, index_buffer)) in render_items.iter().enumerate()
            {
                let is_triangles = matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Triangles
                );
                let is_lines =
                    matches!(render_data.pipeline_type, crate::core::PipelineType::Lines);
                let is_points =
                    matches!(render_data.pipeline_type, crate::core::PipelineType::Points);
                let is_textured = matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Textured
                );
                // Use direct mapping for lines/triangles/points for correct pixel-sized markers in GUI
                let use_direct = ((use_direct_for_triangles && is_triangles)
                    || (use_direct_for_lines && is_lines)
                    || is_points)
                    && bounds_opt.is_some();

                if use_direct {
                    // Safe because we only read pointers here within pass
                    let pipeline_ref: &wgpu::RenderPipeline = unsafe {
                        if is_triangles {
                            direct_tri_pipeline.unwrap().as_ref().unwrap()
                        } else if is_lines {
                            direct_line_pipeline.unwrap().as_ref().unwrap()
                        } else {
                            direct_point_pipeline.unwrap().as_ref().unwrap()
                        }
                    };
                    let uniform_bg = &self.wgpu_renderer.direct_uniform_bind_group;
                    render_pass.set_pipeline(pipeline_ref);
                    render_pass.set_bind_group(0, uniform_bg, &[]);
                } else if is_textured {
                    let pipeline = self
                        .wgpu_renderer
                        .get_pipeline(crate::core::PipelineType::Textured);
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(
                        0,
                        &self.wgpu_renderer.direct_uniform_bind_group,
                        &[],
                    );
                    if let Some(ref bg) = image_bind_groups[idx] {
                        render_pass.set_bind_group(1, bg, &[]);
                    }
                } else {
                    let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(
                        0,
                        self.wgpu_renderer.get_uniform_bind_group(),
                        &[],
                    );
                }

                if is_points && use_direct {
                    if let Some((ref buf, len)) = point_buffers[idx] {
                        if let Some(ref bg) = point_style_bind_groups[idx] {
                            render_pass.set_bind_group(1, bg, &[]);
                        }
                        render_pass.set_vertex_buffer(0, buf.slice(..));
                        render_pass.draw(0..len as u32, 0..1);
                        continue;
                    }
                } else {
                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                }
                if let Some(idx) = index_buffer {
                    render_pass.set_index_buffer(idx.slice(..), wgpu::IndexFormat::Uint32);
                    if let Some(indices) = &render_data.indices {
                        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                    }
                } else {
                    for dc in &render_data.draw_calls {
                        render_pass.draw(
                            dc.vertex_offset as u32..(dc.vertex_offset + dc.vertex_count) as u32,
                            0..dc.instance_count as u32,
                        );
                    }
                }
            }
        }

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Render all axes of a subplot grid into their respective viewport rectangles.
    /// `axes_viewports` is a vector of (x, y, w, h) in physical pixels, length equals rows*cols.
    pub fn render_axes_to_viewports(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        axes_viewports: &[(u32, u32, u32, u32)],
        msaa_samples: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Build map axes_index -> node ids
        let mut axes_to_nodes: std::collections::HashMap<usize, Vec<crate::core::scene::NodeId>> =
            std::collections::HashMap::new();
        for node in self.scene.get_visible_nodes() {
            axes_to_nodes
                .entry(node.axes_index)
                .or_default()
                .push(node.id);
        }

        if self.axes_cameras.is_empty() {
            self.axes_cameras.push(Self::create_default_camera());
        }

        // Pre-collect all node ids
        let all_ids: Vec<crate::core::scene::NodeId> = self
            .scene
            .get_visible_nodes()
            .into_iter()
            .map(|n| n.id)
            .collect();

        for (ax_idx, viewport) in axes_viewports.iter().enumerate() {
            let ids_for_axes = axes_to_nodes.get(&ax_idx).cloned().unwrap_or_default();
            if ids_for_axes.is_empty() {
                continue;
            }

            // Hide nodes not belonging to this axes
            let mut hidden_ids: Vec<crate::core::scene::NodeId> = Vec::new();
            for id in &all_ids {
                if !ids_for_axes.contains(id) {
                    if let Some(node) = self.scene.get_node_mut(*id) {
                        if node.visible {
                            node.visible = false;
                            hidden_ids.push(*id);
                        }
                    }
                }
            }
            // Update camera and bounds
            if let Some(cam) = self.axes_cameras.get(ax_idx).cloned() {
                self.camera = cam;
            }
            let _ = self.calculate_data_bounds();

            // Render this axes into its viewport
            let cfg = PlotRenderConfig {
                width: viewport.2,
                height: viewport.3,
                msaa_samples,
                ..Default::default()
            };
            let _ = self.render_camera_to_viewport(encoder, target_view, *viewport, &cfg)?;

            // Restore hidden nodes visibility
            for id in hidden_ids {
                if let Some(node) = self.scene.get_node_mut(id) {
                    node.visible = true;
                }
            }
        }
        Ok(())
    }

    /// Create default 2D camera for plotting
    fn create_default_camera() -> Camera {
        let mut camera = Camera::new();
        camera.projection = crate::core::camera::ProjectionType::Orthographic {
            left: -5.0,
            right: 5.0,
            bottom: -5.0,
            top: 5.0,
            near: -1.0,
            far: 1.0,
        };
        camera.position = Vec3::new(0.0, 0.0, 5.0);
        camera.target = Vec3::new(0.0, 0.0, 0.0);
        camera.up = Vec3::new(0.0, 1.0, 0.0);
        camera
    }

    // Removed simple data_bounds getter in favor of overlay-aware bounds below

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

    /// Get current view bounds (camera frustum) in world/data space for 2D
    pub fn view_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        match self.camera.projection {
            crate::core::camera::ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                ..
            } => Some((left as f64, right as f64, bottom as f64, top as f64)),
            _ => None,
        }
    }

    /// Overlay configuration getters
    pub fn overlay_show_grid(&self) -> bool {
        self.figure_show_grid
    }
    pub fn overlay_title(&self) -> Option<&String> {
        self.figure_title.as_ref()
    }
    pub fn overlay_x_label(&self) -> Option<&String> {
        self.figure_x_label.as_ref()
    }
    pub fn overlay_y_label(&self) -> Option<&String> {
        self.figure_y_label.as_ref()
    }
    pub fn overlay_show_legend(&self) -> bool {
        self.figure_show_legend
    }
    pub fn overlay_legend_entries(&self) -> &Vec<LegendEntry> {
        &self.legend_entries
    }
    pub fn overlay_x_log(&self) -> bool {
        self.figure_x_log
    }
    pub fn overlay_y_log(&self) -> bool {
        self.figure_y_log
    }
    pub fn overlay_colormap(&self) -> ColorMap {
        self.figure_colormap
    }
    pub fn overlay_colorbar_enabled(&self) -> bool {
        self.figure_colorbar_enabled
    }
    /// Subplot grid
    pub fn figure_axes_grid(&self) -> (usize, usize) {
        self.scene_axes_grid_fallback()
    }

    fn scene_axes_grid_fallback(&self) -> (usize, usize) {
        // We don't retain Figure here; infer grid from number of axes cameras if present, else 1x1
        if !self.axes_cameras.is_empty() {
            // Try best effort: assume rows*cols == axes_cameras.len() with rows contiguous
            // Default to 1 x N for now
            (1, self.axes_cameras.len())
        } else {
            (1, 1)
        }
    }
    /// Return categorical labels if any (is_x_axis, &labels)
    pub fn overlay_categorical_labels(&self) -> Option<(bool, &Vec<String>)> {
        if let (Some(is_x), Some(labels)) = (
            &self.figure_categorical_is_x,
            &self.figure_categorical_labels,
        ) {
            Some((*is_x, labels))
        } else {
            None
        }
    }

    /// Get bounds used for display (manual axis limits override data bounds when provided)
    pub fn data_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        let base = self.data_bounds;
        base.map(|(bx_min, bx_max, by_min, by_max)| {
            let (mut x_min, mut x_max) = (bx_min, bx_max);
            let (mut y_min, mut y_max) = (by_min, by_max);
            if let Some((xl, xr)) = self.figure_x_limits {
                x_min = xl;
                x_max = xr;
            }
            if let Some((yl, yr)) = self.figure_y_limits {
                y_min = yl;
                y_max = yr;
            }
            (x_min, x_max, y_min, y_max)
        })
    }

    /// Get mutable reference to a specific axes camera when using subplots
    pub fn axes_camera_mut(&mut self, idx: usize) -> Option<&mut Camera> {
        self.axes_cameras.get_mut(idx)
    }

    /// Get view bounds for a specific axes camera (l, r, b, t)
    pub fn view_bounds_for_axes(&self, idx: usize) -> Option<(f64, f64, f64, f64)> {
        if let Some(cam) = self.axes_cameras.get(idx) {
            if let crate::core::camera::ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                ..
            } = cam.projection
            {
                return Some((left as f64, right as f64, bottom as f64, top as f64));
            }
        }
        None
    }

    /// Prefer exporting the original figure if available
    pub fn export_figure_clone(&self) -> crate::plots::Figure {
        if let Some(f) = &self.last_figure {
            return f.clone();
        }
        // As a strict fallback, produce an empty figure with current metadata only
        let mut fig = crate::plots::Figure::new();
        fig.title = self.figure_title.clone();
        fig.x_label = self.figure_x_label.clone();
        fig.y_label = self.figure_y_label.clone();
        fig.legend_enabled = self.figure_show_legend;
        fig.grid_enabled = self.figure_show_grid;
        fig.x_limits = self.figure_x_limits;
        fig.y_limits = self.figure_y_limits;
        fig.x_log = self.figure_x_log;
        fig.y_log = self.figure_y_log;
        fig.axis_equal = self.figure_axis_equal;
        fig.colormap = self.figure_colormap;
        fig.colorbar_enabled = self.figure_colorbar_enabled;
        let (rows, cols) = self.figure_axes_grid();
        fig.set_subplot_grid(rows, cols);
        fig
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
        } else if value.abs() >= 1000.0 || value.fract().abs() < 0.001 {
            format!("{value:.0}")
        } else {
            format!("{value:.1}")
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
