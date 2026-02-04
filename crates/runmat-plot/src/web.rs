//! WebGPU renderer integration for browser builds.
//!
//! This module owns the WGPU surface that backs a `<canvas>` element and
//! exposes a light-weight wrapper around [`PlotRenderer`] so wasm callers can
//! drive the full RunMat plotting stack without bouncing through JavaScript
//! custom events.

#![cfg(all(target_arch = "wasm32", feature = "web"))]

use crate::context::SharedWgpuContext;
use crate::core::{camera::MouseButton as CameraMouseButton, CameraController, PlotEvent};
use crate::core::plot_renderer::{PlotRenderConfig, PlotRenderer, RenderTarget};
use crate::plots::Figure;
use log::{debug, warn};
use std::sync::Arc;
use thiserror::Error;
use web_sys::{HtmlCanvasElement, OffscreenCanvas};

#[cfg(feature = "egui-overlay")]
use crate::overlay::plot_overlay::{OverlayConfig, OverlayMetrics, PlotOverlay};
#[cfg(feature = "egui-overlay")]
use egui_wgpu::ScreenDescriptor;

/// Canvas handle accepted by the web renderer.
#[derive(Clone)]
pub enum WebCanvas {
    Html(HtmlCanvasElement),
    Offscreen(OffscreenCanvas),
}

impl WebCanvas {
    fn width(&self) -> u32 {
        match self {
            WebCanvas::Html(canvas) => canvas.width(),
            WebCanvas::Offscreen(canvas) => canvas.width(),
        }
    }

    fn height(&self) -> u32 {
        match self {
            WebCanvas::Html(canvas) => canvas.height(),
            WebCanvas::Offscreen(canvas) => canvas.height(),
        }
    }

    fn set_width(&self, width: u32) {
        match self {
            WebCanvas::Html(canvas) => canvas.set_width(width),
            WebCanvas::Offscreen(canvas) => canvas.set_width(width),
        }
    }

    fn set_height(&self, height: u32) {
        match self {
            WebCanvas::Html(canvas) => canvas.set_height(height),
            WebCanvas::Offscreen(canvas) => canvas.set_height(height),
        }
    }
}

/// Configuration for the WebGPU renderer.
#[derive(Debug, Clone)]
pub struct WebRendererOptions {
    /// Preferred surface width; falls back to the canvas dimensions.
    pub width: Option<u32>,
    /// Preferred surface height; falls back to the canvas dimensions.
    pub height: Option<u32>,
    /// Target power preference when selecting an adapter.
    pub power_preference: wgpu::PowerPreference,
    /// Requested present mode if the surface supports it.
    pub present_mode: wgpu::PresentMode,
    /// Requested MSAA sample count.
    pub msaa_samples: u32,
}

impl Default for WebRendererOptions {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            msaa_samples: 4,
        }
    }
}

/// Errors produced while initializing or rendering with the web renderer.
#[derive(Debug, Error)]
pub enum WebRendererError {
    #[error("canvas has zero area; ensure width/height attributes are set")]
    CanvasZeroArea,
    #[error("surface creation failed: {0}")]
    Surface(#[from] wgpu::CreateSurfaceError),
    #[error("no compatible WebGPU adapter was found")]
    AdapterUnavailable,
    #[error("device creation failed: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("plot renderer initialization failed: {0}")]
    PlotInit(String),
    #[error("surface error: {0}")]
    SurfaceFrame(#[from] wgpu::SurfaceError),
    #[error("rendering failed: {0}")]
    Render(String),
}

/// Owns the WGPU instance/surface backing a `<canvas>` so wasm callers can
/// render figures directly from Rust.
pub struct WebRenderer {
    canvas: WebCanvas,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_config: wgpu::SurfaceConfiguration,
    plot_renderer: PlotRenderer,
    camera_controller: CameraController,
    render_config: PlotRenderConfig,
    options: WebRendererOptions,
    msaa_texture: Option<wgpu::Texture>,
    msaa_extent: (u32, u32),
    pixels_per_point: f32,
    last_axes_viewports_px: Vec<(u32, u32, u32, u32)>,
    last_pointer_position: glam::Vec2,
    #[cfg(feature = "egui-overlay")]
    overlay: Option<WebOverlayState>,
}

#[cfg(feature = "egui-overlay")]
struct WebOverlayState {
    egui_ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    plot_overlay: PlotOverlay,
}

impl WebRenderer {
    /// Initialize the renderer for the provided canvas element.
    pub async fn new(
        canvas: WebCanvas,
        options: WebRendererOptions,
    ) -> Result<Self, WebRendererError> {
        Self::init(canvas, options, None).await
    }

    /// Initialize the renderer using a shared GPU context supplied by the host runtime.
    pub async fn with_shared_context(
        canvas: WebCanvas,
        options: WebRendererOptions,
        context: SharedWgpuContext,
    ) -> Result<Self, WebRendererError> {
        Self::init(canvas, options, Some(context)).await
    }

    async fn init(
        canvas: WebCanvas,
        options: WebRendererOptions,
        shared: Option<SharedWgpuContext>,
    ) -> Result<Self, WebRendererError> {
        let (width, height) = desired_canvas_size(&canvas, options.width, options.height)?;
        let (instance, shared_ctx) = if let Some(ctx) = shared {
            (ctx.instance.clone(), Some(ctx))
        } else {
            (
                Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    ..Default::default()
                })),
                None,
            )
        };
        let surface = match &canvas {
            WebCanvas::Html(element) => {
                instance.create_surface(wgpu::SurfaceTarget::Canvas(element.clone()))?
            }
            WebCanvas::Offscreen(element) => {
                instance.create_surface(wgpu::SurfaceTarget::OffscreenCanvas(element.clone()))?
            }
        };

        let (device, queue, adapter) = if let Some(ctx) = shared_ctx {
            (ctx.device, ctx.queue, ctx.adapter)
        } else {
            let adapter_raw = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: options.power_preference,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .ok_or(WebRendererError::AdapterUnavailable)?;
            let adapter = Arc::new(adapter_raw);
            let limits =
                wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
            let (device_raw, queue_raw) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("runmat-plot-web"),
                        required_features: wgpu::Features::empty(),
                        required_limits: limits.clone(),
                    },
                    None,
                )
                .await?;
            (Arc::new(device_raw), Arc::new(queue_raw), adapter)
        };

        let capabilities = surface.get_capabilities(&adapter);
        let format = pick_surface_format(&capabilities);
        let present_mode = pick_present_mode(&capabilities, options.present_mode);
        let alpha_mode = pick_alpha_mode(&capabilities);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };

        surface.configure(device.as_ref(), &surface_config);
        let plot_renderer =
            PlotRenderer::new(device.clone(), queue.clone(), surface_config.clone())
                .await
                .map_err(|err| WebRendererError::PlotInit(err.to_string()))?;

        #[cfg(feature = "egui-overlay")]
        let overlay = {
            let egui_ctx = egui::Context::default();
            // Match native overlay: modern dark theme + transparent panels.
            let theme = crate::styling::ModernDarkTheme::default();
            theme.apply_to_egui(&egui_ctx);
            let egui_renderer = egui_wgpu::Renderer::new(&device, surface_config.format, None, 1);
            Some(WebOverlayState {
                egui_ctx,
                egui_renderer,
                plot_overlay: PlotOverlay::new(),
            })
        };

        let mut renderer = Self {
            canvas,
            surface,
            device,
            queue,
            surface_config,
            plot_renderer,
            camera_controller: CameraController::new(),
            render_config: PlotRenderConfig {
                width,
                height,
                msaa_samples: options.msaa_samples,
                ..Default::default()
            },
            options,
            msaa_texture: None,
            msaa_extent: (0, 0),
            pixels_per_point: 1.0,
            last_axes_viewports_px: vec![(0, 0, width.max(1), height.max(1))],
            last_pointer_position: glam::Vec2::ZERO,
            #[cfg(feature = "egui-overlay")]
            overlay,
        };
        renderer.sync_renderer_config();
        Ok(renderer)
    }

    fn pick_axes_index(&self, position: glam::Vec2) -> usize {
        for (i, (x, y, w, h)) in self.last_axes_viewports_px.iter().enumerate() {
            let x0 = *x as f32;
            let y0 = *y as f32;
            let x1 = (*x + *w) as f32;
            let y1 = (*y + *h) as f32;
            if position.x >= x0 && position.x <= x1 && position.y >= y0 && position.y <= y1 {
                return i;
            }
        }
        0
    }

    /// Apply a user interaction event (mouse/keyboard) to the renderer state.
    /// Returns `true` when a re-render is recommended.
    pub fn handle_event(&mut self, event: PlotEvent) -> bool {
        match event {
            PlotEvent::MousePress {
                position,
                button,
                modifiers,
            } => {
                #[cfg(target_arch = "wasm32")]
                log::debug!(
                    target: "runmat_plot",
                    "web.handle_event MousePress pos=({:.1},{:.1}) button={:?}",
                    position.x,
                    position.y,
                    button
                );
                self.camera_controller
                    .mouse_press(position, map_mouse_button(button), modifiers);
                self.last_pointer_position = position;
                self.plot_renderer.note_camera_interaction();
                true
            }
            PlotEvent::MouseRelease {
                position,
                button,
                modifiers,
            } => {
                #[cfg(target_arch = "wasm32")]
                log::debug!(target: "runmat_plot", "web.handle_event MouseRelease button={:?}", button);
                self.camera_controller
                    .mouse_release(position, map_mouse_button(button), modifiers);
                true
            }
            PlotEvent::MouseMove {
                position,
                delta,
                modifiers,
            } => {
                #[cfg(target_arch = "wasm32")]
                log::debug!(
                    target: "runmat_plot",
                    "web.handle_event MouseMove pos=({:.1},{:.1}) delta=({:.2},{:.2})",
                    position.x,
                    position.y,
                    delta.x,
                    delta.y
                );
                let axes_index = self.pick_axes_index(position);
                let (vx, vy, vw, vh) = self
                    .last_axes_viewports_px
                    .get(axes_index)
                    .copied()
                    .unwrap_or((0, 0, self.render_config.width.max(1), self.render_config.height.max(1)));
                let viewport = (vw.max(1), vh.max(1));
                if let Some(cam) = self.plot_renderer.axes_camera_mut(axes_index) {
                    self.camera_controller
                        .mouse_move(
                            glam::Vec2::new(position.x - (vx as f32), position.y - (vy as f32)),
                            delta,
                            viewport,
                            modifiers,
                            cam,
                        );
                }
                self.last_pointer_position = position;
                self.plot_renderer.note_camera_interaction();
                true
            }
            PlotEvent::MouseWheel {
                position,
                delta,
                modifiers,
            } => {
                #[cfg(target_arch = "wasm32")]
                {
                    let (proj, pos, target) = match self.plot_renderer.camera().projection {
                        crate::core::camera::ProjectionType::Perspective { .. } => (
                            "Perspective",
                            self.plot_renderer.camera().position,
                            self.plot_renderer.camera().target,
                        ),
                        crate::core::camera::ProjectionType::Orthographic { .. } => (
                            "Orthographic",
                            self.plot_renderer.camera().position,
                            self.plot_renderer.camera().target,
                        ),
                    };
                    log::debug!(
                        target: "runmat_plot",
                        "web.handle_event MouseWheel delta={:.3} proj={} cam.pos=({:.2},{:.2},{:.2}) cam.target=({:.2},{:.2},{:.2})",
                        delta,
                        proj,
                        pos.x,
                        pos.y,
                        pos.z,
                        target.x,
                        target.y,
                        target.z
                    );
                }
                let axes_index = self.pick_axes_index(position);
                if let Some(cam) = self.plot_renderer.axes_camera_mut(axes_index) {
                    let (vx, vy, vw, vh) = self
                        .last_axes_viewports_px
                        .get(axes_index)
                        .copied()
                        .unwrap_or((0, 0, self.render_config.width.max(1), self.render_config.height.max(1)));
                    let local = glam::Vec2::new(position.x - (vx as f32), position.y - (vy as f32));
                    self.camera_controller
                        .mouse_wheel(delta, local, (vw.max(1), vh.max(1)), modifiers, cam);
                }
                self.last_pointer_position = position;
                self.plot_renderer.note_camera_interaction();
                true
            }
            PlotEvent::Resize { .. } => true,
            PlotEvent::KeyPress { .. } | PlotEvent::KeyRelease { .. } => false,
        }
    }

    /// Explicitly resize the underlying surface to the provided dimensions (in
    /// physical pixels). Returns early if the surface already matches the
    /// requested size.
    pub fn resize_surface(&mut self, width: u32, height: u32) -> Result<(), WebRendererError> {
        if width == 0 || height == 0 {
            return Err(WebRendererError::CanvasZeroArea);
        }
        if self.surface_config.width == width && self.surface_config.height == height {
            return Ok(());
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.render_config.width = width;
        self.render_config.height = height;
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.msaa_texture = None;
        self.msaa_extent = (0, 0);
        self.reconfigure_surface()?;
        self.render_current_scene()
    }

    /// Update egui scaling (devicePixelRatio) used by the overlay.
    pub fn set_pixels_per_point(&mut self, pixels_per_point: f32) {
        if pixels_per_point.is_finite() && pixels_per_point > 0.0 {
            self.pixels_per_point = pixels_per_point;
        }
    }

    /// Render a [`Figure`] directly into the canvas.
    pub fn render_figure(&mut self, figure: Figure) -> Result<(), WebRendererError> {
        self.plot_renderer.set_figure(figure);
        self.render_current_scene()
    }

    /// Redraw the last figure that was provided.
    pub fn render_current_scene(&mut self) -> Result<(), WebRendererError> {
        self.sync_canvas_extent()?;
        self.plot_renderer
            .wgpu_renderer
            .ensure_msaa(self.options.msaa_samples);

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                debug!("runmat_plot/web: surface lost or outdated; reconfiguring");
                self.reconfigure_surface()?;
                self.surface.get_current_texture()?
            }
            Err(wgpu::SurfaceError::Timeout) => {
                warn!("runmat_plot/web: surface acquisition timed out");
                return Ok(());
            }
            Err(err) => return Err(WebRendererError::SurfaceFrame(err)),
        };

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        #[cfg(feature = "egui-overlay")]
        let use_overlay = self.overlay.is_some();
        #[cfg(not(feature = "egui-overlay"))]
        let use_overlay = false;

        let requested_samples = self.render_config.msaa_samples.max(1);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-plot-web-encoder"),
            });

        self.render_config.width = self.surface_config.width.max(1);
        self.render_config.height = self.surface_config.height.max(1);
        self.render_config.msaa_samples = self.options.msaa_samples.max(1);

        if !use_overlay {
            // Existing fast path: full-surface render (clears).
            let use_msaa = requested_samples > 1;
            if use_msaa {
                self.ensure_msaa_texture()?;
            }
            let msaa_view_holder = if use_msaa {
                Some(
                    self.msaa_texture
                        .as_ref()
                        .expect("MSAA texture missing")
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                )
            } else {
                None
            };
            let render_target = if let Some(msaa_view) = msaa_view_holder.as_ref() {
                RenderTarget {
                    view: msaa_view,
                    resolve_target: Some(&frame_view),
                }
            } else {
                RenderTarget {
                    view: &frame_view,
                    resolve_target: None,
                }
            };

            self.plot_renderer
                .render(&mut encoder, render_target, &self.render_config)
                .map_err(|err| WebRendererError::Render(err.to_string()))?;
        } else {
            #[cfg(feature = "egui-overlay")]
            {
                // Clear background once per frame.
                {
                    let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("runmat-plot-web-clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &frame_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: self.render_config.background_color.x as f64,
                                    g: self.render_config.background_color.y as f64,
                                    b: self.render_config.background_color.z as f64,
                                    a: self.render_config.background_color.w as f64,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                }

                let Some(overlay) = self.overlay.as_mut() else {
                    unreachable!("use_overlay implies overlay is Some");
                };

                // Ensure bounds are current before drawing overlay (keeps axes in sync with render).
                let _ = self.plot_renderer.calculate_data_bounds();

                overlay.egui_ctx.set_pixels_per_point(self.pixels_per_point);
                let raw_input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::new(0.0, 0.0),
                        egui::Vec2::new(
                            (self.surface_config.width.max(1) as f32) / self.pixels_per_point,
                            (self.surface_config.height.max(1) as f32) / self.pixels_per_point,
                        ),
                    )),
                    ..Default::default()
                };

                // Build overlay UI and capture plot area.
                let scene_stats = self.plot_renderer.scene.statistics();
                let mut plot_area_points: Option<egui::Rect> = None;
                let full_output = overlay.egui_ctx.run(raw_input, |ctx| {
                    let overlay_config = OverlayConfig {
                        // Let the plot pipeline draw grid under data (more efficient).
                        show_grid: false,
                        // Toolbar is handled by the host UI in the wasm IDE.
                        show_toolbar: false,
                        // Make overlay text more readable in the IDE.
                        font_scale: 1.25,
                        show_axes: true,
                        show_title: true,
                        title: self
                            .plot_renderer
                            .overlay_title()
                            .cloned()
                            .or(Some("Plot".to_string())),
                        x_label: self
                            .plot_renderer
                            .overlay_x_label()
                            .cloned()
                            .or(Some("X".to_string())),
                        y_label: self
                            .plot_renderer
                            .overlay_y_label()
                            .cloned()
                            .or(Some("Y".to_string())),
                        // Disable the heavyweight sidebar in IDE surfaces for now.
                        show_sidebar: false,
                        ..Default::default()
                    };
                    let overlay_metrics = OverlayMetrics {
                        vertex_count: scene_stats.total_vertices,
                        triangle_count: scene_stats.total_triangles,
                        render_time_ms: 0.0,
                        fps: 60.0,
                    };
                    let frame_info = overlay.plot_overlay.render(
                        ctx,
                        &self.plot_renderer,
                        &overlay_config,
                        overlay_metrics,
                    );
                    plot_area_points = frame_info.plot_area;
                });

                let paint_jobs = overlay
                    .egui_ctx
                    .tessellate(full_output.shapes, full_output.pixels_per_point);

                for (id, image_delta) in &full_output.textures_delta.set {
                    overlay.egui_renderer.update_texture(
                        &self.device,
                        &self.queue,
                        *id,
                        image_delta,
                    );
                }

                let screen_descriptor = ScreenDescriptor {
                    size_in_pixels: [
                        self.surface_config.width.max(1),
                        self.surface_config.height.max(1),
                    ],
                    pixels_per_point: full_output.pixels_per_point,
                };

                overlay.egui_renderer.update_buffers(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &paint_jobs,
                    &screen_descriptor,
                );

                // Determine plot viewport from overlay, defaulting to full canvas.
                let ppp = self.pixels_per_point.max(0.5);
                let (vx, vy, vw, vh) = if let Some(rect) = plot_area_points {
                    let vx = (rect.min.x * ppp).round().max(0.0) as u32;
                    let vy = (rect.min.y * ppp).round().max(0.0) as u32;
                    let vw = (rect.width() * ppp).round().max(1.0) as u32;
                    let vh = (rect.height() * ppp).round().max(1.0) as u32;
                    (vx, vy, vw, vh)
                } else {
                    (
                        0,
                        0,
                        self.surface_config.width.max(1),
                        self.surface_config.height.max(1),
                    )
                };

                // Align plot camera to the plot area aspect ratio (matching native behavior).
                if vw > 0 && vh > 0 {
                    self.plot_renderer
                        .camera_mut()
                        .update_aspect_ratio((vw as f32) / (vh as f32));
                }

                // Render plot content into the plot area viewport(s) using the camera-to-viewport path.
                let (rows, cols) = self.plot_renderer.figure_axes_grid();
                if rows * cols > 1 {
                    let rect_points = plot_area_points.unwrap_or_else(|| {
                        egui::Rect::from_min_size(
                            egui::Pos2::new(0.0, 0.0),
                            egui::Vec2::new(
                                (self.surface_config.width.max(1) as f32) / ppp,
                                (self.surface_config.height.max(1) as f32) / ppp,
                            ),
                        )
                    });
                    let rects = overlay.plot_overlay.compute_subplot_rects(
                        rect_points,
                        rows,
                        cols,
                        8.0,
                        8.0,
                    );
                    let sw = self.surface_config.width as f32;
                    let sh = self.surface_config.height as f32;
                    let mut viewports: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(rects.len());
                    for r in rects {
                        let rx = (r.min.x * ppp).round().max(0.0);
                        let ry = (r.min.y * ppp).round().max(0.0);
                        let mut rw = (r.width() * ppp).round().max(1.0);
                        let mut rh = (r.height() * ppp).round().max(1.0);
                        if rx + rw > sw {
                            rw = (sw - rx).max(1.0);
                        }
                        if ry + rh > sh {
                            rh = (sh - ry).max(1.0);
                        }
                        viewports.push((rx as u32, ry as u32, rw as u32, rh as u32));
                    }
                    self.last_axes_viewports_px = viewports.clone();
                    self.plot_renderer
                        .render_axes_to_viewports(
                            &mut encoder,
                            &frame_view,
                            &viewports,
                            requested_samples,
                        )
                        .map_err(|err| WebRendererError::Render(err.to_string()))?;
                } else {
                    self.last_axes_viewports_px = vec![(vx, vy, vw.max(1), vh.max(1))];
                    let cfg = PlotRenderConfig {
                        width: vw.max(1),
                        height: vh.max(1),
                        msaa_samples: requested_samples,
                        ..Default::default()
                    };
                    let cam = self.plot_renderer.camera().clone();
                    let _ = self
                        .plot_renderer
                        .render_camera_to_viewport(
                            &mut encoder,
                            &frame_view,
                            (vx, vy, vw, vh),
                            &cfg,
                            &cam,
                        )
                        .map_err(|err| WebRendererError::Render(err.to_string()))?;
                }

                // Render egui on top.
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("runmat-plot-web-egui"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &frame_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    overlay
                        .egui_renderer
                        .render(&mut render_pass, &paint_jobs, &screen_descriptor);
                }

                for id in &full_output.textures_delta.free {
                    overlay.egui_renderer.free_texture(id);
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn sync_canvas_extent(&mut self) -> Result<(), WebRendererError> {
        let (width, height) =
            desired_canvas_size(&self.canvas, self.options.width, self.options.height)?;
        if width == self.surface_config.width && height == self.surface_config.height {
            return Ok(());
        }

        self.surface_config.width = width;
        self.surface_config.height = height;
        self.render_config.width = width;
        self.render_config.height = height;
        self.msaa_texture = None;
        self.msaa_extent = (0, 0);
        self.reconfigure_surface()
    }

    fn reconfigure_surface(&mut self) -> Result<(), WebRendererError> {
        if self.surface_config.width == 0 || self.surface_config.height == 0 {
            return Err(WebRendererError::CanvasZeroArea);
        }
        self.surface.configure(&self.device, &self.surface_config);
        self.sync_renderer_config();
        Ok(())
    }

    fn sync_renderer_config(&mut self) {
        self.plot_renderer.wgpu_renderer.surface_config = self.surface_config.clone();
        self.plot_renderer.on_surface_config_updated();
        self.msaa_texture = None;
        let _ = self.ensure_msaa_texture();
    }

    fn ensure_msaa_texture(&mut self) -> Result<(), WebRendererError> {
        if self.render_config.msaa_samples <= 1 {
            self.msaa_texture = None;
            self.msaa_extent = (0, 0);
            return Ok(());
        }

        let width = self.surface_config.width.max(1);
        let height = self.surface_config.height.max(1);
        if self.msaa_texture.is_some() && self.msaa_extent == (width, height) {
            return Ok(());
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("runmat-plot-msaa-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.render_config.msaa_samples,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.msaa_texture = Some(texture);
        self.msaa_extent = (width, height);
        Ok(())
    }
}

fn map_mouse_button(button: crate::core::interaction::MouseButton) -> CameraMouseButton {
    match button {
        crate::core::interaction::MouseButton::Left => CameraMouseButton::Left,
        crate::core::interaction::MouseButton::Right => CameraMouseButton::Right,
        crate::core::interaction::MouseButton::Middle => CameraMouseButton::Middle,
    }
}

fn desired_canvas_size(
    canvas: &WebCanvas,
    override_width: Option<u32>,
    override_height: Option<u32>,
) -> Result<(u32, u32), WebRendererError> {
    let width = override_width.unwrap_or_else(|| canvas.width());
    let height = override_height.unwrap_or_else(|| canvas.height());
    if width > 0 && height > 0 {
        return Ok((width, height));
    }
    match canvas {
        WebCanvas::Html(element) => {
            let rect = element.get_bounding_client_rect();
            let w = rect.width().round() as u32;
            let h = rect.height().round() as u32;
            if w > 0 && h > 0 {
                Ok((w, h))
            } else {
                Err(WebRendererError::CanvasZeroArea)
            }
        }
        WebCanvas::Offscreen(_) => Err(WebRendererError::CanvasZeroArea),
    }
}

fn pick_surface_format(capabilities: &wgpu::SurfaceCapabilities) -> wgpu::TextureFormat {
    capabilities
        .formats
        .iter()
        .copied()
        .find(|format| format.is_srgb())
        .unwrap_or_else(|| capabilities.formats[0])
}

fn pick_present_mode(
    capabilities: &wgpu::SurfaceCapabilities,
    preferred: wgpu::PresentMode,
) -> wgpu::PresentMode {
    if capabilities.present_modes.contains(&preferred) {
        preferred
    } else if capabilities
        .present_modes
        .contains(&wgpu::PresentMode::AutoNoVsync)
    {
        wgpu::PresentMode::AutoNoVsync
    } else {
        wgpu::PresentMode::Fifo
    }
}

fn pick_alpha_mode(capabilities: &wgpu::SurfaceCapabilities) -> wgpu::CompositeAlphaMode {
    capabilities
        .alpha_modes
        .iter()
        .copied()
        .find(|mode| {
            matches!(
                mode,
                wgpu::CompositeAlphaMode::Opaque | wgpu::CompositeAlphaMode::Auto
            )
        })
        .unwrap_or(wgpu::CompositeAlphaMode::Auto)
}
