use std::sync::Arc;

use crate::core::{Camera, PlotRenderConfig, PlotRenderer, RenderResult};
use crate::gpu::util::map_read_async;
use crate::plots::Figure;
#[cfg(feature = "egui-overlay")]
use crate::styling::ModernDarkTheme;
use crate::styling::PlotThemeConfig;
#[cfg(feature = "egui-overlay")]
use runmat_time::Instant;

#[cfg(feature = "egui-overlay")]
use crate::overlay::plot_overlay::{OverlayConfig, OverlayMetrics, PlotOverlay};
#[cfg(feature = "egui-overlay")]
use egui_wgpu::ScreenDescriptor;

/// Renderer adapter for external/native surface targets owned by a host runtime.
pub struct NativeSurfaceRenderContext {
    renderer: PlotRenderer,
    config: PlotRenderConfig,
    pixels_per_point: f32,
    background_policy: BackgroundPolicy,
    #[cfg(feature = "egui-overlay")]
    overlay: Option<NativeOverlayState>,
}

#[derive(Debug, Clone, Copy)]
enum BackgroundPolicy {
    ThemeDriven,
    Explicit(glam::Vec4),
}

#[cfg(feature = "egui-overlay")]
struct NativeOverlayState {
    egui_ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    plot_overlay: PlotOverlay,
}

impl NativeSurfaceRenderContext {
    /// Create a context that can render into external texture views.
    pub async fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        let renderer = PlotRenderer::new(device, queue, surface_config)
            .await
            .map_err(|err| format!("native surface renderer init failed: {err}"))?;
        let config = PlotRenderConfig {
            width: width.max(1),
            height: height.max(1),
            ..PlotRenderConfig::default()
        };
        #[cfg(feature = "egui-overlay")]
        let overlay = {
            let egui_ctx = egui::Context::default();
            ModernDarkTheme::default().apply_to_egui(&egui_ctx);
            let egui_renderer =
                egui_wgpu::Renderer::new(&renderer.wgpu_renderer.device, format, None, 1);
            Some(NativeOverlayState {
                egui_ctx,
                egui_renderer,
                plot_overlay: PlotOverlay::new(),
            })
        };

        Ok(Self {
            renderer,
            config,
            pixels_per_point: 1.0,
            background_policy: BackgroundPolicy::ThemeDriven,
            #[cfg(feature = "egui-overlay")]
            overlay,
        })
    }

    /// Resize renderer viewport state.
    pub fn resize(&mut self, width: u32, height: u32) {
        let next_width = width.max(1);
        let next_height = height.max(1);
        self.config.width = next_width;
        self.config.height = next_height;
        self.renderer.wgpu_renderer.surface_config.width = next_width;
        self.renderer.wgpu_renderer.surface_config.height = next_height;
        self.renderer.on_surface_config_updated();
    }

    pub fn set_pixels_per_point(&mut self, pixels_per_point: f32) {
        self.pixels_per_point = pixels_per_point.clamp(0.5, 4.0);
    }

    pub fn set_theme_config(&mut self, theme: PlotThemeConfig) {
        self.renderer.theme = theme.clone();
        self.config.theme = theme;
        self.apply_background_policy();
    }

    /// Render a figure directly into an externally-owned texture view.
    pub fn render_to_view(
        &mut self,
        figure: &Figure,
        view: &wgpu::TextureView,
        camera: Option<&Camera>,
        axes_cameras: Option<&[Camera]>,
    ) -> Result<RenderResult, String> {
        self.prepare_scene(figure, camera, axes_cameras);

        let mut encoder = self.renderer.wgpu_renderer.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Native Surface Render Encoder"),
            },
        );

        let render_result = self.render_scene_with_overlay(&mut encoder, view)?;

        self.renderer
            .wgpu_renderer
            .queue
            .submit(std::iter::once(encoder.finish()));

        Ok(render_result)
    }

    /// Render a figure into an offscreen texture and read back RGBA8 bytes.
    pub async fn render_to_rgba(
        &mut self,
        figure: &Figure,
        camera: Option<&Camera>,
        axes_cameras: Option<&[Camera]>,
    ) -> Result<Vec<u8>, String> {
        self.prepare_scene(figure, camera, axes_cameras);

        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let format = self.renderer.wgpu_renderer.surface_config.format;
        let device = self.renderer.wgpu_renderer.device.clone();
        let queue = self.renderer.wgpu_renderer.queue.clone();

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("native_surface_offscreen_color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Native Surface RGBA Render Encoder"),
        });
        self.render_scene_with_overlay(&mut encoder, &color_view)?;
        queue.submit(std::iter::once(encoder.finish()));

        let bytes_per_pixel = 4u32;
        let padded_bytes_per_row = (width * bytes_per_pixel).div_ceil(256) * 256;
        let output_buffer_size = (padded_bytes_per_row * height) as wgpu::BufferAddress;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native_surface_offscreen_readback"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Native Surface RGBA Copy Encoder"),
        });
        copy_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(copy_encoder.finish()));

        let slice = output_buffer.slice(..);
        map_read_async(device.as_ref(), &slice).await?;
        let data = slice.get_mapped_range();
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src_start = row * padded_bytes_per_row as usize;
            let dst_start = row * width as usize * 4;
            pixels[dst_start..dst_start + width as usize * 4]
                .copy_from_slice(&data[src_start..src_start + width as usize * 4]);
        }
        drop(data);
        output_buffer.unmap();
        Ok(pixels)
    }

    fn prepare_scene(
        &mut self,
        figure: &Figure,
        camera: Option<&Camera>,
        axes_cameras: Option<&[Camera]>,
    ) {
        // Keep runtime config aligned with figure metadata, but treat the default figure
        // white background as "unspecified" and prefer active theme background for app parity.
        let bg = figure.background_color;
        self.background_policy = if is_default_figure_bg(bg) {
            BackgroundPolicy::ThemeDriven
        } else {
            BackgroundPolicy::Explicit(bg)
        };
        self.apply_background_policy();
        self.config.show_grid = figure.grid_enabled;
        self.config.show_title = figure.title.is_some();

        self.renderer.set_figure(figure.clone());
        if let Some(camera) = camera {
            *self.renderer.camera_mut() = camera.clone();
        }
        if let Some(overrides) = axes_cameras {
            for (index, override_camera) in overrides.iter().enumerate() {
                if let Some(target) = self.renderer.axes_camera_mut(index) {
                    *target = override_camera.clone();
                }
            }
        }
    }

    fn render_scene_with_overlay(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) -> Result<RenderResult, String> {
        #[cfg(feature = "egui-overlay")]
        {
            let Some(overlay) = self.overlay.as_mut() else {
                return self
                    .renderer
                    .render_scene_to_target(encoder, target_view, &self.config)
                    .map_err(|err| format!("native surface render failed: {err}"));
            };

            let start_time = Instant::now();
            let mut plot_area_points: Option<egui::Rect> = None;
            let scene_stats = self.renderer.scene.statistics();
            let _ = self.renderer.calculate_data_bounds();
            overlay
                .egui_ctx
                .set_pixels_per_point(self.pixels_per_point.max(0.5));
            let ppp = self.pixels_per_point.max(0.5);
            let full_output = overlay.egui_ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::new(0.0, 0.0),
                        egui::Vec2::new(
                            (self.config.width.max(1) as f32) / ppp,
                            (self.config.height.max(1) as f32) / ppp,
                        ),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    overlay
                        .plot_overlay
                        .set_theme_config(self.renderer.theme.clone());
                    overlay.plot_overlay.apply_theme(ctx);
                    let overlay_config = OverlayConfig {
                        // Let plot renderer own grid drawing semantics.
                        show_grid: false,
                        // Toolbar actions are surfaced by host UI, not native overlay.
                        show_toolbar: false,
                        font_scale: 1.25,
                        show_axes: true,
                        show_title: true,
                        title: self
                            .renderer
                            .overlay_title()
                            .cloned()
                            .or(Some("Plot".to_string())),
                        x_label: self
                            .renderer
                            .overlay_x_label()
                            .cloned()
                            .or(Some("X".to_string())),
                        y_label: self
                            .renderer
                            .overlay_y_label()
                            .cloned()
                            .or(Some("Y".to_string())),
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
                        &self.renderer,
                        &overlay_config,
                        overlay_metrics,
                    );
                    plot_area_points = frame_info.plot_area;
                },
            );

            let paint_jobs = overlay
                .egui_ctx
                .tessellate(full_output.shapes, full_output.pixels_per_point);
            for (id, image_delta) in &full_output.textures_delta.set {
                overlay.egui_renderer.update_texture(
                    &self.renderer.wgpu_renderer.device,
                    &self.renderer.wgpu_renderer.queue,
                    *id,
                    image_delta,
                );
            }

            let screen_descriptor = ScreenDescriptor {
                size_in_pixels: [self.config.width.max(1), self.config.height.max(1)],
                pixels_per_point: full_output.pixels_per_point,
            };
            overlay.egui_renderer.update_buffers(
                &self.renderer.wgpu_renderer.device,
                &self.renderer.wgpu_renderer.queue,
                encoder,
                &paint_jobs,
                &screen_descriptor,
            );

            let (vx, vy, vw, vh) = if let Some(rect) = plot_area_points {
                let vx = (rect.min.x * ppp).round().max(0.0) as u32;
                let vy = (rect.min.y * ppp).round().max(0.0) as u32;
                let vw = (rect.width() * ppp).round().max(1.0) as u32;
                let vh = (rect.height() * ppp).round().max(1.0) as u32;
                (vx, vy, vw, vh)
            } else {
                (0, 0, self.config.width.max(1), self.config.height.max(1))
            };
            let max_w = self.config.width.max(1);
            let max_h = self.config.height.max(1);
            let vx = vx.min(max_w.saturating_sub(1));
            let vy = vy.min(max_h.saturating_sub(1));
            let vw = vw.max(1).min(max_w.saturating_sub(vx).max(1));
            let vh = vh.max(1).min(max_h.saturating_sub(vy).max(1));

            if vw > 0 && vh > 0 {
                self.renderer
                    .camera_mut()
                    .update_aspect_ratio((vw as f32) / (vh as f32));
            }

            let (rows, cols) = self.renderer.figure_axes_grid();
            if rows * cols > 1 {
                let rect_points = plot_area_points.unwrap_or_else(|| {
                    egui::Rect::from_min_size(
                        egui::Pos2::new(0.0, 0.0),
                        egui::Vec2::new(
                            (self.config.width.max(1) as f32) / ppp,
                            (self.config.height.max(1) as f32) / ppp,
                        ),
                    )
                });
                let rects =
                    overlay
                        .plot_overlay
                        .compute_subplot_rects(rect_points, rows, cols, 8.0, 8.0);
                let sw = self.config.width as f32;
                let sh = self.config.height as f32;
                let mut viewports: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(rects.len());
                for r in rects {
                    let mut rx = (r.min.x * ppp).round().max(0.0);
                    let mut ry = (r.min.y * ppp).round().max(0.0);
                    let mut rw = (r.width() * ppp).round().max(1.0);
                    let mut rh = (r.height() * ppp).round().max(1.0);
                    if rx >= sw {
                        rx = (sw - 1.0).max(0.0);
                    }
                    if ry >= sh {
                        ry = (sh - 1.0).max(0.0);
                    }
                    if rx + rw > sw {
                        rw = (sw - rx).max(1.0);
                    }
                    if ry + rh > sh {
                        rh = (sh - ry).max(1.0);
                    }
                    viewports.push((rx as u32, ry as u32, rw as u32, rh as u32));
                }
                self.renderer
                    .render_axes_to_viewports(
                        encoder,
                        target_view,
                        &viewports,
                        self.config.msaa_samples.max(1),
                        &self.config,
                    )
                    .map_err(|err| format!("native surface subplot render failed: {err}"))?;
            } else {
                {
                    let clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("runmat-native-single-axes-clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: self.config.background_color.x as f64,
                                    g: self.config.background_color.y as f64,
                                    b: self.config.background_color.z as f64,
                                    a: self.config.background_color.w as f64,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    drop(clear_pass);
                }

                let mut cfg = self.config.clone();
                cfg.width = vw.max(1);
                cfg.height = vh.max(1);
                let cam = self.renderer.camera().clone();
                self.renderer
                    .render_camera_to_viewport(encoder, target_view, (vx, vy, vw, vh), &cfg, &cam)
                    .map_err(|err| format!("native surface viewport render failed: {err}"))?;
            }

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("runmat-native-egui-overlay"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
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

            Ok(RenderResult {
                success: true,
                data_bounds: self.renderer.data_bounds(),
                vertex_count: scene_stats.total_vertices,
                triangle_count: scene_stats.total_triangles,
                render_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            })
        }

        #[cfg(not(feature = "egui-overlay"))]
        {
            self.renderer
                .render_scene_to_target(encoder, target_view, &self.config)
                .map_err(|err| format!("native surface render failed: {err}"))
        }
    }

    fn apply_background_policy(&mut self) {
        self.config.background_color = match self.background_policy {
            BackgroundPolicy::ThemeDriven => {
                self.renderer.theme.build_theme().get_background_color()
            }
            BackgroundPolicy::Explicit(color) => color,
        };
    }
}

fn is_default_figure_bg(bg: glam::Vec4) -> bool {
    const EPS: f32 = 1e-3;
    (bg.x - 1.0).abs() <= EPS
        && (bg.y - 1.0).abs() <= EPS
        && (bg.z - 1.0).abs() <= EPS
        && (bg.w - 1.0).abs() <= EPS
}

async fn create_headless_context(
    width: u32,
    height: u32,
) -> Result<NativeSurfaceRenderContext, String> {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    if let Some(ctx) = crate::context::shared_wgpu_context() {
        return NativeSurfaceRenderContext::new(ctx.device, ctx.queue, width, height, format).await;
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("Failed to find suitable GPU adapter")?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(|err| format!("Failed to create device: {err}"))?;
    NativeSurfaceRenderContext::new(Arc::new(device), Arc::new(queue), width, height, format).await
}

pub async fn render_figure_rgba_bytes_interactive_with_camera(
    figure: Figure,
    width: u32,
    height: u32,
    camera: &Camera,
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context.render_to_rgba(&figure, Some(camera), None).await
}

pub async fn render_figure_rgba_bytes_interactive_with_camera_and_theme(
    figure: Figure,
    width: u32,
    height: u32,
    camera: &Camera,
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context.set_theme_config(theme);
    context.render_to_rgba(&figure, Some(camera), None).await
}

pub async fn render_figure_rgba_bytes_interactive_with_axes_cameras(
    figure: Figure,
    width: u32,
    height: u32,
    axes_cameras: &[Camera],
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context
        .render_to_rgba(&figure, None, Some(axes_cameras))
        .await
}

pub async fn render_figure_rgba_bytes_interactive_with_axes_cameras_and_theme(
    figure: Figure,
    width: u32,
    height: u32,
    axes_cameras: &[Camera],
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context.set_theme_config(theme);
    context
        .render_to_rgba(&figure, None, Some(axes_cameras))
        .await
}

pub async fn render_figure_rgba_bytes_interactive(
    figure: Figure,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context.render_to_rgba(&figure, None, None).await
}

pub async fn render_figure_rgba_bytes_interactive_and_theme(
    figure: Figure,
    width: u32,
    height: u32,
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context.set_theme_config(theme);
    context.render_to_rgba(&figure, None, None).await
}

fn encode_png_bytes(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    use image::{ImageBuffer, ImageFormat, Rgba};

    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| "Failed to create image buffer for PNG encoding".to_string())?;
    let mut out = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut out, ImageFormat::Png)
        .map_err(|err| format!("Failed to encode PNG bytes: {err}"))?;
    Ok(out.into_inner())
}

pub async fn render_figure_png_bytes_interactive(
    figure: Figure,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let rgba = render_figure_rgba_bytes_interactive(figure, width, height).await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}

pub async fn render_figure_png_bytes_interactive_and_theme(
    figure: Figure,
    width: u32,
    height: u32,
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let rgba = render_figure_rgba_bytes_interactive_and_theme(figure, width, height, theme).await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}

pub async fn render_figure_png_bytes_interactive_with_camera(
    figure: Figure,
    width: u32,
    height: u32,
    camera: &Camera,
) -> Result<Vec<u8>, String> {
    let rgba =
        render_figure_rgba_bytes_interactive_with_camera(figure, width, height, camera).await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}

pub async fn render_figure_png_bytes_interactive_with_camera_and_theme(
    figure: Figure,
    width: u32,
    height: u32,
    camera: &Camera,
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let rgba = render_figure_rgba_bytes_interactive_with_camera_and_theme(
        figure, width, height, camera, theme,
    )
    .await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}

pub async fn render_figure_png_bytes_interactive_with_axes_cameras(
    figure: Figure,
    width: u32,
    height: u32,
    axes_cameras: &[Camera],
) -> Result<Vec<u8>, String> {
    let rgba =
        render_figure_rgba_bytes_interactive_with_axes_cameras(figure, width, height, axes_cameras)
            .await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}

pub async fn render_figure_png_bytes_interactive_with_axes_cameras_and_theme(
    figure: Figure,
    width: u32,
    height: u32,
    axes_cameras: &[Camera],
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let rgba = render_figure_rgba_bytes_interactive_with_axes_cameras_and_theme(
        figure,
        width,
        height,
        axes_cameras,
        theme,
    )
    .await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}
