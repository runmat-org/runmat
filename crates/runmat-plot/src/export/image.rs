//! Image export (PNG, JPEG, etc.)
//!
//! Static image export functionality.

use std::path::Path;
use std::sync::Arc;
use wgpu::{Device, Queue, TextureFormat};
use crate::core::plot_renderer::{PlotRenderer, PlotRenderConfig};
use crate::plots::Figure;
#[cfg(feature = "gui")]
use egui::{Align2, Color32, FontId, Pos2};
#[cfg(feature = "gui")]
use egui_wgpu;

/// High-performance image exporter using GPU rendering
pub struct ImageExporter {
    /// GPU device for rendering
    device: Arc<Device>,
    /// Command queue
    queue: Arc<Queue>,
    /// Surface format
    #[allow(dead_code)]
    format: TextureFormat,
    /// Export settings
    settings: ImageExportSettings,
}

/// Image export configuration
#[derive(Debug, Clone)]
pub struct ImageExportSettings {
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels  
    pub height: u32,
    /// Samples for anti-aliasing (1, 4, 8, 16)
    pub samples: u32,
    /// Background color [R, G, B, A] (0.0-1.0)
    pub background_color: [f32; 4],
    /// Image quality (0.0-1.0) for lossy formats
    pub quality: f32,
    /// Include metadata in output
    pub include_metadata: bool,
}

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    WebP,
    Bmp,
}

impl Default for ImageExportSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            samples: 4,                             // 4x MSAA
            background_color: [1.0, 1.0, 1.0, 1.0], // White background
            quality: 0.95,
            include_metadata: true,
        }
    }
}

impl ImageExporter {
    /// Create a new image exporter with GPU acceleration
    pub async fn new() -> Result<Self, String> {
        Self::with_settings(ImageExportSettings::default()).await
    }

    /// Create exporter with custom settings
    pub async fn with_settings(settings: ImageExportSettings) -> Result<Self, String> {
        // Initialize GPU context for headless rendering
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
            .map_err(|e| format!("Failed to create device: {e}"))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            format: TextureFormat::Rgba8UnormSrgb,
            settings,
        })
    }

    /// Export figure to PNG file using the same PlotRenderer pipeline (headless)
    pub async fn export_png<P: AsRef<Path>>(
        &self,
        figure: &mut Figure,
        path: P,
    ) -> Result<(), String> {
        // Create an offscreen texture as color target
        let sc_desc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.format,
            width: self.settings.width,
            height: self.settings.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };

        // Build a WgpuRenderer and PlotRenderer without a surface
        // WgpuRenderer::new expects Arc<Device> and Arc<Queue> we already own Device/Queue;
        // create fresh Arcs by moving clones into Arcs (Device/Queue implement Clone in wgpu 0.19)
        let device: Arc<wgpu::Device> = self.device.clone();
        let queue: Arc<wgpu::Queue> = self.queue.clone();
        let mut plot_renderer = PlotRenderer::new(device.clone(), queue.clone(), sc_desc)
            .await
            .map_err(|e| format!("plot renderer init failed: {e}"))?;

        plot_renderer.set_figure(figure.clone());

        // Render into an offscreen texture
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("image_export_encoder") });
        // Use single-sample textures for readback compatibility
        let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("export_color"),
            size: wgpu::Extent3d { width: self.settings.width, height: self.settings.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth texture for 3D plots if needed
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("export_depth"),
            size: wgpu::Extent3d { width: self.settings.width, height: self.settings.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let _depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Clear background
        {
            let mut clear_pass = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("export_clear_encoder") });
            {
                let rp = clear_pass.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("export_clear_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.settings.background_color[0] as f64,
                            g: self.settings.background_color[1] as f64,
                            b: self.settings.background_color[2] as f64,
                            a: self.settings.background_color[3] as f64,
                        }), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                // nothing drawn; just clear
                drop(rp);
            }
            self.queue.submit(Some(clear_pass.finish()));
        }

        // Render via camera-viewport path. If subplot grid is present, compose per-axes into their viewports.
        let mut cfg = PlotRenderConfig { width: self.settings.width, height: self.settings.height, ..Default::default() };
        // Use 1x sample when writing directly to readback texture; MSAA would require resolve.
        cfg.msaa_samples = 1;

        let (rows, cols) = plot_renderer.figure_axes_grid();
        if rows * cols > 1 {
            // Compute simple grid with small gaps
            let hgap: u32 = 8;
            let vgap: u32 = 8;
            let total_hgap = hgap * (cols.saturating_sub(1) as u32);
            let total_vgap = vgap * (rows.saturating_sub(1) as u32);
            let cell_w = (self.settings.width.saturating_sub(total_hgap)) / (cols as u32);
            let cell_h = (self.settings.height.saturating_sub(total_vgap)) / (rows as u32);
            let mut viewports: Vec<(u32,u32,u32,u32)> = Vec::with_capacity(rows*cols);
            for r in 0..rows {
                for c in 0..cols {
                    let x = c as u32 * (cell_w + hgap);
                    let y = r as u32 * (cell_h + vgap);
                    viewports.push((x, y, cell_w.max(1), cell_h.max(1)));
                }
            }
            plot_renderer
                .render_axes_to_viewports(&mut encoder, &color_view, &viewports, 1)
                .map_err(|e| format!("render subplot failed: {e}"))?;
        } else {
            let viewport = (0u32, 0u32, self.settings.width, self.settings.height);
            plot_renderer
                .render_camera_to_viewport(&mut encoder, &color_view, viewport, &cfg)
                .map_err(|e| format!("render failed: {e}"))?;
        }

        // Draw overlay text (title/x/y labels) using a minimal egui pass onto the target
        #[cfg(feature = "gui")]
        {
            let egui_ctx = egui::Context::default();
            let mut raw_input = egui::RawInput::default();
            raw_input.viewports.insert(egui::viewport::ViewportId::ROOT, egui::ViewportInfo {
                native_pixels_per_point: Some(1.0),
                ..Default::default()
            });
            let full_output = egui_ctx.run(raw_input, |ctx| {
                egui::CentralPanel::default().frame(egui::Frame::none()).show(ctx, |ui| {
                    if let Some(title) = &figure.title {
                        ui.painter().text(
                            Pos2::new(self.settings.width as f32 * 0.5, 24.0),
                            Align2::CENTER_CENTER,
                            title,
                            FontId::proportional(18.0),
                            Color32::BLACK,
                        );
                    }
                    let (rows, cols) = figure.axes_grid();
                    let hgap: f32 = 8.0; let vgap: f32 = 8.0;
                    let total_hgap = hgap * (cols.saturating_sub(1) as f32);
                    let total_vgap = vgap * (rows.saturating_sub(1) as f32);
                    let cell_w = (self.settings.width as f32 - total_hgap).max(1.0) / (cols.max(1) as f32);
                    let cell_h = (self.settings.height as f32 - total_vgap).max(1.0) / (rows.max(1) as f32);
                    for r in 0..rows {
                        for c in 0..cols {
                            let vp_x = c as f32 * (cell_w + hgap);
                            let vp_y = r as f32 * (cell_h + vgap);
                            let vp_center_x = vp_x + cell_w * 0.5;
                            let vp_max_y = vp_y + cell_h;
                            let vp_center_y = vp_y + cell_h * 0.5;
                            let vp_min_x = vp_x;
                            if let Some(xl) = &figure.x_label {
                                ui.painter().text(
                                    Pos2::new(vp_center_x, vp_max_y + 20.0),
                                    Align2::CENTER_CENTER,
                                    xl,
                                    FontId::proportional(12.0),
                                    Color32::BLACK,
                                );
                            }
                            if let Some(yl) = &figure.y_label {
                                ui.painter().text(
                                    Pos2::new(vp_min_x - 24.0, vp_center_y),
                                    Align2::CENTER_CENTER,
                                    yl,
                                    FontId::proportional(12.0),
                                    Color32::BLACK,
                                );
                            }
                        }
                    }
                });
            });

            let mut egui_renderer = egui_wgpu::Renderer::new(
                &self.device,
                self.format,
                None,
                1,
            );
            for (id, image_delta) in &full_output.textures_delta.set {
                egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
            }
            let shapes = egui_ctx.tessellate(full_output.shapes, 1.0);
            let mut enc_overlay = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("image_overlay_encoder") });
            egui_renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut enc_overlay,
                &shapes,
                &egui_wgpu::ScreenDescriptor { size_in_pixels: [self.settings.width, self.settings.height], pixels_per_point: 1.0 },
            );
            {
                let mut rp = enc_overlay.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("image_overlay_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                egui_renderer.render(&mut rp, &shapes, &egui_wgpu::ScreenDescriptor { size_in_pixels: [self.settings.width, self.settings.height], pixels_per_point: 1.0 });
            }
            self.queue.submit(Some(enc_overlay.finish()));
            for id in &full_output.textures_delta.free { egui_renderer.free_texture(id); }
        }

        // Submit and copy to CPU buffer
        self.queue.submit(Some(encoder.finish()));

        // Read back the color texture to CPU via a staging buffer
        let bytes_per_pixel = 4u32;
        let padded_bytes_per_row = ((self.settings.width * bytes_per_pixel + 255) / 256) * 256;
        let output_buffer_size = (padded_bytes_per_row * self.settings.height) as wgpu::BufferAddress;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export_readback"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder2 = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("export_copy_encoder") });
        encoder2.copy_texture_to_buffer(
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
                    rows_per_image: Some(self.settings.height),
                },
            },
            wgpu::Extent3d { width: self.settings.width, height: self.settings.height, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder2.finish()));

        // Map and extract actual bytes (strip row padding)
        let buffer_slice = output_buffer.slice(..);
        // Map synchronously with poll
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|_| "map failed".to_string())?.map_err(|_| "map error".to_string())?;
        let data = buffer_slice.get_mapped_range();
        let mut pixels = vec![0u8; (self.settings.width * self.settings.height * 4) as usize];
        for y in 0..self.settings.height as usize {
            let src_start = y * padded_bytes_per_row as usize;
            let dst_start = y * (self.settings.width as usize) * 4;
            pixels[dst_start..dst_start + (self.settings.width as usize) * 4]
                .copy_from_slice(&data[src_start..src_start + (self.settings.width as usize) * 4]);
        }
        drop(data);
        output_buffer.unmap();

        self.save_png(&pixels, path).await
    }

    /// Save raw RGBA data as PNG
    async fn save_png<P: AsRef<Path>>(&self, data: &[u8], path: P) -> Result<(), String> {
        use image::{ImageBuffer, Rgba};

        let image =
            ImageBuffer::<Rgba<u8>, _>::from_raw(self.settings.width, self.settings.height, data)
                .ok_or("Failed to create image buffer")?;

        image
            .save(path)
            .map_err(|e| format!("Failed to save PNG: {e}"))?;

        log::debug!(target: "runmat_plot", "png export completed");
        Ok(())
    }

    /// Update export settings
    pub fn set_settings(&mut self, settings: ImageExportSettings) {
        self.settings = settings;
    }

    /// Get current export settings
    pub fn settings(&self) -> &ImageExportSettings {
        &self.settings
    }
}
