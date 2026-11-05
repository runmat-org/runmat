use runmat_accelerate_api::{WgpuDeviceAccess, WgpuEmission};
use std::sync::Arc;
use std::{any::Any, panic};

use crate::core::{Camera, PlotRenderConfig, PlotRenderer, RenderResult};
use crate::geometry_scene::{GeometryScene, GeometryScenePresentation};
use crate::gpu::util::readback_bytes;
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

pub const HEADLESS_GPU_ADAPTER_UNAVAILABLE: &str = "Failed to find suitable GPU adapter";
pub const HEADLESS_GPU_DEVICE_CREATION_FAILED_PREFIX: &str = "Failed to create device:";
pub const HEADLESS_GPU_CONTEXT_PANICKED_PREFIX: &str = "Headless GPU context creation panicked:";

/// Renderer adapter for external/native surface targets owned by a host runtime.
pub struct NativeSurfaceRenderContext {
    renderer: PlotRenderer,
    config: PlotRenderConfig,
    pixels_per_point: f32,
    presented_frame: Option<Arc<PresentedFrame>>,
    background_policy: BackgroundPolicy,
    textmark: Option<String>,
    #[cfg(feature = "egui-overlay")]
    host_actions: Vec<NativeSurfaceHostAction>,
    #[cfg(feature = "egui-overlay")]
    pending_overlay_events: Vec<crate::core::PlotEvent>,
    #[cfg(feature = "egui-overlay")]
    overlay: Option<NativeOverlayState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSurfaceHostAction {
    CreateFeaStudy,
}

#[derive(Debug, Clone, Copy)]
enum BackgroundPolicy {
    ThemeDriven,
    Explicit(glam::Vec4),
}

struct PreparedOverlayRender {
    result: RenderResult,
    preparation: Vec<wgpu::CommandBuffer>,
}

impl PreparedOverlayRender {
    fn submit(self, queue: &wgpu::Queue, encoder: wgpu::CommandEncoder) -> RenderResult {
        // egui's buffer preparation may encode work outside the main render encoder.
        // Keep that work ordered before the scene and overlay in one submission.
        queue.submit(
            self.preparation
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        self.result
    }
}

#[cfg(feature = "egui-overlay")]
struct NativeOverlayState {
    egui_ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    plot_overlay: PlotOverlay,
    wants_pointer_input: bool,
    capture_regions_px: Vec<[f32; 4]>,
}

#[derive(Debug, Clone)]
pub struct NativeSurfaceCameraState {
    pub active_camera: Camera,
    pub axes_cameras: Vec<Camera>,
    pub axes_camera_user_controlled: Vec<bool>,
}

/// An immutable GPU copy of a completed native frame. Readback does not rerender
/// or reconstruct layout, so text, antialiasing, and all overlays are preserved.
pub struct PresentedFrame {
    texture: wgpu::Texture,
    access: runmat_accelerate_api::WgpuDeviceAccess,
}

impl PresentedFrame {
    pub fn size(&self) -> (u32, u32) {
        (self.texture.width(), self.texture.height())
    }

    pub async fn rgba(&self) -> Result<Vec<u8>, String> {
        let (width, height) = self.size();
        let stride = (width * 4).div_ceil(256) * 256;
        let byte_len = stride as usize * height as usize;
        let buffer = self
            .access
            .with_shared(|emission| {
                let buffer = emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("presented frame readback"),
                    size: byte_len as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder =
                    emission
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("presented frame readback"),
                        });
                encoder.copy_texture_to_buffer(
                    crate::wgpu_compat::TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    crate::wgpu_compat::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: crate::wgpu_compat::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(stride),
                            rows_per_image: Some(height),
                        },
                    },
                    self.texture.size(),
                );
                emission.queue().submit(Some(encoder.finish()));
                buffer
            })
            .map_err(|error| error.to_string())?;
        let data = crate::gpu::util::readback_bytes(&self.access, &buffer, byte_len).await?;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for row in data.chunks_exact(stride as usize) {
            rgba.extend_from_slice(&row[..(width * 4) as usize]);
        }
        if matches!(
            self.texture.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for pixel in rgba.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        Ok(rgba)
    }
}

impl NativeSurfaceRenderContext {
    /// Create a context that can render into external texture views.
    pub async fn new(
        access: WgpuDeviceAccess,
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
        let renderer = PlotRenderer::new(access.clone(), surface_config)
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
            let egui_renderer = access
                .with_shared(|emission| {
                    crate::wgpu_compat::egui_renderer_new(emission.device(), format, None, 1)
                })
                .map_err(|error| error.to_string())?;
            Some(NativeOverlayState {
                egui_ctx,
                egui_renderer,
                plot_overlay: PlotOverlay::new(),
                wants_pointer_input: false,
                capture_regions_px: Vec::new(),
            })
        };

        Ok(Self {
            renderer,
            config,
            pixels_per_point: 1.0,
            presented_frame: None,
            background_policy: BackgroundPolicy::ThemeDriven,
            textmark: None,
            #[cfg(feature = "egui-overlay")]
            host_actions: Vec::new(),
            #[cfg(feature = "egui-overlay")]
            pending_overlay_events: Vec::new(),
            #[cfg(feature = "egui-overlay")]
            overlay,
        })
    }

    /// Keep the final native output on the GPU; CPU readback is deferred until
    /// an actual overlay handoff. Never overwrite a frame held by a reader.
    pub fn retain_presented_frame(&mut self, source: &wgpu::Texture) {
        let access = self.renderer.wgpu_renderer.access.clone();
        let reusable = self.presented_frame.take().filter(|frame| {
            Arc::strong_count(frame) == 1
                && frame.texture.size() == source.size()
                && frame.texture.format() == source.format()
        });
        let retained = access.with_shared(|emission| {
            let frame = reusable.unwrap_or_else(|| {
                let texture = emission.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("completed native frame"),
                    size: source.size(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: source.format(),
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                Arc::new(PresentedFrame {
                    texture,
                    access: access.clone(),
                })
            });
            let mut encoder =
                emission
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("retain native frame"),
                    });
            encoder.copy_texture_to_texture(
                crate::wgpu_compat::TexelCopyTextureInfo {
                    texture: source,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                crate::wgpu_compat::TexelCopyTextureInfo {
                    texture: &frame.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                source.size(),
            );
            emission.queue().submit(Some(encoder.finish()));
            frame
        });
        // A gate failure leaves no stale frame: readers must not observe it.
        self.presented_frame = retained.ok();
    }

    pub fn presented_frame(&self) -> Option<Arc<PresentedFrame>> {
        self.presented_frame.clone()
    }

    /// Resize renderer viewport state.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        let access = self.renderer.wgpu_renderer.access.clone();
        access
            .with_shared(|emission| {
                let next_width = width.max(1);
                let next_height = height.max(1);
                self.config.width = next_width;
                self.config.height = next_height;
                self.renderer.wgpu_renderer.surface_config.width = next_width;
                self.renderer.wgpu_renderer.surface_config.height = next_height;
                self.renderer.on_surface_config_updated(emission);
            })
            .map_err(|error| error.to_string())
    }

    pub fn set_pixels_per_point(&mut self, pixels_per_point: f32) {
        self.pixels_per_point = pixels_per_point.clamp(0.5, 4.0);
        self.renderer.wgpu_renderer.pixels_per_point = self.pixels_per_point;
    }

    /// Query only modes valid for both the actual output and depth attachment,
    /// including resolve support. Without an adapter, use WebGPU guarantees.
    /// A supplied adapter must be the one that created this context's device.
    pub fn supported_antialiasing(
        &self,
        adapter: Option<&wgpu::Adapter>,
    ) -> Result<Vec<crate::core::Antialiasing>, String> {
        let features = self
            .renderer
            .wgpu_renderer
            .access
            .with_shared(|emission| emission.device().features())
            .map_err(|error| error.to_string())?;
        let get = |format: wgpu::TextureFormat| {
            if features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
                if let Some(adapter) = adapter {
                    return adapter.get_texture_format_features(format);
                }
            }
            format.guaranteed_format_features(features)
        };
        let color = get(self.renderer.wgpu_renderer.surface_config.format);
        let depth = get(crate::core::WgpuRenderer::depth_format());
        Ok([
            crate::core::Antialiasing::Off,
            crate::core::Antialiasing::X2,
            crate::core::Antialiasing::X4,
            crate::core::Antialiasing::X8,
        ]
        .into_iter()
        .filter(|mode| mode.supported_by_formats(color, depth))
        .collect())
    }

    pub fn set_antialiasing(
        &mut self,
        mode: crate::core::Antialiasing,
        adapter: Option<&wgpu::Adapter>,
    ) -> Result<(), String> {
        if !self.supported_antialiasing(adapter)?.contains(&mode) {
            return Err(format!(
                "{}x anti-aliasing is unsupported by the active color/depth formats",
                mode.samples()
            ));
        }
        self.config.msaa_samples = mode.samples();
        // The ordinary render path rebuilds matching color/depth attachments
        // and pipelines through ensure_msaa. Do not reuse a retained old frame.
        self.presented_frame = None;
        Ok(())
    }

    pub fn set_theme_config(&mut self, theme: PlotThemeConfig) {
        self.renderer.set_theme(theme.clone());
        self.config.theme = theme;
        self.apply_background_policy();
    }

    pub fn set_textmark(&mut self, textmark: Option<&str>) {
        self.textmark = textmark
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);
    }

    pub fn camera_state(&self) -> NativeSurfaceCameraState {
        let mut axes_cameras = Vec::new();
        let mut index = 0;
        while let Some(camera) = self.renderer.axes_camera(index) {
            axes_cameras.push(camera.clone());
            index += 1;
        }
        if axes_cameras.is_empty() {
            axes_cameras.push(self.renderer.camera().clone());
        }
        let active_camera = axes_cameras
            .first()
            .cloned()
            .unwrap_or_else(|| self.renderer.camera().clone());
        let mut axes_camera_user_controlled =
            self.renderer.axes_camera_interaction_flags().to_vec();
        axes_camera_user_controlled.resize(axes_cameras.len().max(1), false);
        NativeSurfaceCameraState {
            active_camera,
            axes_cameras,
            axes_camera_user_controlled,
        }
    }

    pub fn overlay_wants_pointer_input(&self) -> bool {
        #[cfg(feature = "egui-overlay")]
        {
            self.overlay
                .as_ref()
                .map(|overlay| overlay.wants_pointer_input)
                .unwrap_or(false)
        }

        #[cfg(not(feature = "egui-overlay"))]
        {
            false
        }
    }

    pub fn overlay_capture_regions_px(&self) -> Vec<[f32; 4]> {
        #[cfg(feature = "egui-overlay")]
        {
            self.overlay
                .as_ref()
                .map(|overlay| overlay.capture_regions_px.clone())
                .unwrap_or_default()
        }

        #[cfg(not(feature = "egui-overlay"))]
        {
            Vec::new()
        }
    }

    pub fn take_host_actions(&mut self) -> Vec<NativeSurfaceHostAction> {
        #[cfg(feature = "egui-overlay")]
        {
            std::mem::take(&mut self.host_actions)
        }

        #[cfg(not(feature = "egui-overlay"))]
        {
            Vec::new()
        }
    }

    #[cfg(feature = "egui-overlay")]
    pub fn push_overlay_events(
        &mut self,
        events: impl IntoIterator<Item = crate::core::PlotEvent>,
    ) {
        self.pending_overlay_events.extend(events);
    }

    /// Render a figure directly into an externally-owned texture view.
    pub fn set_presence(
        &mut self,
        frame: crate::collaboration::PresenceFrame,
    ) -> Result<(), String> {
        self.renderer.set_presence(frame)
    }

    pub fn presence_projections(&self) -> Vec<crate::collaboration::PresenceProjection> {
        #[cfg(feature = "egui-overlay")]
        if let Some(overlay) = &self.overlay {
            return overlay.plot_overlay.presence_projections(&self.renderer);
        }
        Vec::new()
    }

    pub fn capture_presence_pointer(
        &self,
        logical_pixel: [f32; 2],
    ) -> Option<crate::collaboration::PointerPosition> {
        #[cfg(feature = "egui-overlay")]
        if let Some(overlay) = &self.overlay {
            return overlay
                .plot_overlay
                .presence_projections(&self.renderer)
                .iter()
                .rev()
                .find_map(|projection| projection.capture(logical_pixel));
        }
        let _ = logical_pixel;
        None
    }

    pub fn render_to_view(
        &mut self,
        figure: &Figure,
        view: &wgpu::TextureView,
        camera: Option<&Camera>,
        axes_cameras: Option<&[Camera]>,
    ) -> Result<RenderResult, String> {
        self.render_to_view_with_camera_state(figure, view, camera, axes_cameras, None)
    }

    pub fn render_to_view_with_camera_state(
        &mut self,
        figure: &Figure,
        view: &wgpu::TextureView,
        camera: Option<&Camera>,
        axes_cameras: Option<&[Camera]>,
        axes_camera_user_controlled: Option<&[bool]>,
    ) -> Result<RenderResult, String> {
        let access = self.renderer.wgpu_renderer.access.clone();
        access
            .with_shared(|emission| {
                self.prepare_scene(
                    emission,
                    figure,
                    camera,
                    axes_cameras,
                    axes_camera_user_controlled,
                );

                let mut encoder =
                    emission
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Native Surface Render Encoder"),
                        });

                let prepared = self.render_scene_with_overlay(emission, &mut encoder, view)?;
                Ok(prepared.submit(emission.queue(), encoder))
            })
            .map_err(|error| error.to_string())?
    }

    /// Render a chunked geometry scene directly into an externally-owned texture view.
    pub fn render_geometry_scene_to_view(
        &mut self,
        scene: &GeometryScene,
        view: &wgpu::TextureView,
        camera: Option<&Camera>,
    ) -> Result<RenderResult, String> {
        self.render_geometry_scene_to_view_with_presentation(scene, view, camera, None)
    }

    pub fn render_geometry_scene_to_view_with_presentation(
        &mut self,
        scene: &GeometryScene,
        view: &wgpu::TextureView,
        camera: Option<&Camera>,
        presentation: Option<&GeometryScenePresentation>,
    ) -> Result<RenderResult, String> {
        let access = self.renderer.wgpu_renderer.access.clone();
        access
            .with_shared(|emission| {
                self.prepare_geometry_scene(scene, camera, presentation);

                let mut encoder =
                    emission
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Native Geometry Scene Render Encoder"),
                        });

                let prepared = self.render_scene_with_overlay(emission, &mut encoder, view)?;
                Ok(prepared.submit(emission.queue(), encoder))
            })
            .map_err(|error| error.to_string())?
    }

    /// Render a figure into an offscreen texture and read back RGBA8 bytes.
    pub async fn render_to_rgba(
        &mut self,
        figure: &Figure,
        camera: Option<&Camera>,
        axes_cameras: Option<&[Camera]>,
    ) -> Result<Vec<u8>, String> {
        self.render_to_rgba_with_camera_state(figure, camera, axes_cameras, None)
            .await
    }

    pub async fn render_to_rgba_with_camera_state(
        &mut self,
        figure: &Figure,
        camera: Option<&Camera>,
        axes_cameras: Option<&[Camera]>,
        axes_camera_user_controlled: Option<&[bool]>,
    ) -> Result<Vec<u8>, String> {
        log::debug!(
            "runmat-plot: native_surface.render_to_rgba.start width={} height={} axes={} overrides={}",
            self.config.width.max(1),
            self.config.height.max(1),
            figure.axes_metadata.len(),
            axes_cameras.map(|items| items.len()).unwrap_or(0)
        );
        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let format = self.renderer.wgpu_renderer.surface_config.format;
        let access = self.renderer.wgpu_renderer.access.clone();
        let (output_buffer, padded_bytes_per_row) = access.with_shared(|emission| {
        let device = emission.device();
        let queue = emission.queue();
        self.prepare_scene(emission, figure, camera, axes_cameras, axes_camera_user_controlled);

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
        log::debug!(
            "runmat-plot: native_surface.render_to_rgba.render_scene width={} height={}",
            width,
            height
        );
        let prepared = self.render_scene_with_overlay(emission, &mut encoder, &color_view)?;
        log::debug!("runmat-plot: native_surface.render_to_rgba.render_scene_ok");
        prepared.submit(queue, encoder);

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
            crate::wgpu_compat::TexelCopyTextureInfo {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            crate::wgpu_compat::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: crate::wgpu_compat::TexelCopyBufferLayout {
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
        log::debug!(
            "runmat-plot: native_surface.render_to_rgba.copy_submitted padded_bytes_per_row={} height={}",
            padded_bytes_per_row,
            height
        );

        Ok::<_, String>((output_buffer, padded_bytes_per_row))
        }).map_err(|error| error.to_string())??;
        let data = readback_bytes(&access, &output_buffer, output_buffer.size() as usize).await?;
        log::debug!("runmat-plot: native_surface.render_to_rgba.readback_ready");
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src_start = row * padded_bytes_per_row as usize;
            let dst_start = row * width as usize * 4;
            pixels[dst_start..dst_start + width as usize * 4]
                .copy_from_slice(&data[src_start..src_start + width as usize * 4]);
        }
        drop(data);
        log::debug!(
            "runmat-plot: native_surface.render_to_rgba.ok bytes={}",
            pixels.len()
        );
        Ok(pixels)
    }

    pub async fn render_geometry_scene_to_rgba(
        &mut self,
        scene: &GeometryScene,
        camera: Option<&Camera>,
    ) -> Result<Vec<u8>, String> {
        self.render_geometry_scene_to_rgba_with_presentation(scene, camera, None)
            .await
    }

    /// Render the same presentation used by the native surface, including selections,
    /// annotations, visibility, sectioning, and display mode.
    pub async fn render_geometry_scene_to_rgba_with_presentation(
        &mut self,
        scene: &GeometryScene,
        camera: Option<&Camera>,
        presentation: Option<&GeometryScenePresentation>,
    ) -> Result<Vec<u8>, String> {
        self.prepare_geometry_scene(scene, camera, presentation);

        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let format = self.renderer.wgpu_renderer.surface_config.format;
        let access = self.renderer.wgpu_renderer.access.clone();
        let (output_buffer, padded_bytes_per_row) = access
            .with_shared(|emission| {
                let device = emission.device();
                let queue = emission.queue();

                let color_texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("native_surface_geometry_offscreen_color"),
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
                    label: Some("Native Geometry Surface RGBA Render Encoder"),
                });
                let prepared =
                    self.render_scene_with_overlay(emission, &mut encoder, &color_view)?;
                prepared.submit(queue, encoder);

                let bytes_per_pixel = 4u32;
                let padded_bytes_per_row = (width * bytes_per_pixel).div_ceil(256) * 256;
                let output_buffer_size = (padded_bytes_per_row * height) as wgpu::BufferAddress;
                let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("native_surface_geometry_offscreen_readback"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut copy_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Native Geometry Surface RGBA Copy Encoder"),
                    });
                copy_encoder.copy_texture_to_buffer(
                    crate::wgpu_compat::TexelCopyTextureInfo {
                        texture: &color_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    crate::wgpu_compat::TexelCopyBufferInfo {
                        buffer: &output_buffer,
                        layout: crate::wgpu_compat::TexelCopyBufferLayout {
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

                Ok::<_, String>((output_buffer, padded_bytes_per_row))
            })
            .map_err(|error| error.to_string())??;
        let data = readback_bytes(&access, &output_buffer, output_buffer.size() as usize).await?;
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src_start = row * padded_bytes_per_row as usize;
            let dst_start = row * width as usize * 4;
            pixels[dst_start..dst_start + width as usize * 4]
                .copy_from_slice(&data[src_start..src_start + width as usize * 4]);
        }
        drop(data);
        Ok(pixels)
    }

    fn prepare_scene(
        &mut self,
        emission: &WgpuEmission<'_>,
        figure: &Figure,
        camera: Option<&Camera>,
        axes_cameras: Option<&[Camera]>,
        axes_camera_user_controlled: Option<&[bool]>,
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
        self.config.show_grid = figure.grid_enabled
            || figure.minor_grid_enabled
            || figure
                .axes_metadata
                .iter()
                .any(|meta| meta.grid_enabled || meta.minor_grid_enabled);
        self.config.show_title = figure.has_any_titles();

        self.renderer.set_figure(emission, figure.clone());
        if let Some(camera) = camera {
            *self.renderer.camera_mut() = camera.clone();
            self.renderer.cancel_figure_viewport_fit(0);
        }
        if let Some(overrides) = axes_cameras {
            for (index, override_camera) in overrides.iter().enumerate() {
                if let Some(target) = self.renderer.axes_camera_mut(index) {
                    *target = override_camera.clone();
                    self.renderer.cancel_figure_viewport_fit(index);
                }
            }
        }
        if let Some(flags) = axes_camera_user_controlled {
            self.renderer.set_axes_camera_interaction_flags(flags);
        }
    }

    fn prepare_geometry_scene(
        &mut self,
        scene: &GeometryScene,
        camera: Option<&Camera>,
        presentation: Option<&GeometryScenePresentation>,
    ) {
        self.background_policy = BackgroundPolicy::ThemeDriven;
        self.apply_background_policy();
        self.config.show_grid = scene.show_grid;
        self.config.show_title = scene.title.is_some();

        if let Some(presentation) = presentation {
            self.renderer
                .set_geometry_scene_with_presentation(scene.clone(), presentation.clone());
        } else {
            self.renderer.set_geometry_scene(scene.clone());
        }
        if let Some(camera) = camera {
            *self.renderer.camera_mut() = camera.clone();
        }
    }

    fn render_scene_with_overlay(
        &mut self,
        emission: &WgpuEmission<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) -> Result<PreparedOverlayRender, String> {
        #[cfg(feature = "egui-overlay")]
        {
            let Some(overlay) = self.overlay.as_mut() else {
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.branch_no_overlay"
                );
                self.renderer.fit_pending_figure_cameras_for_surface(
                    self.config.width.max(1),
                    self.config.height.max(1),
                );
                let result = self
                    .renderer
                    .render_scene_to_target(emission, encoder, target_view, &self.config)
                    .map_err(|err| format!("native surface render failed: {err}"))?;
                return Ok(PreparedOverlayRender {
                    result,
                    preparation: Vec::new(),
                });
            };

            let start_time = Instant::now();
            log::debug!(
                "runmat-plot: native_surface.render_scene_with_overlay.start width={} height={} ppp={}",
                self.config.width,
                self.config.height,
                self.pixels_per_point
            );
            let mut plot_area_points: Option<egui::Rect> = None;
            let scene_stats = self.renderer.scene.statistics();
            let _ = self.renderer.calculate_data_bounds();
            let ppp = self.pixels_per_point.max(0.5);
            let screen_rect = egui::Rect::from_min_size(
                egui::Pos2::new(0.0, 0.0),
                egui::Vec2::new(
                    (self.config.width.max(1) as f32) / ppp,
                    (self.config.height.max(1) as f32) / ppp,
                ),
            );
            let raw_input = crate::core::interaction::egui_raw_input_from_plot_events(
                screen_rect,
                ppp,
                std::mem::take(&mut self.pending_overlay_events),
            );
            let full_output = overlay.egui_ctx.run(raw_input, |ctx| {
                overlay
                    .plot_overlay
                    .set_theme_config(self.renderer.theme.clone());
                overlay.plot_overlay.apply_theme(ctx);
                let overlay_config = OverlayConfig {
                    // Let plot renderer own grid drawing semantics.
                    show_grid: false,
                    // Toolbar actions are surfaced by host UI, not native overlay.
                    show_toolbar: false,
                    // CAD controls are surfaced by host UI, not native overlay.
                    show_cad_overlay: false,
                    font_scale: 1.25,
                    show_axes: true,
                    show_title: self.renderer.geometry_overlay().is_none(),
                    title: self.renderer.overlay_title().cloned(),
                    x_label: self.renderer.overlay_x_label().cloned(),
                    y_label: self.renderer.overlay_y_label().cloned(),
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
                    &mut self.renderer,
                    &overlay_config,
                    overlay_metrics,
                );
                if let Some(textmark) = self.textmark.as_deref() {
                    let screen = ctx.screen_rect();
                    let anchor = egui::pos2(screen.max.x - 8.0, screen.max.y - 6.0);
                    let font =
                        egui::FontId::proportional(11.0 * overlay_config.font_scale.max(0.8));
                    let layer = egui::LayerId::new(
                        egui::Order::Foreground,
                        egui::Id::new("runmat_export_textmark"),
                    );
                    let painter = ctx.layer_painter(layer);
                    painter.text(
                        anchor + egui::vec2(1.0, 1.0),
                        egui::Align2::RIGHT_BOTTOM,
                        textmark,
                        font.clone(),
                        egui::Color32::from_rgba_premultiplied(0, 0, 0, 72),
                    );
                    painter.text(
                        anchor,
                        egui::Align2::RIGHT_BOTTOM,
                        textmark,
                        font,
                        egui::Color32::from_rgba_premultiplied(226, 234, 245, 96),
                    );
                }
                plot_area_points = frame_info.plot_area;
            });

            let cad_actions = overlay.plot_overlay.take_cad_actions();
            self.host_actions
                .extend(apply_cad_overlay_actions(&mut self.renderer, cad_actions));
            overlay.wants_pointer_input = overlay.plot_overlay.overlay_pointer_captured();
            overlay.capture_regions_px = overlay
                .plot_overlay
                .overlay_capture_regions()
                .into_iter()
                .map(|[x0, y0, x1, y1]| [x0 * ppp, y0 * ppp, x1 * ppp, y1 * ppp])
                .collect();

            let paint_jobs = overlay
                .egui_ctx
                .tessellate(full_output.shapes, full_output.pixels_per_point);
            for (id, image_delta) in &full_output.textures_delta.set {
                overlay.egui_renderer.update_texture(
                    emission.device(),
                    emission.queue(),
                    *id,
                    image_delta,
                );
            }

            let screen_descriptor = ScreenDescriptor {
                size_in_pixels: [self.config.width.max(1), self.config.height.max(1)],
                pixels_per_point: full_output.pixels_per_point,
            };
            let preparation = overlay.egui_renderer.update_buffers(
                emission.device(),
                emission.queue(),
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
            let mut axes_viewports_px = vec![(vx, vy, vw, vh)];

            if vw > 0 && vh > 0 {
                self.renderer
                    .camera_mut()
                    .update_aspect_ratio((vw as f32) / (vh as f32));
            }

            let (rows, cols) = self.renderer.figure_axes_grid();
            log::debug!(
                "runmat-plot: native_surface.render_scene_with_overlay.axes_grid rows={} cols={} plot_area_present={}",
                rows,
                cols,
                plot_area_points.is_some()
            );
            if rows * cols > 1 {
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.branch_subplot_axes"
                );
                let rect_points = plot_area_points.unwrap_or_else(|| {
                    egui::Rect::from_min_size(
                        egui::Pos2::new(0.0, 0.0),
                        egui::Vec2::new(
                            (self.config.width.max(1) as f32) / ppp,
                            (self.config.height.max(1) as f32) / ppp,
                        ),
                    )
                });
                let existing_rect_count = overlay.plot_overlay.axes_plot_rects().len();
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.subplot_rect_source rows={} cols={} existing_rects={} expected_rects={}",
                    rows,
                    cols,
                    existing_rect_count,
                    rows * cols
                );
                let rects = if overlay.plot_overlay.axes_plot_rects().len() == rows * cols {
                    log::debug!(
                        "runmat-plot: native_surface.render_scene_with_overlay.subplot_rect_source_existing rects={}",
                        existing_rect_count
                    );
                    overlay.plot_overlay.axes_plot_rects().to_vec()
                } else {
                    log::debug!(
                        "runmat-plot: native_surface.render_scene_with_overlay.subplot_rect_source_compute rect_points=({}, {})..({}, {})",
                        rect_points.min.x,
                        rect_points.min.y,
                        rect_points.max.x,
                        rect_points.max.y
                    );
                    overlay.plot_overlay.compute_subplot_plot_rects_snapped(
                        rect_points,
                        &self.renderer,
                        1.0,
                        ppp,
                    )
                };
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.subplot_rects_ready rects={}",
                    rects.len()
                );
                let sw = self.config.width as f32;
                let sh = self.config.height as f32;
                let mut viewports: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(rects.len());
                for (rect_index, r) in rects.into_iter().enumerate() {
                    log::debug!(
                        "runmat-plot: native_surface.render_scene_with_overlay.subplot_rect viewport_index={} rect=({}, {})..({}, {})",
                        rect_index,
                        r.min.x,
                        r.min.y,
                        r.max.x,
                        r.max.y
                    );
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
                    log::debug!(
                        "runmat-plot: native_surface.render_scene_with_overlay.subplot_viewport viewport_index={} viewport=({}, {}, {}, {})",
                        rect_index,
                        rx as u32,
                        ry as u32,
                        rw as u32,
                        rh as u32
                    );
                }
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.subplot_viewports_ready count={}",
                    viewports.len()
                );
                axes_viewports_px = viewports.clone();
                let axes_plot_sizes_px: Vec<(u32, u32)> = viewports
                    .iter()
                    .map(|&(_, _, w, h)| (w.max(1), h.max(1)))
                    .collect();
                self.renderer
                    .ensure_scene_viewport_dependent_geometry_for_axes(
                        emission,
                        &axes_plot_sizes_px,
                    );
                self.renderer
                    .render_axes_to_viewports(
                        emission,
                        encoder,
                        target_view,
                        &viewports,
                        self.config.msaa_samples.max(1),
                        &self.config,
                    )
                    .map_err(|err| format!("native surface subplot render failed: {err}"))?;
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.subplot_render_ok viewports={}",
                    viewports.len()
                );
            } else {
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.branch_single_axes"
                );
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
                let cam = self
                    .renderer
                    .axes_camera(0)
                    .cloned()
                    .unwrap_or_else(|| self.renderer.camera().clone());
                let axes_plot_sizes_px = vec![(vw.max(1), vh.max(1))];
                self.renderer
                    .ensure_scene_viewport_dependent_geometry_for_axes(
                        emission,
                        &axes_plot_sizes_px,
                    );
                self.renderer
                    .render_camera_to_viewport(
                        emission,
                        encoder,
                        target_view,
                        (vx, vy, vw, vh),
                        &cfg,
                        &cam,
                        0,
                        true,
                    )
                    .map_err(|err| format!("native surface viewport render failed: {err}"))?;
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.single_axes_render_ok"
                );
            }

            {
                let mut render_pass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
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
                    })
                    .forget_lifetime();
                overlay
                    .egui_renderer
                    .render(&mut render_pass, &paint_jobs, &screen_descriptor);
                log::debug!(
                    "runmat-plot: native_surface.render_scene_with_overlay.overlay_ok paint_jobs={} textures_set={} textures_free={}",
                    paint_jobs.len(),
                    full_output.textures_delta.set.len(),
                    full_output.textures_delta.free.len()
                );
            }

            for id in &full_output.textures_delta.free {
                overlay.egui_renderer.free_texture(id);
            }

            Ok(PreparedOverlayRender {
                result: RenderResult {
                    success: true,
                    data_bounds: self.renderer.data_bounds(),
                    axes_viewports_px,
                    vertex_count: scene_stats.total_vertices,
                    triangle_count: scene_stats.total_triangles,
                    render_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                },
                preparation,
            })
        }

        #[cfg(not(feature = "egui-overlay"))]
        {
            self.renderer.fit_pending_figure_cameras_for_surface(
                self.config.width.max(1),
                self.config.height.max(1),
            );
            let result = self
                .renderer
                .render_scene_to_target(emission, encoder, target_view, &self.config)
                .map_err(|err| format!("native surface render failed: {err}"))?;
            Ok(PreparedOverlayRender {
                result,
                preparation: Vec::new(),
            })
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

#[cfg(all(test, feature = "egui-overlay", not(target_arch = "wasm32")))]
#[path = "native_surface/preparation_tests.rs"]
mod preparation_tests;

pub async fn create_headless_context(
    width: u32,
    height: u32,
) -> Result<NativeSurfaceRenderContext, String> {
    // Export paths provide theme/plot colors in display (sRGB) space, just like interactive
    // surface rendering on most backends. Using an sRGB attachment here applies an extra
    // linear->sRGB conversion and visibly washes out captures.
    let format = wgpu::TextureFormat::Rgba8Unorm;
    if let Some(ctx) = crate::context::shared_wgpu_context() {
        log::debug!(
            "runmat-plot: native_surface.headless_context.branch_shared_context width={} height={}",
            width,
            height
        );
        return NativeSurfaceRenderContext::new(ctx.access, width, height, format).await;
    }

    log::debug!(
        "runmat-plot: native_surface.headless_context.branch_dedicated_context width={} height={}",
        width,
        height
    );

    let instance = panic::catch_unwind(|| {
        crate::wgpu_compat::instance_new(wgpu::InstanceDescriptor::default())
    })
    .map_err(|payload| {
        format!(
            "{HEADLESS_GPU_CONTEXT_PANICKED_PREFIX} {}",
            panic_payload_to_string(payload)
        )
    })?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or(HEADLESS_GPU_ADAPTER_UNAVAILABLE)?;
    let (device, queue) = adapter
        .request_device(&crate::wgpu_compat::default_device_descriptor(), None)
        .await
        .map_err(|err| format!("{HEADLESS_GPU_DEVICE_CREATION_FAILED_PREFIX} {err}"))?;
    let context = NativeSurfaceRenderContext::new(
        WgpuDeviceAccess::new(device, queue),
        width,
        height,
        format,
    )
    .await?;
    log::debug!(
        "runmat-plot: native_surface.headless_context.ready width={} height={}",
        width,
        height
    );
    Ok(context)
}

pub fn is_headless_gpu_unavailable_error(err: &str) -> bool {
    err.contains(HEADLESS_GPU_ADAPTER_UNAVAILABLE)
        || err.contains(HEADLESS_GPU_DEVICE_CREATION_FAILED_PREFIX)
        || err.contains(HEADLESS_GPU_CONTEXT_PANICKED_PREFIX)
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
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

pub async fn render_geometry_scene_rgba_bytes_interactive_with_camera_and_theme(
    scene: GeometryScene,
    width: u32,
    height: u32,
    camera: &Camera,
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context.set_theme_config(theme);
    context
        .render_geometry_scene_to_rgba(&scene, Some(camera))
        .await
}

pub async fn render_geometry_scene_rgba_bytes_interactive_and_theme(
    scene: GeometryScene,
    width: u32,
    height: u32,
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context.set_theme_config(theme);
    context.render_geometry_scene_to_rgba(&scene, None).await
}

pub async fn render_geometry_scene_png_bytes_interactive_with_camera_and_theme(
    scene: GeometryScene,
    width: u32,
    height: u32,
    camera: &Camera,
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let rgba = render_geometry_scene_rgba_bytes_interactive_with_camera_and_theme(
        scene, width, height, camera, theme,
    )
    .await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}

pub async fn render_geometry_scene_png_bytes_interactive_and_theme(
    scene: GeometryScene,
    width: u32,
    height: u32,
    theme: PlotThemeConfig,
) -> Result<Vec<u8>, String> {
    let rgba =
        render_geometry_scene_rgba_bytes_interactive_and_theme(scene, width, height, theme).await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}

pub fn encode_png_bytes(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    use image::{ImageBuffer, ImageFormat, Rgba};

    let mut opaque = rgba.to_vec();
    for pixel in opaque.chunks_exact_mut(4) {
        pixel[3] = 255;
    }

    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, opaque)
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

pub async fn render_figure_png_bytes_interactive_and_theme_and_textmark(
    figure: Figure,
    width: u32,
    height: u32,
    theme: PlotThemeConfig,
    textmark: Option<&str>,
) -> Result<Vec<u8>, String> {
    log::debug!(
        "runmat-plot: render_figure_png_bytes_interactive_and_theme_and_textmark.start width={} height={} axes={} textmark={}",
        width,
        height,
        figure.axes_metadata.len(),
        textmark.unwrap_or("")
    );
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    log::debug!(
        "runmat-plot: render_figure_png_bytes_interactive_and_theme_and_textmark.context_ready width={} height={}",
        width.max(1),
        height.max(1)
    );
    context.set_theme_config(theme);
    context.set_textmark(textmark);
    let rgba = context.render_to_rgba(&figure, None, None).await?;
    log::debug!(
        "runmat-plot: render_figure_png_bytes_interactive_and_theme_and_textmark.rgba_ready bytes={}",
        rgba.len()
    );
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

pub async fn render_figure_png_bytes_interactive_with_camera_and_theme_and_textmark(
    figure: Figure,
    width: u32,
    height: u32,
    camera: &Camera,
    theme: PlotThemeConfig,
    textmark: Option<&str>,
) -> Result<Vec<u8>, String> {
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    context.set_theme_config(theme);
    context.set_textmark(textmark);
    let rgba = context.render_to_rgba(&figure, Some(camera), None).await?;
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

pub async fn render_figure_png_bytes_interactive_with_axes_cameras_and_theme_and_textmark(
    figure: Figure,
    width: u32,
    height: u32,
    axes_cameras: &[Camera],
    theme: PlotThemeConfig,
    textmark: Option<&str>,
) -> Result<Vec<u8>, String> {
    log::debug!(
        "runmat-plot: render_figure_png_bytes_interactive_with_axes_cameras_and_theme_and_textmark.start width={} height={} axes={} camera_overrides={} textmark={}",
        width,
        height,
        figure.axes_metadata.len(),
        axes_cameras.len(),
        textmark.unwrap_or("")
    );
    let mut context = create_headless_context(width.max(1), height.max(1)).await?;
    log::debug!(
        "runmat-plot: render_figure_png_bytes_interactive_with_axes_cameras_and_theme_and_textmark.context_ready width={} height={}",
        width.max(1),
        height.max(1)
    );
    context.set_theme_config(theme);
    context.set_textmark(textmark);
    let rgba = context
        .render_to_rgba(&figure, None, Some(axes_cameras))
        .await?;
    log::debug!(
        "runmat-plot: render_figure_png_bytes_interactive_with_axes_cameras_and_theme_and_textmark.rgba_ready bytes={}",
        rgba.len()
    );
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}

#[cfg(feature = "egui-overlay")]
fn apply_cad_overlay_actions(
    renderer: &mut PlotRenderer,
    actions: crate::overlay::cad_overlay::CadOverlayActions,
) -> Vec<NativeSurfaceHostAction> {
    let mut host_actions = Vec::new();
    if actions.reset_view {
        renderer.reset_geometry_view();
    }
    if actions.create_fea_study {
        host_actions.push(NativeSurfaceHostAction::CreateFeaStudy);
    }
    if let Some(preset) = actions.view_preset {
        renderer.set_camera_view_preset(preset);
    }
    if let Some(enabled) = actions.grid_enabled {
        renderer.set_overlay_grid_enabled(enabled);
    }
    if let Some(enabled) = actions.xray_enabled {
        renderer.set_geometry_xray_enabled(enabled);
    }
    for (owner_id, visible) in actions.owner_visibility {
        renderer.set_geometry_owner_visible(owner_id, visible);
    }
    host_actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headless_gpu_panic_errors_are_fallback_eligible() {
        assert!(is_headless_gpu_unavailable_error(
            "Headless GPU context creation panicked: called `Option::unwrap()` on a `None` value"
        ));
    }
}

#[cfg(all(test, feature = "egui-overlay"))]
mod parity_tests {
    use super::*;
    use crate::core::{Material, Vertex};
    use crate::geometry_scene::*;
    use crate::plots::LinePlot;
    use crate::styling::ThemeVariant;
    use glam::{Vec3, Vec4};

    fn target(context: &NativeSurfaceRenderContext) -> wgpu::Texture {
        context
            .renderer
            .wgpu_renderer
            .access
            .with_shared(|emission| {
                emission.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("parity native target"),
                    size: wgpu::Extent3d {
                        width: context.config.width,
                        height: context.config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: context.renderer.wgpu_renderer.surface_config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            })
            .unwrap()
    }

    async fn peer(
        context: &NativeSurfaceRenderContext,
        dpr: f32,
        theme: &PlotThemeConfig,
    ) -> NativeSurfaceRenderContext {
        let mut peer = NativeSurfaceRenderContext::new(
            context.renderer.wgpu_renderer.access.clone(),
            context.config.width,
            context.config.height,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .await
        .unwrap();
        peer.set_pixels_per_point(dpr);
        peer.set_theme_config(theme.clone());
        peer
    }

    fn assert_pixels(label: &str, actual: &[u8], expected: &[u8], width: u32, height: u32) {
        let differences = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
        if differences > 0 {
            for (suffix, data) in [("native", expected), ("snapshot", actual)] {
                let path = std::env::temp_dir().join(format!("runmat-parity-{label}-{suffix}.png"));
                image::save_buffer(path, data, width, height, image::ColorType::Rgba8).unwrap();
            }
        }
        assert_eq!(
            differences, 0,
            "{label}: native/snapshot channel differences"
        );
    }

    #[tokio::test]
    #[ignore = "requires a GPU; run explicitly with --ignored"]
    async fn native_snapshot_pixel_parity() {
        // Intentionally fail if no GPU is available: this suite must exercise real rendering.
        let seed = create_headless_context(640, 480).await.unwrap();
        // Native Metal/DirectX targets generally use BGRA; the canvas uses RGBA.
        let mut live = NativeSurfaceRenderContext::new(
            seed.renderer.wgpu_renderer.access.clone(),
            640,
            480,
            wgpu::TextureFormat::Bgra8Unorm,
        )
        .await
        .unwrap();
        let mut figure = Figure::new();
        figure.set_subplot_grid(1, 2);
        for axis in 0..2 {
            let line = figure
                .add_line_plot(LinePlot::new(vec![0., 1., 2., 3.], vec![0., 2., 1., 3.]).unwrap());
            figure.assign_plot_to_axes(line, axis).unwrap();
            figure.set_axes_title(axis, format!("Panel {axis}"));
            figure.set_axes_labels(axis, "Time (s)", "Amplitude");
            figure.set_axes_legend_enabled(axis, true);
            figure.set_axes_tick_label_rotations(axis, Some(30.0), None);
        }
        let cameras = vec![
            Camera::new_2d((0.5, 2.5, 0.5, 2.5)),
            Camera::new_2d((0., 3., 0., 3.)),
        ];
        let flags = [true, false];
        let chunk = GeometrySceneChunk::indexed_triangles(
            "face",
            vec![
                Vertex::new(Vec3::new(-1., -1., 0.), Vec4::new(0.6, 0.7, 0.8, 1.)),
                Vertex::new(Vec3::new(1., -1., 0.), Vec4::new(0.6, 0.7, 0.8, 1.)),
                Vertex::new(Vec3::new(0., 1., 0.), Vec4::new(0.6, 0.7, 0.8, 1.)),
            ],
            vec![0, 1, 2],
            Material::default(),
        )
        .with_regions(vec![GeometrySceneRegion::new(
            "support",
            None,
            None,
            vec![GeometrySceneTriangleRange::new(0, 1)],
        )]);
        let scene =
            GeometryScene::new("parity", 1, vec![chunk]).with_overlay(GeometrySceneOverlay {
                source_name: None,
                status: GeometrySceneCompleteness::Complete,
                quality_label: None,
                format: None,
                source_label: None,
                allow_create_fea_study: false,
                byte_count: None,
                mesh_count: 1,
                vertex_count: 3,
                triangle_count: 1,
                progress_percent: None,
                scalar_legend: Some(GeometrySceneScalarLegend {
                    label: "structural.displacement magnitude".into(),
                    unit: "m".into(),
                    minimum: 0.0,
                    maximum: 9.65e-10,
                }),
                region_count: 1,
                mapped_region_count: 1,
                assembly_nodes: vec![],
                regions: vec![],
                warnings: vec![],
            });
        for dpr in [1.0, 1.25, 2.0] {
            let width = (640.0 * dpr) as u32;
            let height = (480.0 * dpr) as u32;
            live.resize(width, height)
                .expect("resize native parity surface");
            for variant in [ThemeVariant::ModernDark, ThemeVariant::ClassicLight] {
                let theme = PlotThemeConfig {
                    variant,
                    ..Default::default()
                };
                live.set_pixels_per_point(dpr);
                live.set_theme_config(theme.clone());
                let texture = target(&live);
                let view = texture.create_view(&Default::default());
                live.render_to_view_with_camera_state(
                    &figure,
                    &view,
                    Some(&cameras[0]),
                    Some(&cameras),
                    Some(&flags),
                )
                .unwrap();
                let projection = live
                    .presence_projections()
                    .into_iter()
                    .next()
                    .expect("2D projection");
                assert!(!projection.is_3d);
                let position = projection
                    .capture([
                        projection.rect[0] + projection.rect[2] * 0.55,
                        projection.rect[1] + projection.rect[3] * 0.45,
                    ])
                    .expect("2D pointer");
                let frame = crate::collaboration::PresenceFrame {
                    revision: "accepted-run".into(),
                    target: None,
                    pointers: vec![crate::collaboration::PresencePointer {
                        position,
                        color: [240, 110, 30],
                        opacity: 1.0,
                        label: Some("Jules".into()),
                        click_progress: Some(0.35),
                        click_position: None,
                    }],
                };
                live.set_presence(frame.clone()).unwrap();
                for _ in 0..2 {
                    live.render_to_view_with_camera_state(
                        &figure,
                        &view,
                        Some(&cameras[0]),
                        Some(&cameras),
                        Some(&flags),
                    )
                    .unwrap();
                }
                live.retain_presented_frame(&texture);
                let retained = live.presented_frame().unwrap();
                let expected = retained.rgba().await.unwrap();
                let mut snapshot = peer(&live, dpr, &theme).await;
                snapshot.set_presence(frame).unwrap();
                let actual = snapshot
                    .render_to_rgba_with_camera_state(
                        &figure,
                        Some(&cameras[0]),
                        Some(&cameras),
                        Some(&flags),
                    )
                    .await
                    .unwrap();
                assert_pixels(
                    &format!("2d-{dpr}-{variant:?}"),
                    &actual,
                    &expected,
                    width,
                    height,
                );
                assert!(
                    snapshot.presence_projections().iter().any(|p| p
                        .project(&snapshot.renderer.presence().pointers[0].position)
                        .is_some()),
                    "cursor must project: {:?}",
                    snapshot.presence_projections()
                );
                snapshot.set_presence(Default::default()).unwrap();
                let without_presence = snapshot
                    .render_to_rgba_with_camera_state(
                        &figure,
                        Some(&cameras[0]),
                        Some(&cameras),
                        Some(&flags),
                    )
                    .await
                    .unwrap();
                assert!(
                    actual != without_presence,
                    "2D collaborator must actually be drawn"
                );
                for mode in [
                    GeometrySceneDisplayMode::Shaded,
                    GeometrySceneDisplayMode::Edges,
                    GeometrySceneDisplayMode::Wireframe,
                ] {
                    let presentation = GeometryScenePresentation {
                        display_mode: mode,
                        selected_region_ids: vec!["support".into()],
                        selected_topology: vec![
                            GeometrySceneTopologySelection::Edge {
                                chunk_id: "face".into(),
                                vertex_indices: [0, 1],
                                segments: vec![],
                            },
                            GeometrySceneTopologySelection::Point {
                                chunk_id: "face".into(),
                                vertex_index: 2,
                            },
                        ],
                        region_annotations: vec![GeometrySceneRegionAnnotation {
                            region_id: "support".into(),
                            kind: GeometrySceneAnnotationKind::FixedSupport,
                            color: None,
                            label: None,
                            direction: None,
                            size: None,
                        }],
                        show_grid: Some(false),
                        ..Default::default()
                    };
                    live.render_geometry_scene_to_view_with_presentation(
                        &scene,
                        &view,
                        None,
                        Some(&presentation),
                    )
                    .unwrap();
                    let camera = live.camera_state().active_camera;
                    let projection = live
                        .presence_projections()
                        .into_iter()
                        .next()
                        .expect("geometry projection");
                    assert!(projection.is_3d, "geometry pointers must be camera-bound");
                    let position = projection
                        .capture([
                            projection.rect[0] + projection.rect[2] * 0.55,
                            projection.rect[1] + projection.rect[3] * 0.45,
                        ])
                        .expect("geometry pointer");
                    let geometry_frame = crate::collaboration::PresenceFrame {
                        revision: "geometry-revision".into(),
                        target: None,
                        pointers: vec![crate::collaboration::PresencePointer {
                            position,
                            color: [240, 110, 30],
                            opacity: 0.8,
                            label: Some("Jules".into()),
                            click_progress: Some(0.35),
                            click_position: None,
                        }],
                    };
                    live.set_presence(geometry_frame.clone()).unwrap();
                    live.render_geometry_scene_to_view_with_presentation(
                        &scene,
                        &view,
                        Some(&camera),
                        Some(&presentation),
                    )
                    .unwrap();
                    live.retain_presented_frame(&texture);
                    let expected_geometry = live.presented_frame().unwrap().rgba().await.unwrap();
                    let mut snapshot = peer(&live, dpr, &theme).await;
                    snapshot.set_presence(geometry_frame).unwrap();
                    let actual = snapshot
                        .render_geometry_scene_to_rgba_with_presentation(
                            &scene,
                            Some(&camera),
                            Some(&presentation),
                        )
                        .await
                        .unwrap();
                    assert_pixels(
                        &format!("geometry-{dpr}-{variant:?}-{mode:?}"),
                        &actual,
                        &expected_geometry,
                        width,
                        height,
                    );
                    snapshot.set_presence(Default::default()).unwrap();
                    let without_presence = snapshot
                        .render_geometry_scene_to_rgba_with_presentation(
                            &scene,
                            Some(&camera),
                            Some(&presentation),
                        )
                        .await
                        .unwrap();
                    assert!(
                        actual != without_presence,
                        "geometry collaborator must actually be drawn"
                    );
                    assert_eq!(
                        retained.rgba().await.unwrap(),
                        expected,
                        "retained frame changed after another present"
                    );
                }
            }
        }
    }
}
