//! WebGPU renderer integration for browser builds.
//!
//! This module owns the WGPU surface that backs a `<canvas>` element and
//! exposes a light-weight wrapper around [`PlotRenderer`] so wasm callers can
//! drive the full RunMat plotting stack without bouncing through JavaScript
//! custom events.

#![cfg(all(target_arch = "wasm32", feature = "web"))]

use crate::context::SharedWgpuContext;
use crate::core::plot_renderer::{PlotRenderConfig, PlotRenderer};
use crate::plots::Figure;
use log::{debug, warn};
use std::sync::Arc;
use thiserror::Error;
use web_sys::HtmlCanvasElement;

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
    canvas: HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_config: wgpu::SurfaceConfiguration,
    plot_renderer: PlotRenderer,
    render_config: PlotRenderConfig,
    options: WebRendererOptions,
}

impl WebRenderer {
    /// Initialize the renderer for the provided canvas element.
    pub async fn new(
        canvas: HtmlCanvasElement,
        options: WebRendererOptions,
    ) -> Result<Self, WebRendererError> {
        Self::init(canvas, options, None).await
    }

    /// Initialize the renderer using a shared GPU context supplied by the host runtime.
    pub async fn with_shared_context(
        canvas: HtmlCanvasElement,
        options: WebRendererOptions,
        context: SharedWgpuContext,
    ) -> Result<Self, WebRendererError> {
        Self::init(canvas, options, Some(context)).await
    }

    async fn init(
        canvas: HtmlCanvasElement,
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
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))?;

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

        let mut renderer = Self {
            canvas,
            surface,
            device,
            queue,
            surface_config,
            plot_renderer,
            render_config: PlotRenderConfig {
                width,
                height,
                msaa_samples: options.msaa_samples,
                ..Default::default()
            },
            options,
        };
        renderer.sync_renderer_config();
        Ok(renderer)
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

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-plot-web-encoder"),
            });

        self.render_config.width = self.surface_config.width.max(1);
        self.render_config.height = self.surface_config.height.max(1);
        self.render_config.msaa_samples = self.options.msaa_samples.max(1);

        self.plot_renderer
            .render(&mut encoder, &view, &self.render_config)
            .map_err(|err| WebRendererError::Render(err.to_string()))?;

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
    }
}

fn desired_canvas_size(
    canvas: &HtmlCanvasElement,
    override_width: Option<u32>,
    override_height: Option<u32>,
) -> Result<(u32, u32), WebRendererError> {
    let width = override_width.unwrap_or_else(|| canvas.width());
    let height = override_height.unwrap_or_else(|| canvas.height());
    if width == 0 || height == 0 {
        let rect = canvas.get_bounding_client_rect();
        let w = rect.width().round() as u32;
        let h = rect.height().round() as u32;
        if w > 0 && h > 0 {
            return Ok((w, h));
        }
        return Err(WebRendererError::CanvasZeroArea);
    }
    Ok((width, height))
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
