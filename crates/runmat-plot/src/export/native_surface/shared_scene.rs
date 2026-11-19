//! Rendering a shared geometry scene, and the reusable offscreen renderer a host uses when it
//! cannot present to a native surface.

use std::sync::Arc;

use super::{create_headless_context, BackgroundPolicy, NativeSurfaceRenderContext};
use crate::core::{Camera, RenderResult};
use crate::geometry_scene::{GeometryScene, GeometryScenePresentation};
use crate::gpu::util::readback_bytes;

impl NativeSurfaceRenderContext {
    /// Render a shared geometry scene directly into an externally-owned texture view.
    ///
    /// The renderer keeps the `Arc`, so a frame whose scene is unchanged (a camera move, for
    /// example) copies no geometry and reuses the GPU buffers it already holds.
    pub fn render_shared_geometry_scene_to_view(
        &mut self,
        scene: &Arc<GeometryScene>,
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

    /// Render a shared geometry scene offscreen with the native surface's presentation and read
    /// back tightly packed RGBA8 rows (`width * height * 4` bytes, top row first).
    pub async fn render_shared_geometry_scene_to_rgba(
        &mut self,
        scene: &Arc<GeometryScene>,
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

    fn prepare_geometry_scene(
        &mut self,
        scene: &Arc<GeometryScene>,
        camera: Option<&Camera>,
        presentation: Option<&GeometryScenePresentation>,
    ) {
        self.background_policy = BackgroundPolicy::ThemeDriven;
        self.apply_background_policy();
        self.config.show_grid = scene.show_grid;
        self.config.show_title = scene.title.is_some();

        if let Some(presentation) = presentation {
            self.renderer
                .set_geometry_scene_with_presentation(Arc::clone(scene), presentation.clone());
        } else {
            self.renderer.set_geometry_scene(Arc::clone(scene));
        }
        if let Some(camera) = camera {
            *self.renderer.camera_mut() = camera.clone();
        }
    }
}

/// One offscreen frame: tightly packed RGBA8 rows, top row first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadlessRgbaFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// An offscreen renderer kept across frames.
///
/// A host that cannot present to a native surface renders each frame offscreen and copies the
/// pixels to its view. Creating a render context compiles the renderer's pipelines and, when no
/// provider device is installed, opens a GPU device, so a host keeps one of these per surface
/// and renders every frame through it. The first frame creates the context (on the provider's
/// shared device when one is installed); later frames reuse it and resize it when the frame
/// size changes. A render error drops the context, so the next frame starts from a new one.
#[derive(Default)]
pub struct HeadlessFrameRenderer {
    context: Option<NativeSurfaceRenderContext>,
}

impl HeadlessFrameRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a context has been created and is kept for the next frame.
    pub fn is_ready(&self) -> bool {
        self.context.is_some()
    }

    /// Drops the kept context, for example after the device was lost or the theme host changed.
    pub fn reset(&mut self) {
        self.context = None;
    }

    /// The kept context sized to `width` x `height`, created on first use. The caller sets the
    /// pixels per point, theme and presence on it before rendering, as for a native surface.
    pub async fn context(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<&mut NativeSurfaceRenderContext, String> {
        let width = width.max(1);
        let height = height.max(1);
        match self.context.as_mut() {
            Some(context) if (context.config.width, context.config.height) != (width, height) => {
                if let Err(error) = context.resize(width, height) {
                    self.context = None;
                    return Err(error);
                }
            }
            Some(_) => {}
            None => self.context = Some(create_headless_context(width, height).await?),
        }
        self.context
            .as_mut()
            .ok_or_else(|| "headless render context unavailable".to_string())
    }

    /// Renders one geometry scene frame and returns its pixels.
    pub async fn render_geometry_scene(
        &mut self,
        scene: &Arc<GeometryScene>,
        width: u32,
        height: u32,
        camera: Option<&Camera>,
        presentation: Option<&GeometryScenePresentation>,
    ) -> Result<HeadlessRgbaFrame, String> {
        let context = self.context(width, height).await?;
        let (width, height) = (context.config.width, context.config.height);
        let result = context
            .render_shared_geometry_scene_to_rgba(scene, camera, presentation)
            .await;
        self.finish(width, height, result)
    }

    /// Records the outcome of a frame rendered through [`Self::context`]: a failure drops the
    /// kept context so the next frame starts from a new one.
    pub fn finish(
        &mut self,
        width: u32,
        height: u32,
        result: Result<Vec<u8>, String>,
    ) -> Result<HeadlessRgbaFrame, String> {
        match result {
            Ok(rgba) => Ok(HeadlessRgbaFrame {
                width,
                height,
                rgba,
            }),
            Err(error) => {
                self.context = None;
                Err(error)
            }
        }
    }
}
