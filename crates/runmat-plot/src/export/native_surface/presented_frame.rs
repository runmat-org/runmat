//! The GPU copy of a presented native frame, and when a present keeps one.

use std::sync::Arc;

use super::NativeSurfaceRenderContext;

/// Which presents keep a GPU copy of their final frame.
///
/// Keeping a copy costs a full-frame texture copy on every retained present, so a host keeps
/// none while the user pans, orbits or zooms, and asks for one only when it needs the exact
/// pixels on screen (a snapshot that must match the display, for example).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PresentedFrameRetention {
    /// Keep no copy. [`NativeSurfaceRenderContext::presented_frame`] returns `None`.
    #[default]
    None,
    /// Keep a copy of the next present only, then return to [`PresentedFrameRetention::None`].
    NextPresent,
    /// Keep a copy of every present.
    EveryPresent,
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
    /// Sets which presents keep a GPU copy of their final frame for
    /// [`NativeSurfaceRenderContext::presented_frame`]. The default keeps none.
    pub fn set_presented_frame_retention(&mut self, retention: PresentedFrameRetention) {
        self.presented_frame_retention = retention;
    }

    pub fn presented_frame_retention(&self) -> PresentedFrameRetention {
        self.presented_frame_retention
    }

    /// Called by the host after it renders a frame into `source` and before it presents it.
    /// Keeps a GPU copy of the frame when the retention policy asks for one; CPU readback is
    /// deferred until a reader asks. Never overwrites a frame held by a reader. When the policy
    /// keeps nothing, any earlier copy is dropped, because it no longer shows the surface.
    pub fn retain_presented_frame(&mut self, source: &wgpu::Texture) {
        match self.presented_frame_retention {
            PresentedFrameRetention::None => {
                self.presented_frame = None;
                return;
            }
            PresentedFrameRetention::NextPresent => {
                self.presented_frame_retention = PresentedFrameRetention::None;
            }
            PresentedFrameRetention::EveryPresent => {}
        }
        self.copy_presented_frame(source);
    }

    fn copy_presented_frame(&mut self, source: &wgpu::Texture) {
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
}
