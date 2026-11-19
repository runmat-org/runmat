//! GPU-resident plot data made on another device.
//!
//! Plot data computed on the accelerator (for example `plot(gpuArray)`) lives in buffers on the
//! provider's device. A render surface may draw on its own device (one that can present to the
//! window), and a device cannot bind another device's buffers. When a buffer records a device
//! other than the renderer's, the renderer reads it back on that device and uploads a copy to its
//! own, once per buffer: the copy is kept while the source buffer is alive, so later frames bind
//! the copy, and new data (a new source buffer) makes a new copy. A buffer that records no
//! device, or the renderer's device, is bound directly.

use super::*;
use runmat_accelerate_api::{WgpuBufferOwner, WgpuEmission};

/// Local copies of buffers made on another device, keyed by the source buffer. The entry keeps
/// the source alive, so its address is never reused while the entry exists.
#[derive(Default)]
pub(super) struct ForeignBufferCopies {
    copies: RefCell<HashMap<usize, (WgpuBufferOwner, WgpuBufferOwner)>>,
}

/// Identifies the vertex data a cached buffer was made from: GPU-resident data by its buffer,
/// CPU data by its vertex storage.
pub(super) fn vertex_signature(render_data: &crate::core::RenderData) -> (usize, usize) {
    match render_data.gpu_vertices.as_ref() {
        Some(gpu) => (WgpuBufferOwner::as_ptr(&gpu.buffer) as usize, 0),
        None => (
            render_data.vertices.as_ptr() as usize,
            render_data.vertices.len(),
        ),
    }
}

impl PlotRenderer {
    /// The vertex buffer to bind for `render_data` on this renderer's device.
    pub(super) fn vertex_buffer_for(
        &self,
        emission: &WgpuEmission<'_>,
        render_data: &crate::core::RenderData,
    ) -> Option<WgpuBufferOwner> {
        match render_data.gpu_vertices.as_ref() {
            Some(gpu) => self.local_buffer(emission, &gpu.buffer),
            None => {
                self.wgpu_renderer
                    .vertex_buffer_from_sources(emission, None, &render_data.vertices)
            }
        }
    }

    /// The indirect draw arguments to use for `render_data` on this renderer's device.
    pub(super) fn gpu_indirect_args(
        &self,
        emission: &WgpuEmission<'_>,
        render_data: &crate::core::RenderData,
    ) -> Option<(WgpuBufferOwner, u64)> {
        let indirect = render_data.gpu_vertices.as_ref()?.indirect.as_ref()?;
        let args = self.local_buffer(emission, &indirect.args)?;
        Some((args, indirect.offset))
    }

    /// `source` itself when this renderer's device can bind it; otherwise a kept local copy.
    /// A copy that fails is logged and the item is not drawn: binding the source would be
    /// invalid on this device.
    fn local_buffer(
        &self,
        emission: &WgpuEmission<'_>,
        source: &WgpuBufferOwner,
    ) -> Option<WgpuBufferOwner> {
        let foreign = source
            .device()
            .is_some_and(|device| !device.same_device(emission.access()));
        if !foreign {
            return Some(source.clone());
        }
        let key = WgpuBufferOwner::as_ptr(source) as usize;
        let mut copies = self.foreign_buffers.copies.borrow_mut();
        if let Some((_, local)) = copies.get(&key) {
            return Some(local.clone());
        }
        match copy_to_device(emission, source) {
            Ok(local) => {
                // Drop copies whose source no one else holds: that data was replaced.
                copies.retain(|_, (kept, _)| WgpuBufferOwner::strong_count(kept) > 1);
                copies.insert(key, (source.clone(), local.clone()));
                Some(local)
            }
            Err(error) => {
                log::error!(
                    target: "runmat_plot",
                    "GPU plot data on another device could not be copied to the render device: {error}"
                );
                None
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn foreign_buffer_copy_count(&self) -> usize {
        self.foreign_buffers.copies.borrow().len()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn copy_to_device(
    emission: &WgpuEmission<'_>,
    source: &WgpuBufferOwner,
) -> Result<WgpuBufferOwner, String> {
    use wgpu::util::DeviceExt;
    let source_device = source
        .device()
        .ok_or_else(|| "buffer records no device".to_string())?;
    let size = source.size();
    let staging = source_device
        .with_shared(|origin| {
            let staging = origin.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-plot foreign buffer readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder =
                origin
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-plot foreign buffer readback"),
                    });
            encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
            origin.queue().submit(Some(encoder.finish()));
            staging
        })
        .map_err(|error| error.to_string())?;
    // Native readback waits on the source device, so the future is ready when polled.
    let bytes = pollster::block_on(crate::gpu::util::readback_bytes(
        source_device,
        &staging,
        size as usize,
    ))?;
    let usage = source
        .usage()
        .difference(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE)
        | wgpu::BufferUsages::COPY_DST;
    let local = emission
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("runmat-plot foreign buffer copy"),
            contents: &bytes,
            usage,
        });
    Ok(WgpuBufferOwner::on_device(local, emission.access()))
}

/// A browser has one device per page, so a buffer from another device cannot be read back here.
#[cfg(target_arch = "wasm32")]
fn copy_to_device(
    _emission: &WgpuEmission<'_>,
    _source: &WgpuBufferOwner,
) -> Result<WgpuBufferOwner, String> {
    Err("a buffer from another device cannot be copied in the browser".to_string())
}
