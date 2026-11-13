use crate::gpu::{GpuInputBuffer, PlotReadbackError, ScalarType};
use futures::channel::oneshot;
use runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind;
use runmat_accelerate_api::WgpuDeviceAccess;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "util/tests.rs"]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "util/browser_tests.rs"]
mod browser_tests;

/// The mapping belongs to the waiting future, not the callback. Cancellation
/// unmaps through the same device authority after registration has returned.
struct Mapping<'a> {
    access: &'a WgpuDeviceAccess,
    buffer: &'a wgpu::Buffer,
    registered: bool,
}
impl Mapping<'_> {
    fn finish(&mut self) -> Result<(), PlotReadbackError> {
        if !self.registered {
            return Ok(());
        }
        // Never retry an unmap that unwound: its completion is uncertain.
        self.registered = false;
        catch_allocation_cleanup_unwind(|| {
            self.access
                .with_cleanup(|cleanup| cleanup.unmap_buffer(self.buffer))
        })
        .map_err(|_| PlotReadbackError::UnmapPanicked)?
        .map_err(PlotReadbackError::CleanupAccess)
    }
}
impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.finish() {
            log::error!("{error}");
        }
    }
}

pub(crate) async fn readback_bytes(
    access: &WgpuDeviceAccess,
    buffer: &wgpu::Buffer,
    byte_len: usize,
) -> Result<Vec<u8>, String> {
    readback_mapped(access, buffer, byte_len, |bytes| Ok(bytes.to_vec()))
        .await
        .map_err(|error| error.to_string())
}

async fn readback_mapped<T>(
    access: &WgpuDeviceAccess,
    buffer: &wgpu::Buffer,
    byte_len: usize,
    decode: impl FnOnce(&[u8]) -> Result<T, PlotReadbackError>,
) -> Result<T, PlotReadbackError> {
    if byte_len == 0 {
        return decode(&[]);
    }
    let byte_len_u64 = u64::try_from(byte_len).map_err(|_| PlotReadbackError::ByteSizeOverflow)?;
    if byte_len_u64 > buffer.size() {
        return Err(PlotReadbackError::BufferTooSmall {
            requested: byte_len,
            available: buffer.size(),
        });
    }
    let slice = buffer.slice(0..byte_len_u64);
    let (tx, rx) = oneshot::channel();
    // Install the cleanup owner before registration, including inline failure.
    let mut mapping = Mapping {
        access,
        buffer,
        registered: false,
    };
    let result = async {
        access
            .with_shared(|_| {
                mapping.registered = true;
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
            })
            .map_err(PlotReadbackError::Access)?;
        #[cfg(not(target_arch = "wasm32"))]
        access.poll(wgpu::Maintain::Wait);
        #[cfg(target_arch = "wasm32")]
        access.poll(wgpu::Maintain::Poll);
        rx.await
            .map_err(|_| PlotReadbackError::MapCallbackDropped)?
            .map_err(PlotReadbackError::Map)?;
        access
            .with_shared(|_| {
                let data = slice.get_mapped_range();
                if data.len() < byte_len {
                    return Err(PlotReadbackError::BufferTooSmall {
                        requested: byte_len,
                        available: data.len() as u64,
                    });
                }
                decode(&data[..byte_len])
            })
            .map_err(PlotReadbackError::Access)?
    }
    .await;
    // The mapped view and emission permit have both ended before cleanup.
    combine_cleanup(result, mapping.finish())
}

fn combine_cleanup<T>(
    result: Result<T, PlotReadbackError>,
    cleanup: Result<(), PlotReadbackError>,
) -> Result<T, PlotReadbackError> {
    match (result, cleanup) {
        (result, Ok(())) => result,
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => Err(PlotReadbackError::WithCleanup {
            primary: Box::new(error),
            cleanup: Box::new(cleanup),
        }),
    }
}

/// Read a scalar from a same-device, MAP_READ staging buffer.
///
/// The caller must supply the buffer's actual device authority; raw buffers do
/// not authenticate that association. Do not poll or drop this future while an
/// emission permit is active: progress and mapping cleanup need that authority.
pub async fn readback_u32(access: &WgpuDeviceAccess, buffer: &wgpu::Buffer) -> Result<u32, String> {
    readback_mapped(access, buffer, 4, |bytes| {
        Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| {
            PlotReadbackError::Decode("invalid u32 readback extent")
        })?))
    })
    .await
    .map_err(|error| error.to_string())
}

/// Read one float under the same raw staging-buffer and future-lifetime
/// requirements as [`readback_u32`].
pub async fn readback_f32(access: &WgpuDeviceAccess, buffer: &wgpu::Buffer) -> Result<f32, String> {
    Ok(f32::from_bits(readback_u32(access, buffer).await?))
}

/// Read floats from a same-device MAP_READ staging buffer. The raw association
/// is caller-established, not authenticated here. As with [`readback_u32`], do
/// not poll or drop the future under an active emission permit.
pub async fn readback_f32_buffer(
    access: &WgpuDeviceAccess,
    buffer: &wgpu::Buffer,
    element_count: usize,
) -> Result<Vec<f32>, String> {
    let byte_len = element_count
        .checked_mul(4)
        .ok_or("readback byte size overflow")?;
    readback_mapped(access, buffer, byte_len, |bytes| {
        Ok(bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect())
    })
    .await
    .map_err(|error| error.to_string())
}

/// Copy and read bytes using the buffer's actual device authority. Exported
/// buffers authenticate that authority; raw renderer buffers rely on the caller.
/// Do not poll or drop this future under an active emission permit.
pub async fn copy_readback_bytes(
    access: &WgpuDeviceAccess,
    buffer: &GpuInputBuffer,
    byte_len: usize,
) -> Result<Vec<u8>, String> {
    copy_readback_bytes_typed(access, buffer, byte_len)
        .await
        .map_err(|error| error.to_string())
}

/// Typed byte export for plot-source consumers that retain device, mapping,
/// and cleanup causes across graphics-property boundaries.
pub(crate) async fn copy_readback_bytes_typed(
    access: &WgpuDeviceAccess,
    buffer: &GpuInputBuffer,
    byte_len: usize,
) -> Result<Vec<u8>, PlotReadbackError> {
    copy_readback_mapped(access, buffer, byte_len, |bytes| Ok(bytes.to_vec())).await
}

/// Exact integer histogram evidence. Mapping/cancellation and original causes
/// belong to the same typed copy-readback owner as the other scalar decoders.
pub async fn copy_readback_u32_buffer(
    access: &WgpuDeviceAccess,
    buffer: &GpuInputBuffer,
    element_count: usize,
) -> Result<Vec<u32>, PlotReadbackError> {
    let byte_len = element_count
        .checked_mul(4)
        .ok_or(PlotReadbackError::ByteSizeOverflow)?;
    copy_readback_mapped(access, buffer, byte_len, |bytes| {
        Ok(bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect())
    })
    .await
}

async fn copy_readback_mapped<T>(
    access: &WgpuDeviceAccess,
    buffer: &GpuInputBuffer,
    byte_len: usize,
    decode: impl FnOnce(&[u8]) -> Result<T, PlotReadbackError>,
) -> Result<T, PlotReadbackError> {
    if byte_len == 0 {
        // Even an empty export carries a device and allocation lease. Check
        // both before reporting success, without creating staging work.
        access
            .with_shared(|emission| {
                buffer
                    .buffer(emission)
                    .map(|_| ())
                    .map_err(PlotReadbackError::Access)
            })
            .map_err(PlotReadbackError::Access)??;
        return decode(&[]);
    }
    let staging = access
        .with_shared(|emission| -> Result<_, PlotReadbackError> {
            let source = buffer.buffer(emission).map_err(PlotReadbackError::Access)?;
            let byte_size =
                u64::try_from(byte_len).map_err(|_| PlotReadbackError::ByteSizeOverflow)?;
            if byte_size > source.size() {
                return Err(PlotReadbackError::CopyBufferTooSmall {
                    requested: byte_len,
                    available: source.size(),
                });
            }
            let staging = emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-plot-readback-staging"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let commands = emission.encode("runmat-plot-readback-encoder", |encoder| {
                encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_size);
            });
            emission.queue().submit(Some(commands));
            Ok(staging)
        })
        .map_err(PlotReadbackError::Access)??;
    readback_mapped(access, &staging, byte_len, decode).await
}

/// Copy and decode scalar data, with the same authority and future-lifetime
/// requirements as [`copy_readback_bytes`].
pub async fn readback_scalar_buffer_f64(
    access: &WgpuDeviceAccess,
    buffer: &GpuInputBuffer,
    element_count: usize,
    scalar: ScalarType,
) -> Result<Vec<f64>, String> {
    readback_scalar_buffer_f64_typed(access, buffer, element_count, scalar)
        .await
        .map_err(|error| error.to_string())
}

/// Typed form for internal export consumers that must retain the original
/// device, map, and cleanup causes.
pub(crate) async fn readback_scalar_buffer_f64_typed(
    access: &WgpuDeviceAccess,
    buffer: &GpuInputBuffer,
    element_count: usize,
    scalar: ScalarType,
) -> Result<Vec<f64>, PlotReadbackError> {
    let width = match scalar {
        ScalarType::F32 => 4,
        ScalarType::F64 => 8,
    };
    let byte_len = element_count
        .checked_mul(width)
        .ok_or(PlotReadbackError::ByteSizeOverflow)?;
    copy_readback_mapped(access, buffer, byte_len, |bytes| match scalar {
        ScalarType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|bytes| {
                f64::from(f32::from_le_bytes(
                    bytes.try_into().expect("four-byte chunk"),
                ))
            })
            .collect()),
        ScalarType::F64 => Ok(bytes
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")))
            .collect()),
    })
    .await
}
