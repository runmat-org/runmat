use crate::gpu::{GpuInputBuffer, ScalarType};
use futures::channel::oneshot;
use runmat_accelerate_api::WgpuDeviceAccess;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "util/tests.rs"]
mod tests;

/// The mapping belongs to the waiting future, not the callback. Cancellation
/// unmaps through the same device authority after registration has returned.
struct Mapping<'a> {
    access: &'a WgpuDeviceAccess,
    buffer: &'a wgpu::Buffer,
    registered: bool,
}
impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        if !self.registered {
            return;
        }
        if let Err(error) = self
            .access
            .with_cleanup(|cleanup| cleanup.unmap_buffer(self.buffer))
        {
            log::error!("plot readback cleanup could not acquire its device: {error}");
        }
    }
}

pub(crate) async fn readback_bytes(
    access: &WgpuDeviceAccess,
    buffer: &wgpu::Buffer,
    byte_len: usize,
) -> Result<Vec<u8>, String> {
    if byte_len == 0 {
        return Ok(Vec::new());
    }
    let byte_len_u64 = u64::try_from(byte_len).map_err(|error| error.to_string())?;
    if byte_len_u64 > buffer.size() {
        return Err("GPU readback buffer too small".to_string());
    }
    let slice = buffer.slice(0..byte_len_u64);
    let (tx, rx) = oneshot::channel();
    // Install the cleanup owner before registration, including inline failure.
    let mut mapping = Mapping {
        access,
        buffer,
        registered: false,
    };
    access
        .with_shared(|_| {
            mapping.registered = true;
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        })
        .map_err(|error| error.to_string())?;
    #[cfg(not(target_arch = "wasm32"))]
    access.poll(wgpu::Maintain::Wait);
    #[cfg(target_arch = "wasm32")]
    access.poll(wgpu::Maintain::Poll);
    rx.await
        .map_err(|_| "map callback dropped".to_string())?
        .map_err(|error| error.to_string())?;
    let bytes = access
        .with_shared(|_| {
            let data = slice.get_mapped_range();
            if data.len() < byte_len {
                return Err("GPU readback buffer too small".to_string());
            }
            Ok(data[..byte_len].to_vec())
        })
        .map_err(|error| error.to_string())??;
    drop(mapping);
    Ok(bytes)
}

pub async fn readback_u32(access: &WgpuDeviceAccess, buffer: &wgpu::Buffer) -> Result<u32, String> {
    let bytes = readback_bytes(access, buffer, 4).await?;
    Ok(u32::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| "invalid u32 readback extent".to_string())?,
    ))
}

pub async fn readback_f32(access: &WgpuDeviceAccess, buffer: &wgpu::Buffer) -> Result<f32, String> {
    Ok(f32::from_bits(readback_u32(access, buffer).await?))
}

pub async fn readback_f32_buffer(
    access: &WgpuDeviceAccess,
    buffer: &wgpu::Buffer,
    element_count: usize,
) -> Result<Vec<f32>, String> {
    let byte_len = element_count
        .checked_mul(4)
        .ok_or("readback byte size overflow")?;
    let bytes = readback_bytes(access, buffer, byte_len).await?;
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

pub async fn copy_readback_bytes(
    access: &WgpuDeviceAccess,
    buffer: &GpuInputBuffer,
    byte_len: usize,
) -> Result<Vec<u8>, String> {
    if byte_len == 0 {
        return Ok(Vec::new());
    }
    let staging = access
        .with_shared(|emission| -> Result<_, String> {
            let source = buffer.buffer(emission).map_err(|error| error.to_string())?;
            let byte_size = u64::try_from(byte_len).map_err(|error| error.to_string())?;
            if byte_size > source.size() {
                return Err("GPU copy readback buffer too small".to_string());
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
        .map_err(|error| error.to_string())??;
    readback_bytes(access, &staging, byte_len).await
}

pub async fn readback_scalar_buffer_f64(
    access: &WgpuDeviceAccess,
    buffer: &GpuInputBuffer,
    element_count: usize,
    scalar: ScalarType,
) -> Result<Vec<f64>, String> {
    let width = match scalar {
        ScalarType::F32 => 4,
        ScalarType::F64 => 8,
    };
    let byte_len = element_count
        .checked_mul(width)
        .ok_or("readback byte size overflow")?;
    let bytes = copy_readback_bytes(access, buffer, byte_len).await?;
    match scalar {
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
    }
}
