use bytemuck::cast_slice;
use std::sync::{mpsc, Arc};

pub fn readback_u32(device: &Arc<wgpu::Device>, buffer: &wgpu::Buffer) -> Result<u32, String> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    match receiver.recv() {
        Ok(Ok(())) => {
            let data = slice.get_mapped_range();
            if data.len() < std::mem::size_of::<u32>() {
                return Err("readback buffer too small".to_string());
            }
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&data[..4]);
            drop(data);
            buffer.unmap();
            Ok(u32::from_le_bytes(bytes))
        }
        Ok(Err(err)) => Err(format!("{err:?}")),
        Err(_) => Err("GPU readback channel closed unexpectedly".to_string()),
    }
}

pub fn readback_f32(device: &Arc<wgpu::Device>, buffer: &wgpu::Buffer) -> Result<f32, String> {
    let bits = readback_u32(device, buffer)?;
    Ok(f32::from_bits(bits))
}

pub fn readback_f32_buffer(
    device: &Arc<wgpu::Device>,
    buffer: &wgpu::Buffer,
    element_count: usize,
) -> Result<Vec<f32>, String> {
    if element_count == 0 {
        return Ok(Vec::new());
    }
    let byte_len = element_count * std::mem::size_of::<f32>();
    let slice = buffer.slice(0..byte_len as u64);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    match receiver.recv() {
        Ok(Ok(())) => {
            let data = slice.get_mapped_range();
            if data.len() < byte_len {
                drop(data);
                buffer.unmap();
                return Err("GPU readback buffer too small".to_string());
            }
            let floats: &[f32] = cast_slice(&data[..byte_len]);
            let out = floats.to_vec();
            drop(data);
            buffer.unmap();
            Ok(out)
        }
        Ok(Err(err)) => Err(format!("{err:?}")),
        Err(_) => Err("GPU readback channel closed unexpectedly".to_string()),
    }
}
