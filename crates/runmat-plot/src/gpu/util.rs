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
