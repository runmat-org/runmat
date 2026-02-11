use crate::gpu::ScalarType;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Axis data source used by GPU plot packers.
///
/// For GPU kernels that support both f32 and f64 data paths, we treat axis precision as matching
/// the input scalar type (f32 axes for f32 shaders, f64 axes for f64 shaders). Shaders can cast
/// to f32 for final vertex positions as needed.
#[derive(Clone)]
pub enum AxisData<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
    Buffer(Arc<wgpu::Buffer>),
}

pub fn axis_storage_buffer(
    device: &Arc<wgpu::Device>,
    label: &'static str,
    axis: &AxisData<'_>,
    scalar: ScalarType,
) -> Result<Arc<wgpu::Buffer>, String> {
    match axis {
        AxisData::Buffer(buffer) => Ok(buffer.clone()),
        AxisData::F32(values) => {
            if scalar != ScalarType::F32 {
                return Err(format!("{label}: expected f64 axis data for f64 shader"));
            }
            Ok(Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(values),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
            )))
        }
        AxisData::F64(values) => {
            if scalar != ScalarType::F64 {
                return Err(format!("{label}: expected f32 axis data for f32 shader"));
            }
            Ok(Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(values),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
            )))
        }
    }
}
