use crate::gpu::ScalarType;
use runmat_accelerate_api::{WgpuDeviceAccess, WgpuEmission};
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
    Buffer(crate::gpu::GpuInputBuffer),
}

#[derive(Clone, Debug)]
pub enum OwnedAxisData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    Buffer(crate::gpu::GpuInputBuffer),
}

impl OwnedAxisData {
    pub fn from_axis(axis: &AxisData<'_>) -> Self {
        match axis {
            AxisData::F32(values) => Self::F32(values.to_vec()),
            AxisData::F64(values) => Self::F64(values.to_vec()),
            AxisData::Buffer(buffer) => Self::Buffer(buffer.clone()),
        }
    }

    pub async fn export_f64(
        &self,
        access: &WgpuDeviceAccess,
        len: usize,
        scalar: ScalarType,
    ) -> Result<Vec<f64>, String> {
        match self {
            Self::F32(values) => Ok(values.iter().map(|value| f64::from(*value)).collect()),
            Self::F64(values) => Ok(values.clone()),
            Self::Buffer(buffer) => {
                crate::gpu::util::readback_scalar_buffer_f64(access, buffer, len, scalar).await
            }
        }
    }
}

pub fn axis_storage_buffer(
    emission: &WgpuEmission<'_>,
    label: &'static str,
    axis: &AxisData<'_>,
    scalar: ScalarType,
) -> Result<crate::gpu::GpuInputBuffer, String> {
    let device = emission.device();
    match axis {
        AxisData::Buffer(buffer) => {
            buffer.buffer(emission).map_err(|error| error.to_string())?;
            Ok(buffer.clone())
        }
        AxisData::F32(values) => {
            if scalar != ScalarType::F32 {
                return Err(format!("{label}: expected f64 axis data for f64 shader"));
            }
            Ok(
                runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytemuck::cast_slice(values),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    },
                ))
                .into(),
            )
        }
        AxisData::F64(values) => {
            if scalar != ScalarType::F64 {
                return Err(format!("{label}: expected f32 axis data for f32 shader"));
            }
            Ok(
                runmat_accelerate_api::WgpuBufferOwner::new(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytemuck::cast_slice(values),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    },
                ))
                .into(),
            )
        }
    }
}
