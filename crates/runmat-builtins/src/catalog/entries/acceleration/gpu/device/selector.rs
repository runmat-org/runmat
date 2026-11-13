//! Exact numeric admission for the RunMat provider-info facade's device index.
//! Runtime values and catalog literals use the same boundary.

pub const GPU_DEVICE_ACTIVE_INDEX: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceIndexError {
    Invalid,
}

pub fn gpu_device_index_from_integer(value: i128) -> Result<u32, GpuDeviceIndexError> {
    u32::try_from(value)
        .ok()
        .filter(|index| *index > 0)
        .ok_or(GpuDeviceIndexError::Invalid)
}

pub fn gpu_device_index_from_float(value: f64) -> Result<u32, GpuDeviceIndexError> {
    if !value.is_finite() {
        return Err(GpuDeviceIndexError::Invalid);
    }
    if value.fract() != 0.0 || value < 1.0 || value > u32::MAX as f64 {
        return Err(GpuDeviceIndexError::Invalid);
    }
    gpu_device_index_from_integer(value as i128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_boundary_matches_accepted_runtime_selector() {
        assert_eq!(gpu_device_index_from_integer(1), Ok(1));
        assert_eq!(
            gpu_device_index_from_integer(u32::MAX as i128),
            Ok(u32::MAX)
        );
        for value in [0, -1, u32::MAX as i128 + 1, u64::MAX as i128] {
            assert_eq!(
                gpu_device_index_from_integer(value),
                Err(GpuDeviceIndexError::Invalid)
            );
        }
        assert_eq!(gpu_device_index_from_float(1.0), Ok(1));
        assert_eq!(gpu_device_index_from_float(u32::MAX as f64), Ok(u32::MAX));
        for value in [
            0.0,
            -0.0,
            -1.0,
            1.0 - 1e-10,
            1.0 + 1e-10,
            1.5,
            f64::NAN,
            f64::INFINITY,
            u32::MAX as f64 + 1.0,
        ] {
            assert_eq!(
                gpu_device_index_from_float(value),
                Err(GpuDeviceIndexError::Invalid)
            );
        }
    }
}
