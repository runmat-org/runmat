//! Host-visible envelope class and geometry admission shared by numeric and
//! statement execution. Finite resident payload values remain a post-gather
//! check, but known-invalid metadata never requires a provider download.

use super::*;

pub(super) fn preflight_signal_metadata(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Tensor(tensor) => {
            signal_geometry(tensor.shape(), tensor.len())?;
        }
        Value::LogicalArray(array) => {
            signal_geometry(&array.shape, array.data.len())?;
        }
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
            }
            let len = handle
                .shape()
                .iter()
                .try_fold(1usize, |count, &extent| count.checked_mul(extent))
                .ok_or_else(|| envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL))?;
            signal_geometry(handle.shape(), len)?;
        }
        Value::Num(value) if !value.is_finite() => {
            return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {}
        _ => return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL)),
    }
    Ok(())
}

pub(super) fn signal_geometry(
    shape: &[usize],
    len: usize,
) -> BuiltinResult<(Vec<usize>, usize, usize)> {
    if len == 0 {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    let shape = tensor::default_shape_for(shape, len);
    if shape.len() > 2 {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    let rows = shape.first().copied().unwrap_or(len);
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(len) {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    Ok((shape, rows, cols))
}
