//! Pre-effect capability for the existing default-precision scan kernels.
use super::WgpuProvider;
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderExtremaDemand};

pub(super) fn admitted(
    provider: &WgpuProvider,
    input: &GpuTensorHandle,
    dimension: usize,
    demand: ProviderExtremaDemand,
) -> Result<bool, runmat_accelerate_api::ProviderExecutionError> {
    // Authenticate the real allocation before interpreting capability metadata.
    // A stale/foreign input is an error, not absence of a kernel.
    let entry = provider.get_entry_raw(input)?;
    if entry.storage != GpuTensorStorage::Real || runmat_accelerate_api::handle_is_logical(input) {
        return Ok(false);
    }
    if entry.integer_type().is_none()
        && (entry.precision != provider.precision
            || (entry.len != 0 && !provider.ieee_words_available(entry.precision)))
    {
        return Ok(false);
    }
    if demand == ProviderExtremaDemand::ValuesAndIndices
        && provider.precision != NumericPrecision::F64
    {
        return Ok(false);
    }
    if dimension >= entry.shape.len() || entry.len > u32::MAX as usize {
        return Ok(false);
    }
    let before = entry.shape[..dimension]
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d));
    let after = entry.shape[dimension + 1..]
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d));
    let Some((before, after)) = before.zip(after) else {
        return Ok(false);
    };
    Ok([
        entry.shape[dimension],
        before,
        before.saturating_mul(after),
        before.saturating_mul(entry.shape[dimension]),
    ]
    .into_iter()
    .all(|n| n <= u32::MAX as usize))
}
