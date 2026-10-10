//! Structural modulation facts frozen before download or scoped preparation.
use super::{anyhow, ensure, GpuTensorStorage, Result, WgpuProvider};
use runmat_accelerate_api::{
    ProviderBitModulationRequest, ProviderExecutionError, ProviderModulationRequest,
};

pub(super) fn unproven(error: impl Into<ProviderExecutionError>) -> ProviderExecutionError {
    error.into().with_unproven_effects()
}

pub(super) struct SymbolPlan {
    pub(super) entry: super::super::backend_types::BufferEntry,
    pub(super) logical_len: usize,
    pub(super) out_len: usize,
    pub(super) order: usize,
}

impl SymbolPlan {
    pub(super) fn new(
        provider: &WgpuProvider,
        request: &ProviderModulationRequest<'_>,
    ) -> Result<Self> {
        ensure!(
            runmat_accelerate_api::handle_storage(request.input) == GpuTensorStorage::Real,
            "modulate_constellation requires a real-valued symbol input"
        );
        ensure!(
            !request.constellation.is_empty() && request.constellation.len().is_multiple_of(2),
            "modulate_constellation requires interleaved real/imag constellation pairs"
        );
        let order = request.constellation.len() / 2;
        ensure!(
            order <= u32::MAX as usize,
            "modulate_constellation: constellation too large"
        );
        let entry = provider.get_entry(request.input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real && entry.precision == provider.precision,
            "modulate_constellation: input storage does not match provider precision"
        );
        let logical_len = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("modulate_constellation: input shape overflow"))?;
        ensure!(
            logical_len == entry.len,
            "modulate_constellation: input data length does not match shape"
        );
        ensure!(
            logical_len <= u32::MAX as usize,
            "modulate_constellation: input too large"
        );
        let out_len = logical_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("modulate_constellation: output length overflow"))?;
        Ok(Self {
            entry,
            logical_len,
            out_len,
            order,
        })
    }
}

pub(super) struct BitPlan {
    pub(super) entry: super::super::backend_types::BufferEntry,
    pub(super) logical_len: usize,
    pub(super) output_logical_len: usize,
    pub(super) output_shape: Vec<usize>,
    pub(super) out_len: usize,
    pub(super) order: usize,
    pub(super) output_rows: usize,
    pub(super) channels: usize,
}

impl BitPlan {
    pub(super) fn new(
        provider: &WgpuProvider,
        request: &ProviderBitModulationRequest<'_>,
    ) -> Result<Self> {
        ensure!(
            runmat_accelerate_api::handle_storage(request.input) == GpuTensorStorage::Real,
            "modulate_bits_constellation requires a real-valued bit input"
        );
        ensure!(
            !request.constellation.is_empty() && request.constellation.len().is_multiple_of(2),
            "modulate_bits_constellation requires interleaved real/imag constellation pairs"
        );
        ensure!(
            request.input_rows > 0 && request.bits_per_symbol > 0,
            "modulate_bits_constellation: invalid bit grouping"
        );
        ensure!(
            request.input_rows.is_multiple_of(request.bits_per_symbol),
            "modulate_bits_constellation: bit rows must be a multiple of bits_per_symbol"
        );
        let order = request.constellation.len() / 2;
        ensure!(
            request.constellation.len() <= u32::MAX as usize && order <= u32::MAX as usize,
            "modulate_bits_constellation: constellation too large"
        );
        ensure!(
            request.input_rows <= u32::MAX as usize && request.bits_per_symbol <= u32::MAX as usize,
            "modulate_bits_constellation: input too large"
        );
        let entry = provider.get_entry(request.input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real && entry.precision == provider.precision,
            "modulate_bits_constellation: input storage does not match provider precision"
        );
        let logical_len = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("modulate_bits_constellation: input shape overflow"))?;
        ensure!(
            logical_len == entry.len,
            "modulate_bits_constellation: input data length does not match shape"
        );
        ensure!(
            entry.shape.first().copied() == Some(request.input_rows),
            "modulate_bits_constellation: input_rows must match the input leading dimension"
        );
        ensure!(
            logical_len <= u32::MAX as usize,
            "modulate_bits_constellation: input too large"
        );
        ensure!(
            logical_len.is_multiple_of(request.input_rows),
            "modulate_bits_constellation: bit input shape is inconsistent"
        );
        let output_rows = request.input_rows / request.bits_per_symbol;
        let channels = logical_len / request.input_rows;
        let output_logical_len = output_rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow!("modulate_bits_constellation: output shape overflow"))?;
        ensure!(
            output_logical_len <= u32::MAX as usize,
            "modulate_bits_constellation: output too large"
        );
        let mut output_shape = entry.shape.clone();
        if output_shape.is_empty() {
            output_shape.push(output_rows);
            output_shape.push(1);
        } else {
            output_shape[0] = output_rows;
        }
        let out_len = output_logical_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("modulate_bits_constellation: output length overflow"))?;
        ensure!(
            out_len <= u32::MAX as usize,
            "modulate_bits_constellation: output too large"
        );
        Ok(Self {
            entry,
            logical_len,
            output_logical_len,
            output_shape,
            out_len,
            order,
            output_rows,
            channels,
        })
    }
}
