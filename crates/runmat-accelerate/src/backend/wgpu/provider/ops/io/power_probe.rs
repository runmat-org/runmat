//! Actual ordinary scalar download observations; no GPU completion claim.
use anyhow::{ensure, Result};
use runmat_accelerate_api::allocation::TensorViewIdentity;
use runmat_accelerate_api::observation::ProviderResourceTicket;

/// Ordinary scalar probe download decodes directly into one f64 lane and a
/// two-axis shape vector. Native mapped/staging backing is provider storage,
/// separately included by the original probe geometry.
pub(crate) fn power_probe_host_workspace_bytes() -> Result<u64> {
    let metadata = std::mem::size_of::<usize>()
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("POWER scalar shape metadata overflow"))?;
    u64::try_from(
        std::mem::size_of::<f64>()
            .checked_add(metadata)
            .ok_or_else(|| anyhow::anyhow!("POWER scalar host workspace overflow"))?,
    )
    .map_err(Into::into)
}

/// The ticket travels with the actual staging buffer across async mapping,
/// including cancellation, and retires only after the readback owner drops.
pub(crate) struct ProbeStaging {
    pub(crate) buffer: runmat_accelerate_api::WgpuBufferOwner,
    ticket: Option<ProviderResourceTicket>,
}

impl ProbeStaging {
    pub(crate) fn new(
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        ticket: Option<ProviderResourceTicket>,
    ) -> Self {
        Self { buffer, ticket }
    }

    pub(crate) fn requested_bytes(&self) -> u64 {
        self.ticket
            .as_ref()
            .map_or(self.buffer.size(), ProviderResourceTicket::requested_bytes)
    }
}

impl std::borrow::Borrow<wgpu::Buffer> for ProbeStaging {
    fn borrow(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl super::super::core::ReadbackResourceOwner for ProbeStaging {
    fn buffer_owner(&self) -> Option<&runmat_accelerate_api::WgpuBufferOwner> {
        Some(&self.buffer)
    }
    fn validate_scope(&self) -> Result<()> {
        self.buffer.validate_creation_scope()?;
        Ok(())
    }
    fn resource_ticket(&self) -> Option<ProviderResourceTicket> {
        self.ticket.clone()
    }
    fn cleanup_failed(&self) {
        if let Some(ticket) = &self.ticket {
            ticket.fail_owner();
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct PowerProbeReadbackObserved {
    pub(crate) input_view: Option<TensorViewIdentity>,
    pub(crate) copied_bytes: Option<u64>,
    pub(crate) staging_buffer_bytes: Option<u64>,
    pub(crate) mapped_bytes: Option<u64>,
    pub(crate) copy_submission_id: Option<u32>,
    pub(crate) copy_fence_requested_bytes: Option<u64>,
    pub(crate) copy_ticket: Option<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
}

impl PowerProbeReadbackObserved {
    pub(crate) fn require_scalar(&self, expected: TensorViewIdentity) -> Result<()> {
        ensure!(
            self.input_view == Some(expected)
                && self.copied_bytes.is_some()
                && self.copied_bytes == self.staging_buffer_bytes
                && self.copied_bytes == self.mapped_bytes
                && self.copy_submission_id.is_some()
                && self.copy_ticket.is_some(),
            "POWER scalar readback observation incomplete"
        );
        Ok(())
    }
}
