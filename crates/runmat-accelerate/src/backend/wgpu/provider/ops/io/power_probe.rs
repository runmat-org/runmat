//! Actual ordinary scalar download observations; no GPU completion claim.
use anyhow::{ensure, Result};
use runmat_accelerate_api::allocation::TensorViewIdentity;
use runmat_accelerate_api::observation::ProviderResourceTicket;

/// The ticket travels with the actual staging buffer across async mapping,
/// including cancellation, and retires only after the readback owner drops.
pub(crate) struct ProbeStaging {
    pub(crate) buffer: wgpu::Buffer,
    ticket: Option<ProviderResourceTicket>,
}

impl ProbeStaging {
    pub(crate) fn new(buffer: wgpu::Buffer, ticket: Option<ProviderResourceTicket>) -> Self {
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
