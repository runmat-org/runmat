//! Private observations from the ordinary two-stage AnyOmit reduction.
//! These are stage evidence, not a completed POWER probe or resource peak.
use crate::backend::wgpu::pipelines::PipelineBundle;
use crate::backend::wgpu::types::NumericPrecision;
use anyhow::{ensure, Result};
use runmat_accelerate_api::{
    allocation::{AllocationIdentity, TensorViewIdentity},
    GpuTensorHandle,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrozenAnySnapshot {
    pub(crate) enabled: bool,
}

impl FrozenAnySnapshot {
    pub(crate) fn capture() -> Self {
        Self {
            enabled: std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_ok(),
        }
    }
}

/// Exact constructor-frozen shader source, distinct from the legacy 512-way
/// dispatch divisor. This private value is execution authority only with its
/// originating owner and the same canonical pipeline bundle.
pub(crate) struct FrozenAnyPipeline {
    owner_device_id: u32,
    precision: NumericPrecision,
    shader_source: String,
    shader_workgroup: u32,
}

impl FrozenAnyPipeline {
    pub(crate) fn physical_identity(&self) -> Result<runmat_execution::Digest> {
        Ok(runmat_execution::Digest::sha256_serialized(&(
            "wgpu-power-any-physical-v1",
            &self.shader_source,
            self.shader_workgroup,
            match self.precision {
                NumericPrecision::F32 => 32_u8,
                NumericPrecision::F64 => 64_u8,
            },
        ))?)
    }

    pub(crate) fn prepare(
        owner_device_id: u32,
        precision: NumericPrecision,
        bundle: &PipelineBundle,
    ) -> Result<Self> {
        bundle.preflight()?;
        let (source, shader_workgroup) = bundle
            .frozen_reduction_source()
            .ok_or_else(|| anyhow::anyhow!("dimensional reduction shader source is not frozen"))?;
        ensure!(
            shader_workgroup > 0,
            "invalid dimensional reduction shader workgroup"
        );
        Ok(Self {
            owner_device_id,
            precision,
            shader_source: source.to_owned(),
            shader_workgroup,
        })
    }

    pub(crate) fn shader_workgroup(&self) -> u32 {
        self.shader_workgroup
    }

    pub(crate) fn require(
        &self,
        owner_device_id: u32,
        precision: NumericPrecision,
        bundle: &PipelineBundle,
    ) -> Result<()> {
        let (source, workgroup) = bundle
            .frozen_reduction_source()
            .ok_or_else(|| anyhow::anyhow!("dimensional reduction shader source changed"))?;
        ensure!(
            self.owner_device_id == owner_device_id
                && self.precision == precision
                && self.shader_workgroup == workgroup
                && self.shader_source == source,
            "prepared dimensional reduction pipeline changed"
        );
        bundle.preflight()
    }
}

#[derive(Debug)]
pub(crate) struct AnyStageObserved {
    pub(crate) input_view: TensorViewIdentity,
    pub(crate) output_allocation: AllocationIdentity,
    pub(crate) output_view: TensorViewIdentity,
    pub(crate) output_buffer_bytes: u64,
    pub(crate) snapshot_copy_bytes: Option<u64>,
    pub(crate) uniform_buffer_bytes: Option<u64>,
    pub(crate) workgroups_x: Option<u32>,
    pub(crate) snapshot_copy_submitted: bool,
    pub(crate) dispatch_submitted: bool,
    pub(crate) snapshot_fence_requested_bytes: Option<u64>,
    pub(crate) dispatch_fence_requested_bytes: Option<u64>,
    pub(crate) snapshot_ticket:
        Option<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
    pub(crate) dispatch_ticket:
        Option<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
}

impl AnyStageObserved {
    pub(crate) fn require_output(&self, output: &GpuTensorHandle) -> Result<()> {
        output.view().validate(output.physical())?;
        output.allocation_lease().validate_live()?;
        ensure!(
            output.allocation_identity() == self.output_allocation
                && output.view_identity() == self.output_view
                && output.shape() == [1, 1]
                && output.physical().requested_bytes() == self.output_buffer_bytes
                && self.input_view.allocation() != self.output_allocation
                && self.uniform_buffer_bytes.is_some()
                && self.workgroups_x == Some(1)
                && self.dispatch_submitted
                && self.snapshot_copy_bytes.is_some() == self.snapshot_copy_submitted
                && self.snapshot_ticket.is_some() == self.snapshot_copy_submitted
                && self.dispatch_ticket.is_some(),
            "POWER AnyOmit stage output/observation mismatch"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "power_probe/tests.rs"]
mod tests;
