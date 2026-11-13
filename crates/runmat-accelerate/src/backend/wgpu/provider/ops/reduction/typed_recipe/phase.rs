//! Request-local observation of real typed-reduction emission boundaries.
use super::*;
use anyhow::Result;
use runmat_accelerate_api::WgpuBufferOwner;

pub(super) trait ReductionObserver {
    #[cfg(test)]
    fn prepare_pending(
        &self,
        _pending: &mut crate::backend::wgpu::dispatch::observation::PendingSubmissions,
    ) {
    }
    fn retaining(
        &mut self,
        resources: SubmittedReductionResources,
    ) -> Result<crate::backend::wgpu::dispatch::observation::SubmittedAllocationRetention> {
        Ok(resources.into_retention())
    }
    fn output_acquired(&mut self, output: &WgpuBufferOwner) -> Result<()>;
    fn partial_acquired(&mut self, partial: &WgpuBufferOwner) -> Result<()>;
    fn submitted(&mut self, stage: Stage, id: u32) -> Result<()>;
}

pub(super) struct NoopObserver;
impl ReductionObserver for NoopObserver {
    #[inline]
    fn output_acquired(&mut self, _output: &WgpuBufferOwner) -> Result<()> {
        Ok(())
    }
    #[inline]
    fn partial_acquired(&mut self, _partial: &WgpuBufferOwner) -> Result<()> {
        Ok(())
    }
    #[inline]
    fn submitted(&mut self, _stage: Stage, _id: u32) -> Result<()> {
        Ok(())
    }
}
