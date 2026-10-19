use crate::backend::wgpu::dispatch::observation::SubmittedAllocationRetention;
use runmat_accelerate_api::{GpuTensorHandle, WgpuBufferOwner};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::sync::Arc;

/// Exact allocation owners consumed by one stochastic tile submission.  The
/// mapped semantic fence is not the lifetime authority: this bundle moves to
/// the dispatch layer's exact queue-completion owner before submit.
pub(super) struct StochasticTileSubmittedAllocations {
    _source: GpuTensorHandle,
    _noise: GpuTensorHandle,
    _output: GpuTensorHandle,
    _uniform: WgpuBufferOwner,
    _semantic_fence: WgpuBufferOwner,
    _completion_control: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

impl StochasticTileSubmittedAllocations {
    pub(super) fn new(
        source: GpuTensorHandle,
        noise: GpuTensorHandle,
        output: GpuTensorHandle,
        uniform: WgpuBufferOwner,
        semantic_fence: WgpuBufferOwner,
        completion_control: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Self {
        Self {
            _source: source,
            _noise: noise,
            _output: output,
            _uniform: uniform,
            _semantic_fence: semantic_fence,
            _completion_control: completion_control,
        }
    }

    pub(super) fn host_upper_bytes() -> Option<u64> {
        SubmittedAllocationRetention::host_upper_bytes::<Self>()
    }

    pub(super) fn into_retention(self) -> SubmittedAllocationRetention {
        SubmittedAllocationRetention::new(self)
    }
}
