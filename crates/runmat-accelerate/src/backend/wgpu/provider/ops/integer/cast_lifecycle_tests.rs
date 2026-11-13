use super::*;
use runmat_accelerate_api::observation::ProviderOperationReceipt;
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
    HostLogicalTensorView, HostNumericDataView, HostNumericTensorView, ProviderPrecision,
};
use std::error::Error as _;

#[cfg(not(target_arch = "wasm32"))]
type WeakBuffer = std::sync::Weak<wgpu::Buffer>;
#[cfg(target_arch = "wasm32")]
type WeakBuffer = std::rc::Weak<wgpu::Buffer>;

fn weak_buffer(buffer: &runmat_accelerate_api::WgpuBufferOwner) -> WeakBuffer {
    #[cfg(not(target_arch = "wasm32"))]
    {
        runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer)
    }
    #[cfg(target_arch = "wasm32")]
    {
        runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer)
    }
}

async fn owner() -> WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    {
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required")
    }
    #[cfg(target_arch = "wasm32")]
    {
        WgpuProvider::new_async(Default::default())
            .await
            .expect("actual browser WebGPU adapter required")
    }
}

fn receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            provider
                .observation_progress
                .as_ref()
                .expect("native progress owner")
                .clone(),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        ProviderOperationReceipt::with_budget(provider.observation_budget.clone())
    }
}

#[path = "cast_lifecycle_tests/cancel.rs"]
mod cancel;
#[path = "cast_lifecycle_tests/numeric.rs"]
mod numeric;
#[path = "cast_lifecycle_tests/phase.rs"]
mod phase;
