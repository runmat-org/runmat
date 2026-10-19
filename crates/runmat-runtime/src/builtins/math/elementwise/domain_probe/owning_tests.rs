use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

struct Provider {
    owner: TestAllocationOwner,
    alias: bool,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "domain probe ownership witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(std::future::pending())
    }
    fn reduce_min<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.alias {
                return Ok(input.with_view(input.view().clone().with_shape(vec![1, 1]))?);
            }
            let output = self.owner.allocate_zeroed(
                vec![1, 1],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.outputs.lock().unwrap().push(output.probe);
            Ok(output.handle)
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_probe_owns_reduction_and_rejects_distinct_view_input_alias() {
    for alias in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            alias,
            outputs: Mutex::new(Vec::new()),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let mut future = Box::pin(probe_gpu_lower_bound(provider, &input.handle, 0.0));
        let waker = futures::task::noop_waker();
        let state = future.as_mut().poll(&mut Context::from_waker(&waker));
        if alias {
            assert!(matches!(
                state,
                Poll::Ready(Err(GpuDomainProbeError::AliasedReduction))
            ));
        } else {
            assert!(state.is_pending());
            assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 0);
        }
        drop(future);
        if !alias {
            assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
        }
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
