use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};

struct Provider {
    owner: TestAllocationOwner,
    unsupported: bool,
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "typed domain probe failure fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no upload after operational probe failure")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no gather retry after operational probe failure") })
    }
    fn reduce_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.unsupported {
                Err(
                    runmat_accelerate_api::unsupported_provider_operation("reduction unavailable")
                        .into(),
                )
            } else {
                // Text containing unsupported must not classify an execution
                // failure as the typed capability rejection.
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "unsupported transport state during reduction",
                )
                .into())
            }
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn all_lower_bound_consumers_preserve_operational_source_and_disable_retry() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        unsupported: false,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for name in [
        "sqrt", "realsqrt", "log", "log2", "log10", "log1p", "gammaln",
    ] {
        let error = futures::executor::block_on(crate::call_builtin_async(
            name,
            &[runmat_value::Value::GpuTensor(input.handle.clone())],
        ))
        .unwrap_err();
        assert_eq!(
            error.gpu_gather_retry,
            crate::GpuGatherRetry::Never,
            "{name}"
        );
        let probe = error
            .source
            .as_ref()
            .expect("operation must retain probe source")
            .downcast_ref::<GpuDomainProbeError>()
            .expect("typed domain probe failure");
        let GpuDomainProbeError::ProviderReduce(source) = probe else {
            panic!("provider reduction source expected");
        };
        assert_eq!(
            std::error::Error::source(source)
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::BrokenPipe,
            "{name}"
        );
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn only_typed_unsupported_probe_is_unknown() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        unsupported: true,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert_eq!(
        futures::executor::block_on(probe_gpu_lower_bound(provider, &input.handle, 0.0)).unwrap(),
        GpuLowerBoundResult::Unknown
    );
    assert_eq!(input.probe.retirements(), 0);
}
