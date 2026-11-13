use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostTensorView, NumericElementType, ProviderInvOptions, ProviderNormOrder,
    ThreadProviderGuard,
};
use runmat_value::Value;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "owning_tests/complex_residency.rs"]
mod complex_residency;

struct Provider {
    owner: TestAllocationOwner,
    alias: bool,
}
impl Provider {
    fn result<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        scalar: bool,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.alias {
                let view = if scalar {
                    input.view().clone().with_shape(vec![1, 1])
                } else {
                    input.view().clone()
                };
                input.with_view(view).map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })
            } else {
                Err(
                    runmat_accelerate_api::ProviderExecutionError::from(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "native linalg failed",
                    ))
                    .with_unproven_effects(),
                )
            }
        })
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "native linalg ownership witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no fallback after native failure") })
    }
    fn inv<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: ProviderInvOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.result(input, false)
    }
    fn norm<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: ProviderNormOrder,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.result(input, true)
    }
}

#[test]
fn inverse_and_norm_keep_provider_causes_and_borrowed_inputs_on_rejection() {
    for name in ["inv", "norm"] {
        for alias in [false, true] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                alias,
            }));
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let error = futures::executor::block_on(crate::call_builtin_async(
                name,
                &[Value::GpuTensor(input.handle.clone())],
            ))
            .unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            if !alias {
                let source = error.source.as_ref().expect("provider source retained");
                let provider = source
                    .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                    .expect("typed provider execution carrier retained");
                let cause = std::error::Error::source(provider)
                    .expect("original provider cause retained")
                    .downcast_ref::<std::io::Error>()
                    .expect("original I/O error remains downcastable");
                assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
                assert_eq!(cause.to_string(), "native linalg failed");
            }
            assert_eq!(input.probe.retirements(), 0);
            input.handle.allocation_lease().validate_live().unwrap();
        }
    }
}
