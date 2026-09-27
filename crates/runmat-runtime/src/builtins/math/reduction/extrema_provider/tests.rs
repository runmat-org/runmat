use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProviderFuture, GpuTensorStorage,
    HostTensorOwned, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
enum Fault {
    Execution,
    Unsupported,
    Download,
}

struct Provider {
    owner: TestAllocationOwner,
    fault: Fault,
    reductions: AtomicUsize,
    downloads: AtomicUsize,
}

impl Provider {
    fn reduce(&self) -> anyhow::Result<ReduceDimResult> {
        self.reductions.fetch_add(1, Ordering::SeqCst);
        match self.fault {
            Fault::Execution => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "unsupported transport state is not a capability rejection",
            )
            .into()),
            Fault::Unsupported | Fault::Download => {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "dimensional reduction unavailable",
                ))
            }
        }
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "extrema execution authority fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unexpected restoration")
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            if matches!(self.fault, Fault::Download) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "download failed",
                )
                .into());
            }
            assert_eq!(input.device_id(), self.device_id());
            input.allocation_lease().validate_live()?;
            Ok(HostTensorOwned {
                data: vec![0.0; input.shape().iter().product()],
                shape: input.shape().to_vec(),
            })
        })
    }
    fn reduce_min_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.reduce() })
    }
    fn reduce_max_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.reduce() })
    }
}

#[test]
fn dimensional_extrema_only_gather_after_typed_unsupported() {
    for name in ["min", "max"] {
        for fault in [Fault::Execution, Fault::Unsupported, Fault::Download] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                fault,
                reductions: AtomicUsize::new(0),
                downloads: AtomicUsize::new(0),
            }));
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let result = futures::executor::block_on(crate::call_builtin_async(
                name,
                &[Value::GpuTensor(input.handle.clone())],
            ));
            assert_eq!(provider.reductions.load(Ordering::SeqCst), 1);
            match fault {
                Fault::Unsupported => assert_eq!(result.unwrap(), Value::Num(0.0)),
                Fault::Execution | Fault::Download => {
                    let error = result.unwrap_err();
                    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                    let mut source = std::error::Error::source(&error);
                    let mut retained = false;
                    while let Some(error) = source {
                        retained |= error.downcast_ref::<std::io::Error>().is_some();
                        source = error.source();
                    }
                    assert!(retained, "original {name} provider failure must survive");
                }
            }
            assert_eq!(
                provider.downloads.load(Ordering::SeqCst),
                usize::from(!matches!(fault, Fault::Execution))
            );
            assert_eq!(input.probe.retirements(), 0);
            input.handle.allocation_lease().validate_live().unwrap();
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[test]
fn dimensional_extrema_never_dispatch_unregistered_input_to_selected_provider() {
    let selected = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        fault: Fault::Execution,
        reductions: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(selected));
    let input = TestAllocationOwner::new()
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for name in ["min", "max"] {
        let error = futures::executor::block_on(crate::call_builtin_async(
            name,
            &[Value::GpuTensor(input.handle.clone())],
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert!(error.message().contains("exact owning provider"));
    }
    assert_eq!(selected.reductions.load(Ordering::SeqCst), 0);
    assert_eq!(selected.downloads.load(Ordering::SeqCst), 0);
    input.handle.allocation_lease().validate_live().unwrap();
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}
