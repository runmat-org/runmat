use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{AccelDownloadFuture, HostTensorView};
use std::sync::atomic::{AtomicUsize, Ordering};

struct FailedProbe {
    inner: InProcessProvider,
    downloads: AtomicUsize,
}

impl AccelProvider for FailedProbe {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "failed-logical-probe".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download(handle)
    }
    fn logical_isnan(&self, _: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(std::io::Error::other("injected NaN probe execution failure").into())
    }
}

#[test]
fn logical_probe_failure_is_not_gathered_and_preserves_source_and_input() {
    let _state = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(FailedProbe {
        inner: InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[0.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let error = block_on(logical_from_gpu(input.clone())).unwrap_err();
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    let mut source: &(dyn std::error::Error + 'static) = &error;
    while source.downcast_ref::<std::io::Error>().is_none() {
        source = source
            .source()
            .expect("original operational source must survive");
    }
    assert_eq!(
        block_on(provider.inner.download(&input)).unwrap().data,
        vec![0.0, 2.0]
    );
}

#[test]
fn logical_single_and_empty_fallback_publish_real_logical_producers() {
    test_support::with_test_provider(|provider| {
        for (values, shape) in [(vec![0.0_f32, -3.0], vec![1, 2]), (vec![], vec![0, 3])] {
            let input = provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(&values),
                    shape: &shape,
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })
                .unwrap()
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            let Value::GpuTensor(output) = block_on(logical_from_gpu(input.clone())).unwrap()
            else {
                panic!("explicit residency")
            };
            assert!(output.view().is_logical());
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(output.shape(), shape);
            assert_eq!(
                block_on(provider.download(&output)).unwrap().data,
                values
                    .iter()
                    .map(|value| if *value == 0.0 { 0.0 } else { 1.0 })
                    .collect::<Vec<_>>()
            );
        }
    });
}
