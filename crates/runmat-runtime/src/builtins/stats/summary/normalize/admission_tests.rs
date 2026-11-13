use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned,
    HostTensorView, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct RejectDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for RejectDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "normalization admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("admission must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("normalize fixture download rejected").into()) })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("normalize fixture download rejected").into()) })
    }
}

#[test]
fn invalid_host_visible_calls_do_not_download_resident_input_or_controls() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let real = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0]),
            shape: vec![3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let complex = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
            shape: vec![3, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let scalar_control = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let array_control = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0]),
            shape: vec![3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let resident = || Value::GpuTensor(real.handle.clone());
    for (data, args, diagnostic) in [
        (
            resident(),
            vec![Value::from("unknown")],
            "unsupported method",
        ),
        (
            resident(),
            vec![Value::from("zscore"), Value::from("unknown")],
            "unsupported zscore method type",
        ),
        (resident(), vec![Value::Num(0.0)], "dimension must be >= 1"),
        (
            resident(),
            vec![
                Value::GpuTensor(scalar_control.handle.clone()),
                Value::from("unknown"),
            ],
            "unsupported method",
        ),
        (
            resident(),
            vec![
                Value::from("norm"),
                Value::GpuTensor(array_control.handle.clone()),
            ],
            "norm method type must be a scalar",
        ),
        (
            resident(),
            vec![
                Value::from("range"),
                Value::GpuTensor(array_control.handle.clone()),
            ],
            "range target must be",
        ),
        (
            Value::GpuTensor(complex.handle.clone()),
            vec![Value::from("range")],
            "not supported for complex",
        ),
        (
            resident(),
            vec![Value::from("center"), Value::Complex(1.0, 2.0)],
            "not supported for real",
        ),
    ] {
        let error = block_on(normalize_builtin(
            data,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .unwrap_err();
        assert!(
            error.message().contains(diagnostic),
            "expected {diagnostic}: {error}"
        );
        assert_eq!(
            error.identifier(),
            NORMALIZE_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }

    let error = block_on(normalize_builtin(
        resident(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(error.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let mut source: &(dyn std::error::Error + 'static) = &error;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("provider cause retained");
    }
    assert_eq!(source.to_string(), "normalize fixture download rejected");
}
