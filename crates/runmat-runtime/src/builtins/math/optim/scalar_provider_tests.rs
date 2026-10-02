use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned,
    HostTensorView, ThreadProviderGuard,
};
use runmat_value::{IntValue, Value};
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
        "scalar optimization admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("admission must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(std::io::Error::other("scalar optimization fixture rejected transfer").into())
        })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(std::io::Error::other("scalar optimization fixture rejected transfer").into())
        })
    }
}

#[test]
fn host_visible_scalar_errors_precede_resident_transfer_and_preserve_borrowed_lease() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(RejectDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let real = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![0.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let complex = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![0.0, 0.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let function = || Value::FunctionHandle("sin".into());
    let resident = || Value::GpuTensor(real.handle.clone());

    let bad_upper = block_on(super::fminbnd::fminbnd_builtin(
        function(),
        resident(),
        Value::Complex(1.0, 1.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(bad_upper.identifier(), Some("RunMat:fminbnd:InvalidInput"));
    assert_eq!(bad_upper.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let wide_upper = block_on(super::fminbnd::fminbnd_builtin(
        function(),
        resident(),
        Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(wide_upper.identifier(), Some("RunMat:fminbnd:InvalidInput"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let bad_initial = block_on(super::fzero::fzero_builtin(
        function(),
        Value::GpuTensor(complex.handle.clone()),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(bad_initial.identifier(), Some("RunMat:fzero:InvalidInput"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let _outputs = crate::output_count::push_output_count(Some(5));
    let bad_outputs = block_on(super::fminbnd::fminbnd_builtin(
        function(),
        resident(),
        Value::Num(1.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(
        bad_outputs.identifier(),
        Some("RunMat:fminbnd:TooManyOutputs")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    drop(_outputs);

    let failed_transfer = block_on(super::fminbnd::fminbnd_builtin(
        function(),
        resident(),
        Value::Num(1.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(
        failed_transfer.identifier(),
        Some("RunMat:gather:DownloadFailed")
    );
    assert_eq!(
        failed_transfer.gpu_gather_retry,
        crate::GpuGatherRetry::Never
    );
    let mut source: &(dyn std::error::Error + 'static) = &failed_transfer;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("original provider cause retained");
    }
    assert_eq!(
        source.to_string(),
        "scalar optimization fixture rejected transfer"
    );
    assert_eq!(real.probe.retirements(), 0);
    assert_eq!(complex.probe.retirements(), 0);
    real.handle.release().unwrap();
    complex.handle.release().unwrap();
}
