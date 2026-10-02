use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostTensorView, ThreadProviderGuard,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum Mode {
    Valid,
    BadShape,
    SharedAliasBadShape,
    BadClass,
    ForeignOwner,
    Stale,
    Operational,
}

struct WindowProvider {
    owner: TestAllocationOwner,
    other: TestAllocationOwner,
    mode: Mode,
    probes: Mutex<Vec<RetirementProbe>>,
    borrowed: Mutex<Option<GpuTensorHandle>>,
}

impl WindowProvider {
    fn new(mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            other: TestAllocationOwner::new(),
            mode,
            probes: Mutex::new(Vec::new()),
            borrowed: Mutex::new(None),
        }))
    }

    fn produce(&self, len: usize) -> anyhow::Result<GpuTensorHandle> {
        if matches!(self.mode, Mode::Operational) {
            anyhow::bail!("device lost while generating window");
        }
        let shape = if matches!(self.mode, Mode::BadShape | Mode::SharedAliasBadShape) {
            vec![1, len]
        } else {
            vec![len, 1]
        };
        let owner = if matches!(self.mode, Mode::ForeignOwner) {
            &self.other
        } else {
            &self.owner
        };
        let element = if matches!(self.mode, Mode::BadClass) {
            NumericElementType::F32
        } else {
            NumericElementType::F64
        };
        let allocation = owner.allocate_zeroed(shape, element, GpuTensorStorage::Real)?;
        if matches!(self.mode, Mode::Stale) {
            allocation.probe.mark_unavailable();
        }
        self.probes.lock().unwrap().push(allocation.probe);
        if matches!(self.mode, Mode::SharedAliasBadShape) {
            *self.borrowed.lock().unwrap() = Some(allocation.handle.clone());
        }
        Ok(allocation.handle)
    }
}

impl AccelProvider for WindowProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "window contract fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("window generation must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("window generation must not download") })
    }
    fn hann_window(&self, len: usize, _: bool) -> anyhow::Result<GpuTensorHandle> {
        self.produce(len)
    }
    fn hamming_window(&self, len: usize, _: bool) -> anyhow::Result<GpuTensorHandle> {
        self.produce(len)
    }
    fn blackman_window(&self, len: usize, _: bool) -> anyhow::Result<GpuTensorHandle> {
        self.produce(len)
    }
}

fn options() -> WindowOptions {
    WindowOptions {
        len: 4,
        sampling: WindowSampling::Symmetric,
        output_type: WindowOutputType::Double,
    }
}

#[test]
fn unsupported_defaults_allow_host_reference_without_masking_operational_failures() {
    let _lock = test_support::accel_test_lock();
    struct UnsupportedProvider(TestAllocationOwner);
    impl AccelProvider for UnsupportedProvider {
        fn device_id(&self) -> u32 {
            self.0.device_id()
        }
        fn device_info(&self) -> String {
            "unsupported window fixture".into()
        }
        fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("unexpected upload")
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("unexpected download") })
        }
    }
    let unsupported = Box::leak(Box::new(UnsupportedProvider(TestAllocationOwner::new())));
    let _active = ThreadProviderGuard::set(Some(unsupported));
    for kind in [
        CosineWindow::Blackman,
        CosineWindow::Hamming,
        CosineWindow::Hann,
    ] {
        assert!(try_generate(kind, options()).unwrap().is_none());
    }
    drop(_active);

    let failed = WindowProvider::new(Mode::Operational);
    let _active = ThreadProviderGuard::set(Some(failed));
    for kind in [
        CosineWindow::Blackman,
        CosineWindow::Hamming,
        CosineWindow::Hann,
    ] {
        let failure = try_generate(kind, options()).unwrap_err();
        assert_eq!(failure.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(failure.identifier(), kind.provider_failed().identifier);
        assert!(failure.source.is_some());
    }
}

#[test]
fn result_admission_retires_only_rejected_outputs() {
    let _lock = test_support::accel_test_lock();
    for mode in [
        Mode::BadShape,
        Mode::BadClass,
        Mode::ForeignOwner,
        Mode::Stale,
    ] {
        let provider = WindowProvider::new(mode);
        let _active = ThreadProviderGuard::set(Some(provider));
        let failure = try_generate(CosineWindow::Hann, options()).unwrap_err();
        assert_eq!(failure.identifier(), HANN_ERROR_INTERNAL.identifier);
        assert_eq!(failure.gpu_gather_retry, GpuGatherRetry::Never);
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].retirements(), 1);
    }
    let provider = WindowProvider::new(Mode::SharedAliasBadShape);
    let _active = ThreadProviderGuard::set(Some(provider));
    let failure = try_generate(CosineWindow::Hann, options()).unwrap_err();
    assert_eq!(failure.identifier(), HANN_ERROR_INTERNAL.identifier);
    let borrowed = provider
        .borrowed
        .lock()
        .unwrap()
        .take()
        .expect("borrowed alias");
    assert!(borrowed.allocation_lease().validate_live().is_ok());
    let probe = provider.probes.lock().unwrap()[0].clone();
    assert_eq!(probe.retirements(), 0);
    drop(borrowed);
    assert_eq!(probe.retirements(), 1);
    drop(_active);
    let provider = WindowProvider::new(Mode::Valid);
    let _active = ThreadProviderGuard::set(Some(provider));
    let output = try_generate(CosineWindow::Blackman, options())
        .unwrap()
        .expect("resident output");
    let Value::GpuTensor(handle) = output else {
        panic!("expected resident window")
    };
    assert_eq!(handle.shape(), [4, 1]);
    assert!(handle.allocation_lease().validate_live().is_ok());
    assert_eq!(provider.probes.lock().unwrap()[0].retirements(), 0);
}
