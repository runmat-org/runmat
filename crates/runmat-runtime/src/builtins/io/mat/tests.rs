//! MAT-only readback witness; the adapter owns authentic allocations while
//! recording whether an invalid call reached its download boundary.

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostTensorView, ProviderPrecision,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Serialize MAT fixtures that touch accelerator, filesystem, or workspace
/// process state. Keep this order identical across every MAT test so a
/// provider callback cannot invert it while another fixture is active.
pub(super) struct MatTestStateGuard {
    _workspace: std::sync::MutexGuard<'static, ()>,
    _filesystem: std::sync::MutexGuard<'static, ()>,
    _global: crate::builtins::common::test_support::AccelTestGuard,
}

pub(super) fn state_guard() -> MatTestStateGuard {
    let global = crate::builtins::common::test_support::accel_test_lock();
    let filesystem = runmat_filesystem::provider_override_lock();
    let workspace = crate::workspace::test_guard();
    MatTestStateGuard {
        _workspace: workspace,
        _filesystem: filesystem,
        _global: global,
    }
}

pub(super) mod receipt;
pub(super) mod pending_provider;

pub(super) struct CountingProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl CountingProvider {
    pub(super) fn new() -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }
    }

    pub(super) fn resident(&self) -> GpuTensorHandle {
        self.upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .expect("owned resident fixture")
    }

    pub(super) fn downloads(&self) -> usize {
        self.downloads.load(Ordering::SeqCst)
    }
}

impl AccelProvider for CountingProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
}
