//! Counted transfer fixture for convolution admission-order tests.

use std::sync::atomic::{AtomicUsize, Ordering};

use runmat_accelerate_api::{
    test_support::{TestAllocation, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned, HostTensorView,
};

pub(crate) struct RejectTransferProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl RejectTransferProvider {
    pub(crate) fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            downloads: AtomicUsize::new(0),
        }))
    }

    pub(crate) fn input(&self, data: Vec<f64>, shape: Vec<usize>) -> TestAllocation {
        self.owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(data),
                shape,
                storage: GpuTensorStorage::Real,
            })
            .expect("fixture input")
    }

    pub(crate) fn downloads(&self) -> usize {
        self.downloads.load(Ordering::SeqCst)
    }
}

impl AccelProvider for RejectTransferProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "convolution transfer admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("admission must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("unexpected download").into()) })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("unexpected numeric download").into()) })
    }
}
