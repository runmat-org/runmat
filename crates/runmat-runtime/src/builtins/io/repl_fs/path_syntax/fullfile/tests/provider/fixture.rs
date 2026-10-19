use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) enum DownloadBehavior {
    Fail,
    Pending,
}

pub(super) struct ControlledDownloadProvider {
    owner: TestAllocationOwner,
    behavior: DownloadBehavior,
    pub(super) downloads: AtomicUsize,
}

impl ControlledDownloadProvider {
    pub(super) fn new(behavior: DownloadBehavior) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            behavior,
            downloads: AtomicUsize::new(0),
        }
    }

    pub(super) fn numeric_character_row(
        &self,
    ) -> runmat_accelerate_api::test_support::TestAllocation {
        self.owner
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![100.0, 97.0]),
                shape: vec![1, 2],
                storage: GpuTensorStorage::Real,
            })
            .expect("numeric character row")
    }

    fn download_result<T: 'static>(&self) -> runmat_accelerate_api::AccelProviderFuture<'_, T> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        match self.behavior {
            DownloadBehavior::Fail => Box::pin(async {
                Err(std::io::Error::other("fullfile fixture rejected download").into())
            }),
            DownloadBehavior::Pending => Box::pin(futures::future::pending()),
        }
    }
}

impl AccelProvider for ControlledDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "fullfile controlled download".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("fixture allocates through its exact owner")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.download_result()
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.download_result()
    }
}
