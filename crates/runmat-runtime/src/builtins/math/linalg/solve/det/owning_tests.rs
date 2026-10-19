use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorStorage, NumericElementType,
    ProviderLuResult, ThreadProviderGuard,
};
use std::{future::Future, sync::Mutex, task::Context};

struct Provider {
    owner: TestAllocationOwner,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl Provider {
    fn factor(&self, shape: Vec<usize>) -> anyhow::Result<GpuTensorHandle> {
        let factor =
            self.owner
                .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)?;
        self.outputs.lock().unwrap().push(factor.probe);
        Ok(factor.handle)
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "determinant factor ownership fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(std::future::pending())
    }
    fn lu<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelProviderFuture<'a, ProviderLuResult> {
        Box::pin(async move {
            Ok(ProviderLuResult {
                combined: self.factor(vec![2, 2])?,
                lower: self.factor(vec![2, 2])?,
                upper: self.factor(vec![2, 2])?,
                perm_matrix: self.factor(vec![2, 2])?,
                perm_vector: self.factor(vec![2, 1])?,
            })
        })
    }
}

#[test]
fn pending_determinant_download_retains_and_releases_entire_factor_bundle() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        outputs: Mutex::new(Vec::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut future = Box::pin(det_gpu_via_provider(provider, &input.handle));
    let waker = futures::task::noop_waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(provider.outputs.lock().unwrap().len(), 5);
    assert!(provider
        .outputs
        .lock()
        .unwrap()
        .iter()
        .all(|probe| probe.retirements() == 0));
    drop(future);
    assert!(provider
        .outputs
        .lock()
        .unwrap()
        .iter()
        .all(|probe| probe.retirements() == 1));
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}
