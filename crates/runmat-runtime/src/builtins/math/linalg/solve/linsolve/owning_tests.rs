use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, GpuTensorStorage, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::sync::Mutex;

struct Provider {
    owner: TestAllocationOwner,
    uploads: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "linsolve operand ownership fixture".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        let output = self.owner.allocate_zeroed(
            host.shape.to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )?;
        self.uploads.lock().unwrap().push(output.probe);
        Ok(output.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused download") })
    }
}

#[test]
fn prepared_linsolve_operand_releases_only_its_last_owned_reference() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        uploads: Mutex::new(Vec::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = Value::GpuTensor(input.handle.clone());
    let borrowed = prepare_gpu_operand(&resident, provider, &[&input.handle])
        .unwrap()
        .unwrap();
    drop(borrowed);
    assert_eq!(input.probe.retirements(), 0);
    let host = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let uploaded = prepare_gpu_operand(&host, provider, &[&input.handle])
        .unwrap()
        .unwrap();
    let published = uploaded.handle().clone();
    drop(uploaded);
    assert_eq!(provider.uploads.lock().unwrap()[0].retirements(), 0);
    drop(published);
    assert_eq!(provider.uploads.lock().unwrap()[0].retirements(), 1);
    input.handle.allocation_lease().validate_live().unwrap();
}
