use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::{future::Future, sync::Mutex, task::Context};

struct Provider {
    owner: TestAllocationOwner,
    uploads: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "matrix product ownership fixture".into()
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
        Box::pin(async { panic!("unexpected download") })
    }
    fn matmul<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(std::future::pending())
    }
}

#[test]
fn matrix_product_pending_drop_retires_uploaded_operand_not_resident_input() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        uploads: Mutex::new(Vec::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let left = Value::GpuTensor(input.handle.clone());
    let right = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let mut future = Box::pin(try_product(&left, &right));
    let waker = futures::task::noop_waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(provider.uploads.lock().unwrap().len(), 1);
    assert_eq!(provider.uploads.lock().unwrap()[0].retirements(), 0);
    drop(future);
    assert_eq!(provider.uploads.lock().unwrap()[0].retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}
