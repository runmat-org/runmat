use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::{
    test_support::{TestAllocation, TestAllocationOwner},
    GpuTensorStorage, HostNumericTensorView, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::Mutex;

struct Provider {
    owner: TestAllocationOwner,
    response: Mutex<Option<GpuTensorHandle>>,
}
impl Provider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            response: Mutex::new(None),
        }))
    }
    fn allocation(&self, shape: Vec<usize>) -> TestAllocation {
        self.owner
            .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
    }
}
impl AccelProvider for Provider {
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { panic!("restore upload must not download") })
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self
            .response
            .lock()
            .unwrap()
            .take()
            .expect("configured upload"))
    }
    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        Ok(self
            .response
            .lock()
            .unwrap()
            .take()
            .expect("configured upload"))
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned restore fixture".into()
    }
}

#[test]
fn protected_restore_rejects_borrowed_allocation_and_preserves_typed_error() {
    let _serial = test_support::accel_test_lock();
    let provider = Provider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider.allocation(vec![1, 1]);
    for value in [
        Value::Num(0.25),
        Value::Complex(0.25, 0.5),
        Value::Tensor(Tensor::new(vec![0.25], vec![1, 1]).unwrap()),
        Value::ComplexTensor(ComplexTensor::new(vec![(0.25, 0.5)], vec![1, 1]).unwrap()),
    ] {
        *provider.response.lock().unwrap() = Some(source.handle.clone());
        let error = upload_value_protected(
            provider,
            value,
            "restore-test",
            std::slice::from_ref(&source.handle),
        )
        .unwrap_err();
        assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
            &error
        ));
        assert_eq!(source.probe.retirements(), 0);
        assert!(source.handle.allocation_lease().validate_live().is_ok());
    }
    drop(source.handle);
    assert_eq!(source.probe.retirements(), 1);
}

#[test]
fn restore_like_always_protects_its_borrowed_prototype() {
    let _serial = test_support::accel_test_lock();
    let provider = Provider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let prototype = provider.allocation(vec![1, 1]);
    *provider.response.lock().unwrap() = Some(prototype.handle.clone());
    assert!(
        upload_value_like(provider, Value::Num(0.0), "restore-test", &prototype.handle).is_err()
    );
    assert_eq!(prototype.probe.retirements(), 0);
    drop(prototype.handle);
    assert_eq!(prototype.probe.retirements(), 1);
}

#[test]
fn restore_rejection_retires_actual_owner_not_selected_provider() {
    let _serial = test_support::accel_test_lock();
    let invoked = Provider::new();
    let unrelated = Provider::new();
    let output = invoked.allocation(vec![1, 2]);
    let borrowed = unrelated.allocation(vec![1, 1]);
    *invoked.response.lock().unwrap() = Some(output.handle);
    let _selected = ThreadProviderGuard::set(Some(unrelated));
    let error = upload_value_protected(invoked, Value::Num(0.25), "restore-test", &[]).unwrap_err();
    assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
        &error
    ));
    assert_eq!(output.probe.retirements(), 1);
    assert_eq!(borrowed.probe.retirements(), 0);
    drop(borrowed.handle);
    assert_eq!(borrowed.probe.retirements(), 1);
}
