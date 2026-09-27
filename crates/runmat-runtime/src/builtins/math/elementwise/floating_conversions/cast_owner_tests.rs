//! Same-type immutable casts may share input ownership, never an unrelated prototype.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{TestAllocation, TestAllocationOwner},
    AccelProvider, AccelProviderFuture, ThreadProviderGuard,
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
    fn allocation(&self) -> TestAllocation {
        self.owner
            .allocate_zeroed(
                vec![1, 2],
                match CAST_PRECISION {
                    ProviderPrecision::F32 => NumericElementType::F32,
                    ProviderPrecision::F64 => NumericElementType::F64,
                },
                GpuTensorStorage::Real,
            )
            .unwrap()
    }
    fn result(&self, input: &GpuTensorHandle) -> GpuTensorHandle {
        self.response
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| input.clone())
    }
}
impl AccelProvider for Provider {
    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("same-type direct cast must not upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { panic!("same-type direct cast must not download") })
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "immutable cast owner".into()
    }
    fn precision(&self) -> ProviderPrecision {
        CAST_PRECISION
    }
    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.result(input)) })
    }
    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.result(input)) })
    }
}

#[test]
fn immutable_cast_adopts_exact_input_alias_without_retiring_caller() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Provider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.allocation();
    let input_view = input.handle.view_identity();
    let output = block_on(cast_from_gpu(input.handle.clone(), &[&input.handle])).unwrap();
    let Value::GpuTensor(output) = output else {
        panic!("cast must stay resident")
    };
    assert!(output.shares_allocation_with(&input.handle));
    assert_eq!(input.handle.view_identity(), input_view);
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(CAST_PRECISION)
    );
    drop(output);
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn immutable_cast_rejects_unrelated_borrowed_prototype_alias() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Provider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.allocation();
    let prototype = provider.allocation();
    *provider.response.lock().unwrap() = Some(prototype.handle.clone());
    assert!(block_on(cast_from_gpu(input.handle.clone(), &[&prototype.handle])).is_err());
    assert_eq!(input.probe.retirements(), 0);
    assert_eq!(prototype.probe.retirements(), 0);
    drop(input.handle);
    drop(prototype.handle);
    assert_eq!(input.probe.retirements(), 1);
    assert_eq!(prototype.probe.retirements(), 1);
}
