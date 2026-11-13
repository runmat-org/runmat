use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericTensorView,
    HostTensorView, NumericElementType, ProviderPrecision, ThreadProviderGuard,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct AliasProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    output: Mutex<Option<GpuTensorHandle>>,
    uploads: AtomicUsize,
}
impl AccelProvider for AliasProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "numeric-limit-alias".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        let output = self.output.lock().unwrap();
        let output = output.as_ref().unwrap();
        Ok(output.with_view(output.view().clone())?)
    }
    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.upload(&HostTensorView {
            data: &[],
            shape: &[0, 0],
        })
        .map_err(|error| {
            runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
        })
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("limit creation must not download its prototype")
    }
}

#[test]
fn limit_uploads_reject_borrowed_aliases_without_freeing_prototypes() {
    let provider = Box::leak(Box::new(AliasProvider {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
        output: Mutex::new(None),
        uploads: AtomicUsize::new(0),
    }));
    let _provider = ThreadProviderGuard::set(Some(provider));
    for (builtin, element) in [
        (
            intmax as fn(Vec<Value>) -> BuiltinResult<Value>,
            NumericElementType::U64,
        ),
        (intmin, NumericElementType::I64),
        (realmax, NumericElementType::F64),
        (realmin, NumericElementType::F32),
        (flintmax, NumericElementType::F64),
        (eps, NumericElementType::F64),
    ] {
        let allocation = provider
            .owner
            .allocate_zeroed(vec![1, 1], element, GpuTensorStorage::Real)
            .unwrap();
        let prototype = allocation.handle;
        let descriptor = prototype.descriptor();
        *provider.output.lock().unwrap() = Some(prototype.clone());
        assert!(builtin(vec![
            Value::from("like"),
            Value::GpuTensor(prototype.clone())
        ])
        .is_err());
        assert_eq!(allocation.probe.retirements(), 0);
        assert_eq!(prototype.descriptor(), descriptor);
        prototype.allocation_lease().validate_live().unwrap();
        let retained = provider.output.lock().unwrap().take();
        drop(retained);
        drop(prototype);
        assert_eq!(allocation.probe.retirements(), 1);
    }
    assert_eq!(provider.uploads.load(Ordering::Relaxed), 6);
}
