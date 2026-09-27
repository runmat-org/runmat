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
    output: Mutex<Option<GpuTensorHandle>>,
    uploads: AtomicUsize,
    frees: AtomicUsize,
}
static PROVIDER: AliasProvider = AliasProvider {
    output: Mutex::new(None),
    uploads: AtomicUsize::new(0),
    frees: AtomicUsize::new(0),
};
impl AccelProvider for AliasProvider {
    fn device_id(&self) -> u32 {
        u32::MAX - 236
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "numeric-limit-alias".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        Ok(self.output.lock().unwrap().as_ref().unwrap().clone())
    }
    fn upload_numeric_raw(&self, _: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.upload(&HostTensorView {
            data: &[],
            shape: &[0, 0],
        })
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("limit creation must not download its prototype")
    }
    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        self.frees.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn limit_uploads_reject_borrowed_aliases_without_freeing_prototypes() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let _provider = ThreadProviderGuard::set(Some(&PROVIDER));
    PROVIDER.frees.store(0, Ordering::Relaxed);
    PROVIDER.uploads.store(0, Ordering::Relaxed);
    for (builtin, element) in [
        (
            intmax as fn(Vec<Value>) -> BuiltinResult<Value>,
            NumericElementType::U64,
        ),
        (intmin, NumericElementType::I64),
        (realmax, NumericElementType::F64),
        (realmin, NumericElementType::F32),
        (flintmax, NumericElementType::F64),
    ] {
        let prototype = GpuTensorHandle::new(vec![1, 1], PROVIDER.device_id(), 1)
            .with_numeric_descriptor(element, GpuTensorStorage::Real);
        let metadata = crate::builtins::common::gpu_helpers::snapshot_handle_metadata(&prototype);
        *PROVIDER.output.lock().unwrap() = Some(prototype.clone());
        assert!(builtin(vec![
            Value::from("like"),
            Value::GpuTensor(prototype.clone())
        ])
        .is_err());
        assert_eq!(PROVIDER.frees.load(Ordering::Relaxed), 0);
        assert_eq!(
            crate::builtins::common::gpu_helpers::snapshot_handle_metadata(&prototype),
            metadata
        );
    }
    assert_eq!(PROVIDER.uploads.load(Ordering::Relaxed), 5);
}
