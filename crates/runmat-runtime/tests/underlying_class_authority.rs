use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostTensorView, NumericElementType, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::Value;

struct MetadataProvider {
    owner: TestAllocationOwner,
    preferred: ProviderPrecision,
}

impl AccelProvider for MetadataProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        self.preferred
    }
    fn device_info(&self) -> String {
        "metadata-only owner".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("class queries must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("class queries must not download")
    }
}

fn assert_class(handle: &GpuTensorHandle, expected: &str) {
    let args = [Value::GpuTensor(handle.clone())];
    let result = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "underlyingType",
        &args,
        1,
    ))
    .unwrap();
    assert_eq!(result, Value::from(expected));
    let args = [Value::GpuTensor(handle.clone()), Value::from(expected)];
    assert_eq!(
        block_on(runmat_runtime::call_builtin_async_with_outputs(
            "isUnderlyingType",
            &args,
            1
        ))
        .unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn native_allocation_class_is_independent_of_owner_preference_without_transfers() {
    use NumericElementType::*;
    for preferred in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let provider = Box::leak(Box::new(MetadataProvider {
            owner: TestAllocationOwner::new(),
            preferred,
        }));
        let _guard = ThreadProviderGuard::set(Some(provider));
        for (element, class) in [
            (F32, "single"),
            (F64, "double"),
            (I8, "int8"),
            (I16, "int16"),
            (I32, "int32"),
            (I64, "int64"),
            (U8, "uint8"),
            (U16, "uint16"),
            (U32, "uint32"),
            (U64, "uint64"),
        ] {
            for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
                let allocation = provider
                    .owner
                    .allocate_zeroed(vec![1, 2], element, storage)
                    .unwrap();
                assert_class(&allocation.handle, class);
                assert_eq!(allocation.probe.retirements(), 0);
                drop(allocation.handle);
                assert_eq!(allocation.probe.retirements(), 1);
            }
        }
        let logical = provider
            .owner
            .allocate_logical(vec![0, 1], vec![1, 2])
            .unwrap();
        assert_class(&logical.handle, "logical");
        drop(logical.handle);
        assert_eq!(logical.probe.retirements(), 1);
    }
}
