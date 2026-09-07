use super::*;

#[cfg(feature = "wgpu")]
#[test]
fn bsxfun_integer_wgpu_fallback_gathers_exactly_and_returns_host_storage() {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("provider");
    let source = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![2, 1],
    )
    .expect("source");
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
        .expect("integer upload");
    let result = call(
        Value::FunctionHandle("plus".to_string()),
        Value::GpuTensor(handle),
        Value::Int(IntValue::U64(1)),
    )
    .expect("gather fallback");
    let Value::Tensor(result) = result else {
        panic!("current fallback must return host tensor, got {result:?}");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_994, u64::MAX]))
    );
}
