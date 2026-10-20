use super::*;

#[test]
#[cfg(feature = "wgpu")]
fn pdf_wgpu_fallback_preserves_residency_for_all_integer_classes() {
    use crate::builtins::common::test_support;

    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_cdf_integer_storages(1) {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer x");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let result = block_on(pdf_builtin(
            Value::String("Normal".into()),
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(1.0),
            ]),
        ))
        .expect("resident integer pdf");
        let Value::GpuTensor(output) = &result else {
            panic!("expected resident pdf output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output));
        let gathered = test_support::gather(result).expect("gather result");
        assert!((gathered.materialize_f64()[0] - 0.241_970_724_519_143_37).abs() < 1.0e-12);
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn cdf_wgpu_fallback_preserves_residency_for_all_integer_classes() {
    use crate::builtins::common::test_support;

    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_cdf_integer_storages(1) {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer x");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let result = block_on(cdf_builtin(
            Value::String("Normal".into()),
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(1.0),
            ]),
        ))
        .expect("resident integer cdf");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather result");
        assert!((gathered.materialize_f64()[0] - 0.841_344_746_068_543).abs() < 1.0e-12);
    }
}
