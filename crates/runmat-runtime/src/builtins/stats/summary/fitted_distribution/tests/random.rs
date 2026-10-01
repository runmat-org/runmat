use super::*;

#[test]
fn random_restores_parameter_residency_but_not_size_control_residency() {
    use crate::builtins::common::test_support;

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let parameter = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1])
            .expect("integer parameter");
        let parameter = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
        let parameter = parameter
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let output = block_on(random_builtin(
            Value::String("Normal".into()),
            vec![
                Value::GpuTensor(parameter),
                Value::Num(1.0),
                Value::Num(2.0),
            ],
        ))
        .expect("resident random output");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident random output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
        let gathered = test_support::gather(output).expect("gather random output");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert!(gathered
            .materialize_f64()
            .iter()
            .all(|value| value.is_finite()));

        let size =
            Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).expect("integer size");
        let size = gpu_helpers::upload_tensor(provider, &size).expect("upload size");
        let size = size
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let output = block_on(random_builtin(
            Value::String("Normal".into()),
            vec![Value::Num(0.0), Value::Num(1.0), Value::GpuTensor(size)],
        ))
        .expect("resident size control");
        assert!(matches!(output, Value::Tensor(_)));
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn random_wgpu_parameter_residency_enforces_double_for_all_integer_classes() {
    use crate::builtins::common::test_support;

    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_cdf_integer_storages(0) {
        let parameter = Tensor::new_integer(storage, vec![1, 1]).expect("integer parameter");
        let handle = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let result = block_on(random_builtin(
            Value::String("Normal".into()),
            vec![Value::GpuTensor(handle), Value::Num(1.0), Value::Num(2.0)],
        ));
        let output = result.expect("resident random output");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident random output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
        assert_eq!(
            runmat_accelerate_api::handle_precision(output_handle),
            Some(ProviderPrecision::F64)
        );
        let gathered = test_support::gather(output).expect("gather random output");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert!(gathered
            .materialize_f64()
            .iter()
            .all(|value| value.is_finite()));
    }
}
