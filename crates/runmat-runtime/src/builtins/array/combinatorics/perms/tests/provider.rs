use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::IntegerStorage;

#[test]
fn provider_outputs_restore_owner_class_and_explicit_residency() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
            vec![1, 2],
        )
        .expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &input)
            .expect("upload")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let output = call(Value::GpuTensor(handle)).expect("resident perms");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident output")
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(output_handle),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let gathered = test_support::gather(output).expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_994,
                9_007_199_254_740_993,
                9_007_199_254_740_993,
                9_007_199_254_740_994,
            ]))
        );
    });
}

#[test]
fn logical_and_complex_provider_values_restore_their_storage() {
    test_support::with_test_provider(|provider| {
        let logical = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 3]).expect("logical");
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &logical.materialize_f64(),
                shape: &logical.shape,
            })
            .expect("upload");
        let output = call(gpu_helpers::logical_gpu_value(handle)).expect("logical perms");
        let Value::GpuTensor(handle) = &output else {
            panic!("expected resident logical output")
        };
        assert!(runmat_accelerate_api::handle_is_logical(handle));

        let complex = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -2.0), (3.0, 0.5)], vec![1, 3])
            .expect("complex");
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let output = call(gpu_helpers::complex_gpu_value(handle)).expect("complex perms");
        let Value::GpuTensor(handle) = &output else {
            panic!("expected resident complex output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let Value::ComplexTensor(output) =
            block_on(gpu_helpers::gather_value_async(&output)).expect("gather")
        else {
            panic!("expected complex tensor")
        };
        assert_eq!(
            complex_rows(&output)[0],
            vec![(3.0, 0.5), (2.0, -2.0), (1.0, 1.0)]
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_fallback_preserves_every_integer_class_and_explicit_residency() {
    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let cases = [
        IntegerStorage::I8(vec![1, 2]),
        IntegerStorage::I16(vec![1, 2]),
        IntegerStorage::I32(vec![1, 2]),
        IntegerStorage::I64(vec![1, 2]),
        IntegerStorage::U8(vec![1, 2]),
        IntegerStorage::U16(vec![1, 2]),
        IntegerStorage::U32(vec![1, 2]),
        IntegerStorage::U64(vec![1, 2]),
    ];
    for input in cases {
        let values = input.exact_values();
        let expected = input
            .from_exact_values_like(vec![
                values[1].clone(),
                values[0].clone(),
                values[0].clone(),
                values[1].clone(),
            ])
            .expect("expected storage");
        let tensor = Tensor::new_integer(input, vec![1, 2]).expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &tensor)
            .expect("upload")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let output = call(Value::GpuTensor(handle)).expect("resident perms");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident output")
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
        let gathered = test_support::gather(output).expect("gather");
        assert_eq!(gathered.integer_storage(), Some(&expected));
    }
}
