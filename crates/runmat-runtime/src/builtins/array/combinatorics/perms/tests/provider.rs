use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::IntegerStorage;

#[test]
fn resident_shape_and_size_errors_precede_provider_access() {
    for (shape, expected) in [
        (vec![2, 2], "RunMat:perms:InvalidInput"),
        (vec![1, 2, 1], "RunMat:perms:InvalidInput"),
        (vec![1, 11], "RunMat:perms:TooLarge"),
    ] {
        let allocation_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let handle = allocation_owner
            .allocate_zeroed(
                shape,
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle;
        let error = call(Value::GpuTensor(handle))
            .expect_err("shape admission before missing owner lookup");
        assert_eq!(error.identifier(), Some(expected));
    }
}

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
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated fixture provenance");
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
        let handle = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1, 1],
                shape: &[1, 3],
            })
            .expect("upload");
        let input_lease = handle.allocation_lease().downgrade();
        let output =
            call(gpu_helpers::logical_gpu_value(handle.clone()).expect("adopt logical producer"))
                .expect("logical perms");
        let Value::GpuTensor(logical_output) = &output else {
            panic!("expected resident logical output")
        };
        assert!(runmat_accelerate_api::handle_is_logical(logical_output));
        let output_lease = logical_output.allocation_lease().downgrade();
        let gathered = test_support::gather(output).expect("logical payload");
        assert_eq!(gathered.shape(), [6, 3]);
        assert_eq!(tensor_rows(&gathered)[0], vec![1.0, 1.0, 0.0]);
        assert!(input_lease.acquire().is_some());
        drop(handle);
        assert!(input_lease.acquire().is_none());
        assert!(output_lease.acquire().is_none());

        let complex = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -2.0), (3.0, 0.5)], vec![1, 3])
            .expect("complex");
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let input_lease = handle.allocation_lease().downgrade();
        let output = call(gpu_helpers::complex_gpu_value(handle).expect("adopt complex producer"))
            .expect("complex perms");
        let Value::GpuTensor(handle) = &output else {
            panic!("expected resident complex output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let output_lease = handle.allocation_lease().downgrade();
        let Value::ComplexTensor(gathered) =
            block_on(gpu_helpers::gather_value_async(&output)).expect("gather")
        else {
            panic!("expected complex tensor")
        };
        assert_eq!(
            complex_rows(&gathered)[0],
            vec![(3.0, 0.5), (2.0, -2.0), (1.0, 1.0)]
        );
        drop(output);
        assert!(input_lease.acquire().is_none());
        assert!(output_lease.acquire().is_none());
    });
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a real WGPU device; run explicitly with --ignored"]
fn wgpu_fallback_preserves_every_integer_class_and_explicit_residency() {
    let _accel_guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU device required for this explicit hardware witness");
    let provider: &dyn runmat_accelerate_api::AccelProvider = provider;
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
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated fixture provenance");
        let output = call(Value::GpuTensor(handle.clone())).expect("resident perms");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident output")
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
        let output_handle = output_handle.clone();
        let gathered = test_support::gather(output).expect("gather");
        assert_eq!(gathered.integer_storage(), Some(&expected));
        output_handle.release().unwrap();
        handle.release().unwrap();
    }
}
