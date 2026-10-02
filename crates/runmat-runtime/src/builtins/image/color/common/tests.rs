use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{IntValue, IntegerStorage};

#[test]
fn image_scalar_collapse_reads_authoritative_storage() {
    let single = Tensor::from_f32(vec![0.25], vec![1, 1]).unwrap();
    assert_eq!(image_value_from_tensor(single), Value::Num(0.25));

    let wide = u64::MAX - 1;
    let integer = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
    assert_eq!(
        image_value_from_tensor(integer),
        Value::Int(IntValue::U64(wide))
    );
}

#[test]
fn image_arrays_remain_tensors() {
    let tensor = Tensor::from_f32(vec![0.25, 0.5], vec![1, 2]).unwrap();
    let Value::Tensor(output) = image_value_from_tensor(tensor) else {
        panic!("expected tensor output");
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), vec![0.25, 0.5]);
}

#[test]
fn rejects_legacy_global_provider_fallback_for_foreign_device_handle() {
    test_support::with_test_provider(|provider| {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F32 => {
                        runmat_accelerate_api::NumericElementType::F32
                    }
                    runmat_accelerate_api::ProviderPrecision::F64 => {
                        runmat_accelerate_api::NumericElementType::F64
                    }
                },
                GpuTensorStorage::Real,
            )
            .expect("admit owned fixture")
            .handle;
        let error = resident_numeric_dtype(&handle, "color-test")
            .expect_err("global fallback is not exact ownership");
        assert!(error.message().contains("not owned"));
    });
}

#[test]
fn native_single_restore_does_not_depend_on_preferred_double_precision() {
    test_support::with_test_provider(|provider| {
        let source_tensor = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &source_tensor).unwrap();
        let single = Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap());

        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("publish provenance view");
        assert!(matches!(
            restore_resident_numeric_result(Some(&source), single.clone(), "color-test").unwrap(),
            Value::GpuTensor(_)
        ));

        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let result = restore_resident_numeric_result(Some(&source), single, "color-test")
            .expect("owner supports actual native Single storage");
        let Value::GpuTensor(output) = result else {
            panic!("resident output")
        };
        assert_eq!(
            output.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        source.release().unwrap();
    });
}
