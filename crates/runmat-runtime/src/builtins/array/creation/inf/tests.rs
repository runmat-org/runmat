use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorStorage;

fn clear_accel_provider_state() -> test_support::AccelTestGuard {
    test_support::accel_test_lock()
}

fn assert_all_pos_inf(tensor: &Tensor) {
    assert!(tensor
        .materialize_f64()
        .iter()
        .all(|value| value.is_infinite() && value.is_sign_positive()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_default_scalar() {
    let _guard = clear_accel_provider_state();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("inf");
    match result {
        Value::Num(value) => assert!(value.is_infinite() && value.is_sign_positive()),
        other => panic!("expected scalar Inf, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_square_from_single_dimension() {
    let _guard = clear_accel_provider_state();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
    ))
    .expect("inf");
    let tensor = test_support::gather(result).expect("gather tensor");
    assert_eq!(tensor.shape(), vec![3, 3]);
    assert_all_pos_inf(&tensor);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_rectangular_from_dims() {
    let _guard = clear_accel_provider_state();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(4.0),
        ]),
    ))
    .expect("inf");
    let tensor = test_support::gather(result).expect("gather tensor");
    assert_eq!(tensor.shape(), vec![2, 4]);
    assert_all_pos_inf(&tensor);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_from_size_vector() {
    let _guard = clear_accel_provider_state();
    let size_vec = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(size_vec)]),
    ))
    .expect("inf");
    let tensor = test_support::gather(result).expect("gather tensor");
    assert_eq!(tensor.shape(), vec![2, 3, 4]);
    assert_all_pos_inf(&tensor);
}

#[test]
fn inf_integer_dimensions_clamp_negative_and_normalize_trailing_singletons() {
    let _guard = clear_accel_provider_state();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(runmat_value::IntValue::I16(-2)),
            Value::Int(runmat_value::IntValue::U8(3)),
            Value::Int(runmat_value::IntValue::U64(1)),
        ]),
    ))
    .expect("inf integer dimensions");
    let Value::Tensor(tensor) = result else {
        panic!("expected empty host tensor");
    };
    assert_eq!(tensor.shape(), vec![0, 3]);
}

#[test]
fn inf_without_explicit_gpu_intent_remains_host_resident() {
    test_support::with_test_provider(|_| {
        let result = block_on(inf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
        ))
        .expect("inf");
        assert!(matches!(result, Value::Tensor(tensor) if tensor.shape() == vec![2, 2]));
    });
}

#[test]
fn inf_column_size_vector_follows_compatibility_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let size = Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![2, 3]), vec![2, 1])
        .expect("column size vector");
    let error = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(size)]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:InfColumnSizeVectorExtension")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_single_output_marks_dtype() {
    let _guard = clear_accel_provider_state();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("single"),
        ]),
    ))
    .expect("inf");
    let tensor = test_support::gather(result).expect("gather tensor");
    assert_eq!(tensor.shape(), vec![2, 2]);
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    assert_all_pos_inf(&tensor);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_implicit_prototype_is_rejected() {
    let _guard = clear_accel_provider_state();
    let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    assert!(block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(proto)])
    ))
    .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_like_complex_scalar() {
    let _guard = clear_accel_provider_state();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ]),
    ))
    .expect("inf");
    match result {
        Value::ComplexTensor(tensor) => {
            assert_eq!(tensor.shape(), vec![3, 3]);
            assert!(tensor
                .materialize_f64()
                .iter()
                .all(|(re, im)| re.is_infinite() && re.is_sign_positive() && *im == 0.0));
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn inf_like_complex_single_preserves_native_single() {
    let prototype = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).expect("complex single");
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::ComplexTensor(prototype),
        ]),
    ))
    .expect("inf complex single like");
    let Value::ComplexTensor(output) = result else {
        panic!("expected complex tensor");
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_like_uses_shape_argument_when_combined_with_like() {
    let _guard = clear_accel_provider_state();
    let shape_source = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let proto = Tensor::new_with_dtype(vec![7.0, 8.0], vec![1, 2], NumericDType::F32).unwrap();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(shape_source),
            Value::from("like"),
            Value::Tensor(proto),
        ]),
    ))
    .expect("inf");
    let tensor = test_support::gather(result).expect("gather tensor");
    assert_eq!(tensor.shape(), vec![2, 3]);
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    assert_all_pos_inf(&tensor);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_like_without_explicit_shape_returns_scalar() {
    let _guard = clear_accel_provider_state();
    let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("like"),
            Value::Tensor(proto),
        ]),
    ))
    .expect("inf");
    assert!(matches!(result, Value::Num(value) if value.is_infinite() && value.is_sign_positive()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_like_single_sparse_preserves_sparse_single() {
    let prototype = SparseTensor::zeros_f32(1, 1);
    let result = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::SparseTensor(prototype),
        ]),
    ))
    .expect("inf single sparse like");
    let Value::SparseTensor(output) = result else {
        panic!("expected sparse tensor");
    };
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
    assert!(output
        .as_f32_slice()
        .expect("single storage")
        .iter()
        .all(|value| value.is_infinite() && value.is_sign_positive()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_rejects_every_integer_like_prototype() {
    let _guard = clear_accel_provider_state();
    let prototypes = [
        runmat_value::IntegerStorage::I8(vec![0]),
        runmat_value::IntegerStorage::I16(vec![0]),
        runmat_value::IntegerStorage::I32(vec![0]),
        runmat_value::IntegerStorage::I64(vec![0]),
        runmat_value::IntegerStorage::U8(vec![0]),
        runmat_value::IntegerStorage::U16(vec![0]),
        runmat_value::IntegerStorage::U32(vec![0]),
        runmat_value::IntegerStorage::U64(vec![0]),
    ];

    for storage in prototypes {
        let proto = Tensor::new_integer(storage, vec![1, 1]).expect("integer prototype");
        let err = block_on(inf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::Tensor(proto),
            ]),
        ))
        .unwrap_err();
        assert!(err.message().contains("integer 'like' prototypes"));
    }

    let prototype = SparseTensor::zeros_with_integer_storage(
        1,
        1,
        &runmat_value::IntegerStorage::U64(Vec::new()),
    );
    let err = block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::SparseTensor(prototype),
        ]),
    ))
    .expect_err("integer sparse like");
    assert!(err.message().contains("integer 'like' prototypes"));
}

#[test]
fn inf_rejects_resident_integer_gpu_like_prototype() {
    test_support::with_test_provider(|provider| {
        let values = [1_u64];
        let prototype = provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
                shape: &[1, 1],
            })
            .expect("integer prototype");
        let err = block_on(inf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(prototype),
            ]),
        ))
        .expect_err("integer gpu like");
        assert!(err.message().contains("integer 'like' prototypes"));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_conflicting_like_and_class_is_error() {
    let _guard = clear_accel_provider_state();
    let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let args = vec![
        Value::Num(2.0),
        Value::from("single"),
        Value::from("like"),
        Value::Tensor(proto),
    ];
    assert!(block_on(inf_builtin(
        crate::builtins::common::test_support::builtin_arguments(args)
    ))
    .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inf_gpu_like_alloc() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = block_on(inf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ]),
        ))
        .expect("inf");
        match result {
            Value::GpuTensor(gpu) => {
                assert_eq!(gpu.shape(), vec![2, 2]);
                let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                assert_all_pos_inf(&gathered);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    });
}

#[test]
fn inf_same_shape_complex_gpu_like_stays_complex_and_resident() {
    test_support::with_f32_test_provider(|provider| {
        let prototype = ComplexTensor::from_f32(vec![(1.0, -1.0); 4], vec![2, 2])
            .expect("complex single prototype");
        for provenance in [
            runmat_accelerate_api::GpuHandleProvenance::Automatic,
            runmat_accelerate_api::GpuHandleProvenance::Explicit,
        ] {
            let handle = gpu_helpers::upload_complex_tensor(provider, &prototype)
                .expect("upload")
                .with_provenance(provenance)
                .expect("prototype provenance");
            let result = block_on(inf_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::GpuTensor(handle),
                ]),
            ))
            .expect("inf complex gpu like");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            assert_eq!(
                runmat_accelerate_api::handle_provenance(&output),
                Some(provenance)
            );
        }
    });
}
