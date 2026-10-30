use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::COV_DESCRIPTOR;
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn cov_output_precision_tracks_observations() {
    let single = Tensor::new_with_dtype(
        vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0],
        vec![3, 2],
        runmat_value::NumericDType::F32,
    )
    .unwrap();
    let double = Tensor::new(vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0], vec![3, 2]).unwrap();
    let Value::Tensor(result) =
        cov_builtin_sync(Value::Tensor(single.clone()), Vec::new()).unwrap()
    else {
        panic!("expected tensor");
    };
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(result.shape(), [2, 2]);
    let Value::Tensor(result) =
        cov_builtin_sync(Value::Tensor(single), vec![Value::Tensor(double)]).unwrap()
    else {
        panic!("expected tensor");
    };
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F64);
}

fn assert_tensor_close(actual: &Tensor, expected: &[f64], tol: f64) {
    let dim = (expected.len() as f64).sqrt() as usize;
    assert_eq!(actual.shape(), [dim, dim], "unexpected tensor shape");
    for (idx, (&got, &want)) in actual
        .materialize_f64()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        if want.is_nan() {
            assert!(
                got.is_nan(),
                "expected NaN at linear index {idx}, found {got}"
            );
        } else {
            assert!(
                (got - want).abs() <= tol,
                "mismatch at linear index {idx}: got {got}, expected {want}"
            );
        }
    }
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Tensor {
    Tensor::new_integer(storage, shape).unwrap()
}

#[test]
fn cov_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = COV_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"C = cov(X)"));
    assert!(labels.contains(&"C = cov(X, normalization)"));
    assert!(labels.contains(&"C = cov(X, Y, w, opt)"));
}

fn cov_builtin_sync(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::cov_builtin(
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_matrix_basic() {
    let tensor = Tensor::new(
        vec![
            4.0, 4.2, 3.9, 4.3, 4.1, //
            2.0, 2.1, 2.0, 2.1, 2.2, //
            0.60, 0.59, 0.58, 0.62, 0.63,
        ],
        vec![5, 3],
    )
    .unwrap();
    let result = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("cov");
    let tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    let expected = [
        0.0250, 0.0075, 0.00175, //
        0.0075, 0.0070, 0.00135, //
        0.00175, 0.00135, 0.00043,
    ];
    assert_tensor_close(&tensor, &expected, 1.0e-6);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_two_vectors() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let y = Tensor::new(vec![10.0, 11.0, 9.0, 12.0], vec![4, 1]).unwrap();
    let result = block_on(cov_builtin(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(y)]),
    ))
    .expect("cov");
    let tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    let expected = [
        1.6666666666666667,
        0.6666666666666666, //
        0.6666666666666666,
        1.6666666666666667,
    ];
    assert_tensor_close(&tensor, &expected, 1.0e-6);
}

#[test]
fn cov_row_vector_scalar_empty_and_paired_matrix_shapes_match_contract() {
    let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let Value::Tensor(row_cov) = block_on(cov_builtin(
        Value::Tensor(row),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("row covariance") else {
        panic!("expected tensor");
    };
    assert_tensor_close(&row_cov, &[1.0], 1.0e-12);

    let scalar = Tensor::new(vec![7.0], vec![1, 1]).unwrap();
    let Value::Tensor(scalar_cov) = block_on(cov_builtin(
        Value::Tensor(scalar),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("scalar covariance") else {
        panic!("expected tensor");
    };
    assert_tensor_close(&scalar_cov, &[0.0], 1.0e-12);

    let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let Value::Tensor(empty_cov) = block_on(cov_builtin(
        Value::Tensor(empty),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("empty covariance") else {
        panic!("expected tensor");
    };
    assert_tensor_close(&empty_cov, &[f64::NAN], 0.0);

    let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let right = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let Value::Tensor(pair_cov) = block_on(cov_builtin(
        Value::Tensor(left),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(right)]),
    ))
    .expect("paired matrix covariance") else {
        panic!("expected tensor");
    };
    let expected = [5.0 / 3.0; 4];
    assert_tensor_close(&pair_cov, &expected, 1.0e-12);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_weighted_vector() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(
        vec![
            4.0, 4.2, 3.9, 4.3, 4.1, //
            2.0, 2.1, 2.0, 2.1, 2.2,
        ],
        vec![5, 2],
    )
    .unwrap();
    let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![5, 1]).unwrap();
    let result = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(weights)]),
    ))
    .expect("cov");
    let tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    let expected = [
        0.022380952380952376,
        0.004999999999999994, //
        0.004999999999999994,
        0.006666666666666678,
    ];
    assert_tensor_close(&tensor, &expected, 1.0e-6);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_accepts_typed_integer_matrix_and_weights() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = poisoned_int_tensor(
        IntegerStorage::I16(vec![
            4, 4, 3, 4, 4, //
            2, 2, 2, 2, 2,
        ]),
        vec![5, 2],
        f64::NAN,
    );
    let weights = poisoned_int_tensor(
        IntegerStorage::U16(vec![1, 1, 1, 2, 2]),
        vec![5, 1],
        f64::NAN,
    );
    let result = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(weights)]),
    ))
    .expect("cov");
    let tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    let expected = [
        1.0 / 7.0,
        0.0, //
        0.0,
        0.0,
    ];
    assert_tensor_close(&tensor, &expected, 1.0e-6);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_rejects_negative_typed_integer_weights() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2], 0.0);
    let weights = poisoned_int_tensor(IntegerStorage::I16(vec![1, -1]), vec![2, 1], 1.0);
    let err = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(weights)]),
    ))
    .expect_err("negative weights should fail");
    assert_eq!(err.identifier(), COV_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn cov_extensions_are_independently_gated_before_computation() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer_error = block_on(cov_builtin(
        Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        integer_error.identifier(),
        Some("RunMat:compatibility:CovIntegerDataExtension")
    );

    let logical_error = block_on(cov_builtin(
        Value::Bool(true),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        logical_error.identifier(),
        Some("RunMat:compatibility:CovLogicalDataExtension")
    );

    let data = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let normalization_error = block_on(cov_builtin(
        Value::Tensor(data.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U8(1))]),
    ))
    .unwrap_err();
    assert_eq!(
        normalization_error.identifier(),
        Some("RunMat:compatibility:CovTypedNormalizationExtension")
    );

    let weights = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let weight_error = block_on(cov_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(weights)]),
    ))
    .unwrap_err();
    assert_eq!(
        weight_error.identifier(),
        Some("RunMat:compatibility:CovVectorWeightsExtension")
    );
}

#[test]
fn cov_supports_all_eight_integer_classes_and_wide_centering() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = vec![
        IntegerStorage::I8(vec![1, 2, 3]),
        IntegerStorage::I16(vec![1, 2, 3]),
        IntegerStorage::I32(vec![1, 2, 3]),
        IntegerStorage::I64(vec![1, 2, 3]),
        IntegerStorage::U8(vec![1, 2, 3]),
        IntegerStorage::U16(vec![1, 2, 3]),
        IntegerStorage::U32(vec![1, 2, 3]),
        IntegerStorage::U64(vec![1, 2, 3]),
    ];
    for storage in storages {
        let class = storage.class_name();
        let data = Tensor::new_integer(storage.clone(), vec![3, 1]).unwrap();
        let Value::Tensor(covariance) = block_on(cov_builtin(
            Value::Tensor(data),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_or_else(|error| panic!("{class}: {error}")) else {
            panic!("{class}: expected tensor");
        };
        assert_tensor_close(&covariance, &[1.0], 1.0e-12);

        let left = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let right = Tensor::new_integer(storage.clone(), vec![3, 1]).unwrap();
        let Value::Tensor(pair) = block_on(cov_builtin(
            Value::Tensor(left),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(right)]),
        ))
        .unwrap_or_else(|error| panic!("{class} paired data: {error}")) else {
            panic!("{class}: expected paired tensor");
        };
        assert_tensor_close(&pair, &[1.0; 4], 1.0e-12);

        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]).unwrap();
        let weights = Tensor::new_integer(storage, vec![3, 1]).unwrap();
        let weighted = block_on(cov_builtin(
            Value::Tensor(matrix),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(weights)]),
        ))
        .unwrap_or_else(|error| panic!("{class} weights: {error}"));
        assert!(matches!(weighted, Value::Tensor(_)));
    }

    let base = 1_u64 << 63;
    let wide = Tensor::new_integer(
        IntegerStorage::U64(vec![base, base + 1, base + 2]),
        vec![3, 1],
    )
    .unwrap();
    let Value::Tensor(covariance) = block_on(cov_builtin(
        Value::Tensor(wide),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("wide covariance") else {
        panic!("expected tensor");
    };
    assert_tensor_close(&covariance, &[1.0], 1.0e-12);
}

#[test]
fn cov_resident_integer_data_rejects_before_provider_dispatch() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(cov_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:CovIntegerDataExtension")
        );
        drop(handle);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_omitrows() {
    let tensor = Tensor::new(
        vec![
            1.0,
            3.0,
            f64::NAN,
            8.0, //
            f64::NAN,
            4.0,
            6.0,
            9.0, //
            2.0,
            5.0,
            7.0,
            10.0,
        ],
        vec![4, 3],
    )
    .unwrap();
    let result = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("omitrows")]),
    ))
    .expect("cov");
    let tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    let expected = [
        12.5, 12.5, 12.5, //
        12.5, 12.5, 12.5, //
        12.5, 12.5, 12.5,
    ];
    assert_tensor_close(&tensor, &expected, 1.0e-6);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_partialrows() {
    let tensor = Tensor::new(
        vec![
            1.0,
            4.0,
            7.0, //
            2.0,
            f64::NAN,
            8.0, //
            f64::NAN,
            6.0,
            9.0,
        ],
        vec![3, 3],
    )
    .unwrap();
    let result = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("partialrows")]),
    ))
    .expect("cov");
    let tensor = match result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    let expected = [
        9.0,
        18.0,
        4.5, //
        18.0,
        18.0,
        f64::NAN, //
        4.5,
        f64::NAN,
        4.5,
    ];
    assert_tensor_close(&tensor, &expected, 1.0e-6);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_mismatched_rows_errors() {
    let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(cov_builtin(
        Value::Tensor(left),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(right)]),
    ))
    .expect_err("expected mismatch error");
    assert_eq!(err.identifier(), COV_ERROR_ROWS_MISMATCH.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_invalid_flag_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.5)]),
    ))
    .expect_err("expected invalid flag error");
    assert_eq!(err.identifier(), COV_ERROR_NORMALIZATION_INVALID.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_weight_vector_length_mismatch_errors() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    let y = Tensor::new(vec![10.0, 11.0, 12.0], vec![3, 1]).unwrap();
    let w = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = block_on(cov_builtin(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(y),
            Value::Tensor(w),
        ]),
    ))
    .expect_err("expected weight length mismatch");
    assert_eq!(
        err.identifier(),
        COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_unknown_rows_option_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("rows"),
            Value::from("bogus"),
        ]),
    ))
    .expect_err("expected unknown rows option error");
    assert_eq!(err.identifier(), COV_ERROR_ROWS_OPTION_UNKNOWN.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_duplicate_normalization_flag_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(cov_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("expected duplicate normalization flag error");
    assert_eq!(
        err.identifier(),
        COV_ERROR_NORMALIZATION_DUPLICATE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_too_many_array_arguments_errors() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
    let w = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
    let z = Tensor::new(vec![7.0, 8.0, 9.0], vec![3, 1]).unwrap();
    let err = block_on(cov_builtin(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(y),
            Value::Tensor(w),
            Value::Tensor(z),
        ]),
    ))
    .expect_err("expected too many array arguments error");
    assert_eq!(
        err.identifier(),
        COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = block_on(cov_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("cov");
        let gathered = test_support::gather(result).expect("gather");
        let expected = [
            0.0250, 0.0075, //
            0.0075, 0.0070,
        ];
        assert_tensor_close(&gathered, &expected, 1.0e-6);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_gpu_host_weights_return_resident_result() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![5, 1]).unwrap();

        let result = block_on(cov_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(weights)]),
        ))
        .expect("weighted cov");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather");
        let expected = [
            0.022380952380952376,
            0.004999999999999994, //
            0.004999999999999994,
            0.006666666666666678,
        ];
        assert_tensor_close(&gathered, &expected, 1.0e-6);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_gpu_resident_weights_return_resident_result() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();
        let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![1, 5]).unwrap();
        let data = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            })
            .expect("upload data");
        let weight_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &weights.materialize_f64(),
                shape: weights.shape(),
            })
            .expect("upload weights");

        let result = block_on(cov_builtin(
            Value::GpuTensor(data),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                weight_handle,
            )]),
        ))
        .expect("weighted cov");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather");
        let expected = [
            0.022380952380952376,
            0.004999999999999994, //
            0.004999999999999994,
            0.006666666666666678,
        ];
        assert_tensor_close(&gathered, &expected, 1.0e-6);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cov_gpu_rejects_negative_resident_weights() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let weights = Tensor::new(vec![1.0, -1.0], vec![2, 1]).unwrap();
        let data = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            })
            .expect("upload data");
        let weight_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &weights.materialize_f64(),
                shape: weights.shape(),
            })
            .expect("upload weights");

        let err = block_on(cov_builtin(
            Value::GpuTensor(data),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                weight_handle,
            )]),
        ))
        .expect_err("negative weights should fail");
        assert_eq!(err.identifier(), COV_ERROR_INVALID_ARGUMENT.identifier);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn cov_wgpu_matches_cpu() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions {
            precision: runmat_accelerate::backend::wgpu::provider::WgpuPrecisionPolicy::RequireF32,
            ..Default::default()
        },
    ) else {
        return;
    };

    // Centered binary fractions are exactly representable in F32. This
    // compares the supported single-precision route without making a broad
    // tolerance allowance for cancellation around an F64-sized mean.
    let tensor = Tensor::new_with_dtype(
        vec![
            -0.5, -0.25, 0.0, 0.25, 0.5, //
            0.25, 0.0, -0.25, 0.5, -0.5,
        ],
        vec![5, 2],
        runmat_value::NumericDType::F32,
    )
    .unwrap();

    let cpu_result = block_on(cov_builtin(
        Value::Tensor(tensor.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("cov");
    let cpu_tensor = match cpu_result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    assert_eq!(cpu_tensor.numeric_dtype(), runmat_value::NumericDType::F32);
    let expected = [0.15625, -0.0625, -0.0625, 0.15625];
    assert_tensor_close(&cpu_tensor, &expected, 0.0);

    let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("typed single upload");
    assert_eq!(
        runmat_accelerate_api::handle_precision(&handle),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );

    let gpu_value = cov_builtin_sync(Value::GpuTensor(handle), Vec::new()).expect("cov");
    assert!(matches!(&gpu_value, Value::GpuTensor(_)));
    let gathered = test_support::gather(gpu_value).expect("gather");
    assert_eq!(gathered.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_tensor_close(&gathered, &expected, 4.0 * f32::EPSILON as f64);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn cov_wgpu_noncentered_f32_cancellation_matches_quantized_oracle() {
    let _accel_guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions {
            precision: runmat_accelerate::backend::wgpu::provider::WgpuPrecisionPolicy::RequireF32,
            ..Default::default()
        },
    )
    .expect("this covariance witness requires a real WGPU adapter");
    for (data, n) in [
        (vec![4.0, 4.2, 3.9, 4.3, 4.1, 2.0, 2.1, 2.0, 2.1, 2.2], 5),
        (vec![1.0e20; 4], 2),
        (vec![1048575.0, 1048577.0, 1048575.0, 1048577.0], 2),
    ] {
        let tensor = Tensor::new_with_dtype(data, vec![n, 2], runmat_value::NumericDType::F32)
            .expect("typed single observations");
        let quantized = tensor.materialize_f64();
        let means = [
            quantized[..n].iter().sum::<f64>() / n as f64,
            quantized[n..].iter().sum::<f64>() / n as f64,
        ];
        let mut expected = [0.0; 4];
        for col in 0..2 {
            for row in 0..2 {
                expected[col * 2 + row] = (0..n)
                    .map(|sample| {
                        (quantized[row * n + sample] - means[row])
                            * (quantized[col * n + sample] - means[col])
                    })
                    .sum::<f64>()
                    / (n - 1) as f64;
            }
        }
        let cpu_result = cov_builtin_sync(Value::Tensor(tensor.clone()), Vec::new())
            .expect("host typed single covariance");
        let Value::Tensor(cpu_tensor) = cpu_result else {
            panic!("host covariance tensor")
        };
        assert_eq!(cpu_tensor.numeric_dtype(), runmat_value::NumericDType::F32);
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("typed single upload");
        assert_eq!(
            runmat_accelerate_api::handle_precision(&handle),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        let gpu_value = cov_builtin_sync(Value::GpuTensor(handle), Vec::new())
            .expect("resident noncentered covariance");
        assert!(matches!(&gpu_value, Value::GpuTensor(_)));
        let gathered = test_support::gather(gpu_value).expect("gather resident covariance");
        assert_eq!(gathered.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(gathered.shape(), [2, 2]);
        for ((host, got), want) in cpu_tensor
            .materialize_f64()
            .iter()
            .zip(gathered.materialize_f64())
            .zip(expected)
        {
            assert!(
                got.is_finite(),
                "finite observations must not overflow uncentered moments"
            );
            if want == 0.0 {
                assert_eq!(got, 0.0, "constant observations have exact zero covariance");
                assert_eq!(*host, 0.0);
            } else {
                // Error is relative to centered products, not the arbitrarily
                // large uncentered offset that the implementation subtracts.
                let tolerance = 32.0 * f32::EPSILON as f64 * want.abs();
                assert!((*host - want).abs() <= 2.0 * f32::EPSILON as f64 * want.abs());
                assert!(
                    (got - want).abs() <= tolerance,
                    "got {got}, quantized oracle {want}, centered bound {tolerance}"
                );
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn cov_wgpu_paired_matrices_vectorize_to_two_variables() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let right = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let left_handle = gpu_helpers::upload_tensor(provider, &left).expect("upload left");
    let right_handle = gpu_helpers::upload_tensor(provider, &right).expect("upload right");
    let result = cov_builtin_sync(
        Value::GpuTensor(left_handle),
        vec![Value::GpuTensor(right_handle)],
    )
    .expect("paired covariance");
    assert!(matches!(result, Value::GpuTensor(_)));
    let gathered = test_support::gather(result).expect("gather");
    assert_tensor_close(&gathered, &[5.0 / 3.0; 4], 1.0e-5);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn cov_wgpu_row_scalar_and_empty_match_host_shapes() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };

    for (data, shape, expected) in [
        (vec![1.0, 2.0, 3.0], vec![1, 3], 1.0),
        (vec![7.0], Vec::new(), 0.0),
        (Vec::new(), vec![0, 0], f64::NAN),
    ] {
        let tensor = Tensor::new(data, shape).expect("tensor");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = cov_builtin_sync(Value::GpuTensor(handle), Vec::new()).expect("covariance");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather");
        assert_tensor_close(&gathered, &[expected], 1.0e-6);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn cov_wgpu_weighted_matches_cpu_and_stays_resident() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };

    let tensor = Tensor::new(
        vec![
            4.0, 4.2, 3.9, 4.3, 4.1, //
            2.0, 2.1, 2.0, 2.1, 2.2,
        ],
        vec![5, 2],
    )
    .unwrap();
    let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![1, 5]).unwrap();

    let cpu_result = block_on(cov_builtin(
        Value::Tensor(tensor.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            weights.clone(),
        )]),
    ))
    .expect("cov");
    let cpu_tensor = match cpu_result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };

    let data_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        })
        .expect("upload data");
    let weight_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &weights.materialize_f64(),
            shape: weights.shape(),
        })
        .expect("upload weights");

    let gpu_value = cov_builtin_sync(
        Value::GpuTensor(data_handle),
        vec![Value::GpuTensor(weight_handle)],
    )
    .expect("weighted cov");
    assert!(matches!(gpu_value, Value::GpuTensor(_)));
    let gathered = test_support::gather(gpu_value).expect("gather");

    assert_tensor_close(&gathered, &cpu_tensor.materialize_f64(), 1.0e-5);
}
