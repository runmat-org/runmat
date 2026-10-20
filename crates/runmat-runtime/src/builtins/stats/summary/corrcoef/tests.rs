use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::{BuiltinOutputMode, CORRCOEF_DESCRIPTOR};
use runmat_value::{
    ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor, Value,
};

#[test]
fn single_observations_keep_all_four_output_classes() {
    let observations = Tensor::new_with_dtype(
        vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 5.0],
        vec![4, 2],
        runmat_value::NumericDType::F32,
    )
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let Value::OutputList(outputs) = block_on(corrcoef_builtin(
        Value::Tensor(observations),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Alpha"),
            Value::Num(0.1),
        ]),
    ))
    .unwrap() else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 4);
    for output in outputs.try_into_owned().expect("unique output list") {
        let Value::Tensor(tensor) = output else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(tensor.shape(), [2, 2]);
    }
}

#[test]
fn mixed_observation_precision_promotes_corrcoef_result() {
    let single = Tensor::new_with_dtype(
        vec![1.0, 2.0, 3.0],
        vec![3, 1],
        runmat_value::NumericDType::F32,
    )
    .unwrap();
    let double = Tensor::new(vec![3.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let Value::Tensor(result) = block_on(corrcoef_builtin(
        Value::Tensor(single),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(double)]),
    ))
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F64);
    assert_eq!(result.shape(), [2, 2]);
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Tensor {
    Tensor::new_integer(storage, shape).unwrap()
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

#[test]
fn corrcoef_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = CORRCOEF_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"R = corrcoef(A)"));
    assert!(labels.contains(&"R = corrcoef(A, B)"));
    assert!(labels.contains(&"[R, P, RL, RU] = corrcoef(A)"));
    assert!(labels.contains(&"___ = corrcoef(A, Name, Value)"));
    assert_eq!(
        CORRCOEF_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_matrix_basic() {
    let tensor = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            2.0, 4.0, 6.0, 8.0, //
            4.0, 1.0, -1.0, 0.0,
        ],
        vec![4, 3],
    )
    .unwrap();
    let result = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("corrcoef");
    match result {
        Value::Tensor(out) => {
            let expected = [
                1.0,
                1.0,
                -0.836_660_026_534,
                1.0,
                1.0,
                -0.836_660_026_534,
                -0.836_660_026_534,
                -0.836_660_026_534,
                1.0,
            ];
            assert_tensor_close(&out, &expected, 1.0e-10);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn corrcoef_accepts_typed_integer_matrix_inputs() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 1, 4]), vec![2, 2], 0.0);
    let result = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("corrcoef");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [2, 2]);
            assert!((out.materialize_f64()[0] - 1.0).abs() < 1.0e-12);
            assert!((out.materialize_f64()[3] - 1.0).abs() < 1.0e-12);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn corrcoef_complete_rows_reads_typed_integer_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = poisoned_int_tensor(
        IntegerStorage::I16(vec![1, 2, 3, 2, 4, 6]),
        vec![3, 2],
        f64::NAN,
    );
    let y = poisoned_int_tensor(
        IntegerStorage::U16(vec![5, 10, 15, 10, 20, 30]),
        vec![3, 2],
        f64::NAN,
    );

    let result = block_on(corrcoef_builtin(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(y),
            Value::from("Rows"),
            Value::from("complete"),
        ]),
    ))
    .expect("corrcoef complete rows");

    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [2, 2]);
            for value in out.materialize_f64() {
                assert!((value - 1.0).abs() < 1.0e-12);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_two_inputs_vectorize_equal_size_matrices() {
    let left = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            4.0, 5.0, 6.0, 7.0,
        ],
        vec![4, 2],
    )
    .unwrap();
    let right = Tensor::new(
        vec![
            8.0, 6.0, 7.0, 5.0, //
            2.0, 9.0, 1.0, 3.0,
        ],
        vec![4, 2],
    )
    .unwrap();
    let combined = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 6.0, 7.0, 5.0, //
            2.0, 9.0, 1.0, 3.0,
        ],
        vec![8, 2],
    )
    .unwrap();

    let via_two = block_on(corrcoef_builtin(
        Value::Tensor(left.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            right.clone(),
        )]),
    ))
    .expect("corrcoef");
    let via_combined = block_on(corrcoef_builtin(
        Value::Tensor(combined),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("corrcoef combined");

    let expected_tensor = match via_combined {
        Value::Tensor(t) => t,
        _ => panic!("expected tensor output"),
    };
    let actual_tensor = match via_two {
        Value::Tensor(t) => t,
        _ => panic!("expected tensor output"),
    };
    assert_tensor_close(&actual_tensor, &expected_tensor.materialize_f64(), 1.0e-10);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_rows_complete_ignores_missing() {
    let tensor = Tensor::new(
        vec![
            1.0,
            f64::NAN,
            3.0,
            4.0, //
            2.0,
            5.0,
            f64::NAN,
            8.0,
        ],
        vec![4, 2],
    )
    .unwrap();
    let result = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("rows"),
            Value::from("complete"),
        ]),
    ))
    .expect("corrcoef");
    match result {
        Value::Tensor(out) => {
            let expected = [
                1.0, 1.0, //
                1.0, 1.0,
            ];
            assert_tensor_close(&out, &expected, 1.0e-10);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_rows_options_do_not_treat_infinity_as_missing() {
    for rows in ["complete", "pairwise"] {
        let tensor = Tensor::new(vec![1.0, f64::INFINITY, 3.0, 1.0, 2.0, 3.0], vec![3, 2]).unwrap();
        let result = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("rows"),
                Value::from(rows),
            ]),
        ))
        .expect("corrcoef");
        let Value::Tensor(out) = result else {
            panic!("expected tensor result");
        };
        let values = out.materialize_f64();
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert!(values[2].is_nan());
        assert_eq!(values[3], 1.0);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_rows_pairwise_staggered_missing() {
    let tensor = Tensor::new(
        vec![
            1.0,
            f64::NAN,
            4.0,
            5.0, //
            2.0,
            5.0,
            f64::NAN,
            8.0, //
            3.0,
            1.0,
            6.0,
            f64::NAN,
        ],
        vec![4, 3],
    )
    .unwrap();
    let result = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("rows"),
            Value::from("pairwise"),
        ]),
    ))
    .expect("corrcoef");
    match result {
        Value::Tensor(out) => {
            let expected = [
                1.0, 1.0, 1.0, //
                1.0, 1.0, -1.0, //
                1.0, -1.0, 1.0,
            ];
            assert_tensor_close(&out, &expected, 1.0e-10);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_numeric_scalar_second_argument_is_data() {
    let result = block_on(corrcoef_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    ))
    .expect("scalar pair");
    let Value::Tensor(result) = result else {
        panic!("expected tensor")
    };
    assert_tensor_close(&result, &[1.0], 0.0);

    let equal = block_on(corrcoef_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    ))
    .expect("equal scalar pair");
    let Value::Tensor(equal) = equal else {
        panic!("expected tensor")
    };
    assert_tensor_close(&equal, &[f64::NAN], 0.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn corrcoef_public_scalar_vector_and_paired_empty_geometry() {
    for shape in [vec![], vec![1, 1]] {
        let scalar = Tensor::new(vec![7.0], shape).unwrap();
        let Value::Tensor(scalar_result) = block_on(corrcoef_builtin(
            Value::Tensor(scalar),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("scalar") else {
            panic!("expected tensor")
        };
        assert_tensor_close(&scalar_result, &[f64::NAN], 0.0);
    }

    for shape in [vec![4], vec![1, 4], vec![4, 1]] {
        let vector = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], shape).unwrap();
        let Value::Tensor(vector_result) = block_on(corrcoef_builtin(
            Value::Tensor(vector),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("vector") else {
            panic!("expected tensor")
        };
        assert_tensor_close(&vector_result, &[1.0], 0.0);
    }

    let left = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let right = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let Value::Tensor(empty) = block_on(corrcoef_builtin(
        Value::Tensor(left),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(right)]),
    ))
    .expect("paired empty") else {
        panic!("expected tensor")
    };
    assert_tensor_close(&empty, &[f64::NAN; 4], 0.0);

    let trailing_singleton = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1, 1]).unwrap();
    let error = block_on(corrcoef_builtin(
        Value::Tensor(trailing_singleton),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("corrcoef admits only rank-zero, vector, and matrix inputs");
    assert_eq!(
        error.identifier(),
        CORRCOEF_ERROR_INVALID_ARGUMENT.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn corrcoef_low_rank_inputs_keep_statistics_outputs_canonical() {
    for (values, shape) in [(vec![7.0], Vec::new()), (vec![1.0, 2.0, 4.0], vec![3])] {
        let _outputs = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
        let Value::OutputList(outputs) = block_on(corrcoef_builtin(
            Value::Tensor(Tensor::new(values, shape).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("low-rank four-output corrcoef") else {
            panic!("expected output list")
        };
        assert_eq!(outputs.len(), 4);
        for output in outputs.try_into_owned().expect("unique output list") {
            let Value::Tensor(output) = output else {
                panic!("statistics output must be a tensor")
            };
            assert_eq!(output.shape(), &[1, 1]);
        }
    }
}

#[test]
fn corrcoef_supports_all_integer_classes_and_exact_wide_centering() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = vec![
        IntegerStorage::I8(vec![1, 2, 3, 3, 2, 1]),
        IntegerStorage::I16(vec![1, 2, 3, 3, 2, 1]),
        IntegerStorage::I32(vec![1, 2, 3, 3, 2, 1]),
        IntegerStorage::I64(vec![1, 2, 3, 3, 2, 1]),
        IntegerStorage::U8(vec![1, 2, 3, 3, 2, 1]),
        IntegerStorage::U16(vec![1, 2, 3, 3, 2, 1]),
        IntegerStorage::U32(vec![1, 2, 3, 3, 2, 1]),
        IntegerStorage::U64(vec![1, 2, 3, 3, 2, 1]),
    ];
    for storage in storages {
        let class = storage.class_name();
        let tensor = Tensor::new_integer(storage, vec![3, 2]).unwrap();
        let Value::Tensor(result) = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_or_else(|error| panic!("{class}: {error}")) else {
            panic!("{class}: expected tensor")
        };
        assert_tensor_close(&result, &[1.0, -1.0, -1.0, 1.0], 1.0e-12);
    }

    let base = u64::MAX - 2;
    let wide = Tensor::new_integer(
        IntegerStorage::U64(vec![base, base + 1, base + 2, base + 2, base + 1, base]),
        vec![3, 2],
    )
    .unwrap();
    let Value::Tensor(result) = block_on(corrcoef_builtin(
        Value::Tensor(wide),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("wide corrcoef") else {
        panic!("expected tensor")
    };
    assert_tensor_close(&result, &[1.0, -1.0, -1.0, 1.0], 1.0e-12);
}

#[test]
fn corrcoef_integer_and_logical_extensions_gate_before_dispatch() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]).unwrap();
    let integer_error = block_on(corrcoef_builtin(
        Value::Tensor(integer),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        integer_error.identifier(),
        Some("RunMat:compatibility:CorrcoefIntegerDataExtension")
    );

    let logical = LogicalArray::new(vec![0, 1, 1], vec![3, 1]).unwrap();
    let logical_error = block_on(corrcoef_builtin(
        Value::LogicalArray(logical),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        logical_error.identifier(),
        Some("RunMat:compatibility:CorrcoefLogicalDataExtension")
    );
}

#[test]
fn corrcoef_resident_integer_gate_precedes_provider_dispatch() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(corrcoef_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:CorrcoefIntegerDataExtension")
        );
        drop(handle);
    });
}

#[test]
fn resident_complex_observations_preserve_host_complex_correlation() {
    test_support::with_test_provider(|provider| {
        let input = ComplexTensor::new(
            vec![
                (1.0, 0.0),
                (2.0, 0.0),
                (3.0, 0.0),
                (0.0, 1.0),
                (0.0, 2.0),
                (0.0, 3.0),
            ],
            vec![3, 2],
        )
        .unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
        let Value::ComplexTensor(result) = block_on(corrcoef_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("admitted complex materialization") else {
            panic!("complex correlation must retain its output class");
        };
        assert_eq!(result.shape(), [2, 2]);
        let values = result.materialize_f64();
        assert_eq!(values[0], (1.0, 0.0));
        assert_eq!(values[3], (1.0, 0.0));
        assert!(values[1].0.abs() < 1e-12 && (values[1].1 + 1.0).abs() < 1e-12);
        assert!(values[2].0.abs() < 1e-12 && (values[2].1 - 1.0).abs() < 1e-12);
        assert!(handle.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn corrcoef_complex_coefficients_are_hermitian() {
    let tensor = ComplexTensor::new(
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (0.0, 3.0),
        ],
        vec![3, 2],
    )
    .unwrap();
    let Value::ComplexTensor(result) = block_on(corrcoef_builtin(
        Value::ComplexTensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("complex corrcoef") else {
        panic!("expected complex tensor")
    };
    assert_eq!(result.shape(), [2, 2]);
    let values = result.materialize_f64();
    assert_eq!(values[0], (1.0, 0.0));
    assert!((values[1].0).abs() < 1.0e-12);
    assert!((values[1].1 + 1.0).abs() < 1.0e-12);
    assert!((values[2].0).abs() < 1.0e-12);
    assert!((values[2].1 - 1.0).abs() < 1.0e-12);
    assert_eq!(values[3], (1.0, 0.0));
}

#[test]
fn corrcoef_complex_integer_components_center_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = u64::MAX - 2;
    let storage = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![base, base + 1, base + 2]),
        IntegerStorage::U64(vec![base + 2, base + 1, base]),
    )
    .unwrap();
    let tensor = ComplexTensor::new_integer(storage, vec![3, 1]).unwrap();
    let Value::ComplexTensor(result) = block_on(corrcoef_builtin(
        Value::ComplexTensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("complex integer corrcoef") else {
        panic!("expected complex tensor")
    };
    let values = result.materialize_f64();
    assert_eq!(result.shape(), [1, 1]);
    assert!((values[0].0 - 1.0).abs() < 1.0e-12);
    assert!(values[0].1.abs() < 1.0e-12);
}

#[test]
fn corrcoef_requested_outputs_include_p_values_and_alpha_bounds() {
    let tensor = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
            1.0, 2.0, 1.0, 3.0, 5.0, 4.0,
        ],
        vec![6, 2],
    )
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let Value::OutputList(outputs) = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Alpha"),
            Value::Num(0.1),
        ]),
    ))
    .expect("four outputs") else {
        panic!("expected output list")
    };
    assert_eq!(outputs.len(), 4);
    let tensors = outputs
        .into_iter()
        .map(|value| match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        })
        .collect::<Vec<_>>();
    for tensor in &tensors {
        assert_eq!(tensor.shape(), [2, 2]);
    }
    let p = tensors[1].materialize_f64();
    assert_eq!(p[0], 1.0);
    assert_eq!(p[3], 1.0);
    assert!(p[1] > 0.0 && p[1] < 1.0);
    let r = tensors[0].materialize_f64()[1];
    let lower = tensors[2].materialize_f64()[1];
    let upper = tensors[3].materialize_f64()[1];
    assert!(lower < r && r < upper);
}

#[test]
fn corrcoef_complex_multi_output_is_rejected() {
    let tensor = ComplexTensor::new(vec![(1.0, 1.0), (2.0, 0.0), (3.0, -1.0)], vec![3, 1]).unwrap();
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(corrcoef_builtin(
        Value::ComplexTensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        CORRCOEF_ERROR_COMPLEX_OUTPUTS.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                2.0, 4.0, 6.0, 8.0, //
                4.0, 1.0, -1.0, 0.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = block_on(corrcoef_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("corrcoef");
        let gathered = test_support::gather(result).expect("gather");
        let expected = [
            1.0,
            1.0,
            -0.836_660_026_534,
            1.0,
            1.0,
            -0.836_660_026_534,
            -0.836_660_026_534,
            -0.836_660_026_534,
            1.0,
        ];
        assert_tensor_close(&gathered, &expected, 1.0e-10);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_mismatched_sizes_error() {
    let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(corrcoef_builtin(
        Value::Tensor(left),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(right)]),
    ))
    .expect_err("expected mismatch error");
    assert_eq!(err.identifier(), CORRCOEF_ERROR_SIZE_MISMATCH.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_invalid_alpha_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Alpha"),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("expected invalid Alpha error");
    assert_eq!(err.identifier(), CORRCOEF_ERROR_ALPHA_INVALID.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_unknown_rows_option_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("rows"),
            Value::from("bogus"),
        ]),
    ))
    .expect_err("expected unknown rows option error");
    assert_eq!(
        err.identifier(),
        CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_duplicate_alpha_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Alpha"),
            Value::Num(0.05),
            Value::from("Alpha"),
            Value::Num(0.1),
        ]),
    ))
    .expect_err("expected duplicate Alpha error");
    assert_eq!(err.identifier(), CORRCOEF_ERROR_OPTION_DUPLICATE.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_unknown_option_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("bogus"),
            Value::from("value"),
        ]),
    ))
    .expect_err("expected unknown option error");
    assert_eq!(err.identifier(), CORRCOEF_ERROR_OPTION_UNKNOWN.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_rows_option_malformed_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let non_string_option = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = block_on(corrcoef_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("rows"),
            Value::Tensor(non_string_option),
        ]),
    ))
    .expect_err("expected malformed rows option error");
    assert_eq!(
        err.identifier(),
        CORRCOEF_ERROR_ROWS_OPTION_MALFORMED.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn corrcoef_too_many_input_arrays_errors() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
    let c = Tensor::new(vec![7.0, 8.0, 9.0], vec![3, 1]).unwrap();
    let err = block_on(corrcoef_builtin(
        Value::Tensor(a),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(b),
            Value::Tensor(c),
        ]),
    ))
    .expect_err("expected too many input arrays error");
    assert_eq!(
        err.identifier(),
        CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn corrcoef_wgpu_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let tensor = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            2.0, 5.0, 6.0, 8.0, //
            4.0, 1.0, 7.0, 0.0,
        ],
        vec![4, 3],
    )
    .unwrap();
    let cpu = corrcoef_from_tensors(
        tensor.clone(),
        None,
        CorrcoefNormalization::Unbiased,
        CorrcoefRows::All,
    )
    .expect("cpu corrcoef");
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let options = CorrcoefOptions {
        normalization: CorrcoefNormalization::Unbiased,
        rows: CorrcoefRows::All,
    };
    let gpu = block_on(provider.corrcoef(&handle, &options)).expect("corrcoef");
    let gathered = test_support::gather(Value::GpuTensor(gpu.clone())).expect("gather");
    assert_tensor_close(&gathered, &cpu.materialize_f64(), 1.0e-6);
    drop(gpu);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn corrcoef_wgpu_public_vector_and_paired_geometry() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };

    let row = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![1, 4]).unwrap();
    let row_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &row.materialize_f64(),
            shape: row.shape(),
        })
        .expect("row upload");
    let row_result = block_on(corrcoef_builtin(
        Value::GpuTensor(row_handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("row corrcoef");
    let row_gathered = test_support::gather(row_result).expect("row gather");
    assert_tensor_close(&row_gathered, &[1.0], 1.0e-6);
    drop(row_handle);

    let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let right = Tensor::new(vec![4.0, 1.0, 2.0, 8.0], vec![2, 2]).unwrap();
    let expected = corrcoef_from_tensors(
        left.clone(),
        Some(right.clone()),
        CorrcoefNormalization::Unbiased,
        CorrcoefRows::All,
    )
    .expect("host pair");
    let left_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &left.materialize_f64(),
            shape: left.shape(),
        })
        .expect("left upload");
    let right_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &right.materialize_f64(),
            shape: right.shape(),
        })
        .expect("right upload");
    let paired = block_on(corrcoef_builtin(
        Value::GpuTensor(left_handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            right_handle.clone(),
        )]),
    ))
    .expect("paired corrcoef");
    let paired_gathered = test_support::gather(paired).expect("paired gather");
    assert_tensor_close(&paired_gathered, &expected.materialize_f64(), 1.0e-6);
    let left_after =
        test_support::gather(Value::GpuTensor(left_handle.clone())).expect("left after corrcoef");
    let right_after =
        test_support::gather(Value::GpuTensor(right_handle.clone())).expect("right after corrcoef");
    assert_eq!(left_after.shape(), [2, 2]);
    assert_eq!(right_after.shape(), [2, 2]);
    drop(left_handle);
    drop(right_handle);
}
