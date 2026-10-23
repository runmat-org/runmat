use super::*;

#[test]
fn retained_gpu_rhs_restores_native_result_to_original_explicit_owner() {
    use crate::builtins::common::gpu_helpers;
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_language_policy(crate::context::RuntimeLanguagePolicy {
            language_mode: crate::context::RuntimeLanguageMode::RunMat,
            extensions_enabled: true,
            top_level_await_enabled: false,
            dynamic_eval_enabled: false,
        });
        for single in [false, true] {
            for complex in [false, true] {
                let source = if complex {
                    let rhs = if single {
                        ComplexTensor::from_f32(vec![(4.0, 5.0), (7.0, 5.0)], vec![2, 1]).unwrap()
                    } else {
                        ComplexTensor::new(vec![(4.0, 5.0), (7.0, 5.0)], vec![2, 1]).unwrap()
                    };
                    gpu_helpers::upload_complex_tensor(provider, &rhs).unwrap()
                } else {
                    let rhs = if single {
                        Tensor::from_f32(vec![4.0, 7.0], vec![2, 1]).unwrap()
                    } else {
                        Tensor::new(vec![4.0, 7.0], vec![2, 1]).unwrap()
                    };
                    gpu_helpers::upload_tensor(provider, &rhs).unwrap()
                }
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
                for kind in ["lu", "chol", "qr", "cod"] {
                    let matrix = if single {
                        Value::Tensor(
                            Tensor::from_f32(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap(),
                        )
                    } else {
                        tensor(&[2.0, 1.0, 1.0, 3.0], 2, 2)
                    };
                    let d = block_on(crate::call_builtin_async_in_context(
                        &runtime,
                        "decomposition",
                        &[matrix, Value::from(kind)],
                    ))
                    .unwrap();
                    for _ in 0..2 {
                        let value = block_on(crate::call_builtin_async_in_context(
                            &runtime,
                            "decomposition.mldivide",
                            &[d.value().clone(), Value::GpuTensor(source.clone())],
                        ))
                        .unwrap();
                        let Value::GpuTensor(output) = value.value() else {
                            panic!("resident RHS must retain resident publication")
                        };
                        assert_eq!(output.shape(), [2, 1]);
                        assert_eq!(
                            runmat_accelerate_api::handle_provenance(output),
                            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                        );
                        assert!(!output.shares_allocation_with(&source));
                        assert!(gpu_helpers::exact_provider_for_handle(output).is_some_and(
                            |owner| runmat_accelerate_api::same_provider_instance(owner, provider)
                        ));
                        assert_eq!(
                            runmat_accelerate_api::handle_precision(output),
                            Some(if single {
                                runmat_accelerate_api::ProviderPrecision::F32
                            } else {
                                runmat_accelerate_api::ProviderPrecision::F64
                            })
                        );
                        let actual =
                            block_on(gpu_helpers::gather_value_async(value.value())).unwrap();
                        let actual = match actual {
                            Value::Tensor(t) => t
                                .materialize_f64()
                                .into_iter()
                                .map(|v| (v, 0.0))
                                .collect::<Vec<_>>(),
                            Value::ComplexTensor(t) => t.materialize_f64(),
                            _ => panic!("native vector result expected"),
                        };
                        for (actual, expected) in actual.iter().zip([
                            (1.0, if complex { 2.0 } else { 0.0 }),
                            (2.0, if complex { 1.0 } else { 0.0 }),
                        ]) {
                            assert!((actual.0 - expected.0).abs() < 3e-5);
                            assert!((actual.1 - expected.1).abs() < 3e-5);
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn retained_public_native_factors_preserve_complex_rhs_and_cross_precision() {
    fn pointer(value: &Value) -> usize {
        use runmat_value::{NativeDenseFactorMatrix as M, NumericFactorCandidate as C};
        let spec = object_to_spec(value).unwrap();
        let data = spec.factor_data.as_ref().unwrap();
        let matrix = match data.parts() {
            C::DenseLu { packed, .. } => packed,
            C::DenseCholesky { lower } => lower,
            C::DenseQr(first) | C::DenseCod { first, .. } => &first.packed,
            _ => panic!("dense retained data expected"),
        };
        match matrix {
            M::Real(t) => {
                if let Some(v) = t.as_f32_slice() {
                    v.as_ptr() as usize
                } else {
                    t.as_f64_slice().unwrap().as_ptr() as usize
                }
            }
            _ => panic!("real factor expected"),
        }
    }
    fn check(value: Value, expected: &[(f64, f64)]) {
        let Value::ComplexTensor(t) = value else {
            panic!("complex RHS must retain its imaginary solution")
        };
        assert_eq!(t.numeric_dtype(), NumericDType::F32);
        for (actual, expected) in t.materialize_f64().iter().zip(expected) {
            assert!((actual.0 - expected.0).abs() < 3e-5);
            assert!((actual.1 - expected.1).abs() < 3e-5);
        }
    }
    for single_factor in [false, true] {
        for kind in ["lu", "chol", "qr", "cod"] {
            let matrix = if single_factor {
                Value::Tensor(Tensor::from_f32(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap())
            } else {
                tensor(&[2.0, 1.0, 1.0, 3.0], 2, 2)
            };
            let d = call_constructor(vec![matrix, Value::from(kind)]).unwrap();
            let original_pointer = pointer(&d);
            let rhs = |shape| {
                if single_factor {
                    Value::ComplexTensor(
                        ComplexTensor::new(vec![(0.0, 5.0), (-5.0, 5.0)], shape).unwrap(),
                    )
                } else {
                    Value::ComplexTensor(
                        ComplexTensor::from_f32(vec![(0.0, 5.0), (-5.0, 5.0)], shape).unwrap(),
                    )
                }
            };
            check(
                call_mldivide(d.clone(), rhs(vec![2, 1])).unwrap(),
                &[(1.0, 2.0), (-2.0, 1.0)],
            );
            let scaled = call_mtimes(Value::Complex(2.0, -1.0), d.clone()).unwrap();
            assert_eq!(
                pointer(&scaled),
                original_pointer,
                "scaling must share the original factor payload"
            );
            check(
                call_mldivide(scaled, rhs(vec![2, 1])).unwrap(),
                &[(0.0, 1.0), (-1.0, 0.0)],
            );
            if kind == "qr" {
                assert!(call_mrdivide(rhs(vec![1, 2]), d.clone()).is_err());
                assert!(
                    call_mldivide(call_ctranspose(d.clone()).unwrap(), rhs(vec![2, 1])).is_err()
                );
            } else {
                check(
                    call_mrdivide(rhs(vec![1, 2]), d.clone()).unwrap(),
                    &[(1.0, 2.0), (-2.0, 1.0)],
                );
                check(
                    call_mldivide(call_ctranspose(d.clone()).unwrap(), rhs(vec![2, 1])).unwrap(),
                    &[(1.0, 2.0), (-2.0, 1.0)],
                );
            }
            assert_eq!(pointer(&d), original_pointer);
        }
    }
}

#[test]
fn retained_public_qr_basic_and_cod_minimum_norm_are_distinct() {
    for single in [false, true] {
        for complex in [false, true] {
            // A has row [1+i,2,0] (or [1,2,0]) and twice that row.
            // COD's independent A^H*b/||row||² oracle differs from a QR
            // basic solution, whose non-pivot variables are exactly zero.
            let imag = if complex { 1.0 } else { 0.0 };
            let entries = [
                (1.0, imag),
                (2.0, 2.0 * imag),
                (2.0, 0.0),
                (4.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ];
            let matrix = match (single, complex) {
                (false, false) => tensor(&entries.map(|v| v.0), 2, 3),
                (true, false) => Value::Tensor(
                    Tensor::from_f32(entries.map(|v| v.0 as f32).to_vec(), vec![2, 3]).unwrap(),
                ),
                (false, true) => complex_tensor(&entries, 2, 3),
                (true, true) => Value::ComplexTensor(
                    ComplexTensor::from_f32(
                        entries.map(|v| (v.0 as f32, v.1 as f32)).to_vec(),
                        vec![2, 3],
                    )
                    .unwrap(),
                ),
            };
            for kind in ["qr", "cod"] {
                let d = call_constructor(vec![matrix.clone(), Value::from(kind)]).unwrap();
                let result = call_mldivide(d, tensor(&[6.0, 12.0], 2, 1)).unwrap();
                let actual = match result {
                    Value::Tensor(t) => t
                        .materialize_f64()
                        .into_iter()
                        .map(|v| (v, 0.0))
                        .collect::<Vec<_>>(),
                    Value::ComplexTensor(t) => t.materialize_f64(),
                    _ => panic!("vector solution expected"),
                };
                let expected = if kind == "qr" {
                    [(0.0, 0.0), (3.0, 0.0), (0.0, 0.0)]
                } else {
                    let norm = 5.0 + imag * imag;
                    [
                        (6.0 / norm, -6.0 * imag / norm),
                        (12.0 / norm, 0.0),
                        (0.0, 0.0),
                    ]
                };
                let tolerance = if single { 3e-5 } else { 3e-12 };
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!((actual.0 - expected.0).abs() < tolerance);
                    assert!((actual.1 - expected.1).abs() < tolerance);
                }
            }
        }
    }
}

#[test]
fn all_integer_classes_and_logical_coefficients_solve_only_in_runmat_mode() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_integer_coefficient_storages() {
        let dtype = storage.numeric_dtype();
        let class = storage.class_name();
        let d = call_constructor(vec![typed_int_tensor(storage, vec![2, 2])])
            .unwrap_or_else(|error| panic!("{class} construction failed: {error}"));
        let stored = object_to_spec(&d).expect("stored decomposition").matrix;
        assert!(
            matches!(stored, Value::Tensor(ref tensor) if tensor.integer_storage().is_some_and(|storage| storage.numeric_dtype() == dtype)),
            "{class} must remain exact before solve"
        );
        let result = call_mldivide(d, tensor(&[8.0, 12.0], 2, 1))
            .unwrap_or_else(|error| panic!("{class} solve failed: {error}"));
        assert_eq!(approx_vec(result), vec![4.0, 3.0], "{class}");
    }

    let logical = Value::LogicalArray(
        LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical identity"),
    );
    let d = call_constructor(vec![logical]).expect("logical decomposition");
    assert!(matches!(
        object_to_spec(&d).expect("stored decomposition").matrix,
        Value::LogicalArray(_)
    ));
    assert_eq!(
        approx_vec(call_mldivide(d, tensor(&[8.0, 12.0], 2, 1)).expect("logical solve")),
        vec![8.0, 12.0]
    );
}

#[test]
fn integer_decomposition_right_solve_promotes_at_the_solve_boundary() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let d = call_constructor(vec![typed_int_tensor(
        IntegerStorage::U64(vec![2, 0, 0, 4]),
        vec![2, 2],
    )])
    .expect("integer decomposition");
    let result = call_mrdivide(tensor(&[8.0, 12.0], 1, 2), d).expect("right solve");
    let Value::Tensor(result) = result else {
        panic!("expected promoted double tensor")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F64);
    assert_eq!(result.materialize_f64(), vec![4.0, 3.0]);
}

#[test]
fn decomposition_left_solve_reuses_mldivide_semantics() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");

    let result = call_mldivide(d, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 4.0).abs() < 1e-12);
    assert!((values[1] - 3.0).abs() < 1e-12);
}

#[test]
fn decomposition_right_solve_uses_rhs_object() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 1, 2);
    let d = call_constructor(vec![a]).expect("decomposition");

    let result = call_mrdivide(b, d).expect("right solve");

    let values = approx_vec(result);
    assert!((values[0] - 4.0).abs() < 1e-12);
    assert!((values[1] - 3.0).abs() < 1e-12);

    // Integer RHS support follows the same scalar-real-double boundary as
    // ordinary division; retained factorization must not broaden it.
    let integer_rhs = || Value::Int(IntValue::I32(8));
    let real = call_constructor(vec![Value::Num(2.0)]).unwrap();
    assert!(matches!(
        call_mrdivide(integer_rhs(), real).unwrap(),
        Value::Int(IntValue::I32(4))
    ));
    let complex = || {
        call_constructor(vec![Value::ComplexTensor(
            ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap(),
        )])
        .unwrap()
    };
    assert!(call_mldivide(complex(), integer_rhs()).is_err());
    assert!(call_mrdivide(integer_rhs(), complex()).is_err());
}

#[test]
fn scalar_multiply_and_divide_adjust_solve_matrix() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");
    let scaled = call_mtimes(Value::Num(2.0), d).expect("scale");
    let divided = call_rdivide(scaled, Value::Num(4.0)).expect("divide");

    let result = call_mldivide(divided, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 8.0).abs() < 1e-12);
    assert!((values[1] - 6.0).abs() < 1e-12);
}

#[test]
fn ctranspose_toggles_property_and_solves_transposed_system() {
    let a = tensor(&[1.0, 0.0, 2.0, 1.0], 2, 2);
    let b = tensor(&[3.0, 1.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");
    let dt = call_ctranspose(d).expect("ctranspose");

    let prop = block_on(decomposition_subsref(
        dt.clone(),
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(IS_CONJUGATE_TRANSPOSED_FIELD.to_string().into()),
    ))
    .expect("property");
    assert_eq!(prop, Value::Bool(true));

    let result = call_mldivide(dt, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 3.0).abs() < 1e-12);
    assert!((values[1] + 5.0).abs() < 1e-12);
}

#[test]
fn complex_decomposition_solves_complex_rhs() {
    let a = complex_tensor(&[(2.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 2.0)], 2, 2);
    let b = complex_tensor(&[(8.0, 2.0), (4.0, 2.0)], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");

    let result = call_mldivide(d, b).expect("complex solve");

    match result {
        Value::ComplexTensor(tensor) => {
            assert!((tensor.materialize_f64()[0].0 - 4.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[0].1 - 1.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[1].0 - 1.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[1].1 + 2.0).abs() < 1e-12);
        }
        other => panic!("expected complex result, got {other:?}"),
    }
}

#[test]
fn decomposition_scaling_preserves_complex_single_storage_through_solve() {
    let coefficient = Value::ComplexTensor(
        ComplexTensor::from_f32(
            vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (4.0, -1.0)],
            vec![2, 2],
        )
        .expect("single coefficient"),
    );
    let rhs = Value::ComplexTensor(
        ComplexTensor::from_f32(vec![(8.0, 2.0), (12.0, -3.0)], vec![2, 1]).expect("single rhs"),
    );
    let d = call_constructor(vec![coefficient]).expect("decomposition");
    let scaled = call_mtimes(Value::Complex(2.0, -1.0), d).expect("scale");
    let Value::ComplexTensor(result) = call_mldivide(scaled, rhs).expect("solve") else {
        panic!("expected complex tensor")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);

    let real_single = Value::Tensor(
        Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("single matrix"),
    );
    let Value::ComplexTensor(scaled) =
        scale_matrix(real_single, (1.0, 1.0)).expect("complex scale")
    else {
        panic!("expected complex tensor")
    };
    assert_eq!(scaled.numeric_dtype(), NumericDType::F32);
}
