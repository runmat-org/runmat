use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

#[test]
fn provider_output_alias_rejection_keeps_borrowed_input_alive() {
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[4.0, 0.0, 0.0, 9.0],
                shape: &[2, 2],
            })
            .unwrap();
        let correlation = input
            .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![2, 2]))
            .unwrap();
        let sigma = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[2.0, 3.0],
                shape: &[2, 1],
            })
            .unwrap();
        let sigma_weak = sigma.downgrade();
        let result = CovarianceGpuEval::from_provider(
            "corrcov",
            provider,
            &input,
            ProviderCovarianceToCorrelationResult { correlation, sigma },
        );
        assert!(result.is_err());
        assert!(input.allocation_lease().validate_live().is_ok());
        assert!(sigma_weak.acquire().is_err());
    });
}

#[test]
fn resident_output_arity_keeps_only_returned_allocation_leases() {
    test_support::with_test_provider(|provider| {
        for (kind, name) in [
            (ConversionKind::Corrcov, "corrcov"),
            (ConversionKind::Cov2corr, "cov2corr"),
        ] {
            for count in [Some(0), Some(1), Some(2), Some(3), None] {
                let correlation = provider
                    .upload(&runmat_accelerate_api::HostTensorView {
                        data: &[1.0, 0.0, 0.0, 1.0],
                        shape: &[2, 2],
                    })
                    .unwrap();
                let sigma = provider
                    .upload(&runmat_accelerate_api::HostTensorView {
                        data: &[1.0, 1.0],
                        shape: &[2, 1],
                    })
                    .unwrap();
                // Publication may mint a provenance/shape view. The owned
                // resource is the physical allocation, not that old view.
                let correlation_weak = correlation.allocation_lease().downgrade();
                let sigma_weak = sigma.allocation_lease().downgrade();
                let _outputs =
                    crate::output_count::push_output_count(count).expect("output-count scope");
                let result = CovarianceGpuEval {
                    provider,
                    correlation,
                    sigma,
                }
                .output(kind, name);
                assert_eq!(result.is_err(), count == Some(3));
                let returns_first = matches!(count, None | Some(1 | 2));
                let returns_second = count == Some(2);
                let (keeps_correlation, keeps_sigma) = match kind {
                    ConversionKind::Corrcov => (returns_first, returns_second),
                    ConversionKind::Cov2corr => (returns_second, returns_first),
                };
                assert_eq!(correlation_weak.acquire().is_some(), keeps_correlation);
                assert_eq!(sigma_weak.acquire().is_some(), keeps_sigma);
                drop(result);
                assert!(correlation_weak.acquire().is_none());
                assert!(sigma_weak.acquire().is_none());
            }
        }
    });
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-10,
        "expected {expected}, got {actual}"
    );
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

fn assert_tensor_values(tensor: &Tensor, expected: &[f64]) {
    let values = tensor.materialize_f64();
    assert_eq!(values.len(), expected.len());
    for (actual, expected) in values.into_iter().zip(expected.iter().copied()) {
        assert_close(actual, expected);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn covariance_matrix_extents_preserve_rank_zero_and_vector_projection() {
    assert_eq!(matrix_validation::covariance_matrix_extents(&[]), (0, 0));
    assert_eq!(matrix_validation::covariance_matrix_extents(&[4]), (1, 4));
    assert_eq!(
        matrix_validation::covariance_matrix_extents(&[2, 3]),
        (2, 3)
    );

    let rank_zero = Value::Tensor(Tensor::new(vec![4.0], vec![]).expect("rank-zero tensor"));
    let Value::Tensor(result) = block_on(corrcov::corrcov_builtin(
        rank_zero,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("rank-zero covariance") else {
        panic!("expected tensor result")
    };
    assert_eq!(result.shape(), [0, 0]);

    let rank_one =
        Value::Tensor(Tensor::new(vec![4.0, 2.0, 2.0, 9.0], vec![4]).expect("rank-one tensor"));
    let error = block_on(corrcov::corrcov_builtin(
        rank_one,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("rank-one covariance must retain row-vector geometry");
    assert!(error.message().contains("must be square"));
}

#[test]
fn corrcov_converts_covariance_and_returns_sigma() {
    let c = Value::Tensor(Tensor::new(vec![4.0, 2.0, 2.0, 9.0], vec![2, 2]).expect("tensor"));
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(corrcov::corrcov_builtin(
        c,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("corrcov");
    match out {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 2);
            match &values[0] {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), [2, 2]);
                    assert_tensor_values(tensor, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
                }
                other => panic!("expected correlation tensor, got {other:?}"),
            }
            match &values[1] {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), [2, 1]);
                    assert_tensor_values(tensor, &[2.0, 3.0]);
                }
                other => panic!("expected sigma tensor, got {other:?}"),
            }
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn corrcov_preserves_native_single_outputs() {
    let covariance = Value::Tensor(
        Tensor::from_f32(vec![4.0, 2.0, 2.0, 9.0], vec![2, 2]).expect("single covariance"),
    );
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let Value::OutputList(outputs) = block_on(corrcov::corrcov_builtin(
        covariance,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("corrcov") else {
        panic!("expected output list");
    };
    let [Value::Tensor(correlation), Value::Tensor(sigma)] = outputs.as_slice() else {
        panic!("expected tensor outputs");
    };
    assert_eq!(correlation.numeric_dtype(), NumericDType::F32);
    assert_eq!(sigma.numeric_dtype(), NumericDType::F32);
    let third = f64::from(1.0_f32 / 3.0);
    assert_tensor_values(correlation, &[1.0, third, third, 1.0]);
    assert_tensor_values(sigma, &[2.0, 3.0]);
}

#[test]
fn corrcov_gpu_input_returns_resident_outputs() {
    test_support::with_test_provider(|provider| {
        let covariance = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[4.0, 2.0, 2.0, 9.0],
                shape: &[2, 2],
            })
            .expect("upload covariance");
        provider.reset_telemetry();

        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let out = block_on(corrcov::corrcov_builtin(
            Value::GpuTensor(covariance),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("corrcov gpu");
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.upload_bytes, 0);
        assert_eq!(telemetry.download_bytes, 0);
        assert!(matches!(values[0], Value::GpuTensor(_)));
        assert!(matches!(values[1], Value::GpuTensor(_)));

        let correlation = test_support::gather(values[0].clone()).expect("correlation");
        let sigma = test_support::gather(values[1].clone()).expect("sigma");
        assert_eq!(correlation.shape(), [2, 2]);
        assert_eq!(sigma.shape(), [2, 1]);
        assert_tensor_values(&correlation, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
        assert_tensor_values(&sigma, &[2.0, 3.0]);
    });
}

#[test]
fn cov2corr_gpu_input_preserves_invalid_covariance_error() {
    test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let covariance = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0, 0.1, 0.2, 1.0],
                shape: &[2, 2],
            })
            .expect("upload covariance");

        let err = block_on(cov2corr::cov2corr_builtin(
            Value::GpuTensor(covariance),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("invalid covariance");

        assert_eq!(err.identifier(), Some("RunMat:cov2corr:InvalidArgument"));
        assert!(err.message().contains("symmetric"));
    });
}

#[test]
fn cov2corr_gpu_extension_returns_finance_output_order() {
    test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let covariance = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[4.0, 2.0, 2.0, 9.0],
                shape: &[2, 2],
            })
            .expect("upload covariance");
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let Value::OutputList(outputs) = block_on(cov2corr::cov2corr_builtin(
            Value::GpuTensor(covariance),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("cov2corr gpu") else {
            panic!("expected output list");
        };
        let sigma = test_support::gather(outputs[0].clone()).expect("sigma");
        let correlation = test_support::gather(outputs[1].clone()).expect("correlation");
        assert_eq!(sigma.shape(), [1, 2]);
        assert_tensor_values(&sigma, &[2.0, 3.0]);
        assert_eq!(correlation.shape(), [2, 2]);
        assert_tensor_values(&correlation, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
    });
}

#[test]
fn cov2corr_returns_finance_output_order_and_shapes() {
    let c = Value::Tensor(Tensor::new(vec![16.0, 4.0, 4.0, 25.0], vec![2, 2]).expect("tensor"));
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let Value::OutputList(outputs) = block_on(cov2corr::cov2corr_builtin(
        c,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("cov2corr") else {
        panic!("expected output list");
    };
    let [Value::Tensor(sigma), Value::Tensor(correlation)] = outputs.as_slice() else {
        panic!("expected tensor outputs");
    };
    assert_eq!(sigma.shape(), [1, 2]);
    assert_tensor_values(sigma, &[4.0, 5.0]);
    assert_eq!(correlation.shape(), [2, 2]);
    assert_tensor_values(correlation, &[1.0, 0.2, 0.2, 1.0]);
}

#[test]
fn covariance_conversion_preserves_empty_shapes_and_cov2corr_default_output() {
    let empty = || Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor"));
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let Value::OutputList(outputs) = block_on(corrcov::corrcov_builtin(
            empty(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("corrcov empty") else {
            panic!("expected output list");
        };
        let [Value::Tensor(correlation), Value::Tensor(sigma)] = outputs.as_slice() else {
            panic!("expected tensor outputs");
        };
        assert_eq!(correlation.shape(), [0, 0]);
        assert_eq!(sigma.shape(), [0, 1]);
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let Value::OutputList(outputs) = block_on(cov2corr::cov2corr_builtin(
            empty(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("cov2corr empty") else {
            panic!("expected output list");
        };
        let [Value::Tensor(sigma), Value::Tensor(correlation)] = outputs.as_slice() else {
            panic!("expected tensor outputs");
        };
        assert_eq!(sigma.shape(), [1, 0]);
        assert_eq!(correlation.shape(), [0, 0]);
    }

    let covariance =
        Value::Tensor(Tensor::new(vec![16.0, 4.0, 4.0, 25.0], vec![2, 2]).expect("tensor"));
    let Value::Tensor(sigma) = block_on(cov2corr::cov2corr_builtin(
        covariance,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("cov2corr") else {
        panic!("expected default sigma output");
    };
    assert_eq!(sigma.shape(), [1, 2]);
    assert_tensor_values(&sigma, &[4.0, 5.0]);
}

#[test]
fn corrcov_zero_variance_produces_nan_correlations() {
    let c = Value::Tensor(Tensor::new(vec![0.0, 0.0, 0.0, 9.0], vec![2, 2]).expect("tensor"));
    let out = block_on(corrcov::corrcov_builtin(
        c,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("corrcov");
    match out {
        Value::Tensor(tensor) => {
            let values = tensor.materialize_f64();
            assert!(values[0].is_nan());
            assert!(values[1].is_nan());
            assert!(values[2].is_nan());
            assert_close(values[3], 1.0);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn corrcov_rejects_invalid_covariance_matrix() {
    let nonsquare =
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).expect("tensor"));
    let err = block_on(corrcov::corrcov_builtin(
        nonsquare,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:corrcov:InvalidArgument"));
    assert!(err.message().contains("square"));

    let negative =
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, -1.0], vec![2, 2]).expect("tensor"));
    let err = block_on(corrcov::corrcov_builtin(
        negative,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message().contains("nonnegative"));

    let asymmetric =
        Value::Tensor(Tensor::new(vec![1.0, 0.1, 0.2, 1.0], vec![2, 2]).expect("tensor"));
    let err = block_on(corrcov::corrcov_builtin(
        asymmetric,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message().contains("symmetric"));

    let one_sided_nan =
        Value::Tensor(Tensor::new(vec![1.0, 2.0, f64::NAN, 1.0], vec![2, 2]).expect("tensor"));
    let err = block_on(corrcov::corrcov_builtin(
        one_sided_nan,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message().contains("finite"));

    let impossible =
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).expect("tensor"));
    let err = block_on(corrcov::corrcov_builtin(
        impossible,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message().contains("variance bounds"));

    let indefinite = Value::Tensor(
        Tensor::new(
            vec![
                1.0, 0.9, 0.9, //
                0.9, 1.0, -0.9, //
                0.9, -0.9, 1.0,
            ],
            vec![3, 3],
        )
        .unwrap(),
    );
    let err = block_on(corrcov::corrcov_builtin(
        indefinite,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message().contains("positive semidefinite"));
}

#[test]
fn corrcov_accepts_typed_integer_tensors_and_scalar_logical_extensions() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    {
        let integer = int_tensor(IntegerStorage::U16(vec![4, 2, 2, 9]), vec![2, 2]);
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let out = block_on(corrcov::corrcov_builtin(
            integer,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
                        assert_eq!(tensor.shape(), [2, 2]);
                        assert_tensor_values(tensor, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
                    }
                    other => panic!("expected correlation tensor, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
                        assert_eq!(tensor.shape(), [2, 1]);
                        assert_tensor_values(tensor, &[2.0, 3.0]);
                    }
                    other => panic!("expected sigma tensor, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    let scalar = block_on(corrcov::corrcov_builtin(
        Value::Num(4.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match scalar {
        Value::Num(value) => assert_close(value, 1.0),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let logical = Value::LogicalArray(
        runmat_value::LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical"),
    );
    let out = block_on(corrcov::corrcov_builtin(
        logical,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
            assert_eq!(tensor.shape(), [2, 2]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn covariance_conversion_extensions_reject_in_matlab_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = int_tensor(IntegerStorage::U16(vec![4, 2, 2, 9]), vec![2, 2]);
    let err = block_on(corrcov::corrcov_builtin(
        integer.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:CorrcovIntegerDataExtension")
    );
    let err = block_on(cov2corr::cov2corr_builtin(
        integer,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:Cov2corrIntegerDataExtension")
    );

    let logical =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap());
    let err = block_on(corrcov::corrcov_builtin(
        logical.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:CorrcovLogicalDataExtension")
    );
    let err = block_on(cov2corr::cov2corr_builtin(
        logical,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:Cov2corrLogicalDataExtension")
    );

    let single = Value::Tensor(Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap());
    let err = block_on(cov2corr::cov2corr_builtin(
        single,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:Cov2corrSingleDataExtension")
    );
}

#[test]
fn covariance_conversion_supports_all_integer_classes_in_runmat_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = vec![
        IntegerStorage::I8(vec![4, 2, 2, 9]),
        IntegerStorage::I16(vec![4, 2, 2, 9]),
        IntegerStorage::I32(vec![4, 2, 2, 9]),
        IntegerStorage::I64(vec![4, 2, 2, 9]),
        IntegerStorage::U8(vec![4, 2, 2, 9]),
        IntegerStorage::U16(vec![4, 2, 2, 9]),
        IntegerStorage::U32(vec![4, 2, 2, 9]),
        IntegerStorage::U64(vec![4, 2, 2, 9]),
    ];
    for storage in storages {
        let covariance = int_tensor(storage, vec![2, 2]);
        let Value::Tensor(correlation) = block_on(corrcov::corrcov_builtin(
            covariance.clone(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("integer corrcov") else {
            panic!("expected correlation tensor");
        };
        assert_eq!(correlation.numeric_dtype(), NumericDType::F64);
        assert_tensor_values(&correlation, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);

        let Value::Tensor(sigma) = block_on(cov2corr::cov2corr_builtin(
            covariance,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("integer cov2corr") else {
            panic!("expected sigma tensor");
        };
        assert_eq!(sigma.numeric_dtype(), NumericDType::F64);
        assert_eq!(sigma.shape(), [1, 2]);
        assert_tensor_values(&sigma, &[2.0, 3.0]);
    }
}

#[test]
fn covariance_conversion_checks_wide_integer_symmetry_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = u64::MAX;
    let symmetric = int_tensor(
        IntegerStorage::U64(vec![base, base - 1, base - 1, base]),
        vec![2, 2],
    );
    let Value::Tensor(result) = block_on(corrcov::corrcov_builtin(
        symmetric,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("wide symmetric") else {
        panic!("expected tensor");
    };
    assert_eq!(result.shape(), [2, 2]);

    let asymmetric = int_tensor(
        IntegerStorage::U64(vec![base, base - 1, base - 2, base]),
        vec![2, 2],
    );
    let err = block_on(corrcov::corrcov_builtin(
        asymmetric,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message().contains("symmetric"));
}

#[test]
fn covariance_conversion_resident_extension_gates_precede_dispatch() {
    test_support::with_test_provider(|provider| {
        let integer =
            Tensor::new_integer(IntegerStorage::U16(vec![4, 2, 2, 9]), vec![2, 2]).unwrap();
        let integer_handle =
            gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = block_on(corrcov::corrcov_builtin(
                Value::GpuTensor(integer_handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap_err();
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:CorrcovIntegerDataExtension")
            );
            let err = block_on(cov2corr::cov2corr_builtin(
                Value::GpuTensor(integer_handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap_err();
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:Cov2corrIntegerDataExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let result = block_on(corrcov::corrcov_builtin(
                Value::GpuTensor(integer_handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("resident integer gather fallback");
            assert!(matches!(result, Value::Tensor(_)));
        }
        drop(integer_handle);

        let double_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[4.0, 2.0, 2.0, 9.0],
                shape: &[2, 2],
            })
            .unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(cov2corr::cov2corr_builtin(
            Value::GpuTensor(double_handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:Cov2corrGpuInputExtension")
        );
        drop(double_handle);
    });
}

#[test]
fn corrcov_rejects_too_many_outputs() {
    let c = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor"));
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let err = block_on(corrcov::corrcov_builtin(
        c,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message().contains("too many output"));
}
