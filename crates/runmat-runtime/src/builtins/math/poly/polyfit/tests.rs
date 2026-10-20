use super::*;

#[test]
fn provider_output_fault_remains_typed_and_non_retryable() {
    let result = ProviderPolyfitResult {
        coefficients: vec![f64::NAN, 1.0],
        r_matrix: None,
        normr: None,
        df: 0.0,
        mu: [0.0, 1.0],
    };
    let error = PolyfitSolution::from_provider(result, ProviderPolyfitOutputs::Coefficients, 1, 2)
        .expect_err("nonfinite successful provider output");
    assert_eq!(error.identifier(), Some("RunMat:polyfit:ProviderFailed"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let source = std::error::Error::source(&error).expect("owned validation cause");
    assert!(source.is::<provider_output::ProviderOutputFault>());

    let valid_stats = ProviderPolyfitResult {
        coefficients: vec![2.0, 1.0],
        r_matrix: Some(vec![1.0, 0.0, 0.0, 1.0]),
        normr: Some(0.0),
        df: 0.0,
        mu: [0.0, 1.0],
    };
    for result in [
        ProviderPolyfitResult {
            r_matrix: Some(vec![f64::NAN, 0.0, 0.0, 1.0]),
            ..valid_stats.clone()
        },
        ProviderPolyfitResult {
            normr: Some(f64::NAN),
            ..valid_stats.clone()
        },
        ProviderPolyfitResult {
            df: 1.0,
            ..valid_stats.clone()
        },
        ProviderPolyfitResult {
            mu: [0.0, -1.0],
            ..valid_stats.clone()
        },
    ] {
        let error =
            PolyfitSolution::from_provider(result, ProviderPolyfitOutputs::Statistics, 1, 2)
                .expect_err("malformed requested statistics");
        assert_eq!(error.identifier(), Some("RunMat:polyfit:ProviderFailed"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error)
            .expect("owned validation cause")
            .is::<provider_output::ProviderOutputFault>());
    }
}

#[test]
fn degree_parser_preserves_representable_uint64() {
    assert_eq!(
        parse_degree(&Value::Int(IntValue::U64(u64::MAX))).ok(),
        usize::try_from(u64::MAX).ok()
    );
    assert!(parse_degree(&Value::Int(IntValue::I64(-1))).is_err());
}

#[test]
fn degree_parser_reads_typed_integer_scalar_storage_exactly() {
    let degree =
        Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("typed degree");

    assert_eq!(parse_degree(&Value::Tensor(degree)).unwrap(), 3);

    let negative =
        Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("negative degree");
    assert!(parse_degree(&Value::Tensor(negative)).is_err());
}

#[test]
fn degree_parser_rejects_unrepresentable_double_boundary() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(parse_degree(&Value::Num(boundary)).is_err());
}

use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage};

fn assert_error_contains(err: crate::RuntimeError, needle: &str) {
    assert!(
        err.message().contains(needle),
        "expected error containing '{needle}', got '{}'",
        err.message()
    );
}

#[test]
fn polyfit_rejects_typed_complex_integer_inputs() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let typed_complex = Value::ComplexTensor(
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MAX]),
                IntegerStorage::I64(vec![-1]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor"),
    );
    let err = evaluate(
        typed_complex,
        Value::Num(1.0),
        Value::Int(IntValue::I32(0)),
        &[],
    )
    .expect_err("typed complex integer input must reject");
    assert_error_contains(err, "complex numbers with integer types");
}

fn evaluate(
    x: Value,
    y: Value,
    degree: Value,
    rest: &[Value],
) -> Result<PolyfitEval, RuntimeError> {
    block_on(super::evaluate(x, y, degree, rest))
}

fn scalar(value: Value) -> f64 {
    match value {
        Value::Num(value) => value,
        Value::Tensor(tensor) if tensor.len() == 1 => tensor.materialize_f64()[0],
        other => panic!("expected scalar numerical result, got {other:?}"),
    }
}

#[test]
fn coefficient_and_qr_basis_follows_requested_output_count() {
    // Independent least-squares oracle: ybar = 24.25, slope = 2.1,
    // intercept = 0.1, SSE = 2.7, Sxx = 5, and df = 2.
    let x = Value::Tensor(Tensor::new(vec![10.0, 11.0, 12.0, 13.0], vec![1, 4]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![21.0, 24.0, 24.0, 28.0], vec![1, 4]).unwrap());
    for outputs in [1, 2, 3] {
        let _requested =
            crate::output_count::push_output_count(Some(outputs)).expect("output-count scope");
        let fit = evaluate(
            x.clone(),
            y.clone(),
            Value::Int(runmat_value::IntValue::I32(1)),
            &[],
        )
        .expect("fit");
        let coefficients = fit.coefficients();
        let coefficient_data = match &coefficients {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected coefficient tensor, got {other:?}"),
        };
        let options = if outputs == 3 {
            vec![fit.stats().expect("statistics"), fit.mu()]
        } else {
            vec![fit.stats().expect("statistics")]
        };
        if outputs == 3 {
            let mu = match fit.mu() {
                Value::Tensor(tensor) => tensor.materialize_f64(),
                other => panic!("expected mu tensor, got {other:?}"),
            };
            assert!((mu[0] - 11.5).abs() < 1e-12);
            assert!((mu[1] - (5.0_f64 / 3.0).sqrt()).abs() < 1e-12);
            assert!((coefficient_data[0] - 2.1 * mu[1]).abs() < 1e-11);
            assert!((coefficient_data[1] - 24.25).abs() < 1e-11);
        } else {
            assert!((coefficient_data[0] - 2.1).abs() < 1e-11);
            assert!((coefficient_data[1] - 0.1).abs() < 1e-10);
        }
        let prediction = block_on(super::super::polyval::evaluate(
            coefficients,
            Value::Num(14.0),
            &options,
            true,
        ))
        .expect("prediction");
        assert!((scalar(prediction.value()) - 29.5).abs() < 1e-10);
        let delta = scalar(prediction.delta().expect("interval"));
        // sqrt((SSE/df) * (1 + 1/n + (x-xbar)^2/Sxx)).
        assert!((delta - 3.375_f64.sqrt()).abs() < 1e-9, "{delta}");
    }
}

#[test]
fn normalized_small_spread_retains_its_actual_basis() {
    let x = Value::Tensor(Tensor::new(vec![1.0e-14, 2.0e-14, 3.0e-14], vec![1, 3]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
    let _requested = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let fit =
        evaluate(x, y, Value::Int(runmat_value::IntValue::I32(1)), &[]).expect("small-spread fit");
    let Value::Tensor(mu) = fit.mu() else {
        panic!("mu row expected");
    };
    let mu_values = mu.materialize_f64();
    assert!((mu_values[0] - 2.0e-14).abs() <= 1.0e-29);
    assert!((mu_values[1] - 1.0e-14).abs() <= 1.0e-29);

    let predicted = block_on(super::super::polyval::evaluate(
        fit.coefficients(),
        Value::Num(4.0e-14),
        &[fit.stats().expect("statistics"), fit.mu()],
        true,
    ))
    .expect("prediction in published QR basis");
    assert!((scalar(predicted.value()) - 4.0).abs() < 1.0e-10);
    assert!(scalar(predicted.delta().expect("interval")).abs() < 1.0e-10);
}

#[test]
fn small_units_preserve_two_output_original_basis() {
    let x = Value::Tensor(Tensor::new(vec![1.0e-14, 2.0e-14, 3.0e-14], vec![1, 3]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
    let _requested = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let fit = evaluate(x, y, Value::Int(runmat_value::IntValue::I32(1)), &[])
        .expect("original-coordinate fit");
    let Value::Tensor(coefficients) = fit.coefficients() else {
        panic!("coefficient row expected");
    };
    let p = coefficients.materialize_f64();
    assert!((p[0] / 1.0e14 - 1.0).abs() < 1.0e-10);
    assert!(p[1].abs() < 1.0e-10);
    let prediction = block_on(super::super::polyval::evaluate(
        Value::Tensor(coefficients),
        Value::Num(4.0e-14),
        &[fit.stats().expect("statistics")],
        true,
    ))
    .expect("original-coordinate prediction");
    assert!((scalar(prediction.value()) - 4.0).abs() < 1.0e-10);
}

#[test]
fn degree_requires_an_exact_integer_not_a_nearby_float() {
    let error = parse_degree(&Value::Num(1.0 + 1.0e-13)).expect_err("fractional degree");
    assert!(error.message().contains("must be an integer"));
}

#[test]
fn coefficient_only_result_reports_absent_statistics_without_panicking() {
    let fit = block_on(super::evaluate_requested(
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
        Value::Num(1.0),
        crate::call::FunctionArgumentView::from_slice(&[]),
        ProviderPolyfitOutputs::Coefficients,
    ))
    .expect("coefficient-only fit");
    assert!(fit.stats().is_none());
}

#[test]
fn householder_reflector_reconstructs_large_small_and_ordinary_columns() {
    for (source, observations, expected) in [
        ([1.0e308, 1.0e307], [1.0, 0.1], 1.0e-308),
        ([1.0e-300, 1.0e-301], [1.0, 0.1], 1.0e300),
        ([1.0, 0.1], [2.0, 0.2], 2.0),
    ] {
        let mut column = source;
        let norm = stable_l2(source);
        let mut rhs = observations.map(|value| Complex64::new(value, 0.0));
        householder_qr(&mut column, 2, 1, &mut rhs).expect("finite reflector");
        assert!(column.iter().all(|value| value.is_finite()));
        assert!(rhs.iter().all(|value| value.re.is_finite()));
        let coefficients = solve_upper(&column, 2, 1, &rhs, &[norm]).expect("finite solve");
        let coefficient = coefficients[0].re;
        assert!((coefficient / expected - 1.0).abs() < 1.0e-13);
        for (input, target) in source.into_iter().zip(observations) {
            assert!(((input * coefficient) / target - 1.0).abs() < 1.0e-13);
        }
    }

    let mut zero = [0.0, 0.0];
    let mut rhs = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    householder_qr(&mut zero, 2, 1, &mut rhs).expect("zero column is unchanged");
    assert_eq!(zero, [0.0, 0.0]);
    assert_eq!(rhs, [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
}

#[test]
fn nonrepresentable_qr_norm_reports_input_error_instead_of_nan() {
    let mut column = [1.5e308, 1.5e308];
    let mut rhs = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
    let error =
        householder_qr(&mut column, 2, 1, &mut rhs).expect_err("norm exceeds binary64 range");
    assert!(matches!(error, PolyfitKernelError::InvalidInput(_)));
    assert!(column.iter().all(|value| value.is_finite()));
    assert!(rhs.iter().all(|value| value.re.is_finite()));

    let mut nonfinite = [f64::INFINITY];
    let mut rhs = [Complex64::new(1.0, 0.0)];
    assert!(matches!(
        householder_qr(&mut nonfinite, 1, 1, &mut rhs),
        Err(PolyfitKernelError::InvalidInput(_))
    ));
}

#[test]
fn finite_extreme_samples_keep_analytic_center_scale_and_coordinates() {
    let maximum = f64::MAX;
    let samples = [maximum, -maximum, -maximum, -maximum, -maximum];
    let mean = normalization::sample_mean(&samples).expect("finite sample mean");
    assert!((mean / maximum + 0.6).abs() < 1.0e-14);
    let scale = compute_scale(&samples, mean).expect("finite sample spread");
    assert!((scale / maximum - 0.8_f64.sqrt()).abs() < 1.0e-14);
    let normalized =
        scaled_coordinates(&samples, mean, scale).expect("finite standardized sample locations");
    assert!((normalized[0] - 1.6 / 0.8_f64.sqrt()).abs() < 1.0e-14);
    for coordinate in &normalized[1..] {
        assert!((*coordinate + 0.4 / 0.8_f64.sqrt()).abs() < 1.0e-14);
    }

    let _requested = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let fit = evaluate(
        Value::Tensor(Tensor::new(samples.to_vec(), vec![1, 5]).unwrap()),
        Value::Tensor(Tensor::new(vec![4.0, 0.0, 0.0, 0.0, 0.0], vec![1, 5]).unwrap()),
        Value::Int(runmat_value::IntValue::I32(1)),
        &[],
    )
    .expect("finite extreme-scale fit");
    let prediction = block_on(super::super::polyval::evaluate(
        fit.coefficients(),
        Value::Num(maximum),
        &[
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            fit.mu(),
        ],
        false,
    ))
    .expect("finite extreme-scale evaluation");
    assert!((scalar(prediction.value()) - 4.0).abs() < 1.0e-11);
}

#[test]
fn tiny_nonzero_imaginary_fit_stays_complex() {
    let x = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    let y = Value::ComplexTensor(
        ComplexTensor::new(vec![(1.0, 1.0e-13), (1.0, 1.0e-13)], vec![1, 2]).unwrap(),
    );
    let fit = evaluate(x, y, Value::Int(runmat_value::IntValue::I32(0)), &[])
        .expect("complex constant fit");
    let Value::ComplexTensor(coefficients) = fit.coefficients() else {
        panic!("nonzero imaginary fit must stay complex");
    };
    let values = coefficients.materialize_f64();
    let [(real, imaginary)] = values.as_slice() else {
        panic!("one complex coefficient expected");
    };
    assert!((*real - 1.0).abs() < 1.0e-12);
    assert!((*imaginary - 1.0e-13).abs() < 1.0e-25);
}

#[test]
fn polyfit_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = POLYFIT_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert!(labels.contains(&"p = polyfit(X, Y, n)"));
    assert!(labels.contains(&"p = polyfit(X, Y, n, weights)"));
    assert!(labels.contains(&"[p, S] = polyfit(X, Y, n)"));
    assert!(labels.contains(&"[p, S, mu] = polyfit(X, Y, n, weights)"));
}

#[test]
fn polyfit_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = POLYFIT_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.POLYFIT.INVALID_ARGUMENT"));
    assert!(codes.contains(&"RM.POLYFIT.INVALID_INPUT"));
    assert!(codes.contains(&"RM.POLYFIT.INTERNAL"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fits_linear_data() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
    let mut y_vals = Vec::new();
    for i in 0..4 {
        y_vals.push(1.5 * i as f64 + 2.0);
    }
    let y = Tensor::new(y_vals, vec![4, 1]).unwrap();
    let eval = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Int(runmat_value::IntValue::I32(1)),
        &[],
    )
    .expect("polyfit");
    match eval.coefficients() {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![1, 2]);
            assert!((t.materialize_f64()[0] - 1.5).abs() < 1e-10);
            assert!((t.materialize_f64()[1] - 2.0).abs() < 1e-10);
        }
        other => panic!("expected tensor coefficients, got {other:?}"),
    }
}

#[test]
fn polyfit_typed_integer_vectors_degree_and_weights_cross_double_boundary_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 2, 3]), vec![4, 1]).unwrap();
    let y = Tensor::new_integer(IntegerStorage::I16(vec![2, 4, 6, 8]), vec![4, 1]).unwrap();
    let degree = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
    let weights = Tensor::new_integer(IntegerStorage::U16(vec![1, 1, 1, 1]), vec![1, 4]).unwrap();
    let eval = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Tensor(degree),
        &[Value::Tensor(weights)],
    )
    .expect("polyfit");
    match eval.coefficients() {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![1, 2]);
            assert!((t.materialize_f64()[0] - 2.0).abs() < 1e-10);
            assert!((t.materialize_f64()[1] - 2.0).abs() < 1e-10);
            assert!(t.integer_storage().is_none());
        }
        other => panic!("expected tensor coefficients, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn returns_struct_and_mu() {
    let x = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]).unwrap();
    let eval = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Int(runmat_value::IntValue::I32(2)),
        &[],
    )
    .expect("polyfit");
    match eval.stats() {
        Some(Value::Struct(s)) => {
            assert!(s.fields.contains_key("R"));
            assert!(s.fields.contains_key("df"));
            assert!(s.fields.contains_key("normr"));
        }
        other => panic!("expected struct, got {other:?}"),
    }
    match eval.mu() {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![1, 2]);
            assert!((t.materialize_f64()[0]).abs() < 1e-10);
            assert!(t.materialize_f64()[1].abs() > 0.0);
        }
        other => panic!("expected tensor mu, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weighted_fit_matches_unweighted_when_weights_equal() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
    let weights = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
    let eval_unweighted = evaluate(
        Value::Tensor(x.clone()),
        Value::Tensor(y.clone()),
        Value::Int(runmat_value::IntValue::I32(2)),
        &[],
    )
    .expect("polyfit");
    let eval_weighted = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Int(runmat_value::IntValue::I32(2)),
        &[Value::Tensor(weights)],
    )
    .expect("polyfit");
    assert_eq!(eval_unweighted.coefficients(), eval_weighted.coefficients());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn accepts_logical_degree_scalar() {
    let x = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
    let logical = runmat_value::LogicalArray::new(vec![1], vec![1, 1]).unwrap();
    let eval = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::LogicalArray(logical),
        &[],
    )
    .expect("polyfit");
    assert!(matches!(eval.coefficients(), Value::Tensor(_)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_non_integer_degree() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
    let err = evaluate(Value::Tensor(x), Value::Tensor(y), Value::Num(1.5), &[])
        .expect_err("polyfit should reject non-integer degree");
    assert_eq!(err.identifier(), POLYFIT_ERROR_INVALID_ARGUMENT.identifier);
    assert_error_contains(err, "integer");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_infinite_weights() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
    let weights = Tensor::new(vec![1.0, f64::INFINITY, 1.0], vec![3, 1]).unwrap();
    let err = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Int(runmat_value::IntValue::I32(2)),
        &[Value::Tensor(weights)],
    )
    .expect_err("polyfit should reject infinite weights");
    assert_error_contains(err, "weight at position 2");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_inputs_are_gathered() {
    test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &x.materialize_f64(),
            shape: x.shape(),
        };
        let x_handle = provider.upload(&view).expect("upload");
        let view_y = runmat_accelerate_api::HostTensorView {
            data: &y.materialize_f64(),
            shape: y.shape(),
        };
        let y_handle = provider.upload(&view_y).expect("upload");
        let eval = evaluate(
            Value::GpuTensor(x_handle),
            Value::GpuTensor(y_handle),
            Value::Int(runmat_value::IntValue::I32(2)),
            &[],
        )
        .expect("polyfit");
        assert!(matches!(eval.coefficients(), Value::Tensor(_)));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_weights_are_gathered() {
    test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();

        let x_view = runmat_accelerate_api::HostTensorView {
            data: &x.materialize_f64(),
            shape: x.shape(),
        };
        let y_view = runmat_accelerate_api::HostTensorView {
            data: &y.materialize_f64(),
            shape: y.shape(),
        };
        let w_view = runmat_accelerate_api::HostTensorView {
            data: &weights.materialize_f64(),
            shape: weights.shape(),
        };

        let x_handle = provider.upload(&x_view).expect("upload x");
        let y_handle = provider.upload(&y_view).expect("upload y");
        let w_handle = provider.upload(&w_view).expect("upload weights");

        let cpu_eval = evaluate(
            Value::Tensor(x.clone()),
            Value::Tensor(y.clone()),
            Value::Int(runmat_value::IntValue::I32(2)),
            &[Value::Tensor(weights.clone())],
        )
        .expect("cpu polyfit");

        let gpu_eval = evaluate(
            Value::GpuTensor(x_handle.clone()),
            Value::GpuTensor(y_handle.clone()),
            Value::Int(runmat_value::IntValue::I32(2)),
            &[Value::GpuTensor(w_handle.clone())],
        )
        .expect("gpu polyfit with weights");

        assert_eq!(cpu_eval.coefficients(), gpu_eval.coefficients());
        assert_eq!(cpu_eval.mu(), gpu_eval.mu());

        drop((x_handle, y_handle, w_handle));
    });
}

#[cfg(feature = "wgpu")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn polyfit_wgpu_matches_cpu() {
    let options = runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default();
    let _provider =
        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(options) {
            Ok(p) => p,
            Err(err) => {
                warn!("polyfit_wgpu_matches_cpu: skipping test ({err})");
                return;
            }
        };
    let x = Tensor::new(vec![10.0, 11.0, 12.0, 13.0], vec![4, 1]).unwrap();
    let y = Tensor::new(vec![21.0, 24.0, 24.0, 28.0], vec![4, 1]).unwrap();

    let trait_provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
    let x_view = runmat_accelerate_api::HostTensorView {
        data: &x.materialize_f64(),
        shape: x.shape(),
    };
    let y_view = runmat_accelerate_api::HostTensorView {
        data: &y.materialize_f64(),
        shape: y.shape(),
    };
    let x_handle = trait_provider.upload(&x_view).expect("upload x");
    let y_handle = trait_provider.upload(&y_view).expect("upload y");

    for output_count in [2, 3] {
        let _requested =
            crate::output_count::push_output_count(Some(output_count)).expect("output-count scope");
        let cpu_eval = evaluate(
            Value::Tensor(x.clone()),
            Value::Tensor(y.clone()),
            Value::Int(runmat_value::IntValue::I32(1)),
            &[],
        )
        .expect("cpu polyfit");
        let gpu_eval = evaluate(
            Value::GpuTensor(x_handle.clone()),
            Value::GpuTensor(y_handle.clone()),
            Value::Int(runmat_value::IntValue::I32(1)),
            &[],
        )
        .expect("gpu polyfit");

        let cpu_coeff = match cpu_eval.coefficients() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor coefficients, got {other:?}"),
        };
        let gpu_coeff = match gpu_eval.coefficients() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor coefficients, got {other:?}"),
        };
        assert_eq!(cpu_coeff.shape(), gpu_coeff.shape());
        for (a, b) in cpu_coeff
            .materialize_f64()
            .iter()
            .zip(gpu_coeff.materialize_f64().iter())
        {
            assert!((a - b).abs() < 1e-9, "coeff mismatch {a} vs {b}");
        }

        let cpu_mu = match cpu_eval.mu() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor mu, got {other:?}"),
        };
        let gpu_mu = match gpu_eval.mu() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor mu, got {other:?}"),
        };
        assert_eq!(cpu_mu.shape(), gpu_mu.shape());
        for (a, b) in cpu_mu
            .materialize_f64()
            .iter()
            .zip(gpu_mu.materialize_f64().iter())
        {
            assert!((a - b).abs() < 1e-9, "mu mismatch {a} vs {b}");
        }

        let cpu_stats = match cpu_eval.stats() {
            Some(Value::Struct(s)) => s,
            other => panic!("expected struct stats, got {other:?}"),
        };
        let gpu_stats = match gpu_eval.stats() {
            Some(Value::Struct(s)) => s,
            other => panic!("expected struct stats, got {other:?}"),
        };
        let cpu_r = match cpu_stats.fields.get("R").expect("R present") {
            Value::Tensor(t) => t.clone(),
            other => panic!("expected tensor R, got {other:?}"),
        };
        let gpu_r = match gpu_stats.fields.get("R").expect("R present") {
            Value::Tensor(t) => t.clone(),
            other => panic!("expected tensor R, got {other:?}"),
        };
        assert_eq!(cpu_r.shape(), gpu_r.shape());
        for (a, b) in cpu_r
            .materialize_f64()
            .iter()
            .zip(gpu_r.materialize_f64().iter())
        {
            assert!((a - b).abs() < 1e-9, "R mismatch {a} vs {b}");
        }
        let cpu_df = match cpu_stats.fields.get("df").expect("df present") {
            Value::Num(n) => *n,
            other => panic!("expected numeric df, got {other:?}"),
        };
        let gpu_df = match gpu_stats.fields.get("df").expect("df present") {
            Value::Num(n) => *n,
            other => panic!("expected numeric df, got {other:?}"),
        };
        assert!((cpu_df - gpu_df).abs() < 1e-9);
        let cpu_normr = match cpu_stats.fields.get("normr").expect("normr present") {
            Value::Num(n) => *n,
            other => panic!("expected numeric normr, got {other:?}"),
        };
        let gpu_normr = match gpu_stats.fields.get("normr").expect("normr present") {
            Value::Num(n) => *n,
            other => panic!("expected numeric normr, got {other:?}"),
        };
        assert!((cpu_normr - gpu_normr).abs() < 1e-9);
    }
    drop((x_handle, y_handle));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_mismatched_lengths() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Int(runmat_value::IntValue::I32(1)),
        &[],
    )
    .expect_err("polyfit should reject mismatched vector lengths");
    assert_error_contains(err, "same length");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_non_vector_inputs() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let err = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Int(runmat_value::IntValue::I32(1)),
        &[],
    )
    .expect_err("polyfit should reject non-vector X");
    assert_error_contains(err, "vector");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_weight_length_mismatch() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
    let weights = Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap();
    let err = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Int(runmat_value::IntValue::I32(2)),
        &[Value::Tensor(weights)],
    )
    .expect_err("polyfit should reject mismatched weights");
    assert_error_contains(err, "weight vector must match");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_negative_weights() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 3.0, 7.0], vec![3, 1]).unwrap();
    let weights = Tensor::new(vec![1.0, -1.0, 1.0], vec![3, 1]).unwrap();
    let err = evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        Value::Int(runmat_value::IntValue::I32(2)),
        &[Value::Tensor(weights)],
    )
    .expect_err("polyfit should reject negative weights");
    assert_error_contains(err, "non-negative");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fits_complex_data() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let complex_values =
        ComplexTensor::new(vec![(0.0, 1.0), (1.0, 0.5), (4.0, -0.25)], vec![3, 1]).unwrap();
    let eval = evaluate(
        Value::Tensor(x),
        Value::ComplexTensor(complex_values),
        Value::Int(runmat_value::IntValue::I32(2)),
        &[],
    )
    .expect("polyfit complex");
    match eval.coefficients() {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape(), vec![1, 3]);
        }
        other => panic!("expected complex tensor coefficients, got {other:?}"),
    }
}
