use super::*;
use runmat_value::NumericDType;

fn input(single: bool, values: &[(f64, f64)], shape: Vec<usize>) -> ComplexTensor {
    if single {
        ComplexTensor::from_f32(
            values
                .iter()
                .map(|&(re, im)| (re as f32, im as f32))
                .collect(),
            shape,
        )
        .unwrap()
    } else {
        ComplexTensor::new(values.to_vec(), shape).unwrap()
    }
}
fn parsed() -> ParsedMoving {
    ParsedMoving {
        window: WindowSpec::Counts {
            before: 1,
            after: 1,
        },
        dim: Some(2),
        nan_mode: NanMode::Include,
        endpoints: Endpoints::Shrink,
        sample_points: None,
        normalization: VarianceNormalization::Sample,
    }
}
fn values(value: &Value, single: bool, real: bool, expected: &[(f64, f64)]) {
    let dtype = if single {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let observed = match value {
        Value::Num(real_part) if real && !single => vec![(*real_part, 0.0)],
        Value::Complex(real_part, imaginary) if !real && !single => {
            vec![(*real_part, *imaginary)]
        }
        Value::Tensor(tensor) if real => {
            assert_eq!(tensor.numeric_dtype(), dtype);
            tensor
                .materialize_f64()
                .iter()
                .map(|&re| (re, 0.0))
                .collect::<Vec<_>>()
        }
        Value::ComplexTensor(tensor) if !real => {
            assert_eq!(tensor.numeric_dtype(), dtype);
            tensor.materialize_f64()
        }
        other => panic!("wrong real/complex publication: {other:?}"),
    };
    assert_eq!(observed.len(), expected.len());
    let tolerance = if single { 2e-6 } else { 1e-12 };
    for (value, expected) in observed.iter().zip(expected) {
        assert!(
            (value.0 - expected.0).abs() < tolerance,
            "{value:?} != {expected:?}"
        );
        assert!(
            (value.1 - expected.1).abs() < tolerance,
            "{value:?} != {expected:?}"
        );
    }
}

#[test]
fn all_complex_window_actions_publish_native_class_and_mathematical_domain() {
    for single in [true, false] {
        let input = input(single, &[(1.0, 1.0), (3.0, 1.0), (5.0, 1.0)], vec![1, 3]);
        for (operation, expected) in [
            (MovingOp::Sum, [(4.0, 2.0), (9.0, 3.0), (8.0, 2.0)]),
            (MovingOp::Mean, [(2.0, 1.0), (3.0, 1.0), (4.0, 1.0)]),
            (MovingOp::Prod, [(2.0, 4.0), (6.0, 22.0), (14.0, 8.0)]),
            (MovingOp::Min, [(1.0, 1.0), (1.0, 1.0), (3.0, 1.0)]),
            (MovingOp::Max, [(3.0, 1.0), (5.0, 1.0), (5.0, 1.0)]),
            (MovingOp::Median, [(2.0, 1.0), (3.0, 1.0), (4.0, 1.0)]),
            (MovingOp::Var, [(2.0, 0.0), (4.0, 0.0), (2.0, 0.0)]),
            (
                MovingOp::Std,
                [(2.0_f64.sqrt(), 0.0), (2.0, 0.0), (2.0_f64.sqrt(), 0.0)],
            ),
        ] {
            let result = apply("moving native fixture", operation, &input, &parsed()).unwrap();
            values(
                &result,
                single,
                operation.uses_variance_normalization(),
                &expected,
            );
        }
        assert_eq!(
            input.materialize_f64(),
            vec![(1.0, 1.0), (3.0, 1.0), (5.0, 1.0)]
        );
    }
}

#[test]
fn native_empty_scalar_and_strided_geometry_do_not_widen_or_gain_complex_variance() {
    for single in [true, false] {
        let empty = input(single, &[], vec![2, 0]);
        for operation in [MovingOp::Sum, MovingOp::Var] {
            let result = apply("moving empty", operation, &empty, &parsed()).unwrap();
            match result {
                Value::Tensor(tensor) => {
                    assert!(operation.uses_variance_normalization());
                    assert_eq!(
                        tensor.numeric_dtype(),
                        if single {
                            NumericDType::F32
                        } else {
                            NumericDType::F64
                        }
                    );
                    assert_eq!(tensor.shape(), &[2, 0]);
                    assert!(tensor.is_empty());
                }
                Value::ComplexTensor(tensor) => {
                    assert!(!operation.uses_variance_normalization());
                    assert_eq!(
                        tensor.numeric_dtype(),
                        if single {
                            NumericDType::F32
                        } else {
                            NumericDType::F64
                        }
                    );
                    assert_eq!(tensor.shape(), &[2, 0]);
                    assert!(tensor.is_empty());
                }
                other => panic!("empty geometry lost: {other:?}"),
            }
        }
        let strided = input(
            single,
            &[(1.0, 1.0), (10.0, 2.0), (3.0, 1.0), (20.0, 2.0)],
            vec![2, 2],
        );
        let result = apply("moving strides", MovingOp::Mean, &strided, &parsed()).unwrap();
        values(
            &result,
            single,
            false,
            &[(2.0, 1.0), (15.0, 2.0), (2.0, 1.0), (15.0, 2.0)],
        );
        let scalar = input(single, &[(3.0, 4.0)], vec![1, 1]);
        if single {
            values(
                &apply("moving scalar sum", MovingOp::Sum, &scalar, &parsed()).unwrap(),
                true,
                false,
                &[(3.0, 4.0)],
            );
        }
        let result = apply("moving scalar", MovingOp::Var, &scalar, &parsed()).unwrap();
        match result {
            Value::Num(value) if !single => assert_eq!(value, 0.0),
            Value::Tensor(tensor) if single => {
                assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
                assert_eq!(tensor.as_f32_slice().unwrap(), &[0.0]);
            }
            other => panic!("scalar native real publication lost: {other:?}"),
        }
    }
}

#[test]
fn native_complex_nan_fill_discard_and_sample_population_keep_shared_options() {
    for single in [true, false] {
        let data = input(
            single,
            &[(1.0, 1.0), (f64::NAN, 0.0), (5.0, 1.0)],
            vec![1, 3],
        );
        let mut options = parsed();
        options.nan_mode = NanMode::Omit;
        options.endpoints = Endpoints::Fill(NumericScalar::F64(2.0));
        values(
            &apply("moving fill", MovingOp::Sum, &data, &options).unwrap(),
            single,
            false,
            &[(3.0, 1.0), (6.0, 2.0), (7.0, 1.0)],
        );
        options.endpoints = Endpoints::Discard;
        values(
            &apply("moving discard", MovingOp::Mean, &data, &options).unwrap(),
            single,
            false,
            &[(3.0, 1.0)],
        );
        options.endpoints = Endpoints::Shrink;
        options.normalization = VarianceNormalization::Population;
        options.sample_points = Some(vec![0.0, 1.0, 3.0]);
        options.window = WindowSpec::Span {
            before: 2.0,
            after: 2.0,
        };
        let result = apply("moving sample population", MovingOp::Var, &data, &options).unwrap();
        values(&result, single, true, &[(0.0, 0.0), (4.0, 0.0), (0.0, 0.0)]);
    }
}
