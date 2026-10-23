use super::*;

fn construct_scalar(kind: &str, value: Value) -> Value {
    let mut args = vec![value, Value::from(kind)];
    if kind == "triangular" {
        args.push(Value::from("upper"));
    }
    call_constructor(args).expect("public retained decomposition")
}

#[test]
fn scalar_multiplied_retained_factors_do_not_prescale_finite_rhs_out_of_range() {
    for kind in ["lu", "chol", "qr", "cod", "diagonal", "triangular"] {
        let large = construct_scalar(kind, tensor(&[1.0e300], 1, 1));
        let large = call_mtimes(Value::Num(1.0e-300), large).expect("tiny scalar multiplier");
        let result = call_mldivide(large, Value::Num(1.0e10)).expect("finite solve");
        let actual = approx_vec(result)[0];
        assert!((actual / 1.0e10 - 1.0).abs() < 1.0e-10, "{kind}: {actual}");

        let small = construct_scalar(kind, tensor(&[1.0e-300], 1, 1));
        let small = call_mtimes(Value::Num(1.0e300), small).expect("large scalar multiplier");
        let result = call_mldivide(small, Value::Num(1.0e-300)).expect("nonzero solve");
        let actual = approx_vec(result)[0];
        assert!(
            (actual / 1.0e-300 - 1.0).abs() < 1.0e-10,
            "{kind}: {actual}"
        );
    }
}

#[test]
fn complex_scale_is_applied_at_coefficient_before_native_output() {
    for kind in ["lu", "chol", "qr", "cod", "diagonal", "triangular"] {
        let factor = construct_scalar(kind, tensor(&[0.5], 1, 1));
        let factor = call_mtimes(Value::Complex(1.0, 1.0), factor).expect("complex scale");
        let Value::Complex(real, imag) =
            call_mldivide(factor, Value::Num(0.75 * f64::MAX)).expect("finite complex solve")
        else {
            panic!("{kind}: complex double scalar output expected");
        };
        assert!((real / f64::MAX - 0.75).abs() < 1.0e-14, "{kind}: {real}");
        assert!((imag / f64::MAX + 0.75).abs() < 1.0e-14, "{kind}: {imag}");
        if kind != "qr" {
            let factor = construct_scalar(kind, tensor(&[0.5], 1, 1));
            let factor = call_mtimes(Value::Complex(1.0, 1.0), factor).expect("right scale");
            let Value::Complex(right_real, right_imag) =
                call_mrdivide(Value::Num(0.75 * f64::MAX), factor)
                    .expect("finite right complex solve")
            else {
                panic!("{kind}: complex right scalar output expected");
            };
            assert!(
                (right_real / f64::MAX - 0.75).abs() < 1.0e-14,
                "{kind}: {right_real}"
            );
            assert!(
                (right_imag / f64::MAX + 0.75).abs() < 1.0e-14,
                "{kind}: {right_imag}"
            );
        }
    }
}

#[test]
fn native_single_scale_forms_physical_coefficient_before_narrowing() {
    for kind in ["lu", "chol", "qr", "cod", "diagonal", "triangular"] {
        let coefficient = Value::Tensor(Tensor::from_f32(vec![1.0e30], vec![1, 1]).unwrap());
        let factor = construct_scalar(kind, coefficient);
        let factor = call_mtimes(Value::Num(1.0e-40), factor).expect("tiny double scale");
        let Value::Tensor(result) =
            call_mldivide(factor, Value::Num(1.0)).expect("finite single solve")
        else {
            panic!("{kind}: native single output expected");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        let actual = result.as_f32_slice().expect("single output")[0];
        assert!(
            (f64::from(actual) / 1.0e10 - 1.0).abs() < 2.0e-5,
            "{kind}: {actual}"
        );
    }
}

#[test]
fn negative_real_cholesky_scale_stays_on_real_native_output() {
    let factor = construct_scalar("chol", tensor(&[0.5], 1, 1));
    let factor = call_mtimes(Value::Num(-1.0), factor).expect("negative real scale");
    let result = call_mldivide(factor, Value::Num(1.0)).expect("real scaled solve");
    assert!(!matches!(
        &result,
        Value::ComplexTensor(_) | Value::Complex(_, _)
    ));
    let result = approx_vec(result);
    assert_eq!(result.len(), 1);
    // Two rounded square-root substitutions need not equal the exact binary
    // quotient bit-for-bit. Bound both error and the original operator residual.
    assert!((result[0] + 2.0).abs() <= 4.0 * f64::EPSILON);
    assert!((-0.5 * result[0] - 1.0).abs() <= 2.0 * f64::EPSILON);
}

#[test]
fn complex_right_solve_conjugates_the_factor_diagonal_before_scaling_it() {
    for kind in ["lu", "cod", "diagonal", "triangular"] {
        let factor = construct_scalar(kind, complex_tensor(&[(1.0, 1.0)], 1, 1));
        let factor = call_mtimes(Value::Complex(2.0, 1.0), factor).expect("complex scale");
        let Value::Complex(real, imag) =
            call_mrdivide(Value::Complex(1.0, 0.0), factor).expect("oriented right solve")
        else {
            panic!("{kind}: complex scalar output expected");
        };
        assert!((real - 0.1).abs() < 1.0e-14, "{kind}: {real}");
        assert!((imag + 0.3).abs() < 1.0e-14, "{kind}: {imag}");
    }
}
