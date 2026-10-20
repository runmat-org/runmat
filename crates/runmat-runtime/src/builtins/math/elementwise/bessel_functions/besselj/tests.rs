use super::*;
use futures::executor::block_on;
use num_complex::Complex64;
use runmat_builtins::BESSELJ_ERROR_INVALID_INPUT;
use runmat_value::{ComplexTensor, NumericDType, Tensor};

fn call(order: Value, argument: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(besselj_builtin(
        None,
        order,
        argument,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn scalar_orders_complex_argument_and_scaling() {
    let Value::Num(j0) = call(Value::Num(0.0), Value::Num(1.0), vec![]).unwrap() else {
        panic!("expected real double scalar");
    };
    close(j0, 0.765_197_686_557_966_6, 2e-14);

    let Value::Num(half) = call(Value::Num(0.5), Value::Num(1.0), vec![]).unwrap() else {
        panic!("expected real fractional-order scalar");
    };
    close(
        half,
        (2.0 / std::f64::consts::PI).sqrt() * 1.0f64.sin(),
        2e-14,
    );

    let Value::Num(negative) = call(Value::Num(-1.0), Value::Num(1.0), vec![]).unwrap() else {
        panic!("expected real negative-order scalar");
    };
    close(negative, -0.440_050_585_744_933_5, 2e-14);

    let Value::Complex(real, imaginary) = call(
        Value::Num(0.0),
        Value::Complex(0.0, 1.0),
        vec![Value::Num(1.0)],
    )
    .unwrap() else {
        panic!("expected scaled complex scalar");
    };
    close(real, 1.266_065_877_752_008_2 / std::f64::consts::E, 2e-14);
    close(imaginary, 0.0, 2e-14);

    let large = 10_000.0f64;
    let Value::Num(half_large) = call(Value::Num(0.5), Value::Num(large), vec![]).unwrap() else {
        panic!("expected real large-argument scalar");
    };
    close(
        half_large,
        (2.0 / (std::f64::consts::PI * large)).sqrt() * large.sin(),
        2e-12,
    );

    let z = Complex64::new(3.0, 4.0);
    let expected = (Complex64::new(2.0 / std::f64::consts::PI, 0.0) / z).sqrt() * z.cos();
    let Value::Complex(real, imaginary) =
        call(Value::Num(-0.5), Value::Complex(z.re, z.im), vec![]).unwrap()
    else {
        panic!("expected complex negative-half-order scalar");
    };
    close(real, expected.re, 2e-12);
    close(imaginary, expected.im, 2e-12);
    let Value::Complex(scaled_real, scaled_imaginary) = call(
        Value::Num(-0.5),
        Value::Complex(z.re, z.im),
        vec![Value::Num(1.0)],
    )
    .unwrap() else {
        panic!("expected scaled complex negative-half-order scalar");
    };
    close(scaled_real, expected.re * (-z.im.abs()).exp(), 2e-12);
    close(scaled_imaginary, expected.im * (-z.im.abs()).exp(), 2e-12);

    let Value::Complex(_, imaginary) =
        call(Value::Num(0.0), Value::Complex(1.0, 0.0), vec![]).unwrap()
    else {
        panic!("explicit complex input must retain complex output class");
    };
    close(imaginary, 0.0, 2e-14);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn arrays_preserve_shape_precision_and_scalar_expansion() {
    let input = Tensor::from_f32(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let Value::Tensor(output) = call(Value::Num(0.0), Value::Tensor(input), vec![]).unwrap() else {
        panic!("expected single tensor");
    };
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    for (actual, expected) in
        output
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip([1.0, 0.765_197_7, 0.223_890_78, -0.260_051_97])
    {
        close(f64::from(*actual), expected, 3e-6);
    }

    let complex = ComplexTensor::from_f32(vec![(0.0, 1.0), (0.0, 2.0)], vec![1, 2]).unwrap();
    let Value::ComplexTensor(output) = call(
        Value::Num(0.0),
        Value::ComplexTensor(complex),
        vec![Value::Num(1.0)],
    )
    .unwrap() else {
        panic!("expected complex single tensor");
    };
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    close(
        f64::from(output.as_f32_slice().unwrap()[0].0),
        1.266_065_877_752_008_2 / std::f64::consts::E,
        3e-6,
    );

    let empty = Tensor::new(vec![], vec![0, 2]).unwrap();
    let Value::Tensor(output) = call(Value::Num(0.0), Value::Tensor(empty), vec![]).unwrap() else {
        panic!("expected empty tensor");
    };
    assert_eq!(output.shape(), &[0, 2]);
    assert!(output.is_empty());

    let singleton = Tensor::new(vec![1.0], vec![1]).unwrap();
    let Value::Tensor(output) = call(Value::Num(0.0), Value::Tensor(singleton), vec![]).unwrap()
    else {
        panic!("singleton array must remain an array");
    };
    assert_eq!(output.shape(), &[1]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_forms_report_typed_errors() {
    let mismatch = call(
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
        vec![],
    )
    .unwrap_err();
    assert_eq!(
        mismatch.identifier(),
        BESSELJ_ERROR_SIZE_MISMATCH.identifier
    );

    let scale = call(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(2.0)]).unwrap_err();
    assert_eq!(
        scale.identifier(),
        BESSELJ_ERROR_INVALID_ARGUMENT.identifier
    );

    let complex_order = call(Value::Complex(0.0, 1.0), Value::Num(1.0), vec![]).unwrap_err();
    assert_eq!(
        complex_order.identifier(),
        BESSELJ_ERROR_INVALID_INPUT.identifier
    );

    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let outputs = call(Value::Num(0.0), Value::Num(1.0), vec![]).unwrap_err();
    assert_eq!(
        outputs.identifier(),
        BESSELJ_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}
