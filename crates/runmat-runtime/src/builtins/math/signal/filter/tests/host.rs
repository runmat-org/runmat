use super::*;
use runmat_value::{ComplexTensor, IntValue};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_fir_basic() {
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new(vec![1.0, 5.0, 2.0, 0.0, 3.0], vec![1, 5]).unwrap();

    let eval = evaluate(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x), &[]).expect("filter");
    let (y, zf) = eval.clone().into_pair();

    let (data, _) = real_tensor_parts(y, "output");
    approx_eq_slice(
        &data,
        &[0.3333333333, 2.0, 2.6666666667, 2.3333333333, 1.6666666667],
    );

    let (z_data, _) = real_tensor_parts(zf, "final-state");
    approx_eq_slice(&z_data, &[1.0, 1.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_reads_typed_integer_coefficients_signal_and_state_exactly() {
    let b = integer_tensor(IntegerStorage::I16(vec![1, 1]), vec![1, 2]);
    let a = integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]);
    let x = integer_tensor(IntegerStorage::I16(vec![2, 3]), vec![1, 2]);
    let zi = integer_tensor(IntegerStorage::I16(vec![7]), vec![1, 1]);

    let eval = evaluate(
        Value::Tensor(b),
        Value::Tensor(a),
        Value::Tensor(x),
        &[Value::Tensor(zi)],
    )
    .expect("filter");
    let (y, zf) = eval.into_pair();

    let (y_data, _) = real_tensor_parts(y, "output");
    approx_eq_slice(&y_data, &[9.0, 5.0]);

    let zf_data = match zf {
        Value::Tensor(t) => real_tensor_parts(Value::Tensor(t), "final-state").0,
        Value::Num(n) => vec![n],
        other => panic!("expected numeric final state, got {other:?}"),
    };
    approx_eq_slice(&zf_data, &[3.0]);
}

#[test]
fn filter_preserves_single_signal_output_and_column_state_layout() {
    let b = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let (y, zf) = evaluate(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x), &[])
        .expect("single filter")
        .into_pair();
    let Value::Tensor(y) = y else {
        panic!("expected tensor output")
    };
    let Value::Tensor(zf) = zf else {
        panic!("expected tensor final state")
    };
    assert_eq!(y.numeric_dtype(), NumericDType::F32);
    assert_eq!(zf.numeric_dtype(), NumericDType::F32);
    assert_eq!(zf.shape(), vec![1, 1]);
}

#[test]
fn filter_accepts_documented_logical_dimension_scalar() {
    let x = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    evaluate(
        Value::Num(1.0),
        Value::Num(1.0),
        Value::Tensor(x),
        &[
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            Value::Bool(true),
        ],
    )
    .expect("logical dimension");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_iir_impulse() {
    let alpha = 0.8;
    let b = Tensor::new(vec![1.0 - alpha], vec![1, 1]).unwrap();
    let a = Tensor::new(vec![1.0, -alpha], vec![1, 2]).unwrap();
    let x = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![1, 5]).unwrap();

    let eval = evaluate(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x), &[]).expect("filter");
    let (y, _) = eval.into_pair();

    let (data, _) = real_tensor_parts(y, "output");
    approx_eq_slice(&data, &[0.2, 0.16, 0.128, 0.1024, 0.08192]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_accepts_empty_typed_integer_initial_state_without_mirror_for_zero_order() {
    let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let zi = integer_tensor(IntegerStorage::I16(Vec::new()), vec![0, 0]);

    let eval = evaluate(
        Value::Tensor(b),
        Value::Tensor(a),
        Value::Tensor(x),
        &[Value::Tensor(zi)],
    )
    .expect("filter");
    let (y, zf) = eval.into_pair();

    let (data, _) = real_tensor_parts(y, "output");
    approx_eq_slice(&data, &[2.0, 3.0]);

    let (data, _) = real_tensor_parts(zf, "final-state");
    assert!(data.is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_with_initial_conditions() {
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x1 = Tensor::new(vec![1.0, 5.0, 2.0], vec![1, 3]).unwrap();

    let eval1 = evaluate(
        Value::Tensor(b.clone()),
        Value::Tensor(a.clone()),
        Value::Tensor(x1),
        &[],
    )
    .expect("filter");
    let (y1, zf1) = eval1.clone().into_pair();
    let (y1_data, _) = real_tensor_parts(y1, "output");
    approx_eq_slice(&y1_data, &[0.3333333333, 2.0, 2.6666666667]);

    let (zf_data, _) = real_tensor_parts(zf1.clone(), "final-state");
    approx_eq_slice(&zf_data, &[2.3333333333, 0.6666666667]);

    let x2 = Tensor::new(vec![0.0, 3.0], vec![1, 2]).unwrap();
    let eval2 = evaluate(
        Value::Tensor(b),
        Value::Tensor(a),
        Value::Tensor(x2),
        &[zf1],
    )
    .expect("filter");
    let (y2, zf2) = eval2.into_pair();

    let (y2_data, _) = real_tensor_parts(y2, "output");
    approx_eq_slice(&y2_data, &[2.3333333333, 1.6666666667]);

    let (zf2_data, _) = real_tensor_parts(zf2, "final-state");
    approx_eq_slice(&zf2_data, &[1.0, 1.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_accepts_empty_initial_placeholder() {
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new(vec![1.0, 5.0, 2.0, 0.0, 3.0], vec![1, 5]).unwrap();
    let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();

    let eval_default = evaluate(
        Value::Tensor(b.clone()),
        Value::Tensor(a.clone()),
        Value::Tensor(x.clone()),
        &[],
    )
    .expect("filter default");
    let eval_placeholder = evaluate(
        Value::Tensor(b),
        Value::Tensor(a),
        Value::Tensor(x),
        &[Value::Tensor(placeholder)],
    )
    .expect("filter with []");

    let (y_default, z_default) = eval_default.into_pair();
    let (y_placeholder, z_placeholder) = eval_placeholder.into_pair();

    let (y_def, _) = real_tensor_parts(y_default, "output");
    let (y_ph, _) = real_tensor_parts(y_placeholder, "output");
    approx_eq_slice(&y_def, &y_ph);

    let (z_def, _) = real_tensor_parts(z_default, "final-state");
    let (z_ph, _) = real_tensor_parts(z_placeholder, "final-state");
    approx_eq_slice(&z_def, &z_ph);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_rows_with_dimension_argument() {
    let b = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new(vec![1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0], vec![2, 4]).unwrap();
    let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();

    let eval = evaluate(
        Value::Tensor(b),
        Value::Tensor(a),
        Value::Tensor(x),
        &[Value::Tensor(placeholder), Value::Int(IntValue::I32(2))],
    )
    .expect("filter");
    let (y, zf) = eval.into_pair();

    let (data, shape) = real_tensor_parts(y, "output");
    assert_eq!(shape, vec![2, 4]);
    approx_eq_slice(&data, &[1.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0]);

    let (state_data, state_shape) = real_tensor_parts(zf, "final-state");
    assert_eq!(state_shape, vec![1, 2]);
    approx_eq_slice(&state_data, &[-4.0, -1.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_complex_signal() {
    let b = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 1.0)], vec![1, 2]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let t = [0.0, 1.0, 2.0, 3.0];
    let x: Vec<(f64, f64)> = t
        .iter()
        .map(|&ti| {
            let re = (std::f64::consts::PI * ti / 4.0).cos();
            let im = (std::f64::consts::PI * ti / 4.0).sin();
            (re, im)
        })
        .collect();
    let x_tensor = ComplexTensor::new(x, vec![1, 4]).unwrap();

    let eval = evaluate(
        Value::ComplexTensor(b),
        Value::Tensor(a),
        Value::ComplexTensor(x_tensor),
        &[],
    )
    .expect("filter");
    let (y, _) = eval.into_pair();

    let data = match y {
        Value::ComplexTensor(t) => t.materialize_f64(),
        other => panic!("expected complex tensor, got {other:?}"),
    };
    let root_half = std::f64::consts::FRAC_1_SQRT_2;
    let one_plus = 1.0 + root_half;
    approx_eq_complex(
        &data,
        &[
            (1.0, 0.0),
            (root_half, one_plus),
            (-root_half, one_plus),
            (-one_plus, root_half),
        ],
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_invalid_initial_shape() {
    let b = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let zi = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap();

    let err = error_message(
        evaluate(
            Value::Tensor(b),
            Value::Tensor(a),
            Value::Tensor(x),
            &[Value::Tensor(zi)],
        )
        .unwrap_err(),
    );
    assert!(err.contains("initial conditions"));
}
