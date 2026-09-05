use super::super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_scalar_pair() {
    let result = complex_call(Value::Num(3.0), vec![Value::Num(4.0)]).expect("complex");
    match result {
        Value::Complex(re, im) => {
            assert_eq!(re, 3.0);
            assert_eq!(im, 4.0);
        }
        other => panic!("expected Complex result, got {other:?}"),
    }
}

#[test]
fn complex_single_components_preserve_native_complex_single_storage() {
    let real = Tensor::from_f32(vec![0.1, 2.0], vec![1, 2]).unwrap();
    let imag = Tensor::from_f32(vec![0.2, -3.0], vec![1, 2]).unwrap();
    let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex tensor");
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.1_f32, 0.2_f32), (2.0_f32, -3.0_f32)])
    );
    assert_eq!(
        result.materialize_f64(),
        vec![(f64::from(0.1_f32), f64::from(0.2_f32)), (2.0, -3.0)]
    );
}

#[test]
fn complex_single_scalar_retains_class_as_complex_tensor() {
    let real = Tensor::from_f32(vec![0.1], vec![1, 1]).unwrap();
    let imag = Tensor::from_f32(vec![0.2], vec![1, 1]).unwrap();
    let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex");
    let Value::ComplexTensor(result) = result else {
        panic!("single complex scalar must retain its class");
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.1_f32, 0.2_f32)])
    );
}

#[test]
fn unary_complex_single_preserves_native_class_shape_and_empty_storage() {
    let real = Tensor::from_f32(vec![0.1, -2.0], vec![2, 1]).unwrap();
    let result = complex_call(Value::Tensor(real), Vec::new()).expect("complex");
    let Value::ComplexTensor(output) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(
        output.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.1_f32, 0.0), (-2.0, 0.0)])
    );
    let empty = Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
    let result = complex_call(Value::Tensor(empty), Vec::new()).expect("complex");
    let Value::ComplexTensor(output) = result else {
        panic!("expected empty complex single tensor");
    };
    assert_eq!(output.shape, vec![0, 3]);
    assert_eq!(output.as_f32_slice(), Some(&[][..]));
}

#[test]
fn complex_floating_composition_preserves_native_or_promoted_storage_without_materialization() {
    let real = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
    let imag = Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex");
    let Value::ComplexTensor(output) = result else {
        panic!("expected broadcast complex single tensor");
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(
        output.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(1.0, 2.0), (1.0, 3.0)])
    );
    let real = Tensor::from_f32(vec![0.1, 2.0], vec![1, 2]).unwrap();
    let imag = Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap();
    let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex");
    let Value::ComplexTensor(output) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        output.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.1_f32, 4.0_f32), (2.0_f32, 5.0_f32)])
    );
    let real = Tensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let imag = Tensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex");
    let Value::ComplexTensor(output) = result else {
        panic!("expected empty complex single tensor");
    };
    assert_eq!(output.shape, vec![0, 2]);
    assert_eq!(output.as_f32_slice(), Some(&[][..]));
}
