use runmat_value::{ComplexTensor, NumericDType, SparseTensor, Value};

use super::{flintmax, realmax, realmin};

#[test]
fn supports_single_and_double() {
    assert_eq!(realmax(Vec::new()).unwrap(), Value::Num(f64::MAX));
    for (output, expected) in [
        (
            realmin(vec![Value::from("single")]).unwrap(),
            f32::MIN_POSITIVE,
        ),
        (
            flintmax(vec![Value::from("single")]).unwrap(),
            2f32.powi(24),
        ),
    ] {
        let Value::Tensor(output) = output else {
            panic!("single limit must retain native single storage")
        };
        assert_eq!(output.as_f32_slice(), Some([expected].as_slice()));
        assert_eq!(output.shape, vec![1, 1]);
    }
}

#[test]
fn like_preserves_complexity_and_sparse_single_storage() {
    let complex = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let output = realmax(vec![Value::from("like"), Value::ComplexTensor(complex)])
        .expect("complex single like form");
    let Value::ComplexTensor(output) = output else {
        panic!("expected complex single scalar")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.shape, vec![1, 1]);

    let sparse = SparseTensor::new_f32(2, 2, vec![0, 1, 1], vec![0], vec![3.0]).unwrap();
    let output = realmin(vec![Value::from("like"), Value::SparseTensor(sparse)])
        .expect("sparse single like form");
    let Value::SparseTensor(output) = output else {
        panic!("expected sparse single scalar")
    };
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
    assert_eq!(output.as_f32_slice(), Some([f32::MIN_POSITIVE].as_slice()));
    assert_eq!((output.rows, output.cols), (1, 1));

    let sparse =
        SparseTensor::new_complex_f32(2, 1, vec![0, 1], vec![1], vec![(3.0, -4.0)]).unwrap();
    let output = flintmax(vec![Value::from("like"), Value::SparseTensor(sparse)])
        .expect("sparse complex single like form");
    let Value::SparseTensor(output) = output else {
        panic!("expected sparse complex single scalar")
    };
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
    assert!(output.is_complex());
    assert_eq!(
        output.as_complex_f32_slice(),
        Some([runmat_value::ComplexElement(2f32.powi(24), 0.0)].as_slice())
    );
}
