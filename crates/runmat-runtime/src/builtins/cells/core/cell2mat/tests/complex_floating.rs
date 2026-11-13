use super::*;
use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor, NumericDType};

#[test]
fn native_complex_single_concatenation_preserves_payload_class_and_shape() {
    let blocks = [
        vec![(1.25_f32, -2.5_f32), (3.5, 4.25)],
        vec![(5.0, 6.0), (-7.0, 8.0)],
    ];
    let values = blocks
        .into_iter()
        .map(|block| Value::ComplexTensor(ComplexTensor::from_f32(block, vec![2, 1]).unwrap()))
        .collect();
    let Value::ComplexTensor(result) = run(crate::make_cell(values, 1, 2).unwrap()).unwrap() else {
        panic!("complex output");
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(result.shape(), &[2, 2]);
    let ComplexStorage::F32(storage) = result.complex_storage() else {
        panic!("native single storage")
    };
    assert_eq!(
        &storage[..],
        &[
            ComplexElement(1.25, -2.5),
            ComplexElement(3.5, 4.25),
            ComplexElement(5.0, 6.0),
            ComplexElement(-7.0, 8.0)
        ]
    );
}

#[test]
fn empty_complex_single_blocks_keep_their_class() {
    let block = Value::ComplexTensor(ComplexTensor::from_f32(vec![], vec![0, 3]).unwrap());
    let Value::ComplexTensor(result) = run(crate::make_cell(vec![block], 1, 1).unwrap()).unwrap()
    else {
        panic!("complex output")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(result.shape(), &[0, 3]);
}

#[test]
fn mixed_complex_floating_classes_are_rejected() {
    let single =
        Value::ComplexTensor(ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap());
    let double = Value::Complex(3.0, 4.0);
    let error = run(crate::make_cell(vec![single, double], 1, 2).unwrap()).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:cell2mat:InvalidContents"));
}
