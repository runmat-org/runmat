use super::super::*;
use runmat_builtins::ArrayfunGpuBinary;
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn varying_double_integer_pair_is_not_whole_array_equivalent() {
    let integer = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 7]),
            vec![1, 2],
        )
        .unwrap(),
    );
    let double = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap());
    for operation in [
        ArrayfunGpuBinary::Add,
        ArrayfunGpuBinary::Subtract,
        ArrayfunGpuBinary::Multiply,
        ArrayfunGpuBinary::RightDivide,
        ArrayfunGpuBinary::LeftDivide,
    ] {
        let operation = ArrayfunGpuCallback::Binary(operation);
        assert!(!equivalent(
            operation,
            "mapped",
            &[integer.clone(), double.clone()]
        ));
        assert!(!equivalent(
            operation,
            "mapped",
            &[double.clone(), integer.clone()]
        ));
        assert!(equivalent(
            operation,
            "mapped",
            &[integer.clone(), Value::Num(2.0)]
        ));
        assert!(equivalent(
            operation,
            "mapped",
            &[integer.clone(), integer.clone()]
        ));
    }
}

#[test]
fn mixed_float_and_broadcast_shapes_remain_admitted() {
    let single = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let double = Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap());
    assert!(equivalent(
        ArrayfunGpuCallback::Binary(ArrayfunGpuBinary::Add),
        "plus",
        &[single.clone(), double.clone()]
    ));
    assert_eq!(shape(&[single, double]).unwrap(), Some(vec![2, 3]));
}
