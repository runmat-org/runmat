use super::super::*;
use runmat_builtins::ArrayfunGpuBinary;
use runmat_value::{IntegerStorage, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn host_and_gpu_admission_share_low_rank_operand_geometry() {
    use crate::builtins::acceleration::gpu::arrayfun::input::ArrayData;
    use runmat_value::ComplexTensor;

    for (dimensions, expected, expanded) in [
        (vec![], vec![1, 1], vec![1, 1]),
        (vec![2], vec![2], vec![1, 2]),
    ] {
        let count = dimensions.iter().product();
        let inputs = [
            Value::Tensor(Tensor::from_f32(vec![2.0; count], dimensions.clone()).unwrap()),
            Value::ComplexTensor(
                ComplexTensor::from_f32(vec![(2.0, 3.0); count], dimensions).unwrap(),
            ),
        ];
        for input in inputs {
            let host = ArrayData::from_value(input.clone()).unwrap();
            assert_eq!(host.shape_vec(), expected);
            assert_eq!(
                shape(crate::call::FunctionArgumentView::from_slice(
                    std::slice::from_ref(&input)
                ))
                .unwrap(),
                Some(expected.clone())
            );
            assert_eq!(
                shape(crate::call::FunctionArgumentView::from_slice(&[
                    input,
                    Value::Num(1.0)
                ]))
                .unwrap(),
                Some(expanded.clone())
            );
        }
    }
}

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
            crate::call::FunctionArgumentView::from_slice(&[integer.clone(), double.clone()])
        ));
        assert!(!equivalent(
            operation,
            "mapped",
            crate::call::FunctionArgumentView::from_slice(&[double.clone(), integer.clone()])
        ));
        assert!(equivalent(
            operation,
            "mapped",
            crate::call::FunctionArgumentView::from_slice(&[integer.clone(), Value::Num(2.0)])
        ));
        assert!(equivalent(
            operation,
            "mapped",
            crate::call::FunctionArgumentView::from_slice(&[integer.clone(), integer.clone()])
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
        crate::call::FunctionArgumentView::from_slice(&[single.clone(), double.clone()])
    ));
    assert_eq!(
        shape(crate::call::FunctionArgumentView::from_slice(&[
            single, double
        ]))
        .unwrap(),
        Some(vec![2, 3])
    );
}
