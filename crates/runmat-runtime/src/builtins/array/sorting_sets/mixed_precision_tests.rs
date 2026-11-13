use super::{intersect, setdiff, union};
use runmat_value::{ComplexTensor, NumericStorage, Tensor, Value};

#[derive(Clone, Copy)]
enum Operation {
    Union,
    Intersect,
    Difference,
}
async fn values(operation: Operation, a: Value, b: Value) -> Value {
    match operation {
        Operation::Union => union::evaluate(a, b, &[])
            .await
            .unwrap()
            .into_values_value(),
        Operation::Intersect => intersect::evaluate(a, b, &[])
            .await
            .unwrap()
            .into_values_value(),
        Operation::Difference => setdiff::evaluate(a, b, &[])
            .await
            .unwrap()
            .into_values_value(),
    }
}
#[test]
fn mixed_single_double_results_follow_common_class_or_setdiff_left_class() {
    futures::executor::block_on(async {
        for (operation, expected) in [
            (Operation::Union, vec![1.0f32, 2.0, 3.0]),
            (Operation::Intersect, vec![2.0]),
            (Operation::Difference, vec![1.0]),
        ] {
            for left_single in [true, false] {
                let a = if left_single {
                    Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap()
                } else {
                    Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()
                };
                let b = if left_single {
                    Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap()
                } else {
                    Tensor::from_f32(vec![2.0, 3.0], vec![2, 1]).unwrap()
                };
                let output = values(operation, Value::Tensor(a), Value::Tensor(b)).await;
                let single = left_single || !matches!(operation, Operation::Difference);
                if single {
                    assert!(matches!(output, Value::Tensor(_)));
                }
                let result =
                    crate::builtins::common::tensor::value_into_tensor_for("fixture", output)
                        .unwrap();
                assert_eq!(result.shape(), &[expected.len(), 1]);
                assert_eq!(
                    result.into_numeric_storage().unwrap(),
                    if single {
                        NumericStorage::F32(expected.clone())
                    } else {
                        NumericStorage::F64(expected.iter().copied().map(f64::from).collect())
                    }
                );
                let a = if left_single {
                    ComplexTensor::from_f32(vec![(1.0, 1.0), (2.0, 1.0)], vec![2, 1]).unwrap()
                } else {
                    ComplexTensor::new(vec![(1.0, 1.0), (2.0, 1.0)], vec![2, 1]).unwrap()
                };
                let b = if left_single {
                    ComplexTensor::new(vec![(2.0, 1.0), (3.0, 1.0)], vec![2, 1]).unwrap()
                } else {
                    ComplexTensor::from_f32(vec![(2.0, 1.0), (3.0, 1.0)], vec![2, 1]).unwrap()
                };
                let output =
                    values(operation, Value::ComplexTensor(a), Value::ComplexTensor(b)).await;
                if single {
                    let Value::ComplexTensor(result) = output else {
                        panic!("native complex single tensor required")
                    };
                    assert_eq!(result.shape(), &[expected.len(), 1]);
                    assert_eq!(
                        result
                            .as_f32_slice()
                            .unwrap()
                            .iter()
                            .copied()
                            .map(<(f32, f32)>::from)
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|value| (*value, 1.0))
                            .collect::<Vec<_>>()
                    );
                } else {
                    assert_eq!(output, Value::Complex(1.0, 1.0));
                }
            }
        }
    });
}
