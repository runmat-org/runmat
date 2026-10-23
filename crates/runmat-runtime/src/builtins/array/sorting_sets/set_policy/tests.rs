use super::*;
use crate::builtins::array::sorting_sets::{intersect, ismember, setdiff, union};
use crate::builtins::common::tensor::value_into_tensor_for;
use futures::executor::block_on;
use runmat_value::{
    CharArray, ComplexTensor, IntegerStorage, LogicalArray, NumericDType, StringArray, Tensor,
};

fn data(single: bool, complex: bool, left: bool, shape: Vec<usize>) -> Value {
    let real = if left {
        vec![f64::NAN, f64::NAN, 1.0, 1.0, 0.0, -0.0, f64::INFINITY]
    } else {
        vec![f64::NAN, 1.0, -0.0, f64::INFINITY]
    };
    if complex {
        let mut values = real
            .into_iter()
            .map(|value| (value, 0.0))
            .collect::<Vec<_>>();
        if left {
            values[1] = (1.0, f64::NAN);
        }
        Value::ComplexTensor(if single {
            ComplexTensor::from_f32(
                values
                    .into_iter()
                    .map(|(re, im)| (re as f32, im as f32))
                    .collect(),
                shape,
            )
            .unwrap()
        } else {
            ComplexTensor::new(values, shape).unwrap()
        })
    } else {
        Value::Tensor(if single {
            Tensor::from_f32(real.into_iter().map(|value| value as f32).collect(), shape).unwrap()
        } else {
            Tensor::new(real, shape).unwrap()
        })
    }
}
fn positions(value: Value) -> Vec<f64> {
    value_into_tensor_for("fixture", value)
        .unwrap()
        .materialize_f64()
}
fn numbers(value: Value, single: bool, complex: bool) -> Vec<f64> {
    if complex {
        let Value::ComplexTensor(value) = value else {
            panic!("complex storage required")
        };
        assert_eq!(value.as_f32_slice().is_some(), single);
        value
            .materialize_f64()
            .into_iter()
            .map(|(re, im)| {
                if re.is_nan() || im.is_nan() {
                    f64::NAN
                } else {
                    re
                }
            })
            .collect()
    } else {
        let value = value_into_tensor_for("fixture", value).unwrap();
        assert_eq!(
            value.numeric_dtype(),
            if single {
                NumericDType::F32
            } else {
                NumericDType::F64
            }
        );
        value.materialize_f64()
    }
}
#[test]
fn distinct_nan_policy_covers_real_complex_single_double_elements_and_rows() {
    block_on(async {
        for single in [false, true] {
            for complex in [false, true] {
                for rows in [false, true] {
                    for stable in [false, true] {
                        let options = if rows {
                            vec![
                                Value::from("rows"),
                                Value::from(if stable { "stable" } else { "sorted" }),
                            ]
                        } else {
                            vec![Value::from(if stable { "stable" } else { "sorted" })]
                        };
                        let a = || data(single, complex, true, vec![7, 1]);
                        let b = || data(single, complex, false, vec![4, 1]);
                        let (c, ia, ib) = union::evaluate(a(), b(), &options)
                            .await
                            .unwrap()
                            .into_triple();
                        let values = numbers(c, single, complex);
                        assert_eq!(values.len(), 6);
                        assert_eq!(values.iter().filter(|v| v.is_nan()).count(), 3);
                        assert_eq!(
                            positions(ia),
                            if stable {
                                vec![1.0, 2.0, 3.0, 5.0, 7.0]
                            } else {
                                vec![5.0, 3.0, 7.0, 1.0, 2.0]
                            }
                        );
                        assert_eq!(positions(ib), vec![1.0]);
                        let (c, ia, ib) = intersect::evaluate(a(), b(), &options)
                            .await
                            .unwrap()
                            .into_triple();
                        let values = numbers(c, single, complex);
                        assert_eq!(
                            values,
                            if stable {
                                vec![1.0, 0.0, f64::INFINITY]
                            } else {
                                vec![0.0, 1.0, f64::INFINITY]
                            }
                        );
                        assert_eq!(
                            positions(ia),
                            if stable {
                                vec![3.0, 5.0, 7.0]
                            } else {
                                vec![5.0, 3.0, 7.0]
                            }
                        );
                        assert_eq!(
                            positions(ib),
                            if stable {
                                vec![2.0, 3.0, 4.0]
                            } else {
                                vec![3.0, 2.0, 4.0]
                            }
                        );
                        let (c, ia) = setdiff::evaluate(a(), b(), &options)
                            .await
                            .unwrap()
                            .into_pair();
                        let values = numbers(c, single, complex);
                        assert_eq!(values.len(), 2);
                        assert!(values.iter().all(|value| value.is_nan()));
                        assert_eq!(positions(ia), vec![1.0, 2.0]);
                        let member_options = if rows {
                            vec![Value::from("rows")]
                        } else {
                            vec![]
                        };
                        let (mask, locations) = ismember::evaluate(a(), b(), &member_options)
                            .await
                            .unwrap()
                            .into_pair();
                        let Value::LogicalArray(mask) = mask else {
                            panic!("logical array required")
                        };
                        assert_eq!(mask.data, vec![0, 0, 1, 1, 1, 1, 1]);
                        assert_eq!(
                            positions(locations),
                            vec![0.0, 0.0, 2.0, 2.0, 3.0, 3.0, 4.0]
                        );
                    }
                }
            }
        }
    });
}
#[test]
fn public_element_orientation_and_setdiff_left_class_do_not_copy_a_through_single() {
    block_on(async {
        for column_b in [false, true] {
            let a = || {
                Value::Tensor(
                    Tensor::new_integer(
                        IntegerStorage::U64(vec![9_007_199_254_740_993, 2, 4]),
                        vec![1, 3],
                    )
                    .unwrap(),
                )
            };
            let b = || {
                Value::Tensor(
                    Tensor::new_integer(
                        IntegerStorage::U64(vec![2, 3, 5]),
                        if column_b { vec![3, 1] } else { vec![1, 3] },
                    )
                    .unwrap(),
                )
            };
            let result = union::evaluate(a(), b(), &[]).await.unwrap();
            let value = value_into_tensor_for("fixture", result.into_values_value()).unwrap();
            assert_eq!(value.shape(), if column_b { &[5, 1] } else { &[1, 5] });
            assert!(matches!(
                value.integer_storage(),
                Some(IntegerStorage::U64(_))
            ));
            let result = setdiff::evaluate(a(), b(), &[]).await.unwrap();
            let value = value_into_tensor_for("fixture", result.into_values_value()).unwrap();
            assert_eq!(value.shape(), &[1, 2]);
            assert_eq!(
                value.integer_storage(),
                Some(&IntegerStorage::U64(vec![4, 9_007_199_254_740_993]))
            );
        }
        let exact = 1.0 + f64::EPSILON;
        let (result, ia) = setdiff::evaluate(
            Value::Tensor(Tensor::new(vec![exact, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
            &[],
        )
        .await
        .unwrap()
        .into_pair();
        let result = value_into_tensor_for("fixture", result).unwrap();
        assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(result.shape(), &[1, 1]);
        assert_eq!(result.materialize_f64(), vec![exact]);
        assert_eq!(positions(ia), vec![1.0]);
        let result = setdiff::evaluate(
            Value::ComplexTensor(
                ComplexTensor::new(vec![(exact, exact), (2.0, 0.0)], vec![1, 2]).unwrap(),
            ),
            Value::ComplexTensor(ComplexTensor::from_f32(vec![(2.0, 0.0)], vec![1, 1]).unwrap()),
            &[],
        )
        .await
        .unwrap()
        .into_values_value();
        assert_eq!(result, Value::Complex(exact, exact));
        let result = setdiff::evaluate(
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap()),
            Value::Bool(false),
            &[],
        )
        .await
        .unwrap()
        .into_values_value();
        assert_eq!(result, Value::Bool(true));
        let chars = union::evaluate(
            Value::CharArray(CharArray::new_row("cab")),
            Value::CharArray(CharArray::new_row("bd")),
            &[],
        )
        .await
        .unwrap()
        .into_values_value();
        let Value::CharArray(chars) = chars else {
            panic!("character result required")
        };
        assert_eq!(chars.row_string().as_deref(), Some("abcd"));
        let strings = intersect::evaluate(
            Value::StringArray(
                StringArray::new(vec!["c".into(), "a".into(), "b".into()], vec![1, 3]).unwrap(),
            ),
            Value::StringArray(StringArray::new(vec!["b".into(), "c".into()], vec![1, 2]).unwrap()),
            &[],
        )
        .await
        .unwrap()
        .into_values_value();
        let Value::StringArray(strings) = strings else {
            panic!("string result required")
        };
        assert_eq!(strings.shape(), &[1, 2]);
        assert_eq!(strings.data(), vec!["b".to_owned(), "c".to_owned()]);
    });
}

#[test]
fn missing_in_a_later_column_makes_every_row_occurrence_distinct() {
    block_on(async {
        let a =
            || Value::Tensor(Tensor::new(vec![1.0, 1.0, f64::NAN, f64::NAN], vec![2, 2]).unwrap());
        let b = || Value::Tensor(Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).unwrap());
        let options = [Value::from("rows")];
        let (c, ia, ib) = union::evaluate(a(), b(), &options)
            .await
            .unwrap()
            .into_triple();
        assert_eq!(
            value_into_tensor_for("fixture", c).unwrap().shape(),
            &[3, 2]
        );
        assert_eq!(positions(ia), vec![1.0, 2.0]);
        assert_eq!(positions(ib), vec![1.0]);
        let (c, ia, ib) = intersect::evaluate(a(), b(), &options)
            .await
            .unwrap()
            .into_triple();
        assert_eq!(
            value_into_tensor_for("fixture", c).unwrap().shape(),
            &[0, 2]
        );
        assert!(positions(ia).is_empty() && positions(ib).is_empty());
        let (c, ia) = setdiff::evaluate(a(), b(), &options)
            .await
            .unwrap()
            .into_pair();
        assert_eq!(
            value_into_tensor_for("fixture", c).unwrap().shape(),
            &[2, 2]
        );
        assert_eq!(positions(ia), vec![1.0, 2.0]);
        let (mask, loc) = ismember::evaluate(a(), b(), &options)
            .await
            .unwrap()
            .into_pair();
        assert_eq!(
            mask,
            Value::LogicalArray(LogicalArray::new(vec![0, 0], vec![2, 1]).unwrap())
        );
        assert_eq!(positions(loc), vec![0.0, 0.0]);
    });
}
