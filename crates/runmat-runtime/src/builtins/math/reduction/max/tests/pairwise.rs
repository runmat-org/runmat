use super::*;
use runmat_builtins::{infer_catalog_call, MAX_CATALOG_ENTRY};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use runmat_value::{
    ComplexStorage, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericStorage,
};

#[test]
fn catalog_pairwise_single_broadcast_shape_and_class_match_runtime() {
    let single = |shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    let inference = infer_catalog_call(
        &MAX_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                single(vec![Some(2), Some(1)]),
                single(vec![Some(1), Some(2)]),
            ],
            literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert_eq!(inference.outputs.len(), 1);
    assert_eq!(
        inference.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        inference.outputs[0].numeric().expect("class").class,
        NumericClass::Single
    );
    let left = Tensor::from_f32(vec![3.0, 1.0], vec![2, 1]).expect("left");
    let right = Tensor::from_f32(vec![2.0, 4.0], vec![1, 2]).expect("right");
    let actual = evaluate(Value::Tensor(left), &[Value::Tensor(right)])
        .expect("runtime max")
        .into_value();
    let Value::Tensor(actual) = actual else {
        panic!("runtime pairwise tensor")
    };
    assert_eq!(actual.shape(), [2, 2]);
    assert_eq!(
        actual.into_numeric_storage().expect("single storage"),
        runmat_value::NumericStorage::F32(vec![3.0, 2.0, 4.0, 4.0])
    );
}

#[test]
fn scalar_complex_pairwise_zero_imaginary_returns_real_scalar() {
    let actual = evaluate(Value::Complex(3.0, 0.0), &[Value::Complex(2.0, 0.0)])
        .expect("pairwise complex")
        .into_value();
    assert_eq!(actual, Value::Num(3.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_elementwise_broadcast() {
    let lhs = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
    let rhs = Tensor::new(vec![2.0, 3.0, 5.0], vec![3, 1]).unwrap();
    let eval = evaluate(Value::Tensor(lhs), &[Value::Tensor(rhs)]).expect("evaluate");
    let (values, indices) = eval.into_pair();
    match values {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [3, 3]);
            assert_eq!(
                [
                    t.materialize_f64()[0],
                    t.materialize_f64()[3],
                    t.materialize_f64()[6]
                ],
                [2.0, 4.0, 7.0]
            );
            assert_eq!(
                [
                    t.materialize_f64()[1],
                    t.materialize_f64()[4],
                    t.materialize_f64()[7]
                ],
                [3.0, 4.0, 7.0]
            );
            assert_eq!(
                [
                    t.materialize_f64()[2],
                    t.materialize_f64()[5],
                    t.materialize_f64()[8]
                ],
                [5.0, 5.0, 7.0]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match indices {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [3, 3]);
            assert_eq!(
                [
                    t.materialize_f64()[0],
                    t.materialize_f64()[3],
                    t.materialize_f64()[6]
                ],
                [2.0, 1.0, 1.0]
            );
            assert_eq!(
                [
                    t.materialize_f64()[1],
                    t.materialize_f64()[4],
                    t.materialize_f64()[7]
                ],
                [2.0, 1.0, 1.0]
            );
            assert_eq!(
                [
                    t.materialize_f64()[2],
                    t.materialize_f64()[5],
                    t.materialize_f64()[8]
                ],
                [2.0, 2.0, 1.0]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_elementwise_preserves_native_single_storage() {
    let lhs = Tensor::from_f32(vec![3.5, -2.0], vec![2, 1]).expect("lhs");
    let rhs = Tensor::from_f32(vec![1.25, 4.0], vec![2, 1]).expect("rhs");
    let (values, indices) = evaluate(Value::Tensor(lhs), &[Value::Tensor(rhs)])
        .expect("evaluate")
        .into_pair();
    let Value::Tensor(values) = values else {
        panic!("expected tensor values");
    };
    assert_eq!(
        values.into_numeric_storage().expect("storage"),
        NumericStorage::F32(vec![3.5, 4.0])
    );
    assert_eq!(
        indices,
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("indices"))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_elementwise_preserves_native_complex_single_storage() {
    let lhs = ComplexTensor::from_f32(vec![(3.0, 4.0), (1.0, 0.0)], vec![2, 1]).expect("lhs");
    let rhs = ComplexTensor::from_f32(vec![(4.0, 0.0), (0.0, 2.0)], vec![2, 1]).expect("rhs");
    let values = evaluate(Value::ComplexTensor(lhs), &[Value::ComplexTensor(rhs)])
        .expect("evaluate")
        .into_value();
    let Value::ComplexTensor(values) = values else {
        panic!("expected complex tensor values");
    };
    assert_eq!(
        values.into_complex_storage(),
        ComplexStorage::F32(vec![(3.0, 4.0), (0.0, 2.0)].into())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_pairwise_public_form_rejects_second_output() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = max_builtin(Value::Num(1.0), vec![Value::Num(2.0)])
        .expect_err("pairwise max has one output");
    assert_eq!(error.identifier(), MAX_ERROR_INVALID_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_elementwise_abs_keeps_uint64_comparison_exact() {
    let lhs = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
        .expect("lhs");
    let rhs = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_992]), vec![1, 1])
        .expect("rhs");
    let args = vec![
        Value::Tensor(rhs),
        Value::from("ComparisonMethod"),
        Value::from("abs"),
    ];
    let (values, indices) = evaluate(Value::Tensor(lhs), &args)
        .expect("evaluate")
        .into_pair();
    assert_eq!(values, Value::Int(IntValue::U64(9_007_199_254_740_993)));
    assert_eq!(indices, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_scalar_double_and_integer_compare_exactly_in_either_order() {
    let rhs = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, 4]),
        vec![1, 2],
    )
    .expect("rhs");
    let (values, indices) = evaluate(Value::Num(9_007_199_254_740_992.0), &[Value::Tensor(rhs)])
        .expect("evaluate")
        .into_pair();
    assert_eq!(
        values,
        Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_992]),
                vec![1, 2],
            )
            .expect("values")
        )
    );
    assert_eq!(
        indices,
        Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).expect("indices"))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_integer_scalar_nan_obeys_include_and_omit_modes() {
    let integer =
        Tensor::new_integer(IntegerStorage::I16(vec![7, -3]), vec![1, 2]).expect("integer");
    let included = evaluate(Value::Tensor(integer.clone()), &[Value::Num(f64::NAN)])
        .expect("include")
        .into_value();
    assert_eq!(
        included,
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![0, 0]), vec![1, 2]).expect("included")
        )
    );

    let omitted = evaluate(
        Value::Tensor(integer.clone()),
        &[Value::Num(f64::NAN), Value::from("omitnan")],
    )
    .expect("omit")
    .into_value();
    assert_eq!(omitted, Value::Tensor(integer));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_integer_pairwise_rejects_incompatible_classes_and_float_shapes() {
    let integer =
        Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).expect("integer");
    let mixed = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("mixed");
    let error = evaluate(Value::Tensor(integer.clone()), &[Value::Tensor(mixed)])
        .expect_err("mixed integer classes reject");
    assert_eq!(error.identifier(), MAX_ERROR_INVALID_INPUT.identifier);

    let single = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single");
    let error = evaluate(Value::Tensor(integer.clone()), &[Value::Tensor(single)])
        .expect_err("scalar single rejects");
    assert_eq!(error.identifier(), MAX_ERROR_INVALID_INPUT.identifier);

    let doubles = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("doubles");
    let error = evaluate(Value::Tensor(integer), &[Value::Tensor(doubles)])
        .expect_err("nonscalar double rejects");
    assert_eq!(error.identifier(), MAX_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_elementwise_abs_comparison() {
    let lhs = Tensor::new(vec![-2.0, 1.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![1.5, -3.0], vec![2, 1]).unwrap();
    let args = vec![
        Value::Tensor(rhs),
        Value::from("ComparisonMethod"),
        Value::from("abs"),
    ];
    let eval = evaluate(Value::Tensor(lhs), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    match values {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![-2.0, -3.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match indices {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![1.0, 2.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_elementwise_omitnan_chooses_non_nan_side() {
    let lhs = Tensor::new(vec![f64::NAN, 2.0, f64::NAN], vec![3, 1]).unwrap();
    let rhs = Tensor::new(vec![3.0, f64::NAN, f64::NAN], vec![3, 1]).unwrap();
    let eval = evaluate(
        Value::Tensor(lhs),
        &[Value::Tensor(rhs), Value::from("omitnan")],
    )
    .expect("evaluate");
    let (values, indices) = eval.into_pair();
    match values {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [3, 1]);
            assert_eq!(t.materialize_f64()[0], 3.0);
            assert_eq!(t.materialize_f64()[1], 2.0);
            assert!(t.materialize_f64()[2].is_nan());
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match indices {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64()[0], 2.0);
            assert_eq!(t.materialize_f64()[1], 1.0);
            assert_eq!(t.materialize_f64()[2], 1.0);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_complex_auto_comparison() {
    let lhs = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let rhs = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
    let eval = evaluate(Value::ComplexTensor(lhs), &[Value::ComplexTensor(rhs)]).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Complex(1.0, 2.0));
    assert_eq!(indices, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_scalar_pair_arguments() {
    let args = vec![Value::Num(2.0)];
    let result = max_builtin(Value::Num(3.0), args).expect("max");
    assert_eq!(result, Value::Num(3.0));
}

#[test]
fn max_scalar_complex_value_reads_typed_integer_complex_storage_without_mirror() {
    let storage =
        IntegerComplexStorage::new(IntegerStorage::I16(vec![-7]), IntegerStorage::I16(vec![2]))
            .expect("complex integer storage");
    let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");

    assert_eq!(
        scalar_complex_value(&Value::ComplexTensor(tensor)),
        Some((-7.0, 2.0))
    );
}
