use super::*;
use runmat_builtins::{infer_catalog_call, MAX_CATALOG_ENTRY};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use runmat_value::{ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_native_integer_reduction_preserves_uint64_values_and_indices() {
    let input = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, 3, 2]),
        vec![2, 2],
    )
    .expect("input");
    let (values, indices) = evaluate(Value::Tensor(input), &[])
        .expect("max")
        .into_pair();
    assert_eq!(
        values,
        Value::Tensor(
            Tensor::new_integer(
                runmat_value::IntegerStorage::U64(vec![u64::MAX, 3]),
                vec![1, 2],
            )
            .expect("values"),
        )
    );
    assert_eq!(
        indices,
        Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).expect("indices")),
    );
}

#[test]
fn catalog_reduction_class_and_shape_match_runtime_uint64_and_double_index() {
    let inference = infer_catalog_call(
        &MAX_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(2), Some(2)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    let input = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, 3, 2]),
        vec![2, 2],
    )
    .expect("input");
    let (value, index) = evaluate(Value::Tensor(input), &[])
        .expect("runtime max")
        .into_pair();
    assert_eq!(inference.outputs.len(), 2);
    for output in &inference.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(1), Some(2)]));
    }
    assert_eq!(
        inference.outputs[0].numeric().expect("value class").class,
        NumericClass::UInt64
    );
    assert_eq!(
        inference.outputs[1].numeric().expect("index class").class,
        NumericClass::Double
    );
    let Value::Tensor(value) = value else {
        panic!("runtime value shape/class")
    };
    assert_eq!(value.shape(), [1, 2]);
    assert_eq!(
        value.into_numeric_storage().expect("integer storage"),
        runmat_value::NumericStorage::U64(vec![u64::MAX, 3])
    );
    let Value::Tensor(index) = index else {
        panic!("runtime index shape/class")
    };
    assert_eq!(index.shape(), [1, 2]);
    assert_eq!(
        index.into_numeric_storage().expect("double storage"),
        runmat_value::NumericStorage::F64(vec![2.0, 1.0])
    );
}

#[test]
fn complex_integer_rejects_with_max_owner_before_numeric_materialization() {
    let storage =
        IntegerComplexStorage::new(IntegerStorage::I16(vec![3]), IntegerStorage::I16(vec![1]))
            .expect("complex integer storage");
    let input = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex integer");
    let error = evaluate(Value::ComplexTensor(input), &[]).expect_err("unsupported class");
    assert_eq!(error.identifier(), MAX_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_native_integer_abs_all_uses_exact_int64_minimum() {
    let input = Tensor::new_integer(
        runmat_value::IntegerStorage::I64(vec![i64::MIN, -3, 3]),
        vec![3, 1],
    )
    .expect("input");
    let args = vec![
        placeholder(),
        Value::from("all"),
        Value::from("ComparisonMethod"),
        Value::from("abs"),
    ];
    let (values, indices) = evaluate(Value::Tensor(input), &args)
        .expect("max")
        .into_pair();
    assert_eq!(values, Value::Int(IntValue::I64(i64::MIN)));
    assert_eq!(indices, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_native_integer_empty_array_retains_its_class() {
    let input = Tensor::new_integer(runmat_value::IntegerStorage::U32(Vec::new()), vec![0, 0])
        .expect("input");
    let (values, indices) = evaluate(Value::Tensor(input), &[])
        .expect("max")
        .into_pair();
    assert_eq!(
        values,
        Value::Tensor(
            Tensor::new_integer(runmat_value::IntegerStorage::U32(Vec::new()), vec![0, 0])
                .expect("values"),
        )
    );
    assert_eq!(
        indices,
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("indices")),
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_native_integer_empty_all_reduction_remains_empty() {
    let input = Tensor::new_integer(runmat_value::IntegerStorage::I16(Vec::new()), vec![0, 0])
        .expect("input");
    let values = evaluate(Value::Tensor(input), &[placeholder(), Value::from("all")])
        .expect("max")
        .into_value();
    assert_eq!(
        values,
        Value::Tensor(
            Tensor::new_integer(runmat_value::IntegerStorage::I16(Vec::new()), vec![0, 0])
                .expect("values"),
        )
    );
}
