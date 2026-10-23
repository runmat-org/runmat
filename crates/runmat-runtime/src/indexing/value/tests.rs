use super::*;
use crate::indexing::plan::{build_assignment_plan, build_index_plan};
use crate::indexing::selectors::SliceSelector;
use futures::executor::block_on;
use runmat_value::{CharArray, StringArray};

#[test]
fn bounded_three_subscript_string_store_uses_full_selector_rank() {
    let plan = build_index_plan(
        &[
            SliceSelector::Scalar(1),
            SliceSelector::Scalar(1),
            SliceSelector::Scalar(2),
        ],
        3,
        &[1, 1, 3],
    )
    .unwrap();
    let source =
        Value::StringArray(StringArray::new(vec![String::new(); 3], vec![1, 1, 3]).unwrap());
    let Value::StringArray(updated) =
        block_on(assign_with_plan(source, &plan, Value::String("mat".into()))).unwrap()
    else {
        panic!("string store must preserve its array carrier")
    };
    assert_eq!(updated.shape, vec![1, 1, 3]);
    assert_eq!(
        updated.data(),
        vec![String::new(), "mat".into(), String::new()]
    );
}

#[test]
fn scalar_numeric_indexed_store_bridge_preserves_class_for_bounded_growth_and_deletion() {
    use crate::indexing::plan::{build_indexed_store_plan, IndexedStorePlanMode};
    use crate::indexing::write_slice::delete_tensor_with_plan;
    use runmat_value::{IntValue, IntegerStorage};

    let cases = [
        (Value::Num(9.0), Value::Num(7.0)),
        (
            Value::Int(IntValue::U64(9_007_199_254_740_993)),
            Value::Int(IntValue::U64(u64::MAX)),
        ),
    ];
    let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
    for (base, rhs) in cases {
        let bounded = block_on(build_indexed_store_plan(
            1,
            0,
            0,
            &[Value::Num(1.0)],
            &[1, 1],
            IndexedStorePlanMode::Grow,
        ))
        .unwrap();
        let Value::Tensor(updated) =
            block_on(assign_with_plan(base.clone(), &bounded, rhs.clone())).unwrap()
        else {
            panic!("numeric scalar bounded assignment must return tensor");
        };
        let growth = block_on(build_indexed_store_plan(
            1,
            0,
            0,
            &[Value::Num(3.0)],
            &[1, 1],
            IndexedStorePlanMode::Grow,
        ))
        .unwrap();
        let Value::Tensor(grown) =
            block_on(assign_with_plan(base.clone(), &growth, rhs.clone())).unwrap()
        else {
            panic!("numeric scalar growth must return tensor");
        };
        let deletion = block_on(build_indexed_store_plan(
            1,
            0,
            0,
            &[Value::Num(1.0)],
            &[1, 1],
            IndexedStorePlanMode::Bounded,
        ))
        .unwrap();
        let Value::Tensor(deleted) =
            delete_tensor_with_plan(numeric_scalar_tensor(base).unwrap(), &deletion, &empty)
                .unwrap()
        else {
            panic!("numeric scalar deletion must return tensor");
        };
        assert_eq!(deleted.len(), 0);
        match (
            updated.integer_storage(),
            grown.integer_storage(),
            deleted.integer_storage(),
        ) {
            (None, None, None) => {
                assert_eq!(updated.as_f64_slice(), Some(&[7.0][..]));
                assert_eq!(grown.as_f64_slice(), Some(&[9.0, 0.0, 7.0][..]));
            }
            (
                Some(IntegerStorage::U64(updated)),
                Some(IntegerStorage::U64(grown)),
                Some(IntegerStorage::U64(deleted)),
            ) => {
                assert_eq!(updated, &[u64::MAX]);
                assert_eq!(grown, &[9_007_199_254_740_993, 0, u64::MAX]);
                assert!(deleted.is_empty());
            }
            _ => panic!("numeric scalar class changed"),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_shape_dispatch_preserves_low_rank_metadata() {
    for input_shape in [Vec::new(), vec![2]] {
        let count = input_shape.iter().product::<usize>();
        let real = Value::Tensor(Tensor::new(vec![7.0; count], input_shape.clone()).unwrap());
        let complex = Value::ComplexTensor(
            runmat_value::ComplexTensor::new(vec![(7.0, 1.0); count], input_shape.clone()).unwrap(),
        );
        assert_eq!(shape(&real), Some(input_shape.clone()));
        assert_eq!(shape(&complex), Some(input_shape));
    }
}

#[test]
fn repeated_scalar_reads_preserve_runtime_family_and_shape() {
    let plan = build_index_plan(
        &[SliceSelector::LinearIndices {
            values: vec![1, 1, 1].into(),
            output_shape: vec![1, 3].into(),
        }],
        1,
        &[1, 1],
    )
    .expect("plan");
    assert!(
        matches!(read_with_plan(Value::Num(4.0), &plan).unwrap(), Value::Tensor(v) if v.shape() == [1, 3])
    );
    assert!(
        matches!(read_with_plan(Value::Bool(true), &plan).unwrap(), Value::LogicalArray(v) if *v.data == [1, 1, 1])
    );
    assert!(
        matches!(read_with_plan(Value::String("entry".into()), &plan).unwrap(), Value::StringArray(v) if v.data() == ["entry", "entry", "entry"])
    );
}

#[test]
fn character_and_logical_scalar_assignment_grow_through_shared_plan() {
    let plan = build_assignment_plan(&[SliceSelector::Scalar(3)], 1, &[1, 1]).unwrap();
    assert!(
        matches!(block_on(assign_with_plan(Value::Bool(true), &plan, Value::Bool(false))).unwrap(), Value::LogicalArray(v) if *v.data == [1, 0, 0])
    );
    assert!(
        matches!(block_on(assign_with_plan(Value::CharArray(CharArray::new_row("a")), &plan, Value::CharArray(CharArray::new_row("z")))).unwrap(), Value::CharArray(v) if v.row_string().as_deref() == Some("a z"))
    );
    assert!(
        matches!(block_on(assign_with_plan(Value::String("a".into()),&plan,Value::String("z".into()))).unwrap(),Value::StringArray(value) if value.data()==["a","","z"])
    );
    let raw = CharArray::from_code_units_with_shape(
        vec![runmat_value::Utf16CodeUnit::from_raw(0xd800)],
        vec![1, 1],
    )
    .unwrap();
    let Value::CharArray(value) = block_on(assign_with_plan(
        Value::CharArray(raw),
        &plan,
        Value::String("\0".into()),
    ))
    .unwrap() else {
        panic!("raw character carrier")
    };
    assert_eq!(
        value
            .iter_column_major()
            .map(runmat_value::Utf16CodeUnit::raw)
            .collect::<Vec<_>>(),
        [0xd800, 32, 0]
    );
}

#[test]
fn numeric_assignment_growth_uses_the_shared_checked_plan_shape() {
    let scalar_plan = build_assignment_plan(&[SliceSelector::Scalar(2)], 1, &[1, 1]).unwrap();
    let Value::Tensor(scalar) = block_on(assign_with_plan(
        Value::Num(1.0),
        &scalar_plan,
        Value::Num(3.0),
    ))
    .unwrap() else {
        panic!("scalar growth must materialize a numeric tensor")
    };
    assert_eq!(scalar.shape(), &[1, 2]);
    assert_eq!(scalar.materialize_f64(), vec![1.0, 3.0]);

    let vector = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let vector_plan = build_assignment_plan(&[SliceSelector::Scalar(4)], 1, &[1, 2]).unwrap();
    let Value::Tensor(vector) = block_on(assign_with_plan(
        Value::Tensor(vector),
        &vector_plan,
        Value::Num(9.0),
    ))
    .unwrap() else {
        panic!("vector growth must retain a numeric tensor")
    };
    assert_eq!(vector.shape(), &[1, 4]);
    assert_eq!(vector.materialize_f64(), vec![1.0, 2.0, 0.0, 9.0]);

    let matrix = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let matrix_plan = build_assignment_plan(
        &[SliceSelector::Scalar(1), SliceSelector::Scalar(3)],
        2,
        &[2, 1],
    )
    .unwrap();
    let Value::Tensor(matrix) = block_on(assign_with_plan(
        Value::Tensor(matrix),
        &matrix_plan,
        Value::Num(7.0),
    ))
    .unwrap() else {
        panic!("matrix growth must retain a numeric tensor")
    };
    assert_eq!(matrix.shape(), &[2, 3]);
    assert_eq!(matrix.materialize_f64(), vec![1.0, 2.0, 0.0, 0.0, 7.0, 0.0]);

    // Two subscripts view the trailing dimensions as one indexed axis. A
    // bounded write keeps the original rank; actual growth collapses that
    // view and re-strides all original columns into the new leading extent.
    let original = Tensor::new((1..=8).map(f64::from).collect(), vec![2, 2, 2]).unwrap();
    let bounded = build_assignment_plan(
        &[SliceSelector::Scalar(1), SliceSelector::Scalar(4)],
        2,
        original.shape(),
    )
    .unwrap();
    let Value::Tensor(updated) = block_on(assign_with_plan(
        Value::Tensor(original),
        &bounded,
        Value::Num(9.0),
    ))
    .unwrap() else {
        panic!("bounded collapsed-view assignment must retain a tensor")
    };
    assert_eq!(updated.shape(), &[2, 2, 2]);
    assert_eq!(
        updated.materialize_f64(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 8.0]
    );

    let growing = build_assignment_plan(
        &[SliceSelector::Scalar(3), SliceSelector::Scalar(5)],
        2,
        updated.shape(),
    )
    .unwrap();
    let Value::Tensor(grown) = block_on(assign_with_plan(
        Value::Tensor(updated),
        &growing,
        Value::Num(10.0),
    ))
    .unwrap() else {
        panic!("grown collapsed-view assignment must retain a tensor")
    };
    assert_eq!(grown.shape(), &[3, 5]);
    assert_eq!(
        grown.materialize_f64(),
        vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 9.0, 8.0, 0.0, 0.0, 0.0, 10.0]
    );
}

#[test]
fn leading_dimension_growth_restrides_typed_and_complex_columns() {
    let plan = build_assignment_plan(
        &[SliceSelector::Scalar(3), SliceSelector::Scalar(2)],
        2,
        &[2, 2],
    )
    .unwrap();
    let original = runmat_value::IntegerStorage::U64(vec![1, 2, u64::MAX, 4]);
    let value = Value::Tensor(Tensor::new_integer(original, vec![2, 2]).unwrap());
    let Value::Tensor(grown) = block_on(assign_with_plan(
        value,
        &plan,
        Value::Int(runmat_value::IntValue::U64(9)),
    ))
    .unwrap() else {
        panic!("typed matrix growth must retain its numeric carrier")
    };
    assert_eq!(grown.shape(), &[3, 2]);
    assert_eq!(
        grown.integer_storage(),
        Some(&runmat_value::IntegerStorage::U64(vec![
            1,
            2,
            0,
            u64::MAX,
            4,
            9
        ]))
    );

    let complex = runmat_value::ComplexTensor::from_f32(
        vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
        vec![2, 2],
    )
    .unwrap();
    let Value::ComplexTensor(grown) = block_on(assign_with_plan(
        Value::ComplexTensor(complex),
        &plan,
        Value::Complex(9.0, 8.0),
    ))
    .unwrap() else {
        panic!("complex matrix growth must retain its carrier")
    };
    assert_eq!(grown.shape(), &[3, 2]);
    assert_eq!(
        grown.as_f32_slice().unwrap(),
        &[
            runmat_value::ComplexElement(1.0, 1.0),
            runmat_value::ComplexElement(2.0, 2.0),
            runmat_value::ComplexElement(0.0, 0.0),
            runmat_value::ComplexElement(3.0, 3.0),
            runmat_value::ComplexElement(4.0, 4.0),
            runmat_value::ComplexElement(9.0, 8.0),
        ]
    );

    let integer_complex = runmat_value::ComplexTensor::new_integer(
        runmat_value::IntegerComplexStorage::new(
            runmat_value::IntegerStorage::U64(vec![1, 2, u64::MAX, 4]),
            runmat_value::IntegerStorage::U64(vec![5, 6, 7, 8]),
        )
        .unwrap(),
        vec![2, 2],
    )
    .unwrap();
    let Value::ComplexTensor(grown) = block_on(assign_with_plan(
        Value::ComplexTensor(integer_complex),
        &plan,
        Value::Int(runmat_value::IntValue::U64(9)),
    ))
    .unwrap() else {
        panic!("exact complex matrix growth must retain its carrier")
    };
    assert_eq!(grown.shape(), &[3, 2]);
    let storage = grown.integer_storage().unwrap();
    assert_eq!(
        storage.real(),
        &runmat_value::IntegerStorage::U64(vec![1, 2, 0, u64::MAX, 4, 9])
    );
    assert_eq!(
        storage.imag(),
        &runmat_value::IntegerStorage::U64(vec![5, 6, 0, 7, 8, 0])
    );
}
