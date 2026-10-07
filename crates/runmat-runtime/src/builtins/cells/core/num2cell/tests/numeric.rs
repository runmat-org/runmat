use super::run;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericStorage, Tensor,
    Value,
};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_numeric_inputs_preserve_cell_geometry_and_native_values() {
    let scalar = Tensor::from_f32(vec![1.25], Vec::new()).unwrap();
    let Value::Cell(scalar) = run(Value::Tensor(scalar), Vec::new()) else {
        panic!("scalar cell expected")
    };
    assert!(scalar.shape.is_empty());
    let Value::Tensor(value) = &scalar.data[0] else {
        panic!("single scalar retains native storage")
    };
    assert_eq!(value.shape(), &[1, 1]);
    assert_eq!(value.as_f32_slice(), Some(&[1.25][..]));

    let vector =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1 << 63]), vec![2]).unwrap();
    let Value::Cell(vector) = run(Value::Tensor(vector), Vec::new()) else {
        panic!("vector cell expected")
    };
    assert_eq!(vector.shape, vec![2]);
    assert_eq!(
        vector.data.as_slice(),
        &[
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Int(IntValue::U64(1 << 63))
        ]
    );
}

#[test]
fn converts_numeric_matrix_in_matlab_order() {
    let matrix = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let Value::Cell(output) = run(Value::Tensor(matrix), Vec::new()) else {
        panic!("expected cell");
    };
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(output.get(0, 0).unwrap(), Value::Num(1.0));
    assert_eq!(output.get(1, 0).unwrap(), Value::Num(3.0));
    assert_eq!(output.get(0, 1).unwrap(), Value::Num(2.0));
    assert_eq!(output.get(1, 1).unwrap(), Value::Num(4.0));
}

#[test]
fn dimension_order_permutates_grouped_block() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let dims = Tensor::new_integer(IntegerStorage::U8(vec![2, 1]), vec![1, 2]).unwrap();
    let Value::Cell(output) = run(Value::Tensor(matrix), vec![Value::Tensor(dims)]) else {
        panic!("expected cell");
    };
    let Value::Tensor(block) = output.get(0, 0).unwrap() else {
        panic!("expected grouped tensor");
    };
    assert_eq!(block.shape(), &[3, 2]);
    assert_eq!(block.materialize_f64(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn preserves_single_and_wide_integer_storage() {
    let single = Tensor::from_f32(vec![1.25], vec![1, 1]).unwrap();
    let Value::Cell(single) = run(Value::Tensor(single), Vec::new()) else {
        panic!("expected cell");
    };
    let Value::Tensor(value) = single.get(0, 0).unwrap() else {
        panic!("single scalar remains typed tensor");
    };
    assert_eq!(
        value.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![1.25])
    );

    let integers = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
    let Value::Cell(integers) = run(Value::Tensor(integers), Vec::new()) else {
        panic!("expected cell");
    };
    assert_eq!(
        integers.get(0, 0).unwrap(),
        Value::Int(IntValue::U64(u64::MAX))
    );
}

#[test]
fn complex_single_keeps_its_component_class() {
    let input = ComplexTensor::from_f32(vec![(1.5, -2.0)], vec![1, 1]).unwrap();
    let Value::Cell(output) = run(Value::ComplexTensor(input), Vec::new()) else {
        panic!("expected cell");
    };
    let Value::ComplexTensor(value) = output.get(0, 0).unwrap() else {
        panic!("single complex scalar remains typed tensor");
    };
    assert!(matches!(value.complex_storage(), ComplexStorage::F32(_)));
}

#[test]
fn character_blocks_follow_requested_dimension_order() {
    let input = CharArray::new(vec!['a', 'c', 'e', 'b', 'd', 'f'], 2, 3).unwrap();
    let dims = Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap();
    let Value::Cell(output) = run(Value::CharArray(input), vec![Value::Tensor(dims)]) else {
        panic!("expected cell");
    };
    let Value::CharArray(block) = output.get(0, 0).unwrap() else {
        panic!("expected character block");
    };
    assert_eq!((block.rows, block.cols), (3, 2));
    assert_eq!(block.data, vec!['a', 'b', 'c', 'd', 'e', 'f']);
}

#[test]
fn empty_arrays_preserve_outer_and_grouped_block_shapes() {
    let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let Value::Cell(elements) = run(Value::Tensor(empty.clone()), Vec::new()) else {
        panic!("expected cell");
    };
    assert_eq!(elements.shape, vec![0, 3]);
    assert!(elements.data.is_empty());

    let Value::Cell(columns) = run(Value::Tensor(empty), vec![Value::Num(1.0)]) else {
        panic!("expected grouped cells");
    };
    assert_eq!(columns.shape, vec![1, 3]);
    for column in columns.data {
        let Value::Tensor(column) = column else {
            panic!("expected empty numeric block");
        };
        assert_eq!(column.shape(), &[0, 1]);
        assert!(column.is_empty());
    }
}
