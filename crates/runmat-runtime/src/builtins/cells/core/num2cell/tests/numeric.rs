use super::run;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericStorage, Tensor,
    Value,
};

mod low_rank;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn grouped_rank_one_and_nd_numeric_arrays_keep_axes_and_column_major_order() {
    let vector =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![2]).unwrap();
    let Value::Cell(grouped) = run(Value::Tensor(vector), vec![Value::Num(1.0)]) else {
        panic!("grouped rank-one cell expected")
    };
    assert_eq!(grouped.shape, vec![1]);
    let Value::Tensor(block) = &grouped.data()[0] else {
        panic!("grouped exact integer block expected")
    };
    assert_eq!(block.shape(), &[2]);
    assert_eq!(
        block.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
    );

    let cube = Tensor::new((1..=8).map(f64::from).collect(), vec![2, 2, 2]).unwrap();
    let dims = Tensor::new_integer(IntegerStorage::U8(vec![3, 1]), vec![1, 2]).unwrap();
    let Value::Cell(grouped) = run(Value::Tensor(cube), vec![Value::Tensor(dims)]) else {
        panic!("grouped N-D cells expected")
    };
    assert_eq!(grouped.shape, vec![1, 2, 1]);
    for (block, expected) in grouped
        .data()
        .iter()
        .zip([vec![1.0, 5.0, 2.0, 6.0], vec![3.0, 7.0, 4.0, 8.0]])
    {
        let Value::Tensor(block) = block else {
            panic!("grouped N-D tensor expected")
        };
        assert_eq!(block.shape(), &[2, 1, 2]);
        assert_eq!(block.materialize_f64(), expected);
    }
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
    let input = CharArray::from_unicode_scalars(vec!['a', 'c', 'e', 'b', 'd', 'f'], 2, 3).unwrap();
    let dims = Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap();
    let Value::Cell(output) = run(Value::CharArray(input), vec![Value::Tensor(dims)]) else {
        panic!("expected cell");
    };
    let Value::CharArray(block) = output.get(0, 0).unwrap() else {
        panic!("expected character block");
    };
    assert_eq!((block.rows, block.cols), (3, 2));
    assert_eq!(
        block.data(),
        runmat_value::encode_utf16_code_units("abcdef")
    );
}

#[test]
fn empty_arrays_preserve_outer_and_grouped_block_shapes() {
    let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let Value::Cell(elements) = run(Value::Tensor(empty.clone()), Vec::new()) else {
        panic!("expected cell");
    };
    assert_eq!(elements.shape, vec![0, 3]);
    assert!(elements.data().is_empty());

    let Value::Cell(columns) = run(Value::Tensor(empty), vec![Value::Num(1.0)]) else {
        panic!("expected grouped cells");
    };
    assert_eq!(columns.shape, vec![1, 3]);
    for column in columns.data() {
        let Value::Tensor(column) = column else {
            panic!("expected empty numeric block");
        };
        assert_eq!(column.shape(), &[0, 1]);
        assert!(column.is_empty());
    }
}
