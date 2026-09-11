use super::*;

#[test]
fn getfield_linear_indexing_follows_matlab_orientation_rules() {
    fn select_numeric(
        target_values: Vec<u64>,
        target_shape: Vec<usize>,
        selector_values: Vec<u64>,
        selector_shape: Vec<usize>,
    ) -> Tensor {
        let target =
            Tensor::new_integer(IntegerStorage::U64(target_values), target_shape).expect("target");
        let selector = Tensor::new_integer(IntegerStorage::U64(selector_values), selector_shape)
            .expect("selector");
        let mut st = StructValue::new();
        st.fields
            .insert("values".to_string(), Value::Tensor(target));
        let index = CellArray::new_with_shape(vec![Value::Tensor(selector)], vec![1, 1])
            .expect("index cell");
        let result = run_getfield(
            Value::Struct(st),
            vec![Value::from("values"), Value::Cell(index)],
        )
        .expect("numeric linear selection");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor result");
        };
        tensor
    }

    fn select_logical(target_shape: Vec<usize>, mask: Vec<u8>, mask_shape: Vec<usize>) -> Tensor {
        let target = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX - 3, u64::MAX - 2, u64::MAX - 1, u64::MAX]),
            target_shape,
        )
        .expect("target");
        let mask = LogicalArray::new(mask, mask_shape).expect("mask");
        let mut st = StructValue::new();
        st.fields
            .insert("values".to_string(), Value::Tensor(target));
        let index = CellArray::new_with_shape(vec![Value::LogicalArray(mask)], vec![1, 1])
            .expect("index cell");
        let result = run_getfield(
            Value::Struct(st),
            vec![Value::from("values"), Value::Cell(index)],
        )
        .expect("logical linear selection");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor result");
        };
        tensor
    }

    let row_from_column = select_numeric(
        vec![u64::MAX - 2, u64::MAX - 1, u64::MAX],
        vec![1, 3],
        vec![3, 1],
        vec![2, 1],
    );
    assert_eq!(row_from_column.shape, vec![1, 2]);
    assert_eq!(
        row_from_column.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX - 2]))
    );

    let column_from_row = select_numeric(
        vec![u64::MAX - 2, u64::MAX - 1, u64::MAX],
        vec![3, 1],
        vec![3, 1],
        vec![1, 2],
    );
    assert_eq!(column_from_row.shape, vec![2, 1]);

    let matrix_from_row = select_numeric(
        vec![u64::MAX - 3, u64::MAX - 2, u64::MAX - 1, u64::MAX],
        vec![2, 2],
        vec![4, 1],
        vec![1, 2],
    );
    assert_eq!(matrix_from_row.shape, vec![1, 2]);
    let matrix_from_column = select_numeric(
        vec![u64::MAX - 3, u64::MAX - 2, u64::MAX - 1, u64::MAX],
        vec![2, 2],
        vec![4, 1],
        vec![2, 1],
    );
    assert_eq!(matrix_from_column.shape, vec![2, 1]);
    let matrix_from_matrix = select_numeric(
        vec![u64::MAX - 3, u64::MAX - 2, u64::MAX - 1, u64::MAX],
        vec![2, 2],
        vec![1, 2, 3, 4],
        vec![2, 2],
    );
    assert_eq!(matrix_from_matrix.shape, vec![2, 2]);

    let scalar_from_row = select_numeric(vec![u64::MAX], vec![1, 1], vec![1, 1], vec![1, 2]);
    assert_eq!(scalar_from_row.shape, vec![1, 2]);
    let scalar_from_column = select_numeric(vec![u64::MAX], vec![1, 1], vec![1, 1], vec![2, 1]);
    assert_eq!(scalar_from_column.shape, vec![2, 1]);

    let empty_from_row = select_numeric(
        vec![u64::MAX - 2, u64::MAX - 1, u64::MAX],
        vec![1, 3],
        Vec::new(),
        vec![0, 1],
    );
    assert_eq!(empty_from_row.shape, vec![1, 0]);
    assert_eq!(
        empty_from_row.integer_storage(),
        Some(&IntegerStorage::U64(Vec::new()))
    );
    let empty_from_matrix = select_numeric(
        vec![u64::MAX - 3, u64::MAX - 2, u64::MAX - 1, u64::MAX],
        vec![2, 2],
        Vec::new(),
        vec![1, 0],
    );
    assert_eq!(empty_from_matrix.shape, vec![1, 0]);

    let logical_row = select_logical(vec![1, 4], vec![1, 0, 1, 0], vec![4, 1]);
    assert_eq!(logical_row.shape, vec![1, 2]);
    assert_eq!(
        logical_row.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX - 3, u64::MAX - 1]))
    );
    let logical_column = select_logical(vec![4, 1], vec![1, 0, 1, 0], vec![1, 4]);
    assert_eq!(logical_column.shape, vec![2, 1]);
    let logical_matrix = select_logical(vec![2, 2], vec![1, 0, 1, 0], vec![2, 2]);
    assert_eq!(logical_matrix.shape, vec![2, 1]);
    let logical_empty_row = select_logical(vec![1, 4], vec![0, 0, 0, 0], vec![4, 1]);
    assert_eq!(logical_empty_row.shape, vec![1, 0]);
    let logical_empty_matrix = select_logical(vec![2, 2], vec![0, 0, 0, 0], vec![2, 2]);
    assert_eq!(logical_empty_matrix.shape, vec![0, 1]);
}
