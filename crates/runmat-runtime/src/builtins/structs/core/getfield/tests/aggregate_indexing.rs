use super::*;

#[test]
fn getfield_cell_field_parentheses_indexing_preserves_cell_container() {
    let payload = CellArray::new_with_shape(
        vec![Value::from("first"), Value::from("second")],
        vec![1, 2],
    )
    .expect("payload cell");
    let mut st = StructValue::new();
    st.fields.insert("values".to_string(), Value::Cell(payload));
    let index = CellArray::new_with_shape(vec![Value::Int(IntValue::U8(2))], vec![1, 1])
        .expect("index cell");

    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("cell parentheses index");
    let Value::Cell(cell) = result else {
        panic!("expected one-element cell result");
    };
    assert_eq!((cell.rows, cell.cols), (1, 1));
    assert_eq!(cell.data, vec![Value::from("second")]);
}

#[test]
fn getfield_logical_cell_indexing_uses_column_major_linear_positions() {
    let payload = CellArray::new_with_shape(
        vec![
            Value::from("row1-col1"),
            Value::from("row1-col2"),
            Value::from("row2-col1"),
            Value::from("row2-col2"),
        ],
        vec![2, 2],
    )
    .expect("payload cell");
    let mut st = StructValue::new();
    st.fields.insert("values".to_string(), Value::Cell(payload));
    let mask = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical mask");
    let index =
        CellArray::new_with_shape(vec![Value::LogicalArray(mask)], vec![1, 1]).expect("index cell");

    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("logical cell index");
    let Value::Cell(cell) = result else {
        panic!("expected cell result");
    };
    assert_eq!((cell.rows, cell.cols), (2, 1));
    assert_eq!(
        cell.data,
        vec![Value::from("row2-col1"), Value::from("row1-col2")]
    );
}

#[test]
fn getfield_matrix_shaped_cell_selection_rebuilds_row_major_storage() {
    let payload = CellArray::new_with_shape(
        vec![
            Value::from("a"),
            Value::from("c"),
            Value::from("b"),
            Value::from("d"),
        ],
        vec![2, 2],
    )
    .expect("payload cell");
    let selector = Tensor::new_integer(IntegerStorage::U8(vec![4, 2, 3, 1]), vec![2, 2])
        .expect("matrix selector");
    let index =
        CellArray::new_with_shape(vec![Value::Tensor(selector)], vec![1, 1]).expect("index cell");
    let mut st = StructValue::new();
    st.fields.insert("values".to_string(), Value::Cell(payload));

    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("matrix-shaped cell selection");
    let Value::Cell(cell) = result else {
        panic!("expected cell result");
    };
    assert_eq!(cell.shape, vec![2, 2]);
    assert_eq!(
        cell.data,
        vec![
            Value::from("d"),
            Value::from("b"),
            Value::from("c"),
            Value::from("a")
        ]
    );
}

#[test]
fn getfield_linear_cell_selection_accounts_for_nd_page_offsets() {
    let payload = CellArray::new_with_shape(
        vec![
            Value::from("page1-row1-col1"),
            Value::from("page1-row1-col2"),
            Value::from("page1-row2-col1"),
            Value::from("page1-row2-col2"),
            Value::from("page2-row1-col1"),
            Value::from("page2-row1-col2"),
            Value::from("page2-row2-col1"),
            Value::from("page2-row2-col2"),
        ],
        vec![2, 2, 2],
    )
    .expect("N-D payload cell");
    let selector =
        Tensor::new_integer(IntegerStorage::U8(vec![2, 6]), vec![1, 2]).expect("page selector");
    let index =
        CellArray::new_with_shape(vec![Value::Tensor(selector)], vec![1, 1]).expect("index cell");
    let mut st = StructValue::new();
    st.fields.insert("values".to_string(), Value::Cell(payload));

    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("N-D page selection");
    let Value::Cell(cell) = result else {
        panic!("expected cell result");
    };
    assert_eq!(cell.shape, vec![1, 2]);
    assert_eq!(
        cell.data,
        vec![
            Value::from("page1-row2-col1"),
            Value::from("page2-row2-col1")
        ]
    );
}
