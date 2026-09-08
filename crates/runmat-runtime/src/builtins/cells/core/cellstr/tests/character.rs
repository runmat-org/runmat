use super::*;
use runmat_value::CharArray;

#[test]
fn converts_character_rows_and_removes_trailing_spaces() {
    let input = CharArray::new(
        vec!['c', 'a', 't', ' ', 'd', 'o', 'g', ' ', 'f', 'o', 'x', ' '],
        3,
        4,
    )
    .unwrap();
    let output = cell(call(Value::CharArray(input)).unwrap());
    assert_eq!(output.shape, vec![3, 1]);
    assert_eq!(strings(&output), vec!["cat", "dog", "fox"]);
}

#[test]
fn empty_character_rows_keep_the_column_cell_shape() {
    let input = CharArray::new(Vec::new(), 0, 5).unwrap();
    let output = cell(call(Value::CharArray(input)).unwrap());
    assert_eq!(output.shape, vec![0, 1]);
    assert!(output.data.is_empty());
}

#[test]
fn all_space_row_becomes_an_empty_character_vector() {
    let input = CharArray::new(vec![' '; 3], 1, 3).unwrap();
    let output = cell(call(Value::CharArray(input)).unwrap());
    let Value::CharArray(row) = &output.data[0] else {
        panic!("expected character row");
    };
    assert_eq!(row.shape, vec![1, 0]);
}

#[test]
fn removes_trailing_tab_but_retains_nonbreaking_space() {
    let input = CharArray::new(vec!['a', '\u{00a0}', '\t'], 1, 3).unwrap();
    let output = cell(call(Value::CharArray(input)).unwrap());
    assert_eq!(strings(&output), vec!["a\u{00a0}"]);
}
