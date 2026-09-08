use runmat_value::{CellArray, CharArray, Value};

pub(super) fn convert(array: CharArray) -> crate::BuiltinResult<Value> {
    let rows = array.rows;
    let columns = array.cols;
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * columns;
        let end = start + columns;
        let text = trimmed_row(&array.data[start..end]);
        values.push(Value::CharArray(CharArray::new_row(&text)));
    }
    CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(super::super::error::internal)
}

fn trimmed_row(characters: &[char]) -> String {
    let end = characters
        .iter()
        .rposition(|character| !trimmable_whitespace(*character))
        .map_or(0, |index| index + 1);
    characters[..end].iter().collect()
}

fn trimmable_whitespace(character: char) -> bool {
    character.is_whitespace() && character != '\u{00a0}'
}
