use runmat_value::{CellArray, CharArray, Utf16CodeUnit, Value};

pub(super) fn convert(array: CharArray) -> crate::BuiltinResult<Value> {
    let rows = array.rows;
    let columns = array.cols;
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * columns;
        let end = start + columns;
        let code_units = trimmed_row(&array.data()[start..end]);
        values.push(Value::CharArray(
            CharArray::from_code_units(code_units.to_vec(), 1, code_units.len())
                .map_err(super::super::error::internal)?,
        ));
    }
    CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(super::super::error::internal)
}

fn trimmed_row(characters: &[Utf16CodeUnit]) -> &[Utf16CodeUnit] {
    let end = characters
        .iter()
        .rposition(|character| !trimmable_whitespace(*character))
        .map_or(0, |index| index + 1);
    &characters[..end]
}

fn trimmable_whitespace(character: Utf16CodeUnit) -> bool {
    character.is_whitespace() && character.raw() != 0x00A0
}
