use runmat_value::{StringArray, Value};

use super::{GroupColumn, VariableResult};

pub(super) fn string_columns(
    name: &str,
    value: StringArray,
    split: bool,
) -> VariableResult<Vec<GroupColumn>> {
    let rows = value.rows();
    let columns = value.cols();
    if !split || columns <= 1 || rows == 1 {
        let rows = value.data.len();
        let value =
            StringArray::new(value.data, vec![rows, 1]).map_err(|error| error.to_string())?;
        return GroupColumn::vector(name, Value::StringArray(value)).map(|column| vec![column]);
    }
    (0..columns)
        .map(|column| {
            let data = (0..rows)
                .map(|row| value.data[row + column * rows].clone())
                .collect();
            let value = StringArray::new(data, vec![rows, 1]).map_err(|error| error.to_string())?;
            GroupColumn::vector(format!("{name}{}", column + 1), Value::StringArray(value))
        })
        .collect()
}
