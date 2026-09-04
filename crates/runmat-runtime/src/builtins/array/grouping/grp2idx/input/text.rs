use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::BuiltinResult;

use super::{ensure_vector, text_atom, GroupingInput, KeyOrder};
use crate::builtins::array::grouping::grp2idx::error;

pub(super) fn strings(value: StringArray) -> BuiltinResult<GroupingInput> {
    let len = value.data.len();
    let rows = value.data.iter().map(|value| text_atom(value)).collect();
    let value = StringArray::new(value.data, vec![len, 1]).map_err(error::invalid)?;
    Ok(GroupingInput::new(
        Value::StringArray(value),
        rows,
        KeyOrder::FirstAppearance,
    ))
}

pub(super) fn scalar(value: String) -> GroupingInput {
    GroupingInput::new(
        Value::String(value.clone()),
        vec![text_atom(&value)],
        KeyOrder::FirstAppearance,
    )
}

pub(super) fn cellstr(value: CellArray) -> BuiltinResult<GroupingInput> {
    ensure_vector(&[value.rows, value.cols], "cell array")?;
    let len = value.data.len();
    let mut rows = Vec::with_capacity(len);
    for item in &value.data {
        let text: String = match item {
            Value::CharArray(chars) if chars.rows <= 1 => chars.data.iter().collect(),
            _ => {
                return Err(error::invalid(
                    "grp2idx: cell input must contain character vectors",
                ))
            }
        };
        rows.push(text_atom(&text));
    }
    let value = CellArray::new(value.data, len, 1).map_err(error::invalid)?;
    Ok(GroupingInput::new(
        Value::Cell(value),
        rows,
        KeyOrder::FirstAppearance,
    ))
}

pub(super) fn characters(value: CharArray) -> BuiltinResult<GroupingInput> {
    if value.shape.len() > 2 {
        return Err(error::invalid(
            "grp2idx: character input must be two-dimensional",
        ));
    }
    let rows = (0..value.rows)
        .map(|row| {
            let start = row * value.cols;
            let text = value.data[start..start + value.cols]
                .iter()
                .collect::<String>();
            text_atom(text.trim_end())
        })
        .collect();
    Ok(GroupingInput::new(
        Value::CharArray(value),
        rows,
        KeyOrder::FirstAppearance,
    ))
}
