use runmat_value::{CellArray, CharArray, StringArray, StringElement, Value};

use crate::BuiltinResult;

use super::{ensure_vector, text_atom, GroupingInput, KeyOrder};
use crate::builtins::array::grouping::grp2idx::error;

fn decode_text(units: &[runmat_value::Utf16CodeUnit]) -> BuiltinResult<String> {
    runmat_value::decode_utf16_code_units(units).map_err(|source| {
        let mut error = error::invalid(format!("grp2idx: invalid character text: {source}"));
        error.source = Some(Box::new(source));
        error
    })
}

pub(super) fn strings(value: StringArray) -> BuiltinResult<GroupingInput> {
    let len = value.data().len();
    let rows = value
        .data()
        .iter()
        .map(|value| value.as_str().and_then(text_atom))
        .collect();
    let value = StringArray::from_shared_storage(value.into_parts().0, vec![len, 1].into())
        .map_err(error::invalid)?;
    Ok(GroupingInput::new(
        Value::StringArray(value),
        rows,
        KeyOrder::FirstAppearance,
    ))
}

pub(super) fn scalar(value: StringElement) -> GroupingInput {
    let rows = vec![value.as_str().and_then(text_atom)];
    GroupingInput::new(Value::String(value), rows, KeyOrder::FirstAppearance)
}

pub(super) fn cellstr(value: CellArray) -> BuiltinResult<GroupingInput> {
    ensure_vector(&[value.rows, value.cols], "cell array")?;
    let len = value.data().len();
    let mut rows = Vec::with_capacity(len);
    for item in value.data() {
        let text: String = match item {
            Value::CharArray(chars) if chars.rows <= 1 => decode_text(chars.data())?,
            _ => {
                return Err(error::invalid(
                    "grp2idx: cell input must contain character vectors",
                ))
            }
        };
        rows.push(text_atom(&text));
    }
    let value = CellArray::from_shared_storage(value.into_parts().0, vec![len, 1].into())
        .map_err(error::invalid)?;
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
            let text = decode_text(&value.data()[start..start + value.cols])?;
            Ok(text_atom(text.trim_end()))
        })
        .collect::<BuiltinResult<_>>()?;
    Ok(GroupingInput::new(
        Value::CharArray(value),
        rows,
        KeyOrder::FirstAppearance,
    ))
}
