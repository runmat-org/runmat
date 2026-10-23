use super::*;

pub fn datetime_char_array(value: &Value) -> BuiltinResult<Option<CharArray>> {
    let Some(array) = datetime_string_array(value)? else {
        return Ok(None);
    };
    let width = array
        .data()
        .iter()
        .map(|s| s.encode_utf16().count())
        .max()
        .unwrap_or(0);
    let rows = array.data().len();
    let mut data = vec![Utf16CodeUnit::from_raw(b' ' as u16); rows * width];
    for (row, text) in array.data().iter().enumerate() {
        for (col, unit) in text.encode_utf16().enumerate() {
            data[row * width + col] = Utf16CodeUnit::from_raw(unit);
        }
    }
    let out = CharArray::from_code_units(data, rows, width)
        .map_err(|err| datetime_error(format!("datetime: {err}")))?;
    Ok(Some(out))
}
