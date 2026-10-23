use runmat_value::{CharArray, Utf16CodeUnit, Value};

use crate::console::{record_console_line, ConsoleStream};
use crate::output_context::requested_output_count;
use crate::BuiltinResult;

pub(super) fn value(rows: &[String]) -> BuiltinResult<Value> {
    if crate::compatibility::runmat_extensions_enabled() || cfg!(windows) {
        padded_rows(rows)
    } else {
        unix_character_vector(rows)
    }
}

fn padded_rows(rows: &[String]) -> BuiltinResult<Value> {
    let width = rows
        .iter()
        .map(|row| row.encode_utf16().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(rows.len() * width);
    for row in rows {
        data.extend(row.encode_utf16().map(Utf16CodeUnit::from_raw));
        data.extend(std::iter::repeat_n(
            Utf16CodeUnit::from_raw(b' ' as u16),
            width.saturating_sub(row.encode_utf16().count()),
        ));
    }
    char_array(data, rows.len(), width)
}

fn unix_character_vector(rows: &[String]) -> BuiltinResult<Value> {
    let text = rows.join("  ");
    let data = text
        .encode_utf16()
        .map(Utf16CodeUnit::from_raw)
        .collect::<Vec<_>>();
    let width = data.len();
    char_array(data, usize::from(width > 0), width)
}

fn char_array(data: Vec<Utf16CodeUnit>, rows: usize, columns: usize) -> BuiltinResult<Value> {
    CharArray::from_code_units(data, rows, columns)
        .map(Value::CharArray)
        .map_err(|error| super::error::operation(format!("ls: {error}")))
}

pub(super) fn emit(rows: &[String]) -> BuiltinResult<()> {
    if !rows.is_empty() {
        record_console_line(ConsoleStream::Stdout, rows.join("\n"))?;
    }
    Ok(())
}

pub(super) fn should_emit_stdout() -> bool {
    requested_output_count().is_none_or(|count| count == 0)
}
