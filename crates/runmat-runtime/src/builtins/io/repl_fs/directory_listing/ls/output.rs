use runmat_value::{CharArray, Value};

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
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(rows.len() * width);
    for row in rows {
        data.extend(row.chars());
        data.extend(std::iter::repeat_n(
            ' ',
            width.saturating_sub(row.chars().count()),
        ));
    }
    char_array(data, rows.len(), width)
}

fn unix_character_vector(rows: &[String]) -> BuiltinResult<Value> {
    let text = rows.join("  ");
    let width = text.chars().count();
    char_array(text.chars().collect(), usize::from(width > 0), width)
}

fn char_array(data: Vec<char>, rows: usize, columns: usize) -> BuiltinResult<Value> {
    CharArray::new(data, rows, columns)
        .map(Value::CharArray)
        .map_err(|error| super::error::operation(format!("ls: {error}")))
}

pub(super) fn emit(rows: &[String]) {
    if !rows.is_empty() {
        record_console_line(ConsoleStream::Stdout, rows.join("\n"));
    }
}

pub(super) fn should_emit_stdout() -> bool {
    requested_output_count().is_none_or(|count| count == 0)
}
