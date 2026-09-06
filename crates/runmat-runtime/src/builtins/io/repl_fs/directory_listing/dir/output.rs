use runmat_value::{CharArray, StructValue, Value};

use crate::console::{record_console_line, ConsoleStream};
use crate::output_context::requested_output_count;
use crate::{make_cell, BuiltinResult};

use super::record::Record;

pub(super) fn value(records: Vec<Record>) -> BuiltinResult<Value> {
    let rows = records.len();
    let values = records.into_iter().map(struct_value).collect();
    make_cell(values, rows, 1).map_err(super::error::operation)
}

fn struct_value(record: Record) -> Value {
    let mut value = StructValue::new();
    value.insert("name", character_row(&record.name));
    value.insert("folder", character_row(&record.folder));
    value.insert("date", character_row(&record.date));
    value.insert("bytes", Value::Num(record.bytes));
    value.insert("isdir", Value::Bool(record.is_dir));
    value.insert("datenum", Value::Num(record.datenum));
    Value::Struct(value)
}

fn character_row(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

pub(super) fn emit(records: &[Record]) {
    let lines = records.iter().map(|record| {
        let size = if record.is_dir {
            "<DIR>".into()
        } else {
            format!("{:>10}", record.bytes as i64)
        };
        let mut name = record.name.clone();
        if record.is_dir && !name.ends_with(std::path::MAIN_SEPARATOR) {
            name.push(std::path::MAIN_SEPARATOR);
        }
        format!("{:<20} {:>10} {}", record.date, size, name)
    });
    let text = lines.collect::<Vec<_>>().join("\n");
    if !text.is_empty() {
        record_console_line(ConsoleStream::Stdout, text);
    }
}

pub(super) fn should_emit_stdout() -> bool {
    requested_output_count().is_none_or(|count| count == 0)
}
