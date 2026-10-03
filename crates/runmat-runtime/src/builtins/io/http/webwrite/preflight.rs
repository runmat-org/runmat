//! Validate host-known options before body or query operands can read back.

use super::*;

pub(super) fn preflight_host_arguments(args: &[Value]) -> BuiltinResult<()> {
    let mut options = WebWriteOptions::default();
    let mut query = Vec::new();
    let mut index = 0;
    if let Some(Value::Struct(fields)) = args.first() {
        for (name, value) in &fields.fields {
            if !crate::dispatcher::value_contains_gpu(value) {
                process_name_value_pair(name, value, &mut options, &mut query)?;
            }
        }
        index = 1;
    } else if let Some(Value::Cell(cell)) = args.first() {
        if cell.cols != 2 {
            return Err(webwrite_error(
                "webwrite: cell array of query parameters must have two columns",
            ));
        }
        if !crate::dispatcher::value_contains_gpu(&args[0]) {
            append_query_from_cell(cell, &mut query)?;
        }
        index = 1;
    }
    if !(args.len() - index).is_multiple_of(2) {
        return Err(webwrite_error_with(
            &WEBWRITE_ERROR_MISSING_OPTION_VALUE,
            "webwrite: missing value for name-value argument",
        ));
    }
    while index < args.len() {
        if !crate::dispatcher::value_contains_gpu(&args[index]) {
            let name = expect_string_scalar(
                &args[index],
                "webwrite: parameter names must be character vectors or strings",
            )?;
            if !crate::dispatcher::value_contains_gpu(&args[index + 1]) {
                process_name_value_pair(&name, &args[index + 1], &mut options, &mut query)?;
            }
        }
        index += 2;
    }
    Ok(())
}
