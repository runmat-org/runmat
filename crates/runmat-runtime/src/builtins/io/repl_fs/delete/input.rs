use super::*;

pub(super) fn parse_delete_options<'a>(
    args: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<(crate::call::FunctionArgumentView<'a>, Option<bool>)> {
    if args.len() >= 2
        && args.last().and_then(option_name).is_some_and(|name| {
            DeleteOption::from_name(&name) == Some(DeleteOption::ResolveSymbolicLinks)
        })
    {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: ResolveSymbolicLinks requires a scalar logical or numeric value",
        ));
    }
    if args.len() >= 3 {
        if let Some(name) = option_name(&args[args.len() - 2]) {
            if DeleteOption::from_name(&name) == Some(DeleteOption::ResolveSymbolicLinks) {
                return Ok((
                    args.range(..args.len() - 2)?,
                    Some(exact_zero_one(&args[args.len() - 1])?),
                ));
            }
            if looks_like_option_value(&args[args.len() - 1]) {
                return Err(delete_error_with(
                    &DELETE_ERROR_INVALID_INPUT,
                    format!("delete: unsupported option '{name}'"),
                ));
            }
        }
    }
    Ok((args, None))
}

fn looks_like_option_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Bool(_)
            | Value::Int(_)
            | Value::Num(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    )
}

fn option_name(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.as_str().to_owned()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data().iter().collect()),
        Value::StringArray(array) if array.data().len() == 1 => {
            Some(array.data()[0].as_str().to_owned())
        }
        _ => None,
    }
}

pub(super) fn exact_zero_one(value: &Value) -> BuiltinResult<bool> {
    crate::builtins::common::exact_logical::decode(value).map_err(|_| {
        delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: ResolveSymbolicLinks must be scalar logical or numeric 0 or 1",
        )
    })
}

pub(super) fn collect_targets(value: &Value, targets: &mut Vec<String>) -> BuiltinResult<()> {
    match value {
        Value::String(text) => push_nonempty_target(text, targets),
        Value::CharArray(array) => collect_char_array_targets(array, targets),
        Value::StringArray(array) => collect_string_array_targets(array, targets),
        Value::Cell(cell) => collect_cell_targets(cell, targets),
        _ => Err(delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: filename must be a character vector, string scalar, string array, or cell array of character vectors",
        )),
    }
}

fn collect_char_array_targets(array: &CharArray, targets: &mut Vec<String>) -> BuiltinResult<()> {
    if array.rows == 0 || array.cols == 0 {
        return Ok(());
    }
    for row in 0..array.rows {
        let mut text = String::with_capacity(array.cols);
        for col in 0..array.cols {
            text.push(array.data()[row * array.cols + col]);
        }
        let trimmed = text.trim_end().to_string();
        if trimmed.is_empty() {
            return Err(delete_error_with(
                &DELETE_ERROR_EMPTY_FILENAME,
                DELETE_ERROR_EMPTY_FILENAME.message,
            ));
        }
        targets.push(trimmed);
    }
    Ok(())
}

fn collect_string_array_targets(
    array: &StringArray,
    targets: &mut Vec<String>,
) -> BuiltinResult<()> {
    for text in array.data() {
        if text.is_empty() {
            return Err(delete_error_with(
                &DELETE_ERROR_EMPTY_FILENAME,
                DELETE_ERROR_EMPTY_FILENAME.message,
            ));
        }
        targets.push(text.as_str().to_owned());
    }
    Ok(())
}

fn collect_cell_targets(cell: &CellArray, targets: &mut Vec<String>) -> BuiltinResult<()> {
    for handle in cell.data() {
        let value = handle;
        collect_targets(value, targets)?;
    }
    Ok(())
}

fn push_nonempty_target(text: &str, targets: &mut Vec<String>) -> BuiltinResult<()> {
    if text.is_empty() {
        Err(delete_error_with(
            &DELETE_ERROR_EMPTY_FILENAME,
            DELETE_ERROR_EMPTY_FILENAME.message,
        ))
    } else {
        targets.push(text.to_string());
        Ok(())
    }
}
