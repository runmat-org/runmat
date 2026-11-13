use super::*;

pub(super) fn parse_delete_options<'a>(
    args: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<(crate::call::FunctionArgumentView<'a>, Option<bool>)> {
    if args.len() >= 2
        && option_name(args.last().expect("validated length"))?.is_some_and(|name| {
            DeleteOption::from_name(&name) == Some(DeleteOption::ResolveSymbolicLinks)
        })
    {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: ResolveSymbolicLinks requires a scalar logical or numeric value",
        ));
    }
    if args.len() >= 3 {
        if let Some(name) = option_name(&args[args.len() - 2])? {
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

fn option_name(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::CharArray(array) if array.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                delete_error_with(
                    &DELETE_ERROR_INVALID_INPUT,
                    "delete: option name contains malformed UTF-16 text",
                )
            })
            .map(Some)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        _ => Ok(None),
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
        Value::String(runmat_value::StringElement::Text(text)) => push_nonempty_target(text, targets),
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
        let start = row * array.cols;
        let text = crate::builtins::io::utf16::decode_utf16_with_error(
            &array.data()[start..start + array.cols],
            || {
                delete_error_with(
                    &DELETE_ERROR_INVALID_INPUT,
                    "delete: filename contains malformed UTF-16 text",
                )
            },
        )?;
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
        let text = text.as_str().ok_or_else(|| {
            delete_error_with(
                &DELETE_ERROR_EMPTY_FILENAME,
                "delete: missing string is not a filename",
            )
        })?;
        if text.is_empty() {
            return Err(delete_error_with(
                &DELETE_ERROR_EMPTY_FILENAME,
                DELETE_ERROR_EMPTY_FILENAME.message,
            ));
        }
        targets.push(text.to_owned());
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
