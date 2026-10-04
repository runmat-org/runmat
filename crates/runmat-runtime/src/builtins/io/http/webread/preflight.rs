//! Validate host-known controls before any operand can trigger a readback.

use super::*;

pub(super) fn preflight_host_arguments(args: &[Value]) -> BuiltinResult<()> {
    let mut options = WebReadOptions::default();
    let mut query = Vec::new();
    let mut index = 0;
    if let Some(Value::Struct(fields)) = args.first() {
        let key_values_known = fields
            .fields
            .iter()
            .filter(|(name, _)| {
                matches!(
                    runmat_builtins::WebOptionField::decode(name),
                    Some(
                        runmat_builtins::WebOptionField::KeyName
                            | runmat_builtins::WebOptionField::KeyValue
                    )
                )
            })
            .all(|(_, value)| !crate::dispatcher::value_contains_gpu(value));
        if key_values_known {
            super::super::api_key::from_options(fields).map_err(|err| {
                webread_error_with_source(
                    &WEBREAD_ERROR_INVALID_OPTION_VALUE,
                    format!("webread: {err}"),
                    err,
                )
            })?;
        }
        for (name, value) in &fields.fields {
            if matches!(
                runmat_builtins::WebOptionField::decode(name),
                Some(
                    runmat_builtins::WebOptionField::KeyName
                        | runmat_builtins::WebOptionField::KeyValue
                )
            ) {
                continue;
            }
            if !crate::dispatcher::value_contains_gpu(value) {
                process_name_value_pair(name, value, &mut options, &mut query)?;
            }
        }
        index = 1;
    } else if let Some(Value::Cell(cell)) = args.first() {
        if cell.cols != 2 {
            return Err(webread_error(
                "webread: cell array of query parameters must have two columns",
            ));
        }
        if !crate::dispatcher::value_contains_gpu(&args[0]) {
            append_query_from_cell(cell, &mut query)?;
        }
        index = 1;
    }
    if !(args.len() - index).is_multiple_of(2) {
        return Err(webread_error_with(
            &WEBREAD_ERROR_MISSING_OPTION_VALUE,
            "webread: missing value for name-value argument",
        ));
    }
    while index < args.len() {
        if !crate::dispatcher::value_contains_gpu(&args[index]) {
            let name = expect_string_scalar(
                &args[index],
                "webread: parameter names must be character vectors or string scalars",
            )?;
            if !crate::dispatcher::value_contains_gpu(&args[index + 1]) {
                process_name_value_pair(&name, &args[index + 1], &mut options, &mut query)?;
            }
        }
        index += 2;
    }
    Ok(())
}
