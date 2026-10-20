//! Validate host-known options before body or query operands can read back.

use super::*;

pub(super) fn preflight_host_arguments(
    args: crate::call::FunctionArgumentView<'_>,
    prepared_api_key: Option<&super::super::api_key::PreparedApiKey>,
) -> BuiltinResult<super::super::request_options::PreparedRequestNames> {
    use super::super::request_options::{PreparedRequestNames, RequestOptionName};
    let mut prepared_names = PreparedRequestNames::default();
    let mut options = WebWriteOptions::default();
    let mut query = Vec::new();
    let mut index = 0;
    if let Some(Value::Struct(fields)) = args.first() {
        if let Some(prepared) = prepared_api_key {
            prepared.preflight_known(fields).map_err(|err| {
                webwrite_error_with_source(
                    &WEBWRITE_ERROR_INVALID_OPTION_VALUE,
                    format!("webwrite: {err}"),
                    err,
                )
            })?;
        }
        for (name, value) in &fields.fields {
            if prepared_api_key.is_some_and(|prepared| prepared.contains_field(name)) {
                continue;
            }
            let field = RequestOptionName::decode(name);
            prepared_names
                .struct_fields
                .push((name.to_owned(), field.clone()));
            if !crate::dispatcher::value_contains_gpu(value) {
                process_name_value_pair(&field, value, &mut options, &mut query)?;
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
            let field = RequestOptionName::decode(&name);
            prepared_names.pair_fields.push(Some(field.clone()));
            if !crate::dispatcher::value_contains_gpu(&args[index + 1]) {
                process_name_value_pair(&field, &args[index + 1], &mut options, &mut query)?;
            }
        } else {
            prepared_names.pair_fields.push(None);
        }
        index += 2;
    }
    Ok(prepared_names)
}
