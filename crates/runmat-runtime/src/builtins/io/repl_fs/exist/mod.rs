//! Workspace, symbol, and filesystem existence queries.

use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    EXIST_DESCRIPTOR, EXIST_ERROR_NAME_ARG, EXIST_ERROR_TOO_MANY_INPUTS,
    EXIST_ERROR_TOO_MANY_OUTPUTS, EXIST_ERROR_TYPE_ARG, EXIST_NONTEXT_HANDLE_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_types::MemberAccess;
use runmat_value::Value;
mod query;
use query::{
    exist_for_query, exist_handle, parse_type_argument, value_to_string, ExistQuery,
    ExistResultKind,
};
#[cfg(test)]
use runmat_builtins::{EXIST_ERROR_INVALID_TYPE, EXIST_SEARCH_TYPE_EXTENSION};
const BUILTIN_NAME: &str = "exist";
#[cfg(test)]
pub(crate) mod tests;

fn exist_error_row(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "exist",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::exist"
)]
async fn exist_builtin(
    name: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &EXIST_DESCRIPTOR,
        &EXIST_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if rest.len() > 1 {
        return Err(exist_error_row(&EXIST_ERROR_TOO_MANY_INPUTS));
    }

    if crate::dispatcher::value_contains_gpu(&name) {
        return Err(exist_error_row(&EXIST_ERROR_NAME_ARG));
    }
    if rest
        .first()
        .is_some_and(crate::dispatcher::value_contains_gpu)
    {
        return Err(exist_error_row(&EXIST_ERROR_TYPE_ARG));
    }
    let type_value = rest.first();

    let query = type_value
        .map(parse_type_argument)
        .transpose()?
        .unwrap_or(ExistQuery::Any);

    let text = value_to_string(&name)?;
    let result = match query {
        ExistQuery::Handle if text.is_none() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EXIST_NONTEXT_HANDLE_EXTENSION,
                BUILTIN_NAME,
            )?;
            exist_handle(&name)
        }
        ExistQuery::Handle => ExistResultKind::NotFound,
        _ => {
            let text = text.ok_or_else(|| exist_error_row(&EXIST_ERROR_NAME_ARG))?;
            exist_for_query(&text, query).await?
        }
    };

    Ok(Value::Num(result.code()))
}
