//! Statistics options structure helpers (`statset` / `statget`).

mod materialization;
mod schema;
#[cfg(test)]
mod tests;
mod validation;

use materialization::*;
use schema::*;
use validation::*;

use runmat_builtins::{
    BuiltinErrorDescriptor, StatOptionField, StatOptionValueKind, STATGET_ERROR_INVALID_ARGUMENT,
    STATGET_ERROR_INVALID_OPTION, STATSET_ERROR_INVALID_ARGUMENT, STATSET_ERROR_INVALID_OPTION,
    STATSET_ERROR_INVALID_STATFUN, STATSET_INTEGER_OPTION_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, StructValue, Tensor, Value};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const STATSET: &str = "statset";
const STATGET: &str = "statget";
const MAX_OPTION_INTEGER: usize = 1_000_000_000;

#[runtime_builtin(
    name = "statset",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::options"
)]
async fn statset_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_statset_output()?;
    preflight_statset(rest.view())?;
    if statset_contains_native_integer_option_value(rest.view()) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STATSET_INTEGER_OPTION_EXTENSION,
            STATSET,
        )?;
    }
    let args = gather_all(rest, runtime.as_ref()).await?;
    Ok(Value::Struct(parse_statset(runtime.as_ref(), args)?))
}

fn statset_contains_native_integer_option_value(
    args: crate::call::FunctionArgumentView<'_>,
) -> bool {
    if args.is_empty() {
        return false;
    }
    let mut index = match &args[0] {
        Value::Struct(_) => {
            if crate::builtins::common::validation::value_contains_native_integer_class(&args[0]) {
                return true;
            }
            if args
                .get(1)
                .is_some_and(|value| matches!(value, Value::Struct(_)))
            {
                if crate::builtins::common::validation::value_contains_native_integer_class(
                    &args[1],
                ) {
                    return true;
                }
                2
            } else {
                1
            }
        }
        first if looks_like_option_name(first) => 0,
        _ => 1,
    };
    while index + 1 < args.len() {
        if crate::builtins::common::validation::value_contains_native_integer_class(
            &args[index + 1],
        ) {
            return true;
        }
        index += 2;
    }
    false
}

#[runtime_builtin(
    name = "statget",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::options"
)]
async fn statget_builtin(
    options: Value,
    field: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_statget_output()?;
    if rest.len() > 1 {
        return Err(statget_error(
            &STATGET_ERROR_INVALID_ARGUMENT,
            "statget: expected at most one default value",
        ));
    }
    let Value::Struct(options) = options else {
        return Err(statget_error(
            &STATGET_ERROR_INVALID_OPTION,
            "statget: options must be a struct",
        ));
    };
    let field = text_scalar(&field).map_err(|err| {
        statget_error(
            &STATGET_ERROR_INVALID_OPTION,
            format!("statget: {}", err.message()),
        )
    })?;
    let default_data = rest.into_iter().next();
    let Some(canonical) = StatOptionField::resolve(&field) else {
        return Ok(empty_numeric());
    };
    let value = lookup_struct_field(&options, canonical.name())
        .cloned()
        .unwrap_or_else(empty_numeric);
    if is_empty_value(&value) {
        if let Some(default_data) = default_data {
            gather_if_needed_async(&default_data)
                .await
                .map_err(statget_materialization_error)
        } else {
            Ok(value)
        }
    } else {
        gather_if_needed_async(&value)
            .await
            .map_err(statget_materialization_error)
    }
}

fn ensure_statset_output() -> BuiltinResult<()> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_ARGUMENT,
            "statset: at most one output is supported",
        ));
    }
    Ok(())
}

fn ensure_statget_output() -> BuiltinResult<()> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(statget_error(
            &STATGET_ERROR_INVALID_ARGUMENT,
            "statget: at most one output is supported",
        ));
    }
    Ok(())
}

fn preflight_statset(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    let Some(first) = args.first() else {
        return Ok(());
    };
    let start = match first {
        Value::Struct(old) => {
            preflight_option_names(old)?;
            if let Some(Value::Struct(new)) = args.get(1) {
                preflight_option_names(new)?;
                2
            } else {
                1
            }
        }
        first if looks_like_option_name(first) => 0,
        _ => 1,
    };
    let rest = args.range(start..)?;
    if !rest.len().is_multiple_of(2) {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_ARGUMENT,
            "statset: expected option name-value pairs",
        ));
    }
    for pair in rest.chunks_exact(2) {
        let name = text_scalar(&pair[0])?;
        canonical_option_name(&name)?;
    }
    Ok(())
}

fn preflight_option_names(options: &StructValue) -> BuiltinResult<()> {
    for name in options.fields.keys() {
        canonical_option_name(name)?;
    }
    Ok(())
}

async fn gather_all(
    values: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    values
        .try_map_preserving_roots_async(values.len(), runtime, async |index| {
            gather_if_needed_async(&values[index])
                .await
                .map_err(statset_materialization_error)
        })
        .await
}

fn parse_statset(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<StructValue> {
    if args.is_empty() {
        return empty_options(runtime);
    }
    let mut index = 0usize;
    let mut options;
    match &args[0] {
        Value::Struct(existing) => {
            options = canonicalize_options(runtime, existing)?;
            index = 1;
            if index < args.len() {
                if let Value::Struct(newopts) = &args[index] {
                    options = merge_old_into_new(runtime, &options, newopts)?;
                    index += 1;
                }
            }
        }
        first if looks_like_option_name(first) => {
            options = empty_options(runtime)?;
        }
        first => {
            let statfun = text_scalar(first).map_err(|err| {
                statset_error(
                    &STATSET_ERROR_INVALID_ARGUMENT,
                    format!("statset: {}", err.message()),
                )
            })?;
            options = defaults_for_statfun(runtime, &statfun)?;
            index = 1;
        }
    }
    let remaining = args.view().range(index..)?;
    if !remaining.is_empty() {
        if !remaining.len().is_multiple_of(2) {
            return Err(statset_error(
                &STATSET_ERROR_INVALID_ARGUMENT,
                "statset: expected option name-value pairs",
            ));
        }
        for pair in remaining.chunks_exact(2) {
            let name = text_scalar(&pair[0]).map_err(|err| {
                statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    format!("statset: {}", err.message()),
                )
            })?;
            let canonical = canonical_option_name(&name)?;
            options = replace_option(
                runtime,
                &options,
                canonical,
                validate_option_value(canonical, &pair[1])?,
            )?;
        }
    }
    Ok(options)
}

fn text_scalar(value: &Value) -> BuiltinResult<String> {
    if let Some(text) = keyword_of(value)? {
        return Ok(text);
    }
    match value {
        Value::CharArray(array) if array.rows == 1 => {
            crate::builtins::stats::decode_utf16_with_error(array.data(), || {
                statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    "option name is not valid UTF-16 text",
                )
            })
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        other => Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("option names must be text scalars, got {other:?}"),
        )),
    }
}

fn looks_like_option_name(value: &Value) -> bool {
    text_scalar(value)
        .ok()
        .and_then(|text| StatOptionField::resolve(&text))
        .is_some()
}

fn lookup_struct_field<'a>(options: &'a StructValue, name: &str) -> Option<&'a Value> {
    options
        .fields
        .iter()
        .find(|(field, _)| field.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        Value::StringArray(array) => array.data().is_empty(),
        Value::CharArray(array) => array.data().is_empty(),
        _ => false,
    }
}

fn empty_numeric() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor"))
}

fn empty_cell() -> Value {
    Value::Cell(CellArray::new(Vec::new(), 0, 0).expect("empty cell"))
}

fn statset_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("statset:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(STATSET);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn statget_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("statget:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(STATGET);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
