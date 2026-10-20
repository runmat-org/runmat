use runmat_builtins::{
    GETPREF_DESCRIPTOR, GETPREF_ERROR_TOO_MANY_OUTPUTS, GETPREF_GROUP_QUERY_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, Value};

use super::controls::{error, preference_names, scalar_text, PreferenceNames};
use super::service;
use crate::context::{PreferenceServiceError, RuntimeContext};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runtime_builtin(
    name = "getpref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::preferences::getpref"
)]
pub(in crate::builtins::io::repl_fs) async fn getpref_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GETPREF_DESCRIPTOR,
        &GETPREF_ERROR_TOO_MANY_OUTPUTS,
        "getpref",
    )?;
    match args.len() {
        0 => {
            let (runtime, preferences) = service::required(runtime, "getpref")?;
            let groups = preferences
                .all_groups(runtime)
                .await
                .map_err(|source| service::service_error("getpref", source))?;
            Ok(Value::Struct(service::values_struct(
                groups
                    .into_iter()
                    .map(|(group, values)| (group, Value::Struct(service::values_struct(values)))),
            )))
        }
        1 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GETPREF_GROUP_QUERY_EXTENSION,
                "getpref",
            )?;
            let group = scalar_text(&args[0], "getpref", "group")?;
            let (runtime, preferences) = service::required(runtime, "getpref")?;
            let values = preferences
                .group_values(runtime, group)
                .await
                .map_err(|source| service::service_error("getpref", source))?;
            Ok(Value::Struct(service::values_struct(
                values.unwrap_or_default(),
            )))
        }
        2 | 3 => named(runtime, args.view()).await,
        _ => Err(error("getpref", "getpref: too many input arguments")),
    }
}

async fn named(
    runtime: Option<RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let group = scalar_text(&args[0], "getpref", "group")?;
    let preferences = preference_names(&args[1], "getpref")?;
    if args
        .get(2)
        .is_some_and(crate::dispatcher::value_contains_gpu)
    {
        return Err(error(
            "getpref",
            "getpref: provider-resident default values are not supported",
        ));
    }
    let (names, shape, defaults) = match preferences {
        PreferenceNames::Scalar(preference) => (
            vec![preference],
            None,
            args.get(2).cloned().map(|value| vec![value]),
        ),
        PreferenceNames::Array { names, shape } => {
            let defaults = match args.get(2) {
                None => None,
                Some(Value::Cell(cell)) if cell.shape == shape => Some(cell.data().to_vec()),
                Some(_) => {
                    return Err(error(
                        "getpref",
                        "getpref: defaults for multiple preferences must be a cell array with matching shape",
                    ));
                }
            };
            (names, Some(shape), defaults)
        }
    };
    let (runtime, preferences) = service::required(runtime, "getpref")?;
    let values = preferences
        .read_or_create_many(runtime, group, names.clone(), defaults)
        .await
        .map_err(named_service_error)?;
    if values.len() != names.len() {
        return Err(service::service_error(
            "getpref",
            PreferenceServiceError::InvalidTransaction(
                "application service returned the wrong number of preference values".into(),
            ),
        ));
    }
    match shape {
        None => values.into_iter().next().ok_or_else(|| {
            service::service_error(
                "getpref",
                PreferenceServiceError::InvalidTransaction(
                    "missing scalar preference result".into(),
                ),
            )
        }),
        Some(shape) => CellArray::new_with_shape(values, shape)
            .map(Value::Cell)
            .map_err(|reason| error("getpref", format!("getpref: {reason}"))),
    }
}

fn named_service_error(source: PreferenceServiceError) -> RuntimeError {
    match source {
        PreferenceServiceError::Missing { group, name } => {
            let cause = PreferenceServiceError::Missing {
                group: group.clone(),
                name: name.clone(),
            };
            build_runtime_error(format!(
                "getpref: preference '{group}/{name}' does not exist"
            ))
            .with_builtin("getpref")
            .with_source(cause)
            .build()
        }
        source => service::service_error("getpref", source),
    }
}
