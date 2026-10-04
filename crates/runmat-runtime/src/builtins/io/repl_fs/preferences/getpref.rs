use runmat_builtins::{
    GETPREF_DESCRIPTOR, GETPREF_ERROR_TOO_MANY_OUTPUTS, GETPREF_GROUP_QUERY_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, Value};

use super::controls::{error, preference_names, scalar_text, PreferenceNames};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "getpref",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::preferences::getpref"
)]
pub(super) async fn getpref_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GETPREF_DESCRIPTOR,
        &GETPREF_ERROR_TOO_MANY_OUTPUTS,
        "getpref",
    )?;
    match args.len() {
        0 => Ok(Value::Struct(super::all_groups())),
        1 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GETPREF_GROUP_QUERY_EXTENSION,
                "getpref",
            )?;
            let group = scalar_text(&args[0], "getpref", "group")?;
            Ok(Value::Struct(super::group_values(&group)))
        }
        2 | 3 => named(&args),
        _ => Err(error("getpref", "getpref: too many input arguments")),
    }
}

fn named(args: &[Value]) -> BuiltinResult<Value> {
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
    match preferences {
        PreferenceNames::Scalar(preference) => {
            super::read_or_create(&group, &preference, args.get(2).cloned())
                .map_err(missing_preference)
        }
        PreferenceNames::Array { names, shape } => {
            let defaults = match args.get(2) {
                None => None,
                Some(Value::Cell(cell)) if cell.shape == shape => Some(cell.data.clone()),
                Some(_) => {
                    return Err(error(
                        "getpref",
                        "getpref: defaults for multiple preferences must be a cell array with matching shape",
                    ));
                }
            };
            let values = super::read_many(&group, &names, defaults.as_deref())
                .map_err(missing_preference)?;
            CellArray::new_with_shape(values, shape)
                .map(Value::Cell)
                .map_err(|reason| error("getpref", format!("getpref: {reason}")))
        }
    }
}

fn missing_preference(missing: super::MissingPreference) -> crate::RuntimeError {
    error(
        "getpref",
        format!(
            "getpref: preference '{}/{}' does not exist",
            missing.group, missing.name
        ),
    )
}
