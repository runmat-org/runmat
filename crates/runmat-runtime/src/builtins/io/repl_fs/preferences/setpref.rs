use runmat_builtins::{
    SETPREF_DESCRIPTOR, SETPREF_ERROR_TOO_MANY_OUTPUTS, SETPREF_STATUS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::controls::{error, map_control_flow, preference_names, scalar_text, PreferenceNames};
use crate::{gather_if_needed_async, BuiltinResult};

#[runtime_builtin(
    name = "setpref",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::preferences::setpref"
)]
pub(super) async fn setpref_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &SETPREF_DESCRIPTOR,
        &SETPREF_ERROR_TOO_MANY_OUTPUTS,
        "setpref",
    )?;
    if args.len() != 3 {
        return Err(error(
            "setpref",
            "setpref: expected group, preference, and value",
        ));
    }
    if crate::output_count::current_output_count().is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SETPREF_STATUS_EXTENSION,
            "setpref",
        )?;
    }
    let group = scalar_text(&args[0], "setpref", "group")?;
    let names = preference_names(&args[1], "setpref")?;
    let entries = match names {
        PreferenceNames::Scalar(name) => vec![(name, args[2].clone())],
        PreferenceNames::Array { names, shape } => match &args[2] {
            Value::Cell(values) if values.shape == shape => {
                names.into_iter().zip(values.data.iter().cloned()).collect()
            }
            _ => {
                return Err(error(
                    "setpref",
                    "setpref: values for multiple preferences must be a cell array with matching shape",
                ));
            }
        },
    };
    let mut prepared = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        let gathered = gather_if_needed_async(&value)
            .await
            .map_err(|source| map_control_flow("setpref", source))?;
        prepared.push((name, gathered));
    }
    super::set_many(group, prepared);
    Ok(Value::Num(0.0))
}
