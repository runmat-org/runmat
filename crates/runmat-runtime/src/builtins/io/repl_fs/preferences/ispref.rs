use runmat_builtins::{ISPREF_DESCRIPTOR, ISPREF_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Value};

use super::controls::{error, preference_names, scalar_text, PreferenceNames};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "ispref",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::preferences::ispref"
)]
pub(super) async fn ispref_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &ISPREF_DESCRIPTOR,
        &ISPREF_ERROR_TOO_MANY_OUTPUTS,
        "ispref",
    )?;
    if args.is_empty() || args.len() > 2 {
        return Err(error(
            "ispref",
            "ispref: expected group and optional preference",
        ));
    }
    let group = scalar_text(&args[0], "ispref", "group")?;
    if args.len() == 1 {
        return Ok(Value::Bool(super::has_group(&group)));
    }
    match preference_names(&args[1], "ispref")? {
        PreferenceNames::Scalar(name) => Ok(Value::Bool(super::has_preference(&group, &name))),
        PreferenceNames::Array { names, shape } => {
            let values = names
                .iter()
                .map(|name| u8::from(super::has_preference(&group, name)))
                .collect();
            LogicalArray::new(values, shape)
                .map(Value::LogicalArray)
                .map_err(|reason| error("ispref", format!("ispref: {reason}")))
        }
    }
}
