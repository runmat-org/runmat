use runmat_builtins::{ISPREF_DESCRIPTOR, ISPREF_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Value};

use super::controls::{error, preference_names, scalar_text, PreferenceNames};
use super::service;
use crate::context::{PreferenceServiceError, RuntimeContext};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "ispref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::preferences::ispref"
)]
pub(in crate::builtins::io::repl_fs) async fn ispref_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
        let (runtime, preferences) = service::required(runtime, "ispref")?;
        return Ok(Value::Bool(
            preferences
                .group_values(runtime, group)
                .await
                .map_err(|source| service::service_error("ispref", source))?
                .is_some(),
        ));
    }
    let (names, shape) = match preference_names(&args[1], "ispref")? {
        PreferenceNames::Scalar(name) => (vec![name], None),
        PreferenceNames::Array { names, shape } => (names, Some(shape)),
    };
    let expected = names.len();
    let (runtime, preferences) = service::required(runtime, "ispref")?;
    let contained = preferences
        .contains_names(runtime, group, names)
        .await
        .map_err(|source| service::service_error("ispref", source))?;
    if contained.len() != expected {
        return Err(service::service_error(
            "ispref",
            PreferenceServiceError::InvalidTransaction(
                "application service returned the wrong number of existence flags".into(),
            ),
        ));
    }
    match shape {
        None => Ok(Value::Bool(contained[0])),
        Some(shape) => {
            let values = contained.into_iter().map(u8::from).collect();
            LogicalArray::new(values, shape)
                .map(Value::LogicalArray)
                .map_err(|reason| error("ispref", format!("ispref: {reason}")))
        }
    }
}
