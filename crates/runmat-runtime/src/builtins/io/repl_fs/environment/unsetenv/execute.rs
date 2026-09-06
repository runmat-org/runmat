use runmat_builtins::{
    UNSETENV_ERROR_ARITY, UNSETENV_ERROR_NAME, UNSETENV_ERROR_TOO_MANY_OUTPUTS,
    UNSETENV_STATUS_OUTPUT_EXTENSION,
};
use runmat_value::Value;

use super::super::{error, mutation, names::*};

const IDENTITY: &str = "unsetenv";

pub(super) fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if requested.is_some_and(|count| count > 1) {
        return Err(error::builtin(IDENTITY, &UNSETENV_ERROR_TOO_MANY_OUTPUTS));
    }
    if requested == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &UNSETENV_STATUS_OUTPUT_EXTENSION,
            IDENTITY,
        )?;
    }
    let [value] = args.as_slice() else {
        return Err(error::builtin(IDENTITY, &UNSETENV_ERROR_ARITY));
    };
    let names = EnvironmentNames::decode(value)
        .map_err(|_| error::builtin(IDENTITY, &UNSETENV_ERROR_NAME))?;
    if names.features().next().is_some() {
        return Err(error::builtin(IDENTITY, &UNSETENV_ERROR_NAME));
    }
    let status = match mutation::removals(names.names()) {
        Ok(plan) => u8::from(mutation::apply(&plan).is_err()),
        Err(_) => 1,
    };
    Ok(match requested {
        None => Value::Num(f64::from(status)),
        Some(0) => Value::OutputList(Vec::new()),
        Some(1) => Value::OutputList(vec![Value::Num(f64::from(status))]),
        Some(_) => unreachable!("output count validated"),
    })
}
