use runmat_builtins::{ISENV_ERROR_ARITY, ISENV_ERROR_INVALID_NAME};
use runmat_value::Value;

use super::super::{error, names::*};

const IDENTITY: &str = "isenv";

pub(super) fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let [value] = args.as_slice() else {
        return Err(error::builtin(IDENTITY, &ISENV_ERROR_ARITY));
    };
    let names = EnvironmentNames::decode(value)
        .map_err(|_| error::builtin(IDENTITY, &ISENV_ERROR_INVALID_NAME))?;
    if names.features().next().is_some() {
        return Err(error::builtin(IDENTITY, &ISENV_ERROR_INVALID_NAME));
    }
    let values = names
        .names()
        .iter()
        .map(|name| crate::builtins::common::env::var(name).is_ok())
        .collect();
    names.logical_output(values).map_err(|detail| {
        error::message(
            IDENTITY,
            &ISENV_ERROR_INVALID_NAME,
            format!("{}: {detail}", ISENV_ERROR_INVALID_NAME.message),
        )
    })
}
