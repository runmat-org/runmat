use runmat_builtins::{ISENV_ERROR_ARITY, ISENV_ERROR_INVALID_NAME};
use runmat_value::Value;

use super::super::{error, names::*};

const IDENTITY: &str = "isenv";

pub(super) fn run(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ISENV_DESCRIPTOR,
        &runmat_builtins::ISENV_ERROR_TOO_MANY_OUTPUTS,
        IDENTITY,
    )?;
    if args.len() != 1 {
        return Err(error::builtin(IDENTITY, &ISENV_ERROR_ARITY));
    }
    let value = args.first().expect("validated one argument");
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
