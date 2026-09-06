use runmat_builtins::{
    MOVEFILE_ERROR_DEST_ARG, MOVEFILE_ERROR_FLAG_ARG, MOVEFILE_ERROR_NOT_ENOUGH_INPUTS,
    MOVEFILE_ERROR_SOURCE_ARG, MOVEFILE_ERROR_TOO_MANY_INPUTS,
};
use runmat_value::Value;

use crate::builtins::common::fs::{contains_wildcards, expand_user_path};
use crate::BuiltinResult;

use super::super::outcome::TransferOutcome;

const BUILTIN: &str = "movefile";

pub(in crate::builtins::io::repl_fs::file_transfer) async fn evaluate(
    args: &[Value],
) -> BuiltinResult<TransferOutcome> {
    let gathered = super::super::input::gather(args, BUILTIN).await?;
    let force = match gathered.as_slice() {
        [_, _] => false,
        [_, _, flag] => super::super::input::force_flag(flag, &MOVEFILE_ERROR_FLAG_ARG, BUILTIN)?,
        [] | [_] => {
            return Err(super::super::input::descriptor_error(
                &MOVEFILE_ERROR_NOT_ENOUGH_INPUTS,
                BUILTIN,
            ))
        }
        _ => {
            return Err(super::super::input::descriptor_error(
                &MOVEFILE_ERROR_TOO_MANY_INPUTS,
                BUILTIN,
            ))
        }
    };
    transfer(&gathered[0], &gathered[1], force).await
}

async fn transfer(
    source: &Value,
    destination: &Value,
    force: bool,
) -> BuiltinResult<TransferOutcome> {
    let source = super::super::input::text(source, &MOVEFILE_ERROR_SOURCE_ARG, BUILTIN)?;
    if source.is_empty() {
        return Ok(super::result::empty_source());
    }
    let destination = super::super::input::text(destination, &MOVEFILE_ERROR_DEST_ARG, BUILTIN)?;
    if destination.is_empty() {
        return Ok(super::result::empty_destination());
    }
    let source = expand_user_path(&source, BUILTIN).map_err(runtime_error)?;
    let destination = expand_user_path(&destination, BUILTIN).map_err(runtime_error)?;
    if contains_wildcards(&source) {
        Ok(super::wildcard::move_matches(&source, &destination, force).await)
    } else {
        Ok(super::single::move_path(&source, &destination, force).await)
    }
}

fn runtime_error(message: String) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(BUILTIN)
        .build()
}
