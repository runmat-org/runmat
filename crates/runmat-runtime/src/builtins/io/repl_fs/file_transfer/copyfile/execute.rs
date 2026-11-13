use runmat_builtins::{
    COPYFILE_ERROR_DEST_ARG, COPYFILE_ERROR_FLAG_ARG, COPYFILE_ERROR_NOT_ENOUGH_INPUTS,
    COPYFILE_ERROR_SOURCE_ARG, COPYFILE_ERROR_TOO_MANY_INPUTS,
};
use runmat_value::Value;

use crate::builtins::common::fs::{contains_wildcards, expand_user_path};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

use super::super::outcome::TransferOutcome;

const BUILTIN: &str = "copyfile";

#[cfg(test)]
pub(in crate::builtins::io::repl_fs::file_transfer) async fn evaluate(
    args: &[Value],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<TransferOutcome> {
    let args = crate::call::BuiltinArguments::from_borrowed(args, runtime)?;
    evaluate_retained(args, runtime).await
}

pub(in crate::builtins::io::repl_fs::file_transfer) async fn evaluate_retained(
    args: crate::call::BuiltinArguments,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<TransferOutcome> {
    let mut cooperation = super::super::cooperation::TransferCooperation::new(
        super::super::authority::TransferOperation::Copy,
        runtime,
    );
    if args.len() < 2 {
        return Err(super::super::input::descriptor_error(
            &COPYFILE_ERROR_NOT_ENOUGH_INPUTS,
            BUILTIN,
        ));
    }
    if args.len() > 3 {
        return Err(super::super::input::descriptor_error(
            &COPYFILE_ERROR_TOO_MANY_INPUTS,
            BUILTIN,
        ));
    }
    if let Some(flag) = args.get(2).filter(|flag| !crate::value_contains_gpu(flag)) {
        super::super::input::force_flag(flag, &COPYFILE_ERROR_FLAG_ARG, BUILTIN)?;
    }
    let gathered = cooperation
        .await_operation(super::super::input::gather(&args, runtime, BUILTIN))
        .await?;
    let force = match gathered.len() {
        2 => false,
        3 => super::super::input::force_flag(&gathered[2], &COPYFILE_ERROR_FLAG_ARG, BUILTIN)?,
        0 | 1 => {
            return Err(super::super::input::descriptor_error(
                &COPYFILE_ERROR_NOT_ENOUGH_INPUTS,
                BUILTIN,
            ))
        }
        _ => {
            return Err(super::super::input::descriptor_error(
                &COPYFILE_ERROR_TOO_MANY_INPUTS,
                BUILTIN,
            ))
        }
    };
    cooperation
        .await_operation(transfer(&gathered[0], &gathered[1], force, runtime))
        .await
}

async fn transfer(
    source: &Value,
    destination: &Value,
    force: bool,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<TransferOutcome> {
    let source = super::super::input::text(source, &COPYFILE_ERROR_SOURCE_ARG, BUILTIN)?;
    if source.is_empty() {
        return Ok(super::result::empty_source());
    }
    let destination = super::super::input::text(destination, &COPYFILE_ERROR_DEST_ARG, BUILTIN)?;
    if destination.is_empty() {
        return Ok(super::result::empty_destination());
    }
    let source = expand_user_path(&source, BUILTIN).map_err(runtime_error)?;
    let destination = expand_user_path(&destination, BUILTIN).map_err(runtime_error)?;
    let mut filesystem = super::TransferFilesystem::capture(
        super::super::authority::TransferOperation::Copy,
        runtime,
    )?;
    if contains_wildcards(&source) {
        Ok(super::wildcard::copy(&mut filesystem, &source, &destination, force).await)
    } else {
        Ok(super::single::copy(&mut filesystem, &source, &destination, force).await)
    }
}

fn runtime_error(message: String) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(BUILTIN)
        .build()
}
