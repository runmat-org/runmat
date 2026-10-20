use runmat_builtins::{
    MOVEFILE_ERROR_DEST_ARG, MOVEFILE_ERROR_FLAG_ARG, MOVEFILE_ERROR_NOT_ENOUGH_INPUTS,
    MOVEFILE_ERROR_SOURCE_ARG, MOVEFILE_ERROR_TOO_MANY_INPUTS,
};

use crate::builtins::common::fs::{contains_wildcards, expand_user_path};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

use super::super::outcome::TransferOutcome;

const BUILTIN: &str = "movefile";

pub(in crate::builtins::io::repl_fs::file_transfer) async fn evaluate(
    args: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<TransferOutcome> {
    let mut cooperation = super::super::cooperation::TransferCooperation::new(
        super::super::authority::TransferOperation::Move,
        runtime,
    );
    if args.len() < 2 {
        return Err(super::super::input::descriptor_error(
            &MOVEFILE_ERROR_NOT_ENOUGH_INPUTS,
            BUILTIN,
        ));
    }
    if args.len() > 3 {
        return Err(super::super::input::descriptor_error(
            &MOVEFILE_ERROR_TOO_MANY_INPUTS,
            BUILTIN,
        ));
    }
    // movefile accepts host text only. Reject resident/numeric path and flag
    // values before any gather, selected-provider lookup, or filesystem effect.
    let source = super::super::input::text(&args[0], &MOVEFILE_ERROR_SOURCE_ARG, BUILTIN)?;
    let destination = super::super::input::text(&args[1], &MOVEFILE_ERROR_DEST_ARG, BUILTIN)?;
    let force = args
        .get(2)
        .map(|flag| super::super::input::force_flag(flag, &MOVEFILE_ERROR_FLAG_ARG, BUILTIN))
        .transpose()?
        .unwrap_or(false);
    cooperation
        .await_operation(transfer(source, destination, force, runtime))
        .await
}

async fn transfer(
    source: String,
    destination: String,
    force: bool,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<TransferOutcome> {
    if source.is_empty() {
        return Ok(super::result::empty_source());
    }
    if destination.is_empty() {
        return Ok(super::result::empty_destination());
    }
    let source = expand_user_path(&source, BUILTIN).map_err(runtime_error)?;
    let destination = expand_user_path(&destination, BUILTIN).map_err(runtime_error)?;
    let mut filesystem = super::super::authority::TransferFilesystem::capture(
        super::super::authority::TransferOperation::Move,
        runtime,
    )?;
    if contains_wildcards(&source) {
        super::wildcard::move_matches(&mut filesystem, &source, &destination, force).await
    } else {
        super::single::move_path(&mut filesystem, &source, &destination, force).await
    }
}

fn runtime_error(message: String) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(BUILTIN)
        .build()
}
