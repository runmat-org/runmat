use std::path::PathBuf;

use runmat_builtins::{
    BuiltinErrorDescriptor, RMDIR_ERROR_ARITY, RMDIR_ERROR_EMPTY_NAME, RMDIR_ERROR_FILESYSTEM,
    RMDIR_ERROR_FOLDER_TYPE, RMDIR_ERROR_OPTION,
};
use runmat_value::Value;

use crate::builtins::common::{exact_logical, fs::expand_user_path};
use crate::{build_runtime_error, BuiltinResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RemoveRequest {
    pub(super) path: PathBuf,
    pub(super) recursive: bool,
    pub(super) resolve_symbolic_links: bool,
}

pub(super) async fn parse(args: Vec<Value>) -> BuiltinResult<RemoveRequest> {
    let args = super::super::input::gather(args, "rmdir").await?;
    if !(1..=4).contains(&args.len()) {
        return Err(error(&RMDIR_ERROR_ARITY));
    }
    let folder =
        super::super::input::text_scalar(&args[0]).map_err(|_| error(&RMDIR_ERROR_FOLDER_TYPE))?;
    if folder.is_empty() {
        return Err(error(&RMDIR_ERROR_EMPTY_NAME));
    }
    let (recursive, option_start) = recursive_prefix(&args)?;
    let resolve_symbolic_links = symbolic_link_option(&args, option_start)?;
    let path = expand_user_path(&folder, "rmdir")
        .map(PathBuf::from)
        .map_err(|cause| error_with_source(&RMDIR_ERROR_FILESYSTEM, cause.into()))?;
    Ok(RemoveRequest {
        path,
        recursive,
        resolve_symbolic_links,
    })
}

fn recursive_prefix(args: &[Value]) -> BuiltinResult<(bool, usize)> {
    let Some(value) = args.get(1) else {
        return Ok((false, 1));
    };
    match super::super::input::text_scalar(value) {
        Ok(text) if text.eq_ignore_ascii_case("s") => Ok((true, 2)),
        Ok(text) if text.eq_ignore_ascii_case("ResolveSymbolicLinks") => Ok((false, 1)),
        _ => Err(error(&RMDIR_ERROR_OPTION)),
    }
}

fn symbolic_link_option(args: &[Value], start: usize) -> BuiltinResult<bool> {
    if args.len() == start {
        return Ok(false);
    }
    if args.len() != start + 2 {
        return Err(error(&RMDIR_ERROR_OPTION));
    }
    let name =
        super::super::input::text_scalar(&args[start]).map_err(|_| error(&RMDIR_ERROR_OPTION))?;
    if !name.eq_ignore_ascii_case("ResolveSymbolicLinks") {
        return Err(error(&RMDIR_ERROR_OPTION));
    }
    exact_logical::decode(&args[start + 1]).map_err(|_| error(&RMDIR_ERROR_OPTION))
}

fn error(descriptor: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    let mut builder = build_runtime_error(descriptor.message).with_builtin("rmdir");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn error_with_source(
    descriptor: &'static BuiltinErrorDescriptor,
    cause: crate::RuntimeError,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(descriptor.message)
        .with_builtin("rmdir")
        .with_source(cause);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
