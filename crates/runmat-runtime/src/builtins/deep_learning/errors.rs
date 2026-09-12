use super::contracts::{ERROR_INVALID_INPUT, ERROR_UNSUPPORTED};
use super::*;

pub(in crate::builtins) fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

pub(in crate::builtins) async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed_async(&value).await?);
    }
    Ok(gathered)
}

pub(in crate::builtins) fn deep_learning_error(
    function: &'static str,
    message: impl Into<String>,
) -> RuntimeError {
    descriptor_error(function, message, &ERROR_INVALID_INPUT)
}

pub(in crate::builtins) fn unsupported_error(
    function: &'static str,
    message: impl Into<String>,
) -> RuntimeError {
    descriptor_error(function, message, &ERROR_UNSUPPORTED)
}

pub(super) fn descriptor_error(
    function: &'static str,
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(function);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
