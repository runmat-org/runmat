//! Structured validator failures and input-count checks.

use super::{build_runtime_error, BuiltinResult, RuntimeError, Value};

pub fn validation_error(builtin: &str, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        format!("{builtin}: validation failed")
    } else {
        format!("{builtin}: {detail}")
    };
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:ValidationFailed"))
        .build()
}

pub(super) fn invalid_argument_error(builtin: &str, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        format!("{builtin}: invalid argument")
    } else {
        format!("{builtin}: {detail}")
    };
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:InvalidArgument"))
        .build()
}

pub(super) fn pass() -> BuiltinResult<Value> {
    Ok(Value::Num(0.0))
}

pub(super) fn require_arg_count(
    builtin: &str,
    args: crate::call::FunctionArgumentView<'_>,
    min: usize,
    max: usize,
) -> BuiltinResult<()> {
    if args.len() < min || args.len() > max {
        return Err(invalid_argument_error(builtin, "invalid number of inputs").into());
    }
    Ok(())
}

pub(super) fn check_validator(builtin: &str, ok: bool) -> BuiltinResult<Value> {
    if ok {
        pass()
    } else {
        Err(validation_error(builtin, "value does not satisfy validator").into())
    }
}
