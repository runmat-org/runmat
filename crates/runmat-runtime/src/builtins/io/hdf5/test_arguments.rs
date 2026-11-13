//! Ordinary direct-test inputs enter the retained API once, in their context.
use super::*;
use crate::context::RuntimeContext;

pub(super) async fn h5read_builtin(
    runtime: Option<RuntimeContext>,
    filename: Value,
    dataset: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, runtime.as_ref())?;
    super::h5read_builtin(runtime, filename, dataset, rest).await
}

pub(super) async fn hdf5read_builtin(
    runtime: Option<RuntimeContext>,
    filename: Value,
    dataset: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, runtime.as_ref())?;
    super::hdf5read_builtin(runtime, filename, dataset, rest).await
}

pub(super) async fn h5write_builtin(
    runtime: Option<RuntimeContext>,
    filename: Value,
    dataset: Value,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, runtime.as_ref())?;
    super::h5write_builtin(runtime, filename, dataset, data, rest).await
}

pub(super) async fn hdf5write_builtin(
    runtime: Option<RuntimeContext>,
    filename: Value,
    dataset: Value,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, runtime.as_ref())?;
    super::hdf5write_builtin(runtime, filename, dataset, data, rest).await
}
