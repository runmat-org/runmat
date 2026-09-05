//! Shared execution policy for `real` and `imag`.

mod host;
mod kind;
mod provider;

use runmat_value::Value;

use crate::BuiltinResult;

pub(super) use kind::ProjectionKind;

pub(super) async fn execute(kind: ProjectionKind, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => provider::execute(kind, handle).await,
        host_value => host::execute(kind, host_value),
    }
}
