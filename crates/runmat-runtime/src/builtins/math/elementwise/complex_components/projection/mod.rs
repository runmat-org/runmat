//! Shared execution policy for `real` and `imag`.

mod host;
mod kind;
mod provider;

use runmat_value::Value;

use crate::BuiltinResult;

pub(super) use kind::ProjectionKind;

pub(super) async fn execute(kind: ProjectionKind, value: Value) -> BuiltinResult<Value> {
    let identity = match kind {
        ProjectionKind::Real => runmat_builtins::REAL_CATALOG_ENTRY.identity,
        ProjectionKind::Imaginary => runmat_builtins::IMAG_CATALOG_ENTRY.identity,
    };
    super::preparation::execute(identity, value, |value| async move {
        match value {
            Value::GpuTensor(handle) => provider::execute(kind, handle).await,
            host_value => host::execute(kind, host_value),
        }
    })
    .await
}
