//! Shared execution policy for `real` and `imag`.

mod host;
mod kind;
mod provider;

use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;

pub(super) use kind::ProjectionKind;

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    kind: ProjectionKind,
    value: Value,
) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    let identity = match kind {
        ProjectionKind::Real => runmat_builtins::REAL_CATALOG_ENTRY.identity,
        ProjectionKind::Imaginary => runmat_builtins::IMAG_CATALOG_ENTRY.identity,
    };
    super::preparation::execute(runtime, identity, value, |value| async move {
        let projected = match value {
            Value::GpuTensor(handle) => provider::execute(kind, handle).await,
            host_value => host::execute(kind, host_value),
        }?;
        RootedValue::new(projected).map_err(|cause| cause.into_runtime_error())
    })
    .await
}
