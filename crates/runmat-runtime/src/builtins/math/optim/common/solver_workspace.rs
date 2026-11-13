//! Fallible ordinary host workspaces. The domain refuses finite managed
//! workspace admission before constructing these; this is not a finite claim.
use crate::{build_runtime_error, BuiltinResult};
use runmat_builtins::BuiltinCatalogIdentity;

pub(crate) fn capacity<T>(count: usize, owner: BuiltinCatalogIdentity) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(count).map_err(|source| {
        build_runtime_error(format!(
            "{}: solver workspace allocation failed",
            owner.name
        ))
        .with_builtin(owner.name)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
    })
}
pub(crate) fn push<T>(
    values: &mut Vec<T>,
    value: T,
    owner: BuiltinCatalogIdentity,
) -> BuiltinResult<()> {
    values.try_reserve(1).map_err(|source| {
        build_runtime_error(format!("{}: solver workspace growth failed", owner.name))
            .with_builtin(owner.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
    })?;
    values.push(value);
    Ok(())
}
