//! Typed diagnostic ownership, without solver/control/callback authority.
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Policy {
    pub identity: BuiltinCatalogIdentity,
    pub invalid: &'static BuiltinErrorDescriptor,
}
impl Policy {
    pub(super) fn invalid(self, message: impl std::fmt::Display) -> RuntimeError {
        let mut builder = crate::build_runtime_error(format!("{}: {message}", self.identity.name))
            .with_builtin(self.identity.name)
            .with_gpu_gather_retry(GpuGatherRetry::Never);
        if let Some(identifier) = self.invalid.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
    pub(super) fn buffer<T>(self, capacity: usize) -> BuiltinResult<Vec<T>> {
        runmat_value::exact_capacity_vec(capacity).map_err(|source| {
            crate::build_runtime_error(format!(
                "{}: workspace allocation failed",
                self.identity.name
            ))
            .with_builtin(self.identity.name)
            .with_identifier(format!("RunMat:{}:AllocationFailed", self.identity.name))
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .with_source(source)
            .build()
        })
    }
    pub(super) fn push<T>(self, values: &mut Vec<T>, value: T) -> BuiltinResult<()> {
        if values.len() == values.capacity() {
            values.try_reserve(1).map_err(|source| {
                crate::build_runtime_error(format!(
                    "{}: workspace growth failed",
                    self.identity.name
                ))
                .with_builtin(self.identity.name)
                .with_identifier(format!("RunMat:{}:AllocationFailed", self.identity.name))
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .with_source(source)
                .build()
            })?;
        }
        values.push(value);
        Ok(())
    }
}
