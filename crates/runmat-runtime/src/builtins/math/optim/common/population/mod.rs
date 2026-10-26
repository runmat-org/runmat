//! Shared physical population/measurement and bounded callback-wave ownership.
//! Algorithms and constraint decoding remain in their actual solver domains.
use super::ScalarCooperation;
use crate::BuiltinResult;
pub(crate) mod fitness;
pub(crate) mod invocation;
pub(crate) mod storage;
pub(crate) mod projection;
pub(crate) use fitness::{Fitness, Measurement};
pub(crate) use invocation::{Counts, Invocation, Wave};
pub(crate) use storage::Population;
#[derive(Clone, Copy)]
pub(crate) struct Policy {
    pub identity: runmat_builtins::BuiltinCatalogIdentity,
    pub error: &'static runmat_builtins::BuiltinErrorDescriptor,
}
impl Policy {
    pub(crate) fn invalid(self, message: impl std::fmt::Display) -> crate::RuntimeError {
        let mut builder = crate::build_runtime_error(format!("{}: {message}", self.identity.name))
            .with_builtin(self.identity.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
        if let Some(identifier) = self.error.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
    pub(crate) fn capacity<T>(
        self,
        count: usize,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Vec<T>> {
        runmat_value::exact_capacity_vec(count).map_err(|source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:HostAllocation:Population",
                self.identity.name,
                crate::context::host_allocation::HostAllocationError::Numeric(source),
            )
        })
    }
}
