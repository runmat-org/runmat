//! Narrow real square-system access to the canonical admitted LU owner.
//! Coefficients are generated directly into LU storage; no matrix mirror,
//! normal equations, inverse, replacement runtime or independent scheduler.
use super::{DenseFactor, DenseFactorError};
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use runmat_builtins::BuiltinCatalogIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NumericalFailure {
    Singular,
    NonFinite,
}

/// Opaque cause preserves the existing typed allocation/execution error chain.
#[derive(Debug)]
pub(crate) struct SystemError(DenseFactorError);
impl SystemError {
    pub(crate) fn numerical(&self) -> Option<NumericalFailure> {
        match &self.0 {
            DenseFactorError::Singular => Some(NumericalFailure::Singular),
            DenseFactorError::NonFinite => Some(NumericalFailure::NonFinite),
            _ => None,
        }
    }

    pub(crate) fn into_runtime_error(
        self,
        identity: BuiltinCatalogIdentity,
        runtime: Option<&RuntimeContext>,
    ) -> crate::RuntimeError {
        match self.0 {
            DenseFactorError::Execution(source) => {
                crate::builtins::common::map_control_flow_with_builtin(
                    source.into_runtime_error_in_context(runtime),
                    identity.name,
                )
            }
            source => {
                let _entry = runtime.map(RuntimeContext::enter);
                crate::build_runtime_error(format!(
                    "{}: square-system factorization failed",
                    identity.name
                ))
                .with_builtin(identity.name)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(SystemError(source))
                .build()
            }
        }
    }
}
impl std::fmt::Display for SystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for SystemError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

pub(crate) struct RealSquare(Box<DenseFactor<f64>>);
impl RealSquare {
    /// Async reads allow a coefficient involving a long numerical reduction
    /// to cooperate on its caller's original budget before the LU CPU job.
    pub(crate) async fn factor(
        order: usize,
        runtime: Option<&RuntimeContext>,
        mut coefficient: impl std::ops::AsyncFnMut(usize, usize) -> Result<f64, ExecutionServiceError>,
    ) -> Result<Self, SystemError> {
        DenseFactor::factor_indexed(order, runtime, async |index| {
            coefficient(index % order, index / order)
                .await
                .map_err(DenseFactorError::Execution)
        })
        .await
        .map(Self)
        .map_err(SystemError)
    }

    pub(crate) async fn solve_into(
        &mut self,
        rhs: &[f64],
        solution: &mut [f64],
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), SystemError> {
        self.0
            .solve_into(rhs, solution, runtime)
            .await
            .map_err(SystemError)
    }
}
