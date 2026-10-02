//! Signal-domain error context for invocation-owned CPU cooperation.

use crate::builtins::common::validation::{self, IntegerBoundaryFailure, PreparedIntegerBoundary};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};
use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_value::Value;

pub(crate) struct SignalCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: &'static str,
}

impl SignalCooperation {
    pub(crate) fn new(runtime: Option<&RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            builtin,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_budget(budget: CpuCooperationBudget, builtin: &'static str) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            builtin,
        }
    }

    pub(crate) async fn completed(&mut self, units: u64) -> Result<(), RuntimeError> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        budget
            .checkpoint(units)
            .await
            .map_err(|source| self.service_error(source))
    }

    /// Shared direct integer preparation uses the very same invocation budget
    /// as the numerical mapper. Callers own compatibility and complex-arithmetic
    /// admission before materialization; this boundary does not gather twice.
    pub(crate) async fn prepare_integer_boundary<'a>(
        &mut self,
        value: &'a Value,
        extension: &'static BuiltinExtensionDescriptor,
        role: &str,
    ) -> Result<PreparedIntegerBoundary<'a>, RuntimeError> {
        if validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, self.builtin)?;
        }
        let prepared = validation::prepare_direct_integer_f64_boundary(value, self.budget.as_mut())
            .await
            .map_err(|error| match error {
                IntegerBoundaryFailure::Execution(source) => self.service_error(source),
                other => other.into_runtime_error(),
            })?;
        if !prepared.is_exact() {
            return Err(build_runtime_error(format!(
                "{}: integer {role} values must be exactly representable as double",
                self.builtin
            ))
            .with_builtin(self.builtin)
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build());
        }
        Ok(prepared)
    }

    fn service_error(&self, source: ExecutionServiceError) -> RuntimeError {
        let mut error = if source == ExecutionServiceError::Cancelled {
            build_runtime_error(format!("{}: execution was cancelled", self.builtin))
                .with_builtin(self.builtin)
                .with_identifier(format!(
                    "{}:Cancelled",
                    self.namespace.as_deref().unwrap_or("RunMat")
                ))
                .build()
        } else {
            let mut error = source.clone().into_runtime_error();
            error.context.builtin = Some(self.builtin.to_string());
            error
        };
        error.source = Some(Box::new(source));
        error.gpu_gather_retry = GpuGatherRetry::Never;
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::CpuCooperationPolicy;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::{IntegerStorage, Tensor};
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    #[test]
    fn integer_boundary_and_waveform_mapper_charge_one_invocation_budget() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let source = CancellationSource::new();
        let turns = Rc::new(Cell::new(0));
        let observed = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4097).unwrap()),
            source.token(),
            move || {
                observed.set(observed.get() + 1);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = SignalCooperation::with_budget(budget, "rectpuls");
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I64(vec![1; 4096]), vec![1, 4096]).unwrap(),
        );
        let prepared = futures::executor::block_on(cooperation.prepare_integer_boundary(
            &value,
            &runmat_builtins::RECTPULS_INTEGER_T_EXTENSION,
            "T",
        ))
        .unwrap();
        assert!(prepared.is_exact());
        assert_eq!(turns.get(), 0);
        futures::executor::block_on(cooperation.completed(1)).unwrap();
        assert_eq!(turns.get(), 1);
        source.cancel(CancellationReason::User);
        let failure = match futures::executor::block_on(cooperation.prepare_integer_boundary(
            &value,
            &runmat_builtins::RECTPULS_INTEGER_T_EXTENSION,
            "T",
        )) {
            Err(error) => error,
            Ok(_) => panic!("same invocation must observe cancellation"),
        };
        assert_eq!(failure.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(failure.identifier(), Some("RunMat:Cancelled"));
        assert!(std::error::Error::source(&failure).is_some());
    }
}
