//! Signal-domain error context for invocation-owned CPU cooperation.

use crate::builtins::common::validation::{self, IntegerBoundaryFailure, PreparedIntegerBoundary};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};
use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_value::Value;

/// Maximum uninterrupted scalar work between checks. The execution service,
/// not this bound, determines when a task turn is yielded.
pub(crate) const SIGNAL_WORK_CHUNK: u64 = 4096;

pub(crate) struct SignalCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: &'static str,
    integer_error_descriptor: Option<&'static BuiltinErrorDescriptor>,
}

impl SignalCooperation {
    pub(crate) fn new(runtime: Option<&RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            builtin,
            integer_error_descriptor: None,
        }
    }

    pub(crate) fn with_integer_error_descriptor(
        mut self,
        descriptor: &'static BuiltinErrorDescriptor,
    ) -> Self {
        self.integer_error_descriptor = Some(descriptor);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_budget(budget: CpuCooperationBudget, builtin: &'static str) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            builtin,
            integer_error_descriptor: None,
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
        let prepared = self.prepare_integer_receipt(value, extension).await?;
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

    /// Integer-specialized numerical operations can consume the authenticated
    /// native payload without first requiring a binary64 conversion. In
    /// particular, sinc maps every nonzero integer exactly to zero.
    pub(crate) async fn prepare_integer_receipt<'a>(
        &mut self,
        value: &'a Value,
        extension: &'static BuiltinExtensionDescriptor,
    ) -> Result<PreparedIntegerBoundary<'a>, RuntimeError> {
        if validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, self.builtin)?;
        }
        let prepared = validation::prepare_direct_integer_f64_boundary(value, self.budget.as_mut())
            .await
            .map_err(|error| match error {
                IntegerBoundaryFailure::Execution(source) => self.service_error(source),
                IntegerBoundaryFailure::Operational(source) => {
                    self.boundary_operational_error(*source)
                }
                other => other.into_runtime_error(),
            })?;
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
                .with_failure_origin(
                    crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation,
                )
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

    fn boundary_operational_error(&self, mut source: RuntimeError) -> RuntimeError {
        let Some(descriptor) = self.integer_error_descriptor else {
            source.context.builtin = Some(self.builtin.to_string());
            return source;
        };
        let span = source.span;
        let mut builder =
            build_runtime_error(format!("{}: {}", descriptor.message, source.message()))
                .with_builtin(self.builtin)
                .with_source(source)
                .with_gpu_gather_retry(GpuGatherRetry::Never);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        let mut owner = builder.build();
        owner.span = span;
        owner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::CpuCooperationPolicy;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::{IntegerStorage, LogicalArray, Tensor};
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

    #[test]
    fn operational_integer_boundary_keeps_owner_identifier_and_original_cause() {
        use std::error::Error;

        let cooperation = SignalCooperation::new(None, "rectpuls")
            .with_integer_error_descriptor(&runmat_builtins::RECTPULS_ERROR_INTERNAL);
        let cause = build_runtime_error("native integer readback failed")
            .with_source(std::io::Error::other("device failure"))
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build();
        let error = cooperation.boundary_operational_error(cause);
        assert_eq!(
            error.identifier(),
            runmat_builtins::RECTPULS_ERROR_INTERNAL.identifier
        );
        assert_eq!(error.context.builtin.as_deref(), Some("rectpuls"));
        assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        assert!(error.source().and_then(|source| source.source()).is_some());
    }

    #[test]
    fn logical_sample_conversion_yields_and_observes_active_cancellation() {
        let source = Rc::new(CancellationSource::new());
        let cancel_on_turn = source.clone();
        let turns = Rc::new(Cell::new(0));
        let count = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4096).unwrap()),
            source.token(),
            move || {
                count.set(count.get() + 1);
                cancel_on_turn.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = SignalCooperation::with_budget(budget, "sinc");
        let input = Value::LogicalArray(LogicalArray::new(vec![1; 8192], vec![8192, 1]).unwrap());
        let error = futures::executor::block_on(super::super::waveform_input_tensor(
            input,
            "sinc",
            &mut cooperation,
        ))
        .expect_err("active cancellation stops logical conversion before mapping");
        assert_eq!(turns.get(), 1);
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        assert_eq!(
            error.failure_origin,
            crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation
        );
        assert!(std::error::Error::source(&error).is_some());
    }
}
